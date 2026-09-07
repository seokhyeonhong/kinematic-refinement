import gc
import os, argparse, tqdm, time
import torch
import torch.nn.functional as F
from mypath import *
from kinref.geo_dataset import PairedDataset, PairedGraph_collate_fn, get_mi_src_tgt_all_graph
from kinref.geo_model import out_post_fwd
from kinref.geo_test import prepare_model_test
from kinref.metric import compute_metric
from utils import geo_utils


def is_oom_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_pen_ratio_depth(out, gt, threshold=1e-8):
    pen_disp, pen_norm = geo_utils.compute_chamfer_normal_collision(
        out["vp"], out["vn"], gt["vidx_part"], gt["v_mask"]
    )
    pen = F.relu((pen_disp * pen_norm).sum(dim=-1))
    valid_mask = ~gt["v_mask"]
    pen_mask = (pen > threshold) & valid_mask
    valid_count = valid_mask.sum().clamp_min(1)
    return pen_mask.to(dtype=pen.dtype).sum() / valid_count, pen.mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", type=str, default="geo")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="test_mesh_ucum/motion/processed")
    parser.add_argument("--oom_frame_chunk_size", type=int, default=16)
    args = parser.parse_args()

    if args.oom_frame_chunk_size < 1:
        raise ValueError("--oom_frame_chunk_size must be >= 1")

    model, cfg, ms_dict = prepare_model_test(args.model_epoch, args.device)

    # Dataset
    ds = PairedDataset()
    data_dir = os.path.join(DATA_DIR, args.data_dir)
    ds.load_data_dir_pairs(data_dir)

    # metric_key = []
    metric_key = ["qR", "ra_xz", "pa", "slide"]
    metric = {key: 0.0 for key in metric_key}
    metric["pen_ratio"] = 0.0
    metric["pen_depth"] = 0.0
    cnt = 0
    oom_state = {
        "oom_cnt": 0,
        "skipped_frame_cnt": 0,
    }

    def compute_batch_metric(src_batch, tgt_batch, consq_n):
        _, _, hatD = model(
            src_batch,
            tgt_batch,
            ms_dict=ms_dict,
            out_rep_cfg=cfg["representation"]["out"],
            consq_n=consq_n,
        )
        out, gt = out_post_fwd(
            {"hatD": hatD},
            tgt_batch,
            ms_dict,
            cfg["representation"]["out"],
            consq_n,
        )

        mi_metric = compute_metric(metric_key, out, gt)
        pen_ratio, pen_depth = compute_pen_ratio_depth(out, gt)
        mi_metric["pen_ratio"] = pen_ratio * 100
        mi_metric["pen_depth"] = pen_depth * 100
        return mi_metric

    def compute_err(mi, src_ri, tgt_ri):
        # https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/retargeting/get_error.py#L47-L55
        (src_batch, tgt_batch), consq_n = get_mi_src_tgt_all_graph(
            dataset=ds, mi=mi, src_ri=src_ri, tgt_ri=tgt_ri, device=args.device
        )
        return compute_batch_metric(src_batch, tgt_batch, consq_n)

    def compute_err_frame_chunk(mi, src_ri, tgt_ri, frame_start, frame_end):
        batch = [
            ds[mi, src_ri, tgt_ri, frame]
            for frame in range(frame_start, frame_end)
        ]
        src_batch, tgt_batch = PairedGraph_collate_fn(batch, device=args.device)
        return compute_batch_metric(src_batch, tgt_batch, frame_end - frame_start)

    def compute_err_by_frame_chunks(mi, src_ri, tgt_ri):
        frame_cnt = ds.frame_cnts[ds.mi_ri_2_fi[mi][src_ri]]
        metric_sum = {key: 0.0 for key in metric_key + ["pen_ratio", "pen_depth"]}
        valid_frame_cnt = 0

        for frame_start in range(0, frame_cnt, args.oom_frame_chunk_size):
            frame_end = min(frame_start + args.oom_frame_chunk_size, frame_cnt)
            chunk_n = frame_end - frame_start
            split_chunk = False
            skip_chunk = False
            try:
                chunk_metric = compute_err_frame_chunk(mi, src_ri, tgt_ri, frame_start, frame_end)
            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise

                if chunk_n > 1:
                    split_chunk = True
                else:
                    oom_state["skipped_frame_cnt"] += 1
                    print(
                        f"[OOM] skip frame "
                        f"(mi={mi}, src_ri={src_ri}, tgt_ri={tgt_ri}, frame={frame_start})"
                    )
                    skip_chunk = True

            if skip_chunk:
                clear_cuda_cache()
                continue

            if split_chunk:
                print(
                    f"[OOM] split chunk into frames "
                    f"(mi={mi}, src_ri={src_ri}, tgt_ri={tgt_ri}, "
                    f"frames={frame_start}:{frame_end})"
                )
                clear_cuda_cache()
                for frame in range(frame_start, frame_end):
                    try:
                        chunk_metric = compute_err_frame_chunk(mi, src_ri, tgt_ri, frame, frame + 1)
                    except RuntimeError as frame_exc:
                        if not is_oom_error(frame_exc):
                            raise
                        oom_state["skipped_frame_cnt"] += 1
                        print(
                            f"[OOM] skip frame "
                            f"(mi={mi}, src_ri={src_ri}, tgt_ri={tgt_ri}, frame={frame})"
                        )
                        clear_cuda_cache()
                        continue
                    for key in metric_key + ["pen_ratio", "pen_depth"]:
                        metric_sum[key] += chunk_metric[key].detach().item()
                    valid_frame_cnt += 1
                    clear_cuda_cache()
                continue

            for key in metric_key + ["pen_ratio", "pen_depth"]:
                metric_sum[key] += chunk_metric[key].detach().item() * chunk_n
            valid_frame_cnt += chunk_n
            clear_cuda_cache()

        if valid_frame_cnt == 0:
            raise RuntimeError(
                f"All fallback frames were skipped due to OOM: "
                f"mi={mi}, src_ri={src_ri}, tgt_ri={tgt_ri}"
            )

        return {
            key: torch.tensor(metric_sum[key] / valid_frame_cnt, device=args.device)
            for key in metric_key + ["pen_ratio", "pen_depth"]
        }

    def compute_err_with_oom_fallback(mi, src_ri, tgt_ri):
        use_frame_fallback = False
        try:
            return compute_err(mi, src_ri, tgt_ri)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise

            oom_state["oom_cnt"] += 1
            use_frame_fallback = True
            # print(
            #     f"[OOM] mi={mi}, src_ri={src_ri}, tgt_ri={tgt_ri}. "
            #     f"retry frame chunks (chunk_size={args.oom_frame_chunk_size})."
            # )

        if use_frame_fallback:
            clear_cuda_cache()
            return compute_err_by_frame_chunks(mi, src_ri, tgt_ri)

    def add_metric(mi_metric):
        for key in metric_key + ["pen_ratio", "pen_depth"]:
            metric[key] += mi_metric[key].detach().item()

    st = time.time()
    # with torch.inference_mode():
    for mi in tqdm.tqdm(range(len(ds.mi_ri_2_fi)), ncols=100):
        R = len(ds.mi_ri_2_fi[mi])
        # Comparison with <Skeleton-Aware Networks for Deep Motion Retargeting>
        # cross: BigVegas -> Goblin_m, Mousey_m, Mremireh_m, Vampire_m
        # internal: Goblin_m, Mousey_m, Mremireh_m, Vampire_m <->
        # https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/retargeting/test.py
        for tgt_ri in range(1, R):
            mi_metric = compute_err_with_oom_fallback(mi, 0, tgt_ri)
            add_metric(mi_metric)
            cnt += 1

    for key in metric_key + ["pen_ratio", "pen_depth"]:
        print(f"{key}: {metric[key]/cnt:.4f}", end="\t")
    if oom_state["oom_cnt"] > 0:
        print(f"oom: {oom_state['oom_cnt']}", end="\t")
    if oom_state["skipped_frame_cnt"] > 0:
        print(f"skipped_frames: {oom_state['skipped_frame_cnt']}", end="\t")

    metric_fp = os.path.join(PRJ_DIR, f"table2_geometry_{args.data_dir.split('/')[0]}.txt")
    if not os.path.exists(metric_fp):
        with open(metric_fp, "w") as f:
            line = "model_epoch,\t"
            for key in metric_key + ["pen_ratio", "pen_depth"]:
                line += f"{key},\t"
            f.write(line + "\n")
    with open(metric_fp, "a") as f:
        line = f"{args.model_epoch},\t"
        for key in metric_key + ["pen_ratio", "pen_depth"]:
            line += f"{metric[key]/cnt:.4f},\t"
        f.write(line + "\n")
