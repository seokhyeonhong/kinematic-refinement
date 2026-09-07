import os, argparse, tqdm, time
import torch
from mypath import *
from kinref.kin_test import prepare_model_test
from kinref.kin_dataset import PairedDataset, get_mi_src_tgt_all_graph
from kinref.kin_model import out_post_fwd
from kinref.metric import compute_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_epoch",
        type=str,
        default="kin",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="test_kin_fixed_sc/motion/processed")
    args = parser.parse_args()

    model, cfg, ms_dict = prepare_model_test(args.model_epoch, args.device)

    # Dataset
    ds = PairedDataset()
    data_dir = os.path.join(DATA_DIR, args.data_dir)
    ds.load_data_dir_pairs(data_dir)

    metric_key = ["qR", "ra_xz", "pa", "slide"]
    metric = {key: 0.0 for key in metric_key}
    cnt = 0

    def compute_err(mi, src_ri, tgt_ri):
        # Metric from <Skeleton-Aware Networks for Deep Motion Retargeting>
        # https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/retargeting/get_error.py#L47-L55
        (src_batch, tgt_batch), consq_n = get_mi_src_tgt_all_graph(
            dataset=ds, mi=mi, src_ri=src_ri, tgt_ri=tgt_ri, device=args.device
        )
        
        z, hatD = model(src_batch, tgt_batch)
        out, gt = out_post_fwd(
            {"hatD": hatD, "z": z},
            tgt_batch,
            ms_dict,
            cfg["representation"]["out"],
            consq_n,
        )
        mi_metric = compute_metric(metric_key, out, gt)
        for key in metric_key:
            metric[key] += mi_metric[key].detach().item()

    st = time.time()
    for mi in tqdm.tqdm(range(len(ds.mi_ri_2_fi)), ncols=100):
        R = len(ds.mi_ri_2_fi[mi])
        # Comparison with <Skeleton-Aware Networks for Deep Motion Retargeting>
        # cross: BigVegas -> Goblin_m, Mousey_m, Mremireh_m, Vampire_m
        # internal: Goblin_m, Mousey_m, Mremireh_m, Vampire_m <->
        # https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/retargeting/test.py
        for tgt_ri in range(1, R):
            compute_err(mi, 0, tgt_ri)
            cnt += 1

    for key in metric_key:
        print(f"{key}: {metric[key]/cnt:.4f}", end="\t")

    metric_fp = os.path.join(PRJ_DIR, f"table2_{args.data_dir.split('/')[0]}.txt")
    if not os.path.exists(metric_fp):
        with open(metric_fp, "w") as f:
            line = "model_epoch,\t"
            for key in metric_key:
                line += f"{key},\t"
            f.write(line + "\n")
    with open(metric_fp, "a") as f:
        line = f"{args.model_epoch},\t"
        for key in metric_key:
            line += f"{metric[key]/cnt:.4f},\t"
        f.write(line + "\n")
