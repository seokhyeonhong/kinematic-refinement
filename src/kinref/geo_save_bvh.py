import copy
import gc
import argparse
import os
from tqdm import tqdm
from pathlib import Path
from utils import tensor_utils
import numpy as np
import torch

from mypath import *
from kinref.geo_model import make_load_model
from kinref.geo_dataset import PairedDataset, PairedGraph_collate_fn, get_mi_src_tgt_all_graph
from kinref.skel_pose_mesh_graph import SkelPoseMeshGraph
from conversions.graph_to_motion import graph_2_skel
from fairmotion.core import motion as motion_class
from utils.motion_utils import simple_stitch


def prepare_model_test(model_epoch, device):
    # device, printoptions
    tensor_utils.set_device(device)
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    model, cfg = make_load_model(model_epoch, device)
    model.eval()

    load_dir = os.path.join(RESULT_DIR, model_epoch.split("/")[0])
    ms_dict = torch.load(os.path.join(load_dir, "ms_dict.pt"))

    # set SkelPoseGraph class variables
    SkelPoseMeshGraph.skel_cfg = cfg["representation"]["skel"]
    SkelPoseMeshGraph.pose_cfg = cfg["representation"]["pose"]
    SkelPoseMeshGraph.mesh_cfg = cfg["representation"]["mesh"]
    SkelPoseMeshGraph.ms_dict = ms_dict

    return model, cfg, ms_dict


""" ================= basic functions commonly needed for tasks ================= """
from conversions.graph_to_motion import gt_recon_motion, hatD_recon_motion
from conversions.motion_to_graph import bvh_2_graph, skel_2_graph
# from torch_geometric.data import Batch

from fairmotion.data.bvh import save as save_bvh


def get_char_motion_from_fi(ds, fi):
    fp = Path(ds.filepaths[fi])
    char_name = fp.parent.name
    motion_name = fp.stem
    return char_name, motion_name


def infer_data_name_from_bvh_path(bvh_path):
    parts = Path(bvh_path).parts
    if "motion" not in parts:
        raise ValueError(f"Cannot infer DATA_NAME from path (missing 'motion'): {bvh_path}")
    motion_i = parts.index("motion")
    if motion_i == 0:
        raise ValueError(f"Cannot infer DATA_NAME from path: {bvh_path}")
    return parts[motion_i - 1]


def load_joint_names_from_char_txt(bvh_path, char_name):
    data_name = infer_data_name_from_bvh_path(bvh_path)
    char_txt = Path(DATA_DIR) / data_name / "character" / "joint_pos" / f"{char_name}.txt"
    if not char_txt.exists():
        raise FileNotFoundError(f"Character joint file not found: {char_txt}")

    joint_names = []
    with open(char_txt, "r") as file:
        for line in file:
            if line.strip() == "":
                continue
            jname = line.split(",")[0].strip().replace("_END", "_End")
            joint_names.append(jname)
    return joint_names


def apply_joint_names(motion, joint_names, bvh_path, char_name):
    n_motion_joints = motion.skel.num_joints()
    if len(joint_names) != n_motion_joints:
        raise ValueError(
            f"Joint count mismatch for {char_name} ({bvh_path}): "
            f"txt={len(joint_names)} vs motion={n_motion_joints}"
        )
    for ji, joint_name in enumerate(joint_names):
        motion.skel.change_joint_name(ji, joint_name)

def is_oom_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def retarget(model, src_batch, tgt_batch, ms_dict, out_rep_cfg, consq_n):
    # src ground truth
    src_motion_list, src_contact_list = gt_recon_motion(src_batch, consq_n)
    # predicted result
    _, _, hatD = model(src_batch, tgt_batch, ms_dict, out_rep_cfg, consq_n)
    out_motion_list, out_contact_list = hatD_recon_motion(
        hatD, tgt_batch, out_rep_cfg, ms_dict, consq_n
    )

    # when tgt ground-truth motion is available
    if hasattr(tgt_batch, "q"):
        tgt_motion_list, tgt_contact_list = gt_recon_motion(tgt_batch, consq_n)
        return src_motion_list[0], tgt_motion_list[0], out_motion_list[0]
    else:
        tgt_skel = graph_2_skel(tgt_batch, 1)[0]
        tgt_motion = motion_class.Motion(skel=tgt_skel)
        tpose = np.eye(4)[None, ...].repeat(tgt_skel.num_joints(), 0)
        tpose[0, 1, 3] = tgt_batch.go[0, 1]  # root height
        tgt_motion.add_one_frame(tpose)

        return src_motion_list[0], tgt_motion, out_motion_list[0]

def stitch_motion_segments(motion_segments):
    if len(motion_segments) == 0:
        raise ValueError("motion_segments must be non-empty")

    stitched_motion = copy.deepcopy(motion_segments[0])
    for next_motion in motion_segments[1:]:
        aligned_next_poses = simple_stitch(
            stitched_motion.poses[-1],
            copy.deepcopy(next_motion.poses),
        )
        for pose in aligned_next_poses:
            stitched_motion.add_one_frame(copy.deepcopy(pose.data))
    return stitched_motion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", type=str, default="geo")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="additional_data_mixamo/motion/processed/")
    parser.add_argument("--oom_frame_chunk_size", type=int, default=16)
    args = parser.parse_args()

    if args.oom_frame_chunk_size < 1:
        raise ValueError("--oom_frame_chunk_size must be >= 1")

    model, cfg, ms_dict = prepare_model_test(args.model_epoch, args.device)

    # `Data`set
    ds = PairedDataset()
    ds.load_data_dir_pairs(
        os.path.join(DATA_DIR, args.data_dir),
    )


    save_root = Path(f"./save_bvh_{args.data_dir.split('/')[0]}")
    save_root.mkdir(parents=True, exist_ok=True)
    
    src_root = save_root / "Source"
    tgt_root = save_root / "Target"
    out_root = save_root / str(args.model_epoch)
    src_root.mkdir(parents=True, exist_ok=True)
    tgt_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    model_device = next(model.parameters()).device

    def run_retarget_batch(src_batch, tgt_batch, consq_n):
        return retarget(
            model,
            src_batch,
            tgt_batch,
            ms_dict,
            out_rep_cfg=cfg["representation"]["out"],
            consq_n=consq_n,
        )

    def run_retarget(mi, src_ri, tgt_ri):
        (src_batch, tgt_batch), consq_n = get_mi_src_tgt_all_graph(
            dataset=ds, mi=mi, src_ri=src_ri, tgt_ri=tgt_ri, device=args.device
        )
        return run_retarget_batch(src_batch, tgt_batch, consq_n)

    def run_retarget_frame_chunk(mi, src_ri, tgt_ri, frame_start, frame_end):
        batch = [
            ds[mi, src_ri, tgt_ri, frame]
            for frame in range(frame_start, frame_end)
        ]
        src_batch, tgt_batch = PairedGraph_collate_fn(batch, device=args.device)
        return run_retarget_batch(src_batch, tgt_batch, frame_end - frame_start)

    def run_retarget_by_frame_chunks(mi, src_ri, tgt_ri):
        frame_cnt = ds.frame_cnts[ds.mi_ri_2_fi[mi][0]]
        src_segments = []
        tgt_segments = []
        out_segments = []

        for frame_start in range(0, frame_cnt, args.oom_frame_chunk_size):
            frame_end = min(frame_start + args.oom_frame_chunk_size, frame_cnt)
            chunk_n = frame_end - frame_start
            split_chunk = False

            try:
                src_motion, tgt_motion, out_motion = run_retarget_frame_chunk(
                    mi, src_ri, tgt_ri, frame_start, frame_end
                )
                src_segments.append(src_motion)
                tgt_segments.append(tgt_motion)
                out_segments.append(out_motion)
            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise
                if chunk_n > 1:
                    split_chunk = True
                else:
                    print(
                        f"[OOM] skip pair "
                        f"(mi={mi}, src_ri={src_ri}, tgt_ri={tgt_ri}, frame={frame_start})"
                    )
                    clear_cuda_cache()
                    return None

            if not split_chunk:
                clear_cuda_cache()
                continue

            print(
                f"[OOM] split chunk into frames "
                f"(mi={mi}, src_ri={src_ri}, tgt_ri={tgt_ri}, "
                f"frames={frame_start}:{frame_end})"
            )
            clear_cuda_cache()

            for frame in range(frame_start, frame_end):
                try:
                    src_motion, tgt_motion, out_motion = run_retarget_frame_chunk(
                        mi, src_ri, tgt_ri, frame, frame + 1
                    )
                except RuntimeError as frame_exc:
                    if not is_oom_error(frame_exc):
                        raise
                    print(
                        f"[OOM] skip pair "
                        f"(mi={mi}, src_ri={src_ri}, tgt_ri={tgt_ri}, frame={frame})"
                    )
                    clear_cuda_cache()
                    return None

                src_segments.append(src_motion)
                tgt_segments.append(tgt_motion)
                out_segments.append(out_motion)
                clear_cuda_cache()

        return (
            stitch_motion_segments(src_segments),
            stitch_motion_segments(tgt_segments),
            stitch_motion_segments(out_segments),
        )

    def run_retarget_with_oom_fallback(mi, src_ri, tgt_ri):
        try:
            return run_retarget(mi, src_ri, tgt_ri)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            if model_device.type != "cuda":
                raise

            print(
                f"[OOM] mi={mi}, src_ri={src_ri}, tgt_ri={tgt_ri}. "
                f"retry frame chunks (chunk_size={args.oom_frame_chunk_size})"
            )
            clear_cuda_cache()
            return run_retarget_by_frame_chunks(mi, src_ri, tgt_ri)

    def retarget_mi(mi):
        R = len(ds.mi_ri_2_fi[mi])
        for src_ri in tqdm(range(R), ncols=100, desc=f"MI {mi}"):
            for tgt_ri in range(R):
                src_fi = ds.mi_ri_2_fi[mi][src_ri]
                tgt_fi = ds.mi_ri_2_fi[mi][tgt_ri]
                src_char, motion = get_char_motion_from_fi(ds, src_fi)
                tgt_char, _ = get_char_motion_from_fi(ds, tgt_fi)

                src_bvh_path = ds.filepaths[src_fi]
                tgt_bvh_path = ds.filepaths[tgt_fi]
                # assert src_bvh_path == tgt_bvh_path, f"Source and target BVH paths do not match: {src_bvh_path} vs {tgt_bvh_path}"

                pair_dir = f"{src_char}2{tgt_char}"
                src_pair_dir = src_root / pair_dir
                tgt_pair_dir = tgt_root / pair_dir
                out_pair_dir = out_root / pair_dir
                src_pair_dir.mkdir(parents=True, exist_ok=True)
                tgt_pair_dir.mkdir(parents=True, exist_ok=True)
                out_pair_dir.mkdir(parents=True, exist_ok=True)
                
                src_bvh_file = src_pair_dir / f"{motion}.bvh"
                tgt_bvh_file = tgt_pair_dir / f"{motion}.bvh"
                out_bvh_file = out_pair_dir / f"{motion}.bvh"

                # if src_bvh_file.exists() and tgt_bvh_file.exists() and out_bvh_file.exists():
                #     continue
                # if tgt_char == "YBot":
                #     continue

                retarget_result = run_retarget_with_oom_fallback(mi, src_ri, tgt_ri)
                if retarget_result is None:
                    continue
                src_motion, tgt_motion, out_motion = retarget_result


                src_joint_names = load_joint_names_from_char_txt(src_bvh_path, src_char)
                tgt_joint_names = load_joint_names_from_char_txt(tgt_bvh_path, tgt_char)
    
                apply_joint_names(src_motion, src_joint_names, src_bvh_path, src_char)
                apply_joint_names(tgt_motion, tgt_joint_names, tgt_bvh_path, tgt_char)
                apply_joint_names(out_motion, tgt_joint_names, tgt_bvh_path, tgt_char)

                src_bvh_file = src_pair_dir / f"{motion}.bvh"
                tgt_bvh_file = tgt_pair_dir / f"{motion}.bvh"
                out_bvh_file = out_pair_dir / f"{motion}.bvh"

                if not src_bvh_file.exists():
                    save_bvh(src_motion, str(src_bvh_file))
                if not tgt_bvh_file.exists():
                    save_bvh(tgt_motion, str(tgt_bvh_file))
                if not out_bvh_file.exists():
                    save_bvh(out_motion, str(out_bvh_file))


    for mi in tqdm(range(len(ds.mi_ri_2_fi)), ncols=100, desc="All MIs"):
        retarget_mi(mi)
