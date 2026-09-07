import argparse
from utils import tensor_utils
import numpy as np
import torch
from mypath import *
from kinref.kin_model import make_load_model
from kinref.kin_dataset import PairedDataset, get_mi_src_tgt_all_graph
from kinref.skel_pose_graph import SkelPoseGraph, SkelPoseBatch, rnd_mask
from conversions.graph_to_motion import graph_2_skel
from fairmotion.core import motion as motion_class


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
    SkelPoseGraph.skel_cfg = cfg["representation"]["skel"]
    SkelPoseGraph.pose_cfg = cfg["representation"]["pose"]
    SkelPoseGraph.ms_dict = ms_dict

    return model, cfg, ms_dict


""" ================= basic functions commonly needed for tasks ================= """
from conversions.graph_to_motion import gt_recon_motion, hatD_recon_motion
from conversions.motion_to_graph import bvh_2_graph, skel_2_graph
# from torch_geometric.data import Batch


def retarget(model, src_batch, tgt_batch, ms_dict, out_rep_cfg, consq_n):
    # src ground truth
    src_motion_list, src_contact_list = gt_recon_motion(src_batch, consq_n)
    # predicted result
    z, hatD = model(src_batch, tgt_batch)
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


##### motion to z #####
def encode_graph(model, graph_batch):
    return model.encoder(graph_batch)


def bvh_2_graph_z(model, bvh_filepath):
    graph_batch = bvh_2_graph(bvh_filepath).to(device=model.device)
    z = encode_graph(model, graph_batch)
    motion_list, contact_list = gt_recon_motion(graph_batch, len(z))
    return motion_list[0], graph_batch, z


##### z to motion #####
def decode_z_skel(model, z, skel, ms_dict):
    return decode_z_skelgraph(model, z, skel_2_graph(skel), ms_dict)


def decode_z_skelgraph(model, z, skel_graph, ms_dict):
    # B_skel_graph = Batch.from_data_list([skel_graph] * len(z)).to(device=model.device)
    B_skel_graph = SkelPoseBatch.from_data_list([skel_graph] * len(z)).to(device=model.device)
    hatD = model.decoder(z, B_skel_graph)
    out_motion_list, out_contact_list = hatD_recon_motion(
        hatD, B_skel_graph, model.rep_cfg["out"], ms_dict, len(z)
    )
    return out_motion_list[0], out_contact_list[0]


##### convert all bvh to z and save as npy #####
def list_bvh_files(directory):
    bvh_files = []
    for root, dirs, files in os.walk(directory):
        if not dirs:  # leaf directory
            relative_path = os.path.relpath(root, directory)
            for file in files:
                if file.endswith(".bvh"):
                    bvh_files.append(os.path.join(relative_path, file))
    return bvh_files


import tqdm, gc


def save_bvh_z(model_epoch, bvh_dir, npy_dir):
    model, cfg, ms_dict = prepare_model_test(model_epoch, "cuda:0")
    bvh_files = list_bvh_files(bvh_dir)
    for bvh_rel_fn in tqdm.tqdm(bvh_files):
        bvh_fp = os.path.join(bvh_dir, bvh_rel_fn)
        npy_fp = os.path.join(npy_dir, bvh_rel_fn[:-4] + ".npy")
        if not os.path.exists(os.path.dirname(npy_fp)):
            os.makedirs(os.path.dirname(npy_fp))
        motion, graph_batch, z = bvh_2_graph_z(model, bvh_fp)
        # print(bvh_fp, npy_fp)
        np.save(npy_fp, z.cpu().detach().numpy())
        del motion, graph_batch, z
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", type=str, default="kin")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="test_kin_fixed_sc/motion/processed/")
    parser.add_argument("--rnd_tgt", type=int, default=0)
    parser.add_argument("--src_mask", type=int, default=1)
    parser.add_argument("--tgt_mask", type=int, default=1)

    args = parser.parse_args()

    model, cfg, ms_dict = prepare_model_test(args.model_epoch, args.device)

    # `Data`set
    ds = PairedDataset(debug=True)
    data_dir = os.path.join(DATA_DIR, args.data_dir)
    ds.load_data_dir_pairs(data_dir)

    from default_viewer import get_default_viewer

    viewer = get_default_viewer(argparse.Namespace(imgui=False))

    def retarget_mi(mi):
        R = len(ds.mi_ri_2_fi[mi])
        src_ri = np.random.randint(0, R, size=1)[0]
        src_fi = ds.mi_ri_2_fi[mi][src_ri]

        tgt_char_name = "BigVegas"
        for i in range(len(ds.mi_ri_2_fi[mi])):
            tgt_ri = i
            tgt_fi = ds.mi_ri_2_fi[mi][tgt_ri]
            if tgt_char_name in ds.filepaths[tgt_fi]:
                break
        tgt_fi = ds.mi_ri_2_fi[mi][tgt_ri]
        src_motion_path = ds.filepaths[src_fi]
        tgt_motion_path = ds.filepaths[tgt_fi]
        print(f"[mi={mi}] src: {src_motion_path}")
        print(f"[mi={mi}] tgt: {tgt_motion_path}")
        (src_batch, tgt_batch), consq_n = get_mi_src_tgt_all_graph(
            dataset=ds, mi=mi, src_ri=src_ri, tgt_ri=tgt_ri, device=args.device
        )
        # option 1) test with data tgt skeleton
        # option 2) test with random skeleton
        if args.rnd_tgt:
            from utils.skel_gen_utils import create_random_skel

            rnd_tgt_skel = create_random_skel()
            rnd_tgt_skel_graph = skel_2_graph(rnd_tgt_skel)
            # rnd_tgt_batch = Batch.from_data_list([rnd_tgt_skel_graph] * consq_n).to(
            #     device=model.device
            # )
            rnd_tgt_batch = SkelPoseBatch.from_data_list([rnd_tgt_skel_graph] * consq_n).to(device=model.device)
            tgt_batch = rnd_tgt_batch

        if args.src_mask:
            src_batch.mask = rnd_mask(src_batch, consq_n=consq_n)

        if args.tgt_mask:
            tgt_batch.mask = rnd_mask(tgt_batch, consq_n=consq_n)

        src_motion, tgt_motion, out_motion = retarget(
            model,
            src_batch,
            tgt_batch,
            ms_dict,
            out_rep_cfg=cfg["representation"]["out"],
            consq_n=consq_n,
        )

        # update viewer
        viewer.update_motions([src_motion, tgt_motion, out_motion], 150, linear=True)
        viewer.mi = mi

    retarget_mi(0)

    def extra_key_callback(key):
        if key == b"m":
            next_mi = (viewer.mi + 1) % len(ds.mi_ri_2_fi)
            retarget_mi(next_mi)
            return True
        return False

    viewer.extra_key_callback = extra_key_callback
    viewer.run()
