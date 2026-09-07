import argparse, sys, yaml, gc, shutil, random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from mypath import *
from utils import file_io, tensor_utils, network_utils
from kinref.geo_dataset import PairedDataset, get_paired_data_loader
from kinref.geo_model import make_finetune_model, out_post_fwd
from kinref.skel_pose_mesh_graph import SkelPoseMeshGraph
from kinref.geo_loss import compute_loss
from kinref.metric import compute_metric


def load_trainer(load_cfg, model, optimizer, scheduler, device, log_path):
    if (load_cfg is None) or (load_cfg["dir"] is None):
        return

    load_dir, load_epoch = load_cfg["dir"], load_cfg["epoch"]
    load_model_name = (
        "last_model.pt" if load_epoch is None else "model_{}.pt".format(load_epoch)
    )
    load_abs_dir = os.path.join(RESULT_DIR, load_dir)
    load_path = os.path.join(load_abs_dir, load_model_name)

    saved = torch.load(load_path, map_location=device)

    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    if "scheduler" in saved and scheduler:
        scheduler.load_state_dict(saved["scheduler"])
    epoch_cnt = saved["epoch"]

    prev_log_path = os.path.join(load_abs_dir, "logs")
    if not os.path.samefile(prev_log_path, log_path):
        shutil.rmtree(log_path)
        shutil.copytree(prev_log_path, log_path)
    print("continue training from ", prev_log_path)
    return epoch_cnt

def compute_grad_stats(model):
    stats = {
        "global_norm": 0.0,
        "max_abs": 0.0,
        "encoder_norm": 0.0,
        "decoder_norm": 0.0,
        "other_norm": 0.0,
    }

    for name, param in model.named_parameters():
        if (param.grad is None) or (not param.requires_grad):
            continue

        grad = param.grad.detach()
        grad_norm = grad.norm(2).item()
        grad_max = grad.abs().max().item()

        stats["global_norm"] += grad_norm ** 2
        stats["max_abs"] = max(stats["max_abs"], grad_max)

        if name.startswith("encoder") or name.startswith("kin_model.encoder"):
            stats["encoder_norm"] += grad_norm ** 2
        elif name.startswith("decoder") or name.startswith("kin_model.decoder"):
            stats["decoder_norm"] += grad_norm ** 2
        else:
            stats["other_norm"] += grad_norm ** 2

    stats["global_norm"] = stats["global_norm"] ** 0.5
    stats["encoder_norm"] = stats["encoder_norm"] ** 0.5
    stats["decoder_norm"] = stats["decoder_norm"] ** 0.5
    stats["other_norm"] = stats["other_norm"] ** 0.5
    return stats


def create_save_dir(exp):
    save_dir = os.path.join(RESULT_DIR, exp)
    os.makedirs(save_dir, exist_ok=False)
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--cfg", type=str, default="geo_v0_kinv4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    cfg = file_io.load_cfg(args.cfg)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tensor_utils.set_device(args.device)
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_cfg = cfg["train"]
    if "copy_orig_contact" in cfg["train"]:
        PairedDataset.copy_orig_contact = cfg["train"]["copy_orig_contact"]

    ## Dataset
    data_dir = os.path.join(DATA_DIR, cfg["train_data"]["dir"])
    dl = get_paired_data_loader(
        data_dir,
        train_cfg["batch_size"],
        train_cfg["consq_n"],
        shuffle=True,
        mask_option=cfg["train_data"]["mask"],
        device=args.device,
        debug=args.debug,
    )

    ## Model, Optimizer, Scheduler
    model, pretrained_cfg, ms_dict = make_finetune_model(cfg, device=args.device)

    # Use pretrained representation as source of truth.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters left. Check freeze settings in cfg['model']['pretrained'].")
    optimizer = torch.optim.Adam(trainable_params, lr=train_cfg["learning_rate"])
    scheduler = network_utils.get_scheduler(
        optimizer, train_cfg["lr_schedule"], train_cfg["epoch_num"]
    )

    ## Log setup
    from torch.utils.tensorboard import SummaryWriter

    save_dir = create_save_dir(args.exp)
    log_path = os.path.join(save_dir, "logs")
    writer = SummaryWriter(log_path)

    with open(os.path.join(save_dir, "para.txt"), "w") as para_file:
        para_file.write(" ".join(sys.argv))
    with open(os.path.join(save_dir, "config.yaml"), "w") as config_file:
        yaml.dump(cfg, config_file)  # save all cfg (not only cfg(==cfg['train]))
    print("SAVE DIR: ", save_dir)
    torch.save(ms_dict, os.path.join(save_dir, "ms_dict.pt"))

    # set SkelPoseMeshGraph class variables
    SkelPoseMeshGraph.skel_cfg = cfg["representation"]["skel"]
    SkelPoseMeshGraph.pose_cfg = cfg["representation"]["pose"]
    SkelPoseMeshGraph.mesh_cfg = cfg["representation"]["mesh"]
    SkelPoseMeshGraph.ms_dict = ms_dict

    epoch_init = 0
    if "load" in train_cfg:
        epoch_init = load_trainer(
            train_cfg["load"], model, optimizer, scheduler, args.device, log_path
        )

    ## debugging
    # torch.autograd.set_detect_anomaly(True)

    print("======================= READY TO TRAIN ===================== ")
    ## Train Loop
    model.train()  # set train mode
    for epoch_cnt in tqdm(range(epoch_init, train_cfg["epoch_num"]), ncols=100):
        epoch_loss = {loss: 0 for loss in list(train_cfg["loss"].keys()) + ["total"]}
        epoch_metric = {metric: 0 for metric in train_cfg["metric"]}
        epoch_grad = {
            "global_norm": 0.0,
            "max_abs": 0.0,
            "encoder_norm": 0.0,
            "decoder_norm": 0.0,
            "other_norm": 0.0,
        }

        for bi, (src_batch, tgt_batch) in tqdm(enumerate(dl), total=len(dl), ncols=100, leave=False):
            optimizer.zero_grad()

            ## forward
            delta_z, delta_joint, hatD = model(
                src_batch,
                tgt_batch,
                ms_dict=ms_dict,
                out_rep_cfg=cfg["representation"]["out"],
                consq_n=train_cfg["consq_n"],
            )
            out = {"hatD": hatD} #, "delta_z": delta_z, "delta_joint": delta_joint}
            if delta_z is not None:
                out["delta_z"] = delta_z
            if delta_joint is not None:
                out["delta_joint"] = delta_joint

            out, gt = out_post_fwd(
                out,
                tgt_batch,
                ms_dict,
                cfg["representation"]["out"],
                train_cfg["consq_n"],
            )
            loss = compute_loss(train_cfg["loss"], out, gt)
            metric = compute_metric(train_cfg["metric"], out, gt)

            ## backward
            loss["total"].backward()
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), train_cfg["grad_max_norm"]
            # )

            # gradient stats logging
            grad_stats = compute_grad_stats(model)

            global_step = epoch_cnt * len(dl) + bi
            writer.add_scalar("grad/global_norm_step", grad_stats["global_norm"], global_step)
            writer.add_scalar("grad/max_abs_step", grad_stats["max_abs"], global_step)
            writer.add_scalar("grad/encoder_norm_step", grad_stats["encoder_norm"], global_step)
            writer.add_scalar("grad/decoder_norm_step", grad_stats["decoder_norm"], global_step)
            writer.add_scalar("grad/other_norm_step", grad_stats["other_norm"], global_step)

            # update
            optimizer.step()

            ## log
            for k, v in loss.items():
                epoch_loss[k] += tensor_utils.cdn(v)
            for k, v in metric.items():
                epoch_metric[k] += tensor_utils.cdn(v)
            for k, v in grad_stats.items():
                epoch_grad[k] += v

            del src_batch, tgt_batch, out, loss, metric

        ## Write Log
        for k, v in epoch_loss.items():
            writer.add_scalar("loss/" + k, v / len(dl), epoch_cnt)
        for k, v in epoch_metric.items():
            writer.add_scalar("metric/" + k, v / len(dl), epoch_cnt)
        for k, v in epoch_grad.items():
            writer.add_scalar("grad/" + k, v / len(dl), epoch_cnt)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_cnt)

        ## Save
        if (epoch_cnt % train_cfg["save_per"] == 0) or (
            epoch_cnt + 1 == train_cfg["epoch_num"]
        ):
            save_state = {
                "epoch": epoch_cnt,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(
                save_state, os.path.join(save_dir, "model_{}.pt".format(epoch_cnt))
            )
            torch.save(save_state, os.path.join(save_dir, "last_model.pt"))
            print("save done: ", epoch_cnt)

        if scheduler is not None:
            scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
