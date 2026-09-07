
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from mypath import RESULT_DIR
from kinref.skel_pose_graph import rep_dim
from utils.file_io import load_model_cfg


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if not use_sdpa else attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, num_heads, N, head_dim]
        
        if attn_mask is not None:
            # attn_mask.shape = [B, N], True for masked positions
            attn_mask = attn_mask[:, None, None, :].expand(-1, self.num_heads, N, -1)  # [B, num_heads, N, N]

        with torch.backends.cuda.sdp_kernel():
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop
            )
            
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, qkv_bias=False, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=ff_dim, drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

    
class TransformerEnc(nn.Module):
    def __init__(self, rep_cfg, z_dim, enc_cfg, normalize_z=False):
        super().__init__()
        skel_dim = sum(rep_dim[k] for k in rep_cfg["skel"])
        pose_dim = sum(rep_dim[k] for k in rep_cfg["pose"])
        input_dim = skel_dim + pose_dim
        self.normalize_z = normalize_z

        d_model = enc_cfg.get("d_model", 256)
        num_heads = enc_cfg.get("num_heads", 4)
        num_layers = enc_cfg.get("num_layers", 3)
        ff_dim = enc_cfg.get("ff_dim", 4 * d_model)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList(
            [Block(d_model, num_heads, ff_dim, qkv_bias=True) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.to_z = nn.Linear(d_model, z_dim)

    def forward(self, src_graph):
        x = src_graph.src_x  # [B, J, D]
        keep = ~src_graph.mask  # True for active joints
        x = self.input_proj(x)

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        key_padding = torch.cat(
            [torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device), ~keep], dim=1
        )
        for layer in self.layers:
            x = layer(x, attn_mask=~key_padding)
        x = self.norm(x)
        cls_out = x[:, 0]
        z = self.to_z(cls_out)
        if self.normalize_z:
            z = F.normalize(z, dim=-1)
        return z


class TransformerDec(nn.Module):
    def __init__(self, rep_cfg, z_dim, dec_cfg):
        super().__init__()
        skel_dim = sum(rep_dim[k] for k in rep_cfg["skel"])
        out_dim = sum(rep_dim[k] for k in rep_cfg["out"])

        d_model = dec_cfg.get("d_model", 256)
        num_heads = dec_cfg.get("num_heads", 4)
        num_layers = dec_cfg.get("num_layers", 3)
        ff_dim = dec_cfg.get("ff_dim", 4 * d_model)
        dropout = float(dec_cfg.get("dropout", 0.0))

        self.input_proj = nn.Linear(skel_dim, d_model)
        self.global_token_proj = nn.Linear(z_dim, d_model)
        self.layers = nn.ModuleList(
            [Block(d_model, num_heads, ff_dim, qkv_bias=True, drop=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, src_z, tgt_graph, delta_joint=None):
        x = self.input_proj(tgt_graph.tgt_x)
        if delta_joint is not None:
            x = x + delta_joint
        global_token = self.global_token_proj(src_z).unsqueeze(1)
        x = torch.cat([global_token, x], dim=1)

        keep = ~tgt_graph.mask  # True for active joints
        key_padding = torch.cat(
            [torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device), ~keep], dim=1
        )
        for layer in self.layers:
            x = layer(x, attn_mask=~key_padding)
        x = self.norm(x)[:, 1:]
        out = self.out_proj(x)
        return out * keep.unsqueeze(-1)


class Model(nn.Module):
    def __init__(self, model_cfg, rep_cfg):
        super().__init__()
        z_dim = model_cfg["z_dim"]
        normalize_z = model_cfg["normalize_z"]
        enc_cfg = model_cfg["Encoder"]
        dec_cfg = model_cfg["Decoder"]

        self.encoder = TransformerEnc(rep_cfg, z_dim, enc_cfg, normalize_z)
        self.decoder = TransformerDec(rep_cfg, z_dim, dec_cfg)

        self.rep_cfg = rep_cfg
        self.z_dim = z_dim

        if "load" in model_cfg:
            for load_i in model_cfg["load"]:
                self.load_params(
                    load_i["dir"], load_i["epoch"], load_i["prefix"], load_i["freeze"]
                )

        # print(self)

    def forward(self, src_graph, tgt_graph):
        z = self.encoder(src_graph)
        hatD = self.decoder(z, tgt_graph)
        return z, hatD

    def encode(self, src_graph):
        return self.encoder(src_graph)

    def decode(self, src_z, tgt_graph, delta_joint=None):
        return self.decoder(src_z, tgt_graph, delta_joint=delta_joint)

    @property
    def device(self):
        return next(self.parameters()).device

    def load_params(self, dir, epoch=None, prefix="", freeze=False):
        load_model_name = "last_model.pt" if epoch is None else f"model_{epoch}.pt"
        load_path = os.path.join(RESULT_DIR, dir, load_model_name)

        saved = torch.load(load_path, map_location=self.device)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v
            for k, v in saved["model"].items()
            if k.startswith(prefix) and k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

        if freeze:
            for name, param in self.named_parameters():
                if name in pretrained_dict:
                    param.requires_grad = False
        print("load model from : ", load_path, " DONE")
        print("prefix : ", prefix, ", freeze:", freeze, "\n")


def make_load_model(model_epoch, device="cuda"):
    if (len(model_epoch.split("/")) == 2) and (model_epoch.split("/")[-1].isdigit()):
        load_epoch = model_epoch.split("/")[-1]
        load_model = model_epoch[: model_epoch.find(load_epoch) - 1]
        load_epoch = int(load_epoch)
    else:
        load_epoch = None
        load_model = model_epoch
        print(model_epoch)
    config = load_model_cfg(load_model)
    model = Model(config["model"], config["representation"]).to(device=device)
    model.load_params(load_model, load_epoch)
    return model, config


######################## model fwd result post-processing functions ########################
from utils import tensor_utils
from fairmotion.utils import constants


def parse_hatD(hatD, mask, out_rep_cfg, ms_dict):
    out = {}
    k_start = 0
    keep = (~mask).unsqueeze(-1)
    for k in out_rep_cfg:
        k_dim = rep_dim[k]
        val = hatD[..., k_start : k_start + k_dim]
        if k + "_s" in ms_dict:
            device = val.device
            out[k + "_n"] = val
            out[k] = val * ms_dict[k + "_s"].to(device=device) + ms_dict[k + "_m"].to(device=device)
        else:
            out[k] = val

        if k == "r":
            out[k + "_n"] = out[k + "_n"][:, 0]
            out[k] = out[k][:, 0]
        else:
            out[k] = out[k] * keep
            if k + "_n" in out:
                out[k + "_n"] = out[k + "_n"] * keep
        k_start += k_dim
    return out


def reshape_consq(v, consq_n):
    return v.reshape(consq_n, -1, *v.shape[1:])


def reshape_dict_consq(result_dict, consq_n):
    for k, v in result_dict.items():
        result_dict[k] = reshape_consq(v, consq_n)
    return result_dict

def FK(lo, qR, r, mask, skel_depth, parent_index):
    keep = ~mask
    B, J = lo.shape[:2]

    joint_T = tensor_utils.Tensor(constants.eye_T())[None, None, ...].repeat(B, J, 1, 1)
    joint_T[..., :3, :3] = qR
    joint_T[..., :3, 3] = lo
    joint_T[:, 0, 1, 3] = r[..., 3]
    joint_T[..., 3, 3] = 1

    max_depth = int(torch.clamp(skel_depth.max(), min=0).item())
    for depth in range(max_depth + 1):
        is_depth = (skel_depth == depth) & keep
        batch_idx, child_idx = torch.where(is_depth)

        if child_idx.numel() == 0:
            continue

        parent_idx = parent_index[batch_idx, child_idx]

        non_root = parent_idx != child_idx
        if not non_root.any():
            continue

        batch_idx = batch_idx[non_root]
        child_idx = child_idx[non_root]
        parent_idx = parent_idx[non_root]

        parent_T = joint_T[batch_idx, parent_idx].clone()
        child_T = joint_T[batch_idx, child_idx].clone()
        joint_T[batch_idx, child_idx] = parent_T @ child_T

    return joint_T

def accum_root(r, consq_n, apply_height=False):
    rT = tensor_utils.tensor_r_to_rT(r)  # [T, B, 4, 4]
    rT_accum = rT.clone()
    for i in range(1, consq_n):
        rT_accum[i] = rT_accum[i - 1].clone() @ rT_accum[i].clone()
    if apply_height:
        rT_accum[..., 1, 3] = r[..., 3]
    return rT_accum


def compute_pa_pv_ra(r, p, consq_n):
    rT_accum = accum_root(r, consq_n, apply_height=False)
    Ta = rT_accum.unsqueeze(2) @ tensor_utils.tensor_p2T(p)
    pa = Ta[..., :3, 3]
    pv = pa[1:] - pa[:-1]
    ra = torch.stack(
        (
            rT_accum[..., 0, 0],
            rT_accum[..., 0, 2],
            rT_accum[..., 0, 3],
            rT_accum[..., 2, 3],
        ),
        dim=-1,
    )
    return pa, pv, ra


def out_post_fwd(out, tgt_batch, ms_dict, out_rep_cfg, consq_n):
    out.update(parse_hatD(out["hatD"], tgt_batch.mask, out_rep_cfg, ms_dict))
    out["qR"] = tensor_utils.tensor_q2qR(out["q"])
    out["fk_T"] = FK(
        tgt_batch.lo,
        out["qR"],
        out["r"],
        tgt_batch.mask,
        tgt_batch.skel_depth,
        tgt_batch.parent_index,
    )
    reshape_dict_consq(out, consq_n)
    out["p"] = out["fk_T"][..., :3, 3]
    out["pa"], out["pv"], out["ra"] = compute_pa_pv_ra(out["r"], out["p"], consq_n)

    gt = {"r": tgt_batch.r_nopad, "p": tgt_batch.p, "c": tgt_batch.c, "mask": tgt_batch.mask}
    gt["qR"] = tensor_utils.tensor_q2qR(tgt_batch.q)
    r_m = ms_dict["r_m"].to(device=gt["r"].device)
    r_s = ms_dict["r_s"].to(device=gt["r"].device)
    gt["r_n"] = (gt["r"] - r_m) / r_s
    gt = reshape_dict_consq(gt, consq_n)
    gt["pa"], gt["pv"], gt["ra"] = compute_pa_pv_ra(gt["r"], gt["p"], consq_n)
    gt["qbi"] = (~tgt_batch.mask) & tgt_batch.qb
    if consq_n > 1:
        gt["qbi"] = reshape_consq(gt["qbi"], consq_n)[0]
    return out, gt
