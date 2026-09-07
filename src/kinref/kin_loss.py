import torch
import torch.nn.functional as F
from functools import partial
from utils import tensor_utils

from typing import Union


def _get_invalid_mask(gt):
    return gt.get("mask", None)


def _align_joint_mask(mask: torch.Tensor, ref: torch.Tensor):
    if mask is None:
        return None
    if ref.shape == mask.shape:
        return mask
    if ref.ndim >= 1 and tuple(ref.shape[:-1]) == tuple(mask.shape):
        return mask
    return None


def _masked_mean(x: torch.Tensor, valid: Union[torch.Tensor, None]):
    if valid is None:
        return x.mean()
    valid_f = valid.to(dtype=x.dtype)
    while valid_f.ndim < x.ndim:
        valid_f = valid_f.unsqueeze(-1)
    denom = valid_f.sum().clamp_min(1.0)
    return (x * valid_f).sum() / denom


def default_mse_loss(loss_key, out, gt):
    y_gt, y_out = gt[loss_key], out[loss_key]

    if y_gt.dtype == torch.bool:
        y_gt = y_gt.float()
    if y_out.dtype == torch.bool:
        y_out = y_out.float()

    joint_invalid = None
    if loss_key not in {"r", "ra", "z", "z_tgt", "z_gauss"}:
        joint_invalid = _align_joint_mask(_get_invalid_mask(gt), y_gt)

    if joint_invalid is None:
        return F.mse_loss(y_out, y_gt)
    
    keep = ~joint_invalid
    sq = (y_out - y_gt) ** 2
    return _masked_mean(sq, keep)


def compute_q_loss(out, gt):
    qbi = gt["qbi"]                 # joint/bone selection mask
    invalid = _get_invalid_mask(gt) # True = invalid joint

    qR_gt = gt["qR"]
    qR_out = out["qR"]
    target_shape = qR_gt.shape[:-2]   # e.g. [T, B, J] or [B, J]

    # Expand qbi to match target joint shape
    if qbi.shape == target_shape:
        qbi_keep = qbi
    elif qbi.ndim + 1 == len(target_shape) and tuple(qbi.shape) == tuple(target_shape[1:]):
        qbi_keep = qbi.unsqueeze(0).expand(target_shape[0], *qbi.shape)
    else:
        raise ValueError(f"Unexpected qbi shape {qbi.shape} for qR shape {qR_gt.shape}")

    # Expand invalid mask if needed
    if invalid.shape == target_shape:
        invalid_mask = invalid
    elif invalid.ndim + 1 == len(target_shape) and tuple(invalid.shape) == tuple(target_shape[1:]):
        invalid_mask = invalid.unsqueeze(0).expand(target_shape[0], *invalid.shape)
    else:
        raise ValueError(f"Unexpected mask shape {invalid.shape} for qR shape {qR_gt.shape}")

    keep = qbi_keep & (~invalid_mask)

    sq = (qR_out - qR_gt) ** 2
    return _masked_mean(sq, keep)


def compute_q_ortho_loss(out, gt):
    qbi = gt["qbi"]                 # joint/bone selection mask
    invalid = _get_invalid_mask(gt) # True = invalid joint

    qR_gt = gt["qR"]
    qR_out = out["qR"]
    target_shape = qR_gt.shape[:-2]   # e.g. [T, B, J] or [B, J]

    # Expand qbi to match target joint shape
    if qbi.shape == target_shape:
        qbi_keep = qbi
    elif qbi.ndim + 1 == len(target_shape) and tuple(qbi.shape) == tuple(target_shape[1:]):
        qbi_keep = qbi.unsqueeze(0).expand(target_shape[0], *qbi.shape)
    else:
        raise ValueError(f"Unexpected qbi shape {qbi.shape} for qR shape {qR_gt.shape}")

    # Expand invalid mask if needed
    if invalid.shape == target_shape:
        invalid_mask = invalid
    elif invalid.ndim + 1 == len(target_shape) and tuple(invalid.shape) == tuple(target_shape[1:]):
        invalid_mask = invalid.unsqueeze(0).expand(target_shape[0], *invalid.shape)
    else:
        raise ValueError(f"Unexpected mask shape {invalid.shape} for qR shape {qR_gt.shape}")

    keep = qbi_keep & (~invalid_mask)

    rel = torch.matmul(qR_out, qR_gt.transpose(-1, -2))
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    loss = torch.acos(((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))
    return _masked_mean(loss, keep)


def compute_cv_loss(out, gt):
    gt_c = gt["c"][..., 0].float()
    pv_norm = torch.norm(out["pv"], dim=-1)

    if gt_c.ndim == pv_norm.ndim + 1:
        gt_c = gt_c[..., 0]
    if gt_c.shape[0] == pv_norm.shape[0] + 1:
        gt_c = gt_c[1:]

    joint_invalid = _align_joint_mask(_get_invalid_mask(gt), pv_norm)
    if joint_invalid is not None and joint_invalid.shape[0] == gt["mask"].shape[0] and pv_norm.shape[0] == joint_invalid.shape[0] - 1:
        joint_invalid = joint_invalid[1:]
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(gt_c * pv_norm, keep)


def compute_cv_loss_v2(out, gt):
    c = out["c"][..., 0].float().detach()
    pv_norm = torch.norm(out["pv"], dim=-1)

    if c.ndim == pv_norm.ndim + 1:
        c = c[..., 0]
    if c.shape[0] == pv_norm.shape[0] + 1:
        c = c[1:]

    joint_invalid = _align_joint_mask(_get_invalid_mask(gt), pv_norm)
    if joint_invalid is not None and joint_invalid.shape[0] == gt["mask"].shape[0] and pv_norm.shape[0] == joint_invalid.shape[0] - 1:
        joint_invalid = joint_invalid[1:]
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(c * pv_norm, keep)


def compute_pen_loss(out, gt):
    out_pa = out["pa"]
    penalty = torch.minimum(out_pa[..., 1], tensor_utils.Tensor([0]).to(out_pa.device)) ** 2
    joint_invalid = _align_joint_mask(_get_invalid_mask(gt), penalty)
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(penalty, keep)


def compute_jerk_loss(out, gt, fps=30):
    CM2KM = 0.01 * 0.001
    out_pa = out["pa"]
    if out_pa.shape[0] < 4:
        return out_pa.new_tensor(0.0)

    jerk = (out_pa[3:] - 3 * out_pa[2:-1] + 3 * out_pa[1:-2] - out_pa[:-3]) * (fps**3)
    jerk = jerk.norm(dim=-1) * CM2KM

    joint_invalid = _get_invalid_mask(gt)
    if joint_invalid is not None:
        joint_invalid = _align_joint_mask(joint_invalid[3:], jerk)
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(jerk**2, keep)


def compute_slide_loss(out, gt):
    H = 3
    out_pa = out["pa"]
    if out_pa.shape[0] < 2:
        return out_pa.new_tensor(0.0)

    pv = out_pa[1:] - out_pa[:-1]
    h = out_pa[1:, ..., 1]
    contact = torch.clamp(1 - h / H, 0, 1).unsqueeze(-1)
    slide = (pv * contact).norm(dim=-1)

    joint_invalid = _get_invalid_mask(gt)
    if joint_invalid is not None:
        joint_invalid = _align_joint_mask(joint_invalid[1:], slide)
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(slide**2, keep)


def compute_z_loss(out, gt):
    return torch.nn.MSELoss()(out["z"], out["z_tgt"])


def compute_z_gauss_loss(out, gt):
    z = out["z"]
    mu = z.mean(dim=0)
    std = z.std(dim=0)
    return torch.mean(mu**2 + (std - 1) ** 2)


def compute_z_contrastive_loss(out, gt):
    z = out["z"]         # [consq_n, B, z_dim]
    z_tgt = out["z_tgt"] # [consq_n, B, z_dim]

    if z.ndim != 3 or z_tgt.ndim != 3:
        raise ValueError(f"Expected z/z_tgt to be 3D, got {z.shape} and {z_tgt.shape}")
    if z.shape != z_tgt.shape:
        raise ValueError(f"z and z_tgt shape mismatch: {z.shape} vs {z_tgt.shape}")

    consq_n, B, _ = z.shape
    temperature = 0.07

    # Treat same batch index as positives across all consq_n groups and both views.
    feat = torch.cat([z, z_tgt], dim=0).reshape(2 * consq_n * B, -1)
    feat = F.normalize(feat, dim=-1)

    labels = torch.arange(B, device=feat.device).repeat(2 * consq_n)
    sim = torch.matmul(feat, feat.T) / temperature

    n = sim.shape[0]
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~eye)

    # Multi-positive InfoNCE over cosine similarities.
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    loss = -(log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_count)
    return loss.mean()


def compute_z_contrastive_eucl_loss(out, gt):
    z = out["z"]         # [consq_n, B, z_dim]
    z_tgt = out["z_tgt"] # [consq_n, B, z_dim]

    if z.ndim != 3 or z_tgt.ndim != 3:
        raise ValueError(f"Expected z/z_tgt to be 3D, got {z.shape} and {z_tgt.shape}")
    if z.shape != z_tgt.shape:
        raise ValueError(f"z and z_tgt shape mismatch: {z.shape} vs {z_tgt.shape}")

    consq_n, B, _ = z.shape
    margin = 1.0

    # Same batch index across all groups/views is treated as positive.
    feat = torch.cat([z, z_tgt], dim=0).reshape(2 * consq_n * B, -1)
    labels = torch.arange(B, device=feat.device).repeat(2 * consq_n)

    # Pairwise Euclidean distance between all embeddings.
    dist = torch.cdist(feat, feat, p=2)

    n = dist.shape[0]
    eye = torch.eye(n, device=dist.device, dtype=torch.bool)
    pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~eye)
    neg_mask = (~labels.unsqueeze(0).eq(labels.unsqueeze(1))) & (~eye)

    # Pull positives together and push negatives at least margin apart.
    pos_count = pos_mask.sum().clamp_min(1)
    neg_count = neg_mask.sum().clamp_min(1)

    pos_loss = (dist.pow(2) * pos_mask.to(dist.dtype)).sum() / pos_count
    neg_loss = (F.relu(margin - dist).pow(2) * neg_mask.to(dist.dtype)).sum() / neg_count
    return pos_loss + neg_loss


def compute_z_linear(out, gt):
    # encourages linear delta z over time

    z = out["z"]         # [consq_n, B, z_dim] 
    z_tgt = out["z_tgt"] # [consq_n, B, z_dim]

    if z.ndim != 3 or z_tgt.ndim != 3:
        raise ValueError(f"Expected z/z_tgt to be 3D, got {z.shape} and {z_tgt.shape}")
    if z.shape != z_tgt.shape:
        raise ValueError(f"z and z_tgt shape mismatch: {z.shape} vs {z_tgt.shape}")
    
    z_accel = z[2:] - 2 * z[1:-1] + z[:-2]
    z_tgt_accel = z_tgt[2:] - 2 * z_tgt[1:-1] + z_tgt[:-2]
    loss = (z_accel ** 2).mean() + (z_tgt_accel ** 2).mean()
    return loss
    

_loss_matching_ = {
    "p": partial(default_mse_loss, "p"),
    "r": partial(default_mse_loss, "r_n"),
    "c": partial(default_mse_loss, "c"),
    "pv": partial(default_mse_loss, "pv"),
    "z": compute_z_loss,
    "q": compute_q_loss,
    "q_ortho": compute_q_ortho_loss,
    "cv": compute_cv_loss,
    "cv_v2": compute_cv_loss_v2,
    "pen": compute_pen_loss,
    "jerk": compute_jerk_loss,
    "slide": compute_slide_loss,
    "z_gauss": compute_z_gauss_loss,
    "z_contrast": compute_z_contrastive_loss,
    "z_contrast_eucl": compute_z_contrastive_eucl_loss,
    "z_linear": compute_z_linear,
}


def get_loss_function(ltype):
    return _loss_matching_[ltype]


def get_loss_names():
    return list(_loss_matching_.keys())


def compute_loss(loss_cfg, out, gt):
    losses = dict()
    loss_sum = 0.0
    for key, weight in loss_cfg.items():
        if weight > 1e-8:
            ftn = get_loss_function(key)
            try:
                loss = ftn(out, gt)
            except Exception as e:
                print(key, "error occured")
                print(e)
                breakpoint()
                exit()
            if torch.isnan(loss):
                print(key, "NaN occured")
                breakpoint()
                exit()
            weighted_loss = loss * weight
            losses[key] = weighted_loss
            loss_sum += weighted_loss
    losses["total"] = loss_sum
    return losses
