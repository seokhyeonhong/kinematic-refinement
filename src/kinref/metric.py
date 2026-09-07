import torch
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


def compute_p_metric(out, gt):
    gt_p, out_p = gt["p"], out["p"]
    p_dist = torch.linalg.norm(out_p - gt_p, dim=-1)
    joint_invalid = _align_joint_mask(_get_invalid_mask(gt), p_dist)
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(p_dist, keep)


def compute_pa_metric(out, gt):
    gt_pa, out_pa = gt["pa"], out["pa"]
    pa_dist = torch.linalg.norm(out_pa - gt_pa, dim=-1)
    joint_invalid = _align_joint_mask(_get_invalid_mask(gt), pa_dist)
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(pa_dist, keep)


def compute_ra_xz_metric(out, gt):
    gt_ra_xz, out_ra_xz = gt["ra"][..., [2, 3]], out["ra"][..., [2, 3]]
    ra_xz_dist = torch.linalg.norm(gt_ra_xz - out_ra_xz, dim=-1)
    return torch.mean(ra_xz_dist)


def compute_ra_theta_metric(out, gt):
    gt_ra_dt, out_ra_dt = gt["ra"][..., [0, 1]], out["ra"][..., [0, 1]]
    ra_dt_dist = torch.linalg.norm(gt_ra_dt - out_ra_dt, dim=-1)
    return torch.mean(ra_dt_dist)


def compute_rtheta_metric(out, gt):
    return torch.abs(gt["r"][..., 0] - out["r"][..., 0]).mean()


def compute_rdx_metric(out, gt):
    return torch.abs(gt["r"][..., 1] - out["r"][..., 1]).mean()


def compute_rdz_metric(out, gt):
    return torch.abs(gt["r"][..., 2] - out["r"][..., 2]).mean()


def compute_rh_metric(out, gt):
    return torch.abs(gt["r"][..., 3] - out["r"][..., 3]).mean()


def compute_pen_metric(out, gt):
    out_pa = out["pa"]
    penetration = torch.minimum(out_pa[..., 1], tensor_utils.Tensor([0]).to(out_pa.device))
    joint_invalid = _align_joint_mask(_get_invalid_mask(gt), penetration)
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(penetration, keep)


def compute_jerk_metric(out, gt, fps=30):
    """Positional jitter (3rd derivative): smoothness of motion."""
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
    return _masked_mean(jerk, keep)


def compute_slide_metric(out, gt):
    """Foot skating metric."""
    H = 3
    out_pa = out["pa"]
    if out_pa.shape[0] < 2:
        return out_pa.new_tensor(0.0)

    pv = out_pa[1:] - out_pa[:-1]
    h = out_pa[1:, ..., 1]
    contact = torch.clamp(2 - 2 ** (h / H), 0, 1).unsqueeze(-1)
    slide = (pv * contact).norm(dim=-1)

    joint_invalid = _get_invalid_mask(gt)
    if joint_invalid is not None:
        joint_invalid = _align_joint_mask(joint_invalid[1:], slide)
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(slide, keep)


def compute_pa_metric_skel_aware_version(out, gt):
    gt_pa, out_pa = gt["pa"], out["pa"]
    err = (gt_pa - out_pa) * (gt_pa - out_pa)
    err /= (gt["height"] ** 2)[..., None, None]
    joint_invalid = _align_joint_mask(_get_invalid_mask(gt), err)
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(err, keep) * 1000


def compute_p_metric_skel_aware_version(out, gt):
    gt_p, out_p = gt["p"], out["p"]
    err = (gt_p - out_p) * (gt_p - out_p)
    err /= (gt["height"] ** 2)[..., None, None]
    joint_invalid = _align_joint_mask(_get_invalid_mask(gt), err)
    keep = None if joint_invalid is None else ~joint_invalid
    return _masked_mean(err, keep) * 1000


def compute_ra_metric_skel_aware_version(out, gt):
    gt_ra_xz, out_ra_xz = gt["ra"][..., [2, 3]], out["ra"][..., [2, 3]]
    err = (gt_ra_xz - out_ra_xz) * (gt_ra_xz - out_ra_xz)
    err /= (gt["height"] ** 2)[..., None]
    return err.mean() * 1000


def compute_qR_metric(out, gt):
    gt_qR, out_qR = gt["qR"], out["qR"]
    qbi = gt["qbi"]
    invalid = _get_invalid_mask(gt) # True = invalid joint
    target_shape = gt_qR.shape[:-2]

    # Expand qbi to match target joint shape
    if qbi.shape == target_shape:
        qbi_keep = qbi
    elif qbi.ndim + 1 == len(target_shape) and tuple(qbi.shape) == tuple(target_shape[1:]):
        qbi_keep = qbi.unsqueeze(0).expand(target_shape[0], *qbi.shape)
    else:
        raise ValueError(f"Unexpected qbi shape {qbi.shape} for qR shape {gt_qR.shape}")

    # Expand invalid mask if needed
    if invalid.shape == target_shape:
        invalid_mask = invalid
    elif invalid.ndim + 1 == len(target_shape) and tuple(invalid.shape) == tuple(target_shape[1:]):
        invalid_mask = invalid.unsqueeze(0).expand(target_shape[0], *invalid.shape)
    else:
        raise ValueError(f"Unexpected mask shape {invalid.shape} for qR shape {gt_qR.shape}")

    keep = qbi_keep & (~invalid_mask)

    qR_diff = gt_qR.transpose(-2, -1) @ out_qR
    qR_diff_aa = tensor_utils.matrix_to_axis_angle(qR_diff)
    qR_diff_aa_norm = torch.linalg.norm(qR_diff_aa, dim=-1)
    return _masked_mean(qR_diff_aa_norm, keep)


_metric_matching_ = {
    "p": compute_p_metric,
    "pa": compute_pa_metric,
    "ra_xz": compute_ra_xz_metric,
    "ra_theta": compute_ra_theta_metric,
    "r_theta": compute_rtheta_metric,
    "r_dx": compute_rdx_metric,
    "r_dz": compute_rdz_metric,
    "r_h": compute_rh_metric,
    "pen": compute_pen_metric,
    "jerk": compute_jerk_metric,
    "slide": compute_slide_metric,
    "pa_skel_aware": compute_pa_metric_skel_aware_version,
    "p_skel_aware": compute_p_metric_skel_aware_version,
    "ra_skel_aware": compute_ra_metric_skel_aware_version,
    "qR": compute_qR_metric,
}


def get_metric_function(ltype):
    return _metric_matching_[ltype]


def get_metric_names():
    return list(_metric_matching_.keys())


def compute_metric(metric_cfg, out, gt):
    metrics = dict()
    for key in metric_cfg:
        ftn = get_metric_function(key)
        metrics[key] = ftn(out, gt)
    return metrics
