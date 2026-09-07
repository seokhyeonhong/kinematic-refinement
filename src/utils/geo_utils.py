import torch
import torch.nn.functional as F

from enum import Enum


class PartIndex(Enum):
    eBody = 0
    eHead = 1
    eLArm = 2
    eRArm = 3
    eLLeg = 4
    eRLeg = 5
    eLHand = 6
    eRHand = 7


def compute_chamfer_normal_collision(
    vp,
    vn,
    vidx_part,
    v_mask,
    eps=1e-8,
    dist_threshold=5.0,
    normal_sim_threshold=0.0,
):
    """
    Part-aware chamfer-NN penetration query.

    Args:
        vp:         [B, V, 3]
        vn:         [B, V, 3]
        vidx_part:  [B, V]
        v_mask:     [B, V]   True=invalid, False=valid

    Returns:
        pen_mask:      [B, V] bool
        pen_disp:      [B, V, 3] float
        pen_ref_norm:  [B, V, 3] float
    """
    B, V, _ = vp.shape
    device = vp.device
    dtype = vp.dtype

    vn = F.normalize(vn, dim=-1, eps=eps)
    valid = ~v_mask

    def _make_mask(part_tensor, vals):
        if len(vals) == 0:
            return torch.zeros_like(part_tensor, dtype=torch.bool)
        m = torch.zeros_like(part_tensor, dtype=torch.bool)
        for v in vals:
            m |= (part_tensor == v)
        return m

    def _pack_by_mask(x, mask):
        """
        x:    [B, V] or [B, V, C]
        mask: [B, V]

        Returns:
            packed_x:    [B, M] or [B, M, C]
            packed_mask: [B, M]
            packed_idx:  [B, M]
        """
        counts = mask.sum(dim=1)
        max_count = int(counts.max().item())

        if max_count == 0:
            if x.dim() == 2:
                packed_x = x[:, :0]
            else:
                packed_x = x[:, :0, :]
            packed_mask = mask[:, :0]
            packed_idx = torch.empty((B, 0), dtype=torch.long, device=x.device)
            return packed_x, packed_mask, packed_idx

        order = mask.long().argsort(dim=1, descending=True)

        if x.dim() == 2:
            packed_x = torch.gather(x, 1, order)[:, :max_count]
        elif x.dim() == 3:
            gather_idx = order.unsqueeze(-1).expand(-1, -1, x.size(-1))
            packed_x = torch.gather(x, 1, gather_idx)[:, :max_count, :]
        else:
            raise ValueError(f"Unsupported x.dim()={x.dim()}")

        packed_idx = order[:, :max_count]
        packed_mask = torch.arange(max_count, device=x.device)[None, :] < counts[:, None]

        return packed_x, packed_mask, packed_idx

    # part rules
    larm_q = [PartIndex.eLArm.value, PartIndex.eLHand.value]
    rarm_q = [PartIndex.eRArm.value, PartIndex.eRHand.value]
    lleg_q = [PartIndex.eLLeg.value]
    rleg_q = [PartIndex.eRLeg.value]
    body_ref = [PartIndex.eBody.value, PartIndex.eHead.value]

    part_rules = [
        (larm_q, body_ref + rarm_q + lleg_q + rleg_q),
        (rarm_q, body_ref + larm_q + lleg_q + rleg_q),
        (lleg_q, body_ref),
        (rleg_q, body_ref),
    ]

    pen_mask = torch.zeros((B, V), dtype=torch.bool, device=device)
    pen_disp = torch.zeros((B, V, 3), dtype=dtype, device=device)
    pen_ref_norm = torch.zeros((B, V, 3), dtype=dtype, device=device)

    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)

    for query_part_vals, ref_part_vals in part_rules:
        query_mask = valid & _make_mask(vidx_part, query_part_vals)   # [B, V]
        ref_mask   = valid & _make_mask(vidx_part, ref_part_vals)     # [B, V]

        active_batch = query_mask.any(dim=1) & ref_mask.any(dim=1)
        if not active_batch.any():
            continue

        # 1) pack queries
        qpos, qmask, qidx = _pack_by_mask(vp, query_mask)             # [B, Q, 3], [B, Q], [B, Q]
        qnrm, _, _        = _pack_by_mask(vn, query_mask)

        if qpos.size(1) == 0:
            continue

        # 2) AABB prune refs using query bbox
        qmin = torch.where(qmask.unsqueeze(-1), qpos, inf).amin(dim=1)      # [B, 3]
        qmax = torch.where(qmask.unsqueeze(-1), qpos, neg_inf).amax(dim=1)  # [B, 3]

        # in_box = ((vp >= qmin[:, None, :]) & (vp <= qmax[:, None, :])).all(dim=-1)  # [B, V]
        in_box = ((vp >= (qmin - dist_threshold)[:, None, :]) & (vp <= (qmax + dist_threshold)[:, None, :])).all(dim=-1)  # [B, V]
        ref_keep = ref_mask & in_box & active_batch[:, None]

        if not ref_keep.any():
            continue

        # 3) pack refs
        rpos, rmask, ridx = _pack_by_mask(vp.detach(), ref_keep)      # [B, R, 3], [B, R], [B, R]
        rnrm, _, _        = _pack_by_mask(vn.detach(), ref_keep)
        rpart, _, _       = _pack_by_mask(vidx_part, ref_keep)

        if rpos.size(1) == 0:
            continue

        # 4) nearest reference for each query
        dist = torch.cdist(qpos, rpos, p=2)                           # [B, Q, R]

        qpart, _, _ = _pack_by_mask(vidx_part, query_mask)

        same_part = qpart[:, :, None] == rpart[:, None, :]
        same_idx  = qidx[:, :, None] == ridx[:, None, :]

        pair_valid = (
            qmask[:, :, None]
            & rmask[:, None, :]
            & (~same_part)
            & (~same_idx)
            & active_batch[:, None, None]
        )

        has_valid = pair_valid.any(dim=-1)                            # [B, Q]
        if not has_valid.any():
            continue

        dist_masked = dist.masked_fill(~pair_valid, float("inf"))
        nn_dist, nn_idx = dist_masked.min(dim=-1)                     # [B, Q]

        gather_idx3 = nn_idx.unsqueeze(-1).expand(-1, -1, 3)
        ref_pos_nn = torch.gather(rpos, 1, gather_idx3)               # [B, Q, 3]
        ref_nrm_nn = torch.gather(rnrm, 1, gather_idx3)               # [B, Q, 3]

        # GATED REF
        cos_sim = torch.sum(qnrm * ref_nrm_nn, dim=-1)   # [B, Q]
        #

        # refer_vector = ref - query  (same convention as your reference code)
        disp = ref_pos_nn - qpos                                      # [B, Q, 3]
        sdf_dot = torch.sum(disp * ref_nrm_nn, dim=-1)                # [B, Q]

        # coll = (sdf_dot > 0.0) & has_valid & qmask                    # [B, Q]
        coll = (
            (sdf_dot > 0.0)
            & (nn_dist < dist_threshold)
            & (cos_sim < normal_sim_threshold)
            & has_valid
            & qmask
        )                    # [B, Q]

        # scatter mask
        mask_updates = torch.zeros((B, V), dtype=torch.bool, device=device)
        mask_updates.scatter_(1, qidx, coll)
        pen_mask |= mask_updates

        # scatter disp
        disp_updates = torch.zeros((B, V, 3), dtype=dtype, device=device)
        disp_updates.scatter_add_(
            1,
            qidx.unsqueeze(-1).expand(-1, -1, 3),
            disp * coll.unsqueeze(-1).to(dtype)
        )
        pen_disp += disp_updates

        # scatter reference normal
        norm_updates = torch.zeros((B, V, 3), dtype=dtype, device=device)
        norm_updates.scatter_add_(
            1,
            qidx.unsqueeze(-1).expand(-1, -1, 3),
            ref_nrm_nn * coll.unsqueeze(-1).to(dtype)
        )
        pen_ref_norm += norm_updates

    # normalize normals where multiple rules contributed
    norm = pen_ref_norm.norm(dim=-1, keepdim=True).clamp_min(eps)
    pen_ref_norm = torch.where(
        pen_mask.unsqueeze(-1),
        pen_ref_norm / norm,
        pen_ref_norm
    )

    return pen_disp, pen_ref_norm

