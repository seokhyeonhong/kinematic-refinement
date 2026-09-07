import copy
import json
import os
import tempfile
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kinref.skel_pose_mesh_graph import SkelPoseMeshGraph, rep_dim
from kinref.kin_model import (
    Model as KinModel,
    FK,
    compute_pa_pv_ra,
    make_load_model as make_load_kin_model,
    out_post_fwd as kin_out_post_fwd,
    parse_hatD,
    reshape_consq,
    reshape_dict_consq,
)
from utils import tensor_utils
from utils.file_io import load_model_cfg
from utils.geo_utils import (
    PartIndex,
    compute_chamfer_normal_collision,
)
from mypath import RESULT_DIR


class Model(nn.Module):
    """Geometry model composed of a pretrained kinematic backbone and refinement branch."""

    def __init__(self, kin_model_cfg, rep_cfg, geo_cfg):
        super().__init__()

        # kinematic model (pre-trained backbone)
        self.kin_model = KinModel(kin_model_cfg, rep_cfg)
        self.kin_model_cfg = kin_model_cfg
        self.rep_cfg = rep_cfg
        self.z_dim = self.kin_model.z_dim

        # geometry branch
        self.use_skin_w = geo_cfg["use_skin_w"]
        self.use_jacobian = geo_cfg["use_jacobian"]
        self.use_static = geo_cfg["use_static"]

        d_model = geo_cfg.get("d_model", 256)
        mesh_dim = sum(rep_dim[k] for k in rep_cfg["mesh"])

        if self.use_static:
            self.static_mlp = nn.Sequential(
                nn.Linear(mesh_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, kin_model_cfg["Encoder"]["d_model"]),
            )
            nn.init.zeros_(self.static_mlp[-1].weight)
            nn.init.zeros_(self.static_mlp[-1].bias)

        if self.use_jacobian:
            self.jacobian_mlp = nn.Sequential(
                nn.Linear(self.z_dim * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.z_dim),
            )
            nn.init.zeros_(self.jacobian_mlp[-1].weight)
            nn.init.zeros_(self.jacobian_mlp[-1].bias)


    @property
    def device(self):
        return next(self.parameters()).device


    def _run_motion_postprocess(self, z, hatD, tgt_graph, ms_dict=None, out_rep_cfg=None, consq_n=1):
        if out_rep_cfg is None:
            out_rep_cfg = self.rep_cfg.get("out", None)
        if out_rep_cfg is None:
            return None

        if ms_dict is None:
            ms_dict = getattr(SkelPoseMeshGraph, "ms_dict", None)
        if ms_dict is None or len(ms_dict) == 0:
            return None

        out = { "hatD": hatD, "z": z }
        out, _ = kin_out_post_fwd(out, tgt_graph, ms_dict, out_rep_cfg, consq_n)
        return out
    
    
    def _compute_dynamic_geometry_features(
        self,
        z,
        posed_vp,
        posed_vn,
        tgt_graph,
        face=None, # DEBUG
        v_mask=None, # DEBUG
    ):
        if not self.use_jacobian:
            return None

        # tpose_collision = compute_chamfer_normal_collision(
        #     tgt_graph.vp,
        #     tgt_graph.vn,
        #     tgt_graph.vidx_part,
        #     tgt_graph.v_mask,
        # )
        # tpose_collision_mask = (tpose_collision.norm(dim=-1) > 0)   # [B, V] bool
        # tpose_collision_mask = expand_mask_by_radius(
        #     tgt_graph.vp,
        #     tpose_collision_mask,
        #     v_mask=tgt_graph.v_mask,
        #     # radius=radius,
        # )   # [B, V]
        
        pen_disp, _ = compute_chamfer_normal_collision(
            posed_vp,
            posed_vn,
            tgt_graph.vidx_part,
            tgt_graph.v_mask
        )

        # # ========================== DEBUG ========================== 
        # # pen_disp, pen_ref_norm = compute_chamfer_normal_collision_v2(posed_vp,posed_vn,tgt_graph.vidx_part,tgt_graph.v_mask,dist_threshold=5.0,)
        # collision = pen_disp * _
        # pen_mask = (pen_disp.norm(dim=-1) > 0)   # [B, V] bool
        # i = 0
        # # for i in range(len(pen_mask)):
        # debug_out_path = "debug_collision_spheres_posed.glb"
        # export_collision_vertex_pair_vis_mesh(posed_vp[i].cpu(),tgt_graph.face[i].cpu(),pen_mask[i].cpu(),pen_disp[i].cpu(),tgt_graph.v_mask[i].cpu(),out_path=debug_out_path,)
        # breakpoint()
        # # # =========================================================== 

        if self.use_jacobian:
            delta_x = pen_disp.detach()
            correction_hint = torch.autograd.grad(
                posed_vp,
                z,
                grad_outputs=delta_x,
                retain_graph=True,
                create_graph=False,
            )[0]
            correction_hint = F.normalize(correction_hint, dim=-1)
            correction_hint = self.jacobian_mlp(torch.cat([correction_hint, z], dim=-1))

        return correction_hint
    

    def _compute_static_geometry_features(
        self,
        mesh_x,
        tgt_graph,
    ):
        if not self.use_static:
            return None
        
        static_feat = self.static_mlp(mesh_x)
        if self.use_skin_w:
            skin_w_inv = tgt_graph.skin_w.transpose(1, 2)     # [TB, J, V]
            skin_w_inv = F.normalize(skin_w_inv, p=1, dim=-1)
            static_feat = torch.matmul(skin_w_inv, static_feat)  # [TB, J, z_dim]
        else:
            static_feat = static_feat.mean(dim=1, keepdim=True)  # [TB, 1, z_dim]
        return static_feat
            
    
    def forward(
        self,
        src_graph,
        tgt_graph,
        ms_dict=None,
        out_rep_cfg=None,
        consq_n=1,
    ):
        # forward kinematic model
        z = self.kin_model.encode(src_graph)
        z = z.detach().requires_grad_(self.use_jacobian)
        if self.use_jacobian:
            with torch.enable_grad():
                hatD = self.kin_model.decode(z, tgt_graph)
        else:
            hatD = self.kin_model.decode(z, tgt_graph)

        # run motion post-processing
        out_motion = self._run_motion_postprocess(
            z,
            hatD,
            tgt_graph,
            ms_dict=ms_dict,
            out_rep_cfg=out_rep_cfg,
            consq_n=consq_n,
        )

        # skinning
        posed_vp, posed_vn = apply_skinning(
            tgt_graph.vp.requires_grad_(self.use_jacobian),
            tgt_graph.vn.requires_grad_(self.use_jacobian),
            tgt_graph.skin_w,
            tgt_graph.bind_xform_inv,
            out_motion["fk_T"],
            v_mask=tgt_graph.v_mask,
        )

        # static geometry features
        static_feat = self._compute_static_geometry_features(
            mesh_x=tgt_graph.mesh_x,    
            tgt_graph=tgt_graph,
        )

        # correction hints
        correction_feat = self._compute_dynamic_geometry_features(
            z=z,
            posed_vp=posed_vp,
            posed_vn=posed_vn,
            tgt_graph=tgt_graph,
            face=tgt_graph.face,
            v_mask=tgt_graph.v_mask,
        )

        delta_z = correction_feat
        delta_joint = static_feat

        z = z + delta_z if delta_z is not None else z
        hatD = self.kin_model.decode(z, tgt_graph, delta_joint=delta_joint)

        return delta_z, delta_joint, hatD

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


def _freeze_by_prefix(model: nn.Module, prefixes):
    if prefixes is None:
        return
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    if len(prefixes) == 0:
        return

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad = False

    print("[finetune] freeze prefixes:", prefixes)


def make_finetune_model(cfg: Dict, device: str = "cuda") -> Tuple[Model, Dict]:
    """Build a refinement model initialized from a pretrained kinematic backbone.

    Expected config:
      cfg["model"]["pretrained"]["model_epoch"]: <exp_name> or <exp_name/epoch>
    Optional:
      cfg["model"]["pretrained"]["freeze_all"]: bool
      cfg["model"]["pretrained"]["freeze_prefix"]: str or list[str]
    """

    model_cfg = copy.deepcopy(cfg.get("model", {}))
    pretrained_cfg = model_cfg.get("pretrained", {})
    geo_cfg = model_cfg.get("geometry")
    model_epoch = pretrained_cfg.get("model_epoch")

    load_dir = os.path.join(RESULT_DIR, model_epoch.split("/")[0])
    ms_dict = torch.load(os.path.join(load_dir, "ms_dict.pt"))

    if model_epoch is None:
        raise ValueError(
            "cfg['model']['pretrained']['model_epoch'] must be set "
            "(e.g. 'v3_base_lr5e-4_bs64_ep300/299')."
        )

    pretrained_model, pretrained_full_cfg = make_load_kin_model(
        model_epoch, device=device
    )

    model = Model(
        pretrained_full_cfg["model"],
        cfg["representation"],
        geo_cfg=geo_cfg,
    ).to(device=device)

    incompatible = model.kin_model.load_state_dict(
        pretrained_model.state_dict(), strict=False
    )
    if len(incompatible.missing_keys) > 0:
        print("[finetune] missing keys:", incompatible.missing_keys)
    if len(incompatible.unexpected_keys) > 0:
        print("[finetune] unexpected keys:", incompatible.unexpected_keys)

    if bool(pretrained_cfg.get("freeze_all", False)):
        for param in model.kin_model.parameters():
            param.requires_grad = False
        print("[finetune] freeze all pretrained kin_model parameters")
    else:
        _freeze_by_prefix(model.kin_model, pretrained_cfg.get("freeze_prefix", []))

    print("[finetune] initialized from pretrained kinematic model:", model_epoch)
    return model, pretrained_full_cfg, ms_dict


# Keep compatible entry point name.
def make_load_model(model_epoch, device="cuda"):
    if (len(model_epoch.split("/")) == 2) and (model_epoch.split("/")[-1].isdigit()):
        load_epoch = model_epoch.split("/")[-1]
        load_model = model_epoch[: model_epoch.find(load_epoch) - 1]
        load_epoch = int(load_epoch)
    else:
        load_epoch = None
        load_model = model_epoch
    geo_cfg = load_model_cfg(load_model)
    pretrained_model_epoch = geo_cfg.get("model", {}).get("pretrained", {}).get("model_epoch")
    kin_model_epoch = pretrained_model_epoch if pretrained_model_epoch is not None else model_epoch
    geo_model_cfg = geo_cfg.get("model", {})
    geo_branch_cfg = geo_model_cfg.get("geometry")

    _, kin_cfg = make_load_kin_model(
        kin_model_epoch, device=device
    )
    model = Model(kin_cfg["model"], geo_cfg["representation"], geo_branch_cfg).to(device=device)
    model.load_params(load_model, load_epoch)
    return model, geo_cfg


def apply_skinning(vp, vn, skin_w, bind_xform_inv, fk_T, v_mask=None):
    """
    vp: [consq_n*B, V, 3]
    vn: [consq_n*B, V, 3]
    skin_w: [consq_n*B, V, J]
    bind_xform_inv: [consq_n*B, J, 4, 4]
    fk_T: [consq_n, B, J, 4, 4]
    v_mask: [consq_n*B, V] (optional)
    """
    
    if fk_T.dim() == 5:
        # [T, B, J, 4, 4] -> [T*B, J, 4, 4]
        fk_T = fk_T.reshape(-1, fk_T.shape[2], fk_T.shape[3], fk_T.shape[4])
    elif fk_T.dim() != 4:
        return None, None

    ones = torch.ones(
        vp.shape[0], vp.shape[1], 1, dtype=vp.dtype, device=vp.device
    )
    vp_h = torch.cat([vp, ones], dim=-1)  # [B, V, 4]

    # [B, J, 4, 4] x [B, V, 4] -> [B, V, J, 4]
    xform_matrix = torch.matmul(fk_T, bind_xform_inv)  # [B, J, 4, 4]
    posed_joint = torch.einsum("bjmn,bvn->bvjm", xform_matrix, vp_h)
    posed_vp = (posed_joint[..., :3] * skin_w.unsqueeze(-1)).sum(dim=2)

    # Normals are transformed only by rotation.
    xform_rot = xform_matrix[..., :3, :3]  # [B, J, 3, 3]
    posed_joint_n = torch.einsum("bjmn,bvn->bvjm", xform_rot, vn)  # [B, V, J, 3]
    posed_vn = (posed_joint_n * skin_w.unsqueeze(-1)).sum(dim=2)
    posed_vn = F.normalize(posed_vn, dim=-1)

    # v_mask: True for padded vertices, False for valid vertices
    if v_mask is not None:
        posed_vp = posed_vp * (~v_mask).unsqueeze(-1)
        posed_vn = posed_vn * (~v_mask).unsqueeze(-1)

    return posed_vp, posed_vn

def get_bounding_boxes(vertices):
    num_people = vertices.shape[0]
    boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
    for i in range(num_people):
        boxes[i, 0, :] = vertices[i].min(dim=0)[0]
        boxes[i, 1, :] = vertices[i].max(dim=0)[0]
    return boxes

# ...existing code...

def compute_sdf(
    vp: torch.Tensor,
    vn: torch.Tensor,
    vidx_part: torch.Tensor,
    v_mask: torch.Tensor = None,
):
    """
    Approximate signed distance from (LArm+LHand) vertices to (Body+Head) surface.

    Args:
        vp: [N, V, 3]
        vn: [N, V, 3]
        vidx_part: [N, V] (PartIndex value per vertex)
        v_mask: [N, V] True = padded/invalid (optional)
        chunk_size: chunk size for nearest-neighbor search

    Returns:
        sdf: [N, V] (only LArm/LHand valid; others = +inf)
        collision_v_mask: [N, V] bool (True if sdf < 0 on LArm/LHand vertices)
        collision_batch_mask: [N] bool (any collision in each sample)
    """
    assert vp.dim() == 3 and vn.dim() == 3 and vidx_part.dim() == 2
    assert vp.shape[:2] == vn.shape[:2] == vidx_part.shape[:2]

    B, V, _ = vp.shape
    device = vp.device
    dtype = vp.dtype

    # Normalize normals for stable sign test
    vn = F.normalize(vn, dim=-1)

    # Part masks
    larm = (vidx_part == PartIndex.eLArm.value)
    lhand = (vidx_part == PartIndex.eLHand.value)
    body = (vidx_part == PartIndex.eBody.value)
    head = (vidx_part == PartIndex.eHead.value)

    src_mask = larm | lhand               # query set
    tgt_mask = body | head                # surface set

    if v_mask is not None:
        valid = ~v_mask
        src_mask = src_mask & valid
        tgt_mask = tgt_mask & valid

    sdf = torch.full((B, V), float("inf"), dtype=dtype, device=device)
    collision_v_mask = torch.zeros((B, V), dtype=torch.bool, device=device)

    for b in range(B):
        src_idx = torch.where(src_mask[b])[0]
        tgt_idx = torch.where(tgt_mask[b])[0]
        if src_idx.numel() == 0 or tgt_idx.numel() == 0:
            continue

        p = vp[b, src_idx]         # [P, 3]
        q = vp[b, tgt_idx]         # [Q, 3]
        n = vn[b, tgt_idx]         # [Q, 3]

        P = p.shape[0]
        Q = q.shape[0]

        chamfer_dist = torch.cdist(p, q)  # [P, Q]
        min_d, nn_idx = chamfer_dist.min(dim=1)  # [P], [P]
        n_nn = n[nn_idx]            # [P, 3]
        vec = p - q[nn_idx]         # [P, 3]

        # sign from target normal / magnitude from euclidean NN distance
        signed_plane = (vec * n_nn).sum(dim=-1)        # [P]
        signed_dist = torch.sign(signed_plane) * min_d # [P]

        sdf[b, src_idx] = signed_dist
        collision_v_mask[b, src_idx] = signed_dist < 0

    collision_batch_mask = collision_v_mask.any(dim=1)
    return sdf, collision_v_mask, collision_batch_mask

def export_collision_vis_mesh(
    vp: torch.Tensor,
    face: torch.Tensor,
    collision_v_mask: torch.Tensor,
    v_mask: torch.Tensor = None,
    out_path: str = "debug_collision.glb",
    emit_timestamp_copy: bool = False,
):
    """
    Export mesh with collision vertices highlighted in red, others in gray.
    
    Args:
        vp: [N, V, 3] or [V, 3]
        face: [N, F, 3] or [F, 3]
        collision_v_mask: [N, V] or [V] bool
        v_mask: [N, V] or [V] bool, True = padded/invalid (optional)
        out_path: output mesh path
    """
    import numpy as np
    import trimesh

    # Extract batch 0 if needed
    if vp.dim() == 3:
        vp = vp[0]
    if face.dim() == 3:
        face = face[0]
    if collision_v_mask.dim() == 2:
        collision_v_mask = collision_v_mask[0]
    if v_mask is not None and v_mask.dim() == 2:
        v_mask = v_mask[0]

    v = vp.detach().cpu().numpy()
    f = face.detach().cpu().numpy()
    col_mask = collision_v_mask.detach().cpu().numpy().astype(bool)

    # Valid faces only
    if f.ndim == 2:
        f = f[(f >= 0).all(axis=1)]

    # Face-discrete coloring
    Fn = f.shape[0]
    v_face = v[f.reshape(-1)]  # [Fn*3, 3]
    f_new = np.arange(Fn * 3, dtype=np.int64).reshape(Fn, 3)

    # Default gray color
    face_colors = np.full((Fn, 4), [150, 150, 150, 255], dtype=np.uint8)

    # Check if face has any collision vertex -> red
    for i in range(Fn):
        tri_verts = f[i]  # [3]
        if col_mask[tri_verts].any():
            face_colors[i] = np.array([255, 50, 50, 255], dtype=np.uint8)  # red
        elif v_mask is not None:
            if v_mask[tri_verts].any():
                face_colors[i] = np.array([0, 0, 0, 0], dtype=np.uint8)  # transparent

    vertex_colors = np.repeat(face_colors, 3, axis=0)
    mesh = trimesh.Trimesh(vertices=v_face, faces=f_new, process=False, vertex_colors=vertex_colors)
    export_info = _export_mesh_atomic(mesh, out_path, emit_timestamp_copy=emit_timestamp_copy)
    print(
        f"[export] collision mesh saved to {export_info['main_path']} "
        f"(mtime={export_info['main_mtime']:.6f})"
    )



def export_collision_spheres_vis_mesh(
    vp: torch.Tensor,
    face: torch.Tensor,
    collision_v_mask: torch.Tensor,
    v_mask: torch.Tensor = None,
    out_path: str = "debug_collision_spheres.glb",
    emit_timestamp_copy: bool = False,
):
    """
    Keep the character mesh unchanged and place red spheres on collision vertices.
    """
    import numpy as np
    import trimesh

    if vp.dim() == 3:
        vp = vp[0]
    if face.dim() == 3:
        face = face[0]
    if collision_v_mask.dim() == 2:
        collision_v_mask = collision_v_mask[0]
    if v_mask is not None and v_mask.dim() == 2:
        v_mask = v_mask[0]

    v = vp.detach().cpu().numpy()
    f = face.detach().cpu().numpy()
    col = collision_v_mask.detach().cpu().numpy().astype(bool)

    if v_mask is not None:
        valid_v = ~v_mask.detach().cpu().numpy().astype(bool)
        col = col & valid_v

    vertex_colors = np.full((v.shape[0], 4), [150, 150, 150, 255], dtype=np.uint8)
    vertex_colors[col] = np.array([255, 50, 50, 255], dtype=np.uint8)  # red

    # valid faces only
    if f.ndim == 2:
        f = f[(f >= 0).all(axis=1)]

    # base character mesh: keep as-is
    base_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, vertex_colors=vertex_colors)

    scene = trimesh.Scene()
    scene.add_geometry(base_mesh, node_name="character")

    export_info = _export_mesh_atomic(scene, out_path, emit_timestamp_copy=emit_timestamp_copy)
    print(
        f"[export] collision spheres saved to {export_info['main_path']} "
        f"(mtime={export_info['main_mtime']:.6f})"
    )


def export_collision_vertex_pair_vis_mesh(
    vp: torch.Tensor,
    face: torch.Tensor,
    pen_mask: torch.Tensor,
    pen_disp: torch.Tensor,
    v_mask: torch.Tensor = None,
    out_path: str = "debug_collision_pairs.glb",
    pair_meta_path: str = "debug_collision_pairs.json",
    sphere_radius: float = 0.001,
    line_radius: float = .5,
    max_pairs: int = 1024,
    emit_timestamp_copy: bool = False,
):
    """
    Export collision-line visualization from penetration mask and displacement.
    - penetrated vertices: red
    - displacement endpoints: blue
    - displacement lines: yellow
    Also saves line metadata to JSON.
    """
    import numpy as np
    import trimesh

    if vp.dim() == 3:
        vp = vp[0]
    if face.dim() == 3:
        face = face[0]
    if pen_mask.dim() == 2:
        pen_mask = pen_mask[0]
    if pen_disp.dim() == 3:
        pen_disp = pen_disp[0]
    if v_mask is not None and v_mask.dim() == 2:
        v_mask = v_mask[0]

    v = vp.detach().cpu().numpy()
    f = face.detach().cpu().numpy()
    pen = pen_mask.detach().cpu().numpy().astype(bool)
    disp = pen_disp.detach().cpu().numpy()

    if v_mask is not None:
        valid = ~v_mask.detach().cpu().numpy().astype(bool)
    else:
        valid = np.ones(v.shape[0], dtype=bool)

    disp_norm = np.linalg.norm(disp, axis=1)
    src_idx = np.where(pen & valid & (disp_norm > 1e-8))[0]
    total_pen_vertices = int(src_idx.size)
    if src_idx.size > max_pairs:
        src_idx = src_idx[:max_pairs]

    pair_rows = []
    pair_lines = []
    end_points = np.zeros((0, 3), dtype=np.float32)
    end_near_idx = np.array([], dtype=np.int64)
    if src_idx.size > 0:
        src_points = v[src_idx]
        src_disp = disp[src_idx]
        end_points = src_points + src_disp

        if valid.any():
            valid_idx = np.where(valid)[0]
            valid_v = v[valid_idx]
            dmat_end = np.linalg.norm(end_points[:, None, :] - valid_v[None, :, :], axis=-1)
            nn_local = dmat_end.argmin(axis=1)
            end_near_idx = np.unique(valid_idx[nn_local])

        for si, p0, d0, p1 in zip(src_idx.tolist(), src_points, src_disp, end_points):
            pair_rows.append(
                {
                    "src_vertex": int(si),
                    "pen_disp": [float(d0[0]), float(d0[1]), float(d0[2])],
                    "pen_disp_norm": float(np.linalg.norm(d0)),
                    "line_start": [float(p0[0]), float(p0[1]), float(p0[2])],
                    "line_end": [float(p1[0]), float(p1[1]), float(p1[2])],
                }
            )
            pair_lines.append((p0, p1))

    # valid faces only
    if f.ndim == 2:
        f = f[(f >= 0).all(axis=1)]

    vertex_colors = np.full((v.shape[0], 4), [255, 255, 255, 255], dtype=np.uint8)
    vertex_colors[src_idx] = np.array([255, 0, 0, 255], dtype=np.uint8)  # source: red
    if end_near_idx.size > 0:
        vertex_colors[end_near_idx] = np.array([255, 50, 50, 255], dtype=np.uint8)  # end-near: blue
        vertex_colors[src_idx] = np.array([255, 50, 50, 255], dtype=np.uint8)  # keep source red
    if v_mask is not None:
        vertex_colors[~valid] = np.array([0, 0, 0, 0], dtype=np.uint8)

    base_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, vertex_colors=vertex_colors)
    scene = trimesh.Scene()
    scene.add_geometry(base_mesh, node_name="character")

    # src_sphere = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
    # src_sphere.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
    # for i, vid in enumerate(src_idx.tolist()):
    #     s = src_sphere.copy()
    #     s.apply_translation(v[vid])
    #     scene.add_geometry(s, node_name=f"src_col_{i}")

    # if end_points.shape[0] > 0:
    #     end_sphere = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius * 0.85)
    #     end_sphere.visual.vertex_colors = np.array([0, 80, 255, 255], dtype=np.uint8)
    #     for i, p in enumerate(end_points):
    #         s = end_sphere.copy()
    #         s.apply_translation(p)
    #         scene.add_geometry(s, node_name=f"col_end_{i}")

    # for i, (p0, p1) in enumerate(pair_lines):
    #     seg = p1 - p0
    #     seg_len = np.linalg.norm(seg)
    #     if seg_len < 1e-8:
    #         continue
    #     cyl = trimesh.creation.cylinder(radius=line_radius, height=float(seg_len), sections=8)
    #     z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    #     dir_axis = (seg / seg_len).astype(np.float32)
    #     tf = trimesh.geometry.align_vectors(z_axis, dir_axis)
    #     if tf is None:
    #         tf = np.eye(4, dtype=np.float32)
    #     tf[:3, 3] = (p0 + p1) * 0.5
    #     cyl.apply_transform(tf)
    #     cyl.visual.vertex_colors = np.array([255, 220, 60, 220], dtype=np.uint8)
    #     scene.add_geometry(cyl, node_name=f"pair_line_{i}")

    export_info = _export_mesh_atomic(scene, out_path, emit_timestamp_copy=emit_timestamp_copy)

    meta = {
        "mesh_path": out_path,
        "num_pairs": len(pair_rows),
        "max_pairs": int(max_pairs),
        # "num_penetration_vertices": total_pen_vertices,
        "num_exported_vertices": len(pair_rows),
        "num_end_near_vertices": int(end_near_idx.size),
        "end_near_vertices": end_near_idx.tolist(),
        "pairs": pair_rows,
    }
    with open(pair_meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=2)

    print(
        f"[export] collision pair mesh saved to {export_info['main_path']} "
        f"(mtime={export_info['main_mtime']:.6f}), pairs={len(pair_rows)}"
    )
    print(f"[export] collision pair metadata saved to {pair_meta_path}")


def export_part_vis_obj(
    vp: torch.Tensor,
    face: torch.Tensor,
    vidx_part: torch.Tensor,
    v_mask: torch.Tensor = None,
    out_path: str = "debug_part.glb",
    emit_timestamp_copy: bool = False,
):
    """
    Color vertices by PartIndex and export mesh.

    Args:
        vp: [V, 3] or [N, V, 3]
        face: [F, 3] or [N, F, 3]
        vidx_part: [V] or [N, V]
        v_mask: [V] or [N, V], True = padded/invalid (optional)
        out_path: output mesh path (.obj/.glb/.ply ...)
    """
    import numpy as np
    import trimesh

    # If batched, use first sample
    if vp.dim() == 3:
        vp = vp[0]
    if face.dim() == 3:
        face = face[0]
    if vidx_part.dim() == 2:
        vidx_part = vidx_part[0]
    if v_mask is not None and v_mask.dim() == 2:
        v_mask = v_mask[0]

    v = vp.detach().cpu().numpy()
    f = face.detach().cpu().numpy()
    p = vidx_part.detach().cpu().numpy()

    # valid faces only
    if f.ndim == 2:
        f = f[(f >= 0).all(axis=1)]

    # RGBA palette by part index
    part_colors = {
        PartIndex.eBody.value:  np.array([180, 180, 180, 255], dtype=np.uint8),  # gray
        PartIndex.eHead.value:  np.array([255, 220, 120, 255], dtype=np.uint8),  # yellow
        PartIndex.eLArm.value:  np.array([255, 80, 80, 255], dtype=np.uint8),    # red
        PartIndex.eRArm.value:  np.array([80, 80, 255, 255], dtype=np.uint8),    # blue
        PartIndex.eLLeg.value:  np.array([80, 220, 80, 255], dtype=np.uint8),    # green
        PartIndex.eRLeg.value:  np.array([180, 80, 255, 255], dtype=np.uint8),   # purple
        PartIndex.eLHand.value: np.array([255, 140, 140, 255], dtype=np.uint8),  # light red
        PartIndex.eRHand.value: np.array([140, 140, 255, 255], dtype=np.uint8),  # light blue
    }

    colors = np.zeros((v.shape[0], 4), dtype=np.uint8)
    colors[:] = np.array([40, 40, 40, 255], dtype=np.uint8)  # unknown/default

    for part_idx, c in part_colors.items():
        colors[p == part_idx] = c

    # padded/invalid vertices (optional): transparent black
    if v_mask is not None:
        m = v_mask.detach().cpu().numpy().astype(bool)
        colors[m] = np.array([0, 0, 0, 0], dtype=np.uint8)

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False, vertex_colors=colors)
    export_info = _export_mesh_atomic(mesh, out_path, emit_timestamp_copy=emit_timestamp_copy)
    print(
        f"[export] part mesh saved to {export_info['main_path']} "
        f"(mtime={export_info['main_mtime']:.6f})"
    )


def _export_mesh_atomic(mesh_or_scene, out_path: str, emit_timestamp_copy: bool = False):
    """
    Export via temp file + atomic replace.
    This improves file-change detection in editors/viewers when overwriting the same filename.
    """
    out_dir = os.path.dirname(out_path) or "."
    out_base = os.path.basename(out_path)
    stem, ext = os.path.splitext(out_base)
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=f".{stem}.", suffix=ext, dir=out_dir)
    os.close(fd)

    ts_path = None
    try:
        mesh_or_scene.export(tmp_path)
        os.replace(tmp_path, out_path)
        os.utime(out_path, None)

        if emit_timestamp_copy:
            ts_path = os.path.join(out_dir, f"{stem}_{int(time.time() * 1000)}{ext}")
            mesh_or_scene.export(ts_path)
            os.utime(ts_path, None)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    out_stat = os.stat(out_path)
    result = {
        "main_path": out_path,
        "main_inode": out_stat.st_ino,
        "main_mtime": out_stat.st_mtime,
        "main_mtime_ns": out_stat.st_mtime_ns,
        "timestamp_copy_path": ts_path,
    }
    if ts_path is not None and os.path.exists(ts_path):
        ts_stat = os.stat(ts_path)
        result["timestamp_copy_mtime"] = ts_stat.st_mtime
        result["timestamp_copy_mtime_ns"] = ts_stat.st_mtime_ns
    return result


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
    out["vp"], out["vn"] = apply_skinning(
        tgt_batch.vp,
        tgt_batch.vn,
        tgt_batch.skin_w,
        tgt_batch.bind_xform_inv,
        out["fk_T"],
        v_mask=tgt_batch.v_mask,
    )
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
    gt["vidx_part"] = tgt_batch.vidx_part
    gt["v_mask"] = tgt_batch.v_mask
    return out, gt
