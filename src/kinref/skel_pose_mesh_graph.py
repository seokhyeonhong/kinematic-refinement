import torch
from typing import Iterable, List, Union

rep_dim = {
    "lo": 3,
    "go": 3,
    "q": 6,
    "r": 4,
    "c": 1,
    "p": 3,
    "pprev": 3,
    "pv": 3,
    "qv": 6,

    "vp": 3,
    "vn": 3,
}


class SkelPoseMeshGraph:
    """Single skeleton-pose sample.

    Node-aligned tensors are stored as [J, D]. Batching pads them to [B, J, D].
    """

    skel_cfg: List[str] = []
    pose_cfg: List[str] = []
    mesh_cfg: List[str] = []
    ms_dict = {}

    def __init__(self, skel_data=None, pose_data=None, mesh_data=None):
        self.skel_data = skel_data
        self.pose_data = pose_data
        self.mesh_data = mesh_data

        if skel_data is not None:
            self.lo = skel_data.lo
            self.go = skel_data.go
            self.edge_index = skel_data.edge_index  # [2, E]
            self.edge_feature = skel_data.edge_feature  # [E, F]
            self.qb = skel_data.qb

        if pose_data is not None:
            self.q = pose_data.q
            self.p = pose_data.p
            self.qv = pose_data.qv
            self.pv = pose_data.pv
            self.pprev = pose_data.pprev
            self.c = pose_data.c
            self.r_nopad = pose_data.r  # [1, 4]
        
        if mesh_data is not None:
            self.vp = mesh_data.vp         # [V, 3], rest-pose vertex positions
            self.vn = mesh_data.vn         # [V, 3], rest-pose vertex normals
            self.bind_xform_inv = mesh_data.bind_xform_inv # [J, 4, 4], inverse bind pose transforms
            self.skin_w = mesh_data.skin_w # [V, J], skinning weights
            self.vidx_part = mesh_data.vidx_part # [V,], associated joint class for each vertex
            self.face = mesh_data.face     # [F, 3], triangle face indices
            self.name = mesh_data.name     # str, optional name for the mesh
        
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to(self, device: Union[torch.device, str]):
        for key, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self

    def clone(self):
        cloned = self.__class__()
        for key, value in self.__dict__.items():
            setattr(cloned, key, value.clone() if torch.is_tensor(value) else value)
        return cloned

    @property
    def device(self):
        return self.lo.device

    @property
    def num_joints(self) -> int:
        return int(self.lo.shape[0])
    
    @property
    def num_verts(self) -> int:
        return int(self.vp.shape[0])
    
    @property
    def num_faces(self) -> int:
        return int(self.face.shape[0])

    def normalize_x(self, key):
        val = getattr(self, key)
        if key + "_m" in self.ms_dict:
            m = self.ms_dict[key + "_m"].to(device=val.device)
            s = self.ms_dict[key + "_s"].to(device=val.device)
            val = (val - m) / s
        return val

    @property
    def skel_x(self):
        assert len(self.skel_cfg) > 0, "skel_cfg is not set"
        return torch.hstack([self.normalize_x(var) for var in self.skel_cfg])

    @property
    def pose_x(self):
        assert len(self.pose_cfg) > 0, "pose_cfg is not set"
        return torch.hstack([self.normalize_x(var) for var in self.pose_cfg])

    @property
    def src_x(self):
        return torch.hstack((self.skel_x, self.pose_x))

    @property
    def tgt_x(self):
        return self.skel_x
    
    @property
    def mesh_x(self):
        assert len(self.mesh_cfg) > 0, "mesh_cfg is not set"
        return torch.hstack([getattr(self, var) for var in self.mesh_cfg])

    @property
    def parent_index(self):
        parent = torch.arange(self.num_joints, device=self.device, dtype=torch.long)
        if hasattr(self, "edge_index"):
            parent[self.edge_index[1]] = self.edge_index[0]
        return parent

    @property
    def skel_depth(self):
        return self.edge_feature[:, 0]
    
    @property
    def mask(self):
        return torch.zeros(self.num_joints, dtype=torch.bool, device=self.device)
    
    @property
    def v_mask(self):
        if hasattr(self, "vp"):
            return torch.zeros(self.num_verts, dtype=torch.bool, device=self.device)
        return None

    @property
    def end_effector_mask(self):
        return self.edge_feature[:, 1] == 0

    @property
    def r(self):
        r_ = (
            self.ms_dict["r_m"]
            .repeat(self.num_joints, 1)
            .to(dtype=self.r_nopad.dtype, device=self.r_nopad.device)
        )
        r_[0] = self.r_nopad[0]
        return r_


class SkelPoseMeshBatch:
    """Padded batch container.

    Node-aligned tensors are padded to [B, J, D] and accompanied by a `mask` of shape [B, J].
    """

    @classmethod
    def from_data_list(cls, data_list: Iterable[SkelPoseMeshGraph]):
        data_list = list(data_list)
        if len(data_list) == 0:
            raise ValueError("data_list must be non-empty")

        batch = cls()
        device = data_list[0].device
        B = len(data_list)
        num_joints = torch.tensor([g.num_joints for g in data_list], dtype=torch.long, device=device)
        num_verts = torch.tensor([g.num_verts for g in data_list], dtype=torch.long, device=device)
        num_faces = torch.tensor([g.num_faces for g in data_list], dtype=torch.long, device=device)
        max_joints = int(num_joints.max().item())
        max_verts = int(num_verts.max().item())
        max_faces = int(num_faces.max().item())
        # ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), num_joints.cumsum(0)], dim=0)

        def pad_field(name, max_elements, trailing_shape, dtype, fill_value=0):
            shape = (B, max_elements) + trailing_shape
            return torch.full(shape, fill_value=fill_value, dtype=dtype, device=device)
        
        # skeleton fields
        batch.lo = pad_field("lo", max_joints, (3,), torch.float32)
        batch.go = pad_field("go", max_joints, (3,), torch.float32)
        batch.qb = pad_field("qb", max_joints, tuple(), torch.bool, False)

        # pose fields
        batch.q = pad_field("q", max_joints, (6,), torch.float32)
        batch.p = pad_field("p", max_joints, (3,), torch.float32)
        batch.qv = pad_field("qv", max_joints, (6,), torch.float32)
        batch.pv = pad_field("pv", max_joints, (3,), torch.float32)
        batch.pprev = pad_field("pprev", max_joints, (3,), torch.float32)
        batch.c = pad_field("c", max_joints, (1,), torch.bool, False)
        batch.mask = pad_field("mask", max_joints, tuple(), torch.bool, True)
        batch.r_nopad = torch.zeros(B, 4, dtype=torch.float32, device=device)
            
        batch.parent_index = pad_field("parent_index", max_joints, tuple(), torch.long, -1)
        batch.skel_depth = pad_field("skel_depth", max_joints, tuple(), torch.long, -1)
        batch.edge_feature = pad_field("edge_feature", max_joints, (2,), torch.float32, -1)

        # mesh fields
        batch.vp = pad_field("vp", max_verts, (3,), torch.float32)
        batch.vn = pad_field("vn", max_verts, (3,), torch.float32)
        batch.skin_w = pad_field("skin_w", max_verts, (max_joints,), torch.float32)
        batch.bind_xform_inv = pad_field("bind_xform_inv", max_joints, (4, 4), torch.float32)
        batch.vidx_part = pad_field("vidx_part", max_verts, tuple(), torch.long, -1)
        batch.face = pad_field("face", max_faces, (3,), torch.long, -1)
        batch.v_mask = pad_field("v_mask", max_verts, tuple(), torch.bool, True)
        batch.name = []
        
        # # legacy flattened helpers
        # batch.ptr = ptr
        # batch.batch = torch.repeat_interleave(torch.arange(B, device=device), lengths)

        for b, graph in enumerate(data_list):
            j = graph.num_joints

            if graph.skel_data is not None:
                batch.lo[b, :j] = graph.lo
                batch.go[b, :j] = graph.go
                batch.qb[b, :j] = graph.qb

                batch.mask[b, :j] = graph.mask
                batch.parent_index[b, :j] = graph.parent_index
                batch.skel_depth[b, :j] = graph.skel_depth

                ef = graph.edge_feature
                if ef.shape[-1] >= 2:
                    batch.edge_feature[b, :j, :2] = ef[:, :2]
                else:
                    batch.edge_feature[b, :j, 0] = ef[:, 0]

            if graph.pose_data is not None:
                batch.q[b, :j] = graph.q
                batch.p[b, :j] = graph.p
                batch.qv[b, :j] = graph.qv
                batch.pv[b, :j] = graph.pv
                batch.pprev[b, :j] = graph.pprev
                batch.c[b, :j] = graph.c
                batch.r_nopad[b] = graph.r_nopad[0]

            if graph.mesh_data is not None:
                v = graph.num_verts
                f = graph.num_faces
                batch.vp[b, :v] = graph.vp
                batch.vn[b, :v] = graph.vn
                batch.skin_w[b, :v, :j] = graph.skin_w
                batch.bind_xform_inv[b, :j] = graph.bind_xform_inv
                batch.face[b, :f] = graph.face
                batch.v_mask[b, :v] = graph.v_mask
                batch.vidx_part[b, :v] = graph.vidx_part
                batch.name.append(graph.name)
                
        return batch

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to(self, device: Union[torch.device, str]):
        for key, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self

    @property
    def device(self):
        return self.lo.device

    @property
    def shape(self):
        return self.lo.shape[:2]

    @property
    def num_graphs(self):
        return int(self.lo.shape[0])

    def normalize_x(self, key):
        val = getattr(self, key)
        if key + "_m" in SkelPoseMeshGraph.ms_dict:
            m = SkelPoseMeshGraph.ms_dict[key + "_m"].to(device=val.device)
            s = SkelPoseMeshGraph.ms_dict[key + "_s"].to(device=val.device)
            view_shape = (1,) * (val.ndim - 1) + (-1,)
            val = (val - m.view(view_shape)) / s.view(view_shape)
        return val

    @property
    def skel_x(self):
        assert len(SkelPoseMeshGraph.skel_cfg) > 0, "skel_cfg is not set"
        return torch.cat([self.normalize_x(var) for var in SkelPoseMeshGraph.skel_cfg], dim=-1)

    @property
    def pose_x(self):
        assert len(SkelPoseMeshGraph.pose_cfg) > 0, "pose_cfg is not set"
        return torch.cat([self.normalize_x(var) for var in SkelPoseMeshGraph.pose_cfg], dim=-1)

    @property
    def src_x(self):
        return torch.cat((self.skel_x, self.pose_x), dim=-1)

    @property
    def tgt_x(self):
        return self.skel_x

    @property
    def mesh_x(self):
        assert len(SkelPoseMeshGraph.mesh_cfg) > 0, "mesh_cfg is not set"
        return torch.cat([getattr(self, var) for var in SkelPoseMeshGraph.mesh_cfg], dim=-1)

    @property
    def end_effector_mask(self):
        return self.edge_feature[..., 1] == 0

    @property
    def r(self):
        r_mean = SkelPoseMeshGraph.ms_dict["r_m"].to(dtype=self.r_nopad.dtype, device=self.r_nopad.device)
        r_ = r_mean.view(1, 1, -1).repeat(self.num_graphs, self.lo.shape[1], 1)
        r_[:, 0] = self.r_nopad
        return r_


def rnd_mask(B_skel, consq_n, mask_prob=0.5, edge_thres=4, demo=None):
    # mask single frame and repeat (to avoid flickering masks for the same joints among consecutive frames)
    device = B_skel.lo.device
    B, J = B_skel.mask.shape
    if demo == "no_mask":
        return torch.zeros(B, J, device=device, dtype=torch.bool)

    nB_sf = B // consq_n
    mask_sf = torch.zeros((nB_sf, J), device=device, dtype=torch.bool)
    do_mask = torch.rand(nB_sf, device=device) < mask_prob

    for b in range(nB_sf):
        if not do_mask[b]:
            continue

        ee = torch.where(~B_skel.mask[b] & B_skel.end_effector_mask[b])[0]
        if len(ee) == 0:
            continue

        sel = ee[torch.randint(0, len(ee), (1,), device=device)].item()
        reach = int(torch.randint(0, edge_thres, (1,), device=device).item())
        cur = sel
        for _ in range(reach + 1):
            if B_skel.mask[b, cur]: # already masked by other path
                break
            mask_sf[b, cur] = True
            parent = int(B_skel.parent_index[b, cur].item())
            if parent == cur: # root node
                break
            cur = parent

    return mask_sf.repeat(consq_n, 1)


def find_ee(skel_graph):
    if isinstance(skel_graph, SkelPoseMeshBatch):
        return skel_graph.end_effector_mask
    return torch.where(skel_graph.end_effector_mask)[0]


def find_feet(skel_graph):
    # CAUTION; this function assumes a single skel
    ee = find_ee(skel_graph)
    go = skel_graph.go
    left_foot, right_foot = None, None
    for foot in ee[torch.argsort(go[ee, 1])[:2]]:
        if go[foot, 0] > 0:
            left_foot = foot
        else:
            right_foot = foot
    assert left_foot != right_foot
    assert (left_foot is not None) and (right_foot is not None)
    return left_foot, right_foot
