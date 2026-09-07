import os
from os.path import join as pjoin
import sys
import argparse
import subprocess
import traceback
import faulthandler

import numpy as np
import torch

from aPyOpenGL import agl

# from mypath import DATA_DIR

faulthandler.enable()



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


def show_part_mesh(
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
    mesh.export(out_path)
    # breakpoint()

JOINT_GROUPS = {
    "body": {"Hips", "Spine", "Spine1", "Spine2"},
    "head": {"Neck", "Head", "Head_End"},
    "lleg": {
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToe",
        "LeftToeBase",
        "LeftToeBase_End",
    },
    "rleg": {
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToe",
        "RightToeBase",
        "RightToeBase_End",
    },
    "larm": {
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
    },
    "lhand": {
        "LeftHand",
        "LeftHand_End",
    },
    "rarm": {
        "RightShoulder",
        "RightArm",
        "RightForeArm",
    },
    "rhand": {
        "RightHand",
        "RightHand_End",
    }
}
PART_NAMES = tuple(JOINT_GROUPS.keys())
PART_TO_COLUMN = {part_name: idx for idx, part_name in enumerate(PART_NAMES)}
PART_NAME_TO_INDEX = {
    "body": 0,
    "head": 1,
    "larm": 2,
    "rarm": 3,
    "lleg": 4,
    "rleg": 5,
    "lhand": 6,
    "rhand": 7,
}
JOINT_NAME_REPLACEMENTS = {
    "Toe_End": "ToeBase_End",
    "HeadTop_End": "Head_End",
}
def normalize_joint_name(joint_name, strip_namespace=False):
    if joint_name is None:
        return None

    normalized_name = joint_name.split(":")[-1] if strip_namespace else joint_name
    for source_name, target_name in JOINT_NAME_REPLACEMENTS.items():
        normalized_name = normalized_name.replace(source_name, target_name)
    return normalized_name


def load_template_joint_names(joint_pos_txt):
    template_joint_names = []
    with open(joint_pos_txt, "r") as file:
        for line in file:
            joint_name = line.split(",")[0].strip()
            if joint_name:
                template_joint_names.append(normalize_joint_name(joint_name))
    return template_joint_names


def extract_submesh(vertices, faces, selected_indices):
    selected_indices = np.asarray(selected_indices, dtype=np.int32)
    selected_vertices = vertices[selected_indices]

    if selected_indices.size == 0:
        return selected_vertices, np.empty((0, 3), dtype=np.int32)

    selected_mask = np.zeros(len(vertices), dtype=bool)
    selected_mask[selected_indices] = True

    face_mask = selected_mask[faces].all(axis=1)
    selected_faces = faces[face_mask]

    vertex_remap = -np.ones(len(vertices), dtype=np.int32)
    vertex_remap[selected_indices] = np.arange(len(selected_indices), dtype=np.int32)
    return selected_vertices, vertex_remap[selected_faces]


def get_vertex_skinning(vert):
    skin_indices = [
        vert.skinning_indices1.x,
        vert.skinning_indices1.y,
        vert.skinning_indices1.z,
        vert.skinning_indices1.w,
        vert.skinning_indices2.x,
        vert.skinning_indices2.y,
        vert.skinning_indices2.z,
        vert.skinning_indices2.w,
    ]
    skin_weights = [
        vert.skinning_weights1.x,
        vert.skinning_weights1.y,
        vert.skinning_weights1.z,
        vert.skinning_weights1.w,
        vert.skinning_weights2.x,
        vert.skinning_weights2.y,
        vert.skinning_weights2.z,
        vert.skinning_weights2.w,
    ]
    return skin_indices, skin_weights


def collect_mesh_vertex_data(mesh, basis_pos, num_joints, joint_name_to_index):
    mesh_gl = mesh.mesh_gl
    num_vertices = len(mesh_gl.vertices)

    vertex_positions, vertex_normals, vertex_uvs, vertex_albedos = [], [], [], []
    vertex_skin_weights = np.zeros((num_vertices, num_joints), dtype=np.float32)

    for vertex_idx, vert in enumerate(mesh_gl.vertices):
        vertex_positions.append(vert.position - basis_pos)
        vertex_normals.append(vert.normal)
        vertex_uvs.append(vert.uv)
        vertex_albedos.append(mesh.materials[vert.material_id].albedo)

        skin_indices, skin_weights = get_vertex_skinning(vert)
        for local_joint_idx, weight in zip(skin_indices, skin_weights):
            if local_joint_idx == -1:
                continue

            joint_name = mesh_gl.joint_names[local_joint_idx]
            vertex_skin_weights[vertex_idx, joint_name_to_index[joint_name]] = weight

    return (
        mesh_gl,
        vertex_positions,
        vertex_normals,
        vertex_uvs,
        vertex_albedos,
        vertex_skin_weights,
    )


def collect_control_point_data(
    mesh_gl,
    vertex_positions,
    vertex_normals,
    vertex_uvs,
    vertex_albedos,
    vertex_skin_weights,
    control_point_offset,
):
    rest_positions, rest_normals = [], []
    rest_skin_weights, rest_uvs, rest_albedos = [], [], []

    control_point_to_vertices = [
        list(vertex_ids)
        for _, vertex_ids in sorted(mesh_gl.control_point_idx_to_vertex_idx.items())
    ]
    for vertex_ids in control_point_to_vertices:
        rest_positions.append(vertex_positions[vertex_ids[0]])

        averaged_normal = np.mean(
            np.stack([vertex_normals[vertex_idx] for vertex_idx in vertex_ids], axis=0),
            axis=0,
        )
        normal_norm = np.linalg.norm(averaged_normal)
        if normal_norm > 0:
            averaged_normal = averaged_normal / normal_norm

        rest_normals.append(averaged_normal)
        rest_skin_weights.append(vertex_skin_weights[vertex_ids[0]])
        rest_uvs.append(vertex_uvs[vertex_ids[0]])
        rest_albedos.append(vertex_albedos[vertex_ids[0]])

    vertex_to_control_point = [
        control_point_idx
        for _, control_point_idx in sorted(mesh_gl.vertex_idx_to_control_point_idx.items())
    ]
    faces = []
    for vertex_start in range(0, len(vertex_to_control_point), 3):
        faces.append(
            np.array(
                [
                    vertex_to_control_point[vertex_start + 0] + control_point_offset,
                    vertex_to_control_point[vertex_start + 1] + control_point_offset,
                    vertex_to_control_point[vertex_start + 2] + control_point_offset,
                ],
                dtype=np.int32,
            )
        )

    return {
        "rest_positions": rest_positions,
        "rest_normals": rest_normals,
        "rest_skin_weights": rest_skin_weights,
        "rest_uvs": rest_uvs,
        "rest_albedos": rest_albedos,
        "faces": faces,
        "num_control_points": len(control_point_to_vertices),
    }


def collect_rest_mesh_features(model):
    skeleton = model.skeleton
    raw_joint_names = [joint.name for joint in skeleton.joints]
    parent_indices = skeleton.parent_idx
    joint_name_to_index = {joint_name: idx for idx, joint_name in enumerate(raw_joint_names)}
    num_joints = len(raw_joint_names)

    basis_pos = skeleton.joints[0].local_pos * np.array([1, 0, 1])
    bind_xform_inv = np.repeat(np.eye(4, dtype=np.float32)[None, ...], num_joints, axis=0)

    rest_positions, rest_normals = [], []
    rest_skin_weights, rest_uvs, rest_albedos = [], [], []
    faces = []

    control_point_offset = 0
    for mesh in model.meshes:
        (
            mesh_gl,
            vertex_positions,
            vertex_normals,
            vertex_uvs,
            vertex_albedos,
            vertex_skin_weights,
        ) = collect_mesh_vertex_data(
            mesh,
            basis_pos,
            num_joints,
            joint_name_to_index,
        )

        control_point_data = collect_control_point_data(
            mesh_gl,
            vertex_positions,
            vertex_normals,
            vertex_uvs,
            vertex_albedos,
            vertex_skin_weights,
            control_point_offset,
        )
        rest_positions.extend(control_point_data["rest_positions"])
        rest_normals.extend(control_point_data["rest_normals"])
        rest_skin_weights.extend(control_point_data["rest_skin_weights"])
        rest_uvs.extend(control_point_data["rest_uvs"])
        rest_albedos.extend(control_point_data["rest_albedos"])
        faces.extend(control_point_data["faces"])
        control_point_offset += control_point_data["num_control_points"]

        for local_joint_idx, joint_name in enumerate(mesh_gl.joint_names):
            global_joint_idx = joint_name_to_index[joint_name]
            bind_xform_inv[global_joint_idx] = np.asarray(
                mesh_gl.bind_xform_inv[local_joint_idx], dtype=np.float32
            )

    mesh_features = {
        "vp": np.stack(rest_positions, axis=0).astype(np.float32),
        "vn": np.stack(rest_normals, axis=0).astype(np.float32),
        "skin_w": np.stack(rest_skin_weights, axis=0).astype(np.float32),
        "uv": np.stack(rest_uvs, axis=0).astype(np.float32),
        "albedo": np.stack(rest_albedos, axis=0).astype(np.float32),
        "face": np.stack(faces, axis=0).astype(np.int32),
    }
    return raw_joint_names, parent_indices, bind_xform_inv, mesh_features


def find_valid_parent_joint(parent_indices, normalized_joint_names, template_joint_set, joint_idx):
    parent_idx = parent_indices[joint_idx]
    while parent_idx != -1 and normalized_joint_names[parent_idx] not in template_joint_set:
        parent_idx = parent_indices[parent_idx]

    if parent_idx == -1:
        raise ValueError(
            f"Could not find a valid template parent for joint {normalized_joint_names[joint_idx]}"
        )
    return parent_idx


def get_joint_group_column(joint_name):
    for part_name, joint_names in JOINT_GROUPS.items():
        if joint_name in joint_names:
            return PART_TO_COLUMN[part_name]
    raise ValueError(f"Joint annotation error: {joint_name}")


def build_joint_annotations(joint_names):
    joint_annotations = np.zeros((len(joint_names), len(PART_NAMES)), dtype=np.float32)
    for joint_idx, joint_name in enumerate(joint_names):
        joint_annotations[joint_idx, get_joint_group_column(joint_name)] = 1.0
    return joint_annotations


def align_joint_data_to_template(
    raw_joint_names,
    parent_indices,
    template_joint_names,
    skin_weights,
    bind_xform_inv,
):
    normalized_joint_names = [
        normalize_joint_name(joint_name, strip_namespace=True)
        for joint_name in raw_joint_names
    ]
    template_joint_set = set(template_joint_names)

    valid_indices = []
    for joint_idx, joint_name in enumerate(normalized_joint_names):
        if joint_name in template_joint_set:
            valid_indices.append(joint_idx)
            continue

        valid_parent_idx = find_valid_parent_joint(
            parent_indices,
            normalized_joint_names,
            template_joint_set,
            joint_idx,
        )
        skin_weights[:, valid_parent_idx] += skin_weights[:, joint_idx]

    valid_joint_names = [normalized_joint_names[idx] for idx in valid_indices]
    valid_skin_weights = skin_weights[:, valid_indices]
    valid_bind_xform_inv = bind_xform_inv[valid_indices]
    valid_joint_annotations = build_joint_annotations(valid_joint_names)

    valid_joint_name_to_index = {
        joint_name: idx for idx, joint_name in enumerate(valid_joint_names)
    }
    ordered_skin_weights = np.zeros(
        (valid_skin_weights.shape[0], len(template_joint_names)),
        dtype=valid_skin_weights.dtype,
    )
    ordered_bind_xform_inv = np.zeros(
        (len(template_joint_names), 4, 4),
        dtype=valid_bind_xform_inv.dtype,
    )
    ordered_joint_annotations = np.zeros(
        (len(template_joint_names), len(PART_NAMES)),
        dtype=valid_joint_annotations.dtype,
    )

    previous_bind_xform = None
    for template_idx, joint_name in enumerate(template_joint_names):
        source_idx = valid_joint_name_to_index.get(joint_name)
        if source_idx is None:
            if previous_bind_xform is None:
                raise ValueError(
                    f"Cannot infer bind transform for missing template joint {joint_name}"
                )
            ordered_bind_xform_inv[template_idx] = previous_bind_xform
            ordered_joint_annotations[template_idx, get_joint_group_column(joint_name)] = 1.0
            continue

        ordered_skin_weights[:, template_idx] = valid_skin_weights[:, source_idx]
        ordered_bind_xform_inv[template_idx] = valid_bind_xform_inv[source_idx]
        ordered_joint_annotations[template_idx] = valid_joint_annotations[source_idx]
        previous_bind_xform = valid_bind_xform_inv[source_idx]

    return ordered_skin_weights, ordered_bind_xform_inv, ordered_joint_annotations


def compute_vertex_part_indices(skin_weights, joint_annotations):
    dominant_joint_indices = np.argmax(skin_weights, axis=1)
    dominant_part_columns = np.argmax(joint_annotations[dominant_joint_indices], axis=1)

    vertex_part_indices = np.full((skin_weights.shape[0],), -1, dtype=np.int32)
    for part_name, part_column in PART_TO_COLUMN.items():
        part_mask = dominant_part_columns == part_column
        vertex_part_indices[part_mask] = PART_NAME_TO_INDEX[part_name]
    return vertex_part_indices


def compute_part_face_indices(vertices, faces, vertex_part_indices):
    part_face_indices = {}
    for part_name in PART_NAMES:
        selected_indices = np.where(
            vertex_part_indices == PART_NAME_TO_INDEX[part_name]
        )[0].astype(np.int32)
        _, part_faces = extract_submesh(vertices, faces, selected_indices)
        part_face_indices[part_name] = part_faces.astype(np.int32)
    return part_face_indices


def save_mesh_features(
    save_path,
    mesh_features,
    bind_xform_inv,
    joint_annotations,
    vertex_part_indices,
    part_face_indices,
):
    save_payload = {
        "vp": mesh_features["vp"].astype(np.float32),
        "vn": mesh_features["vn"].astype(np.float32),
        "skin_w": mesh_features["skin_w"].astype(np.float32),
        "uv": mesh_features["uv"].astype(np.float32),
        "albedo": mesh_features["albedo"].astype(np.float32),
        "bind_xform_inv": bind_xform_inv.astype(np.float32),
        "face": mesh_features["face"].astype(np.int32),
        "joint_class": joint_annotations.astype(np.float32),
        "vidx_part": vertex_part_indices.astype(np.int32),
    }

    for part_name in PART_NAMES:
        save_payload[f"fidx_{part_name}"] = part_face_indices[part_name].astype(np.int32)

    np.savez(save_path, **save_payload)


def prepare_save_path(save_dir, subject_name):
    os.makedirs(save_dir, exist_ok=True)
    return pjoin(save_dir, subject_name)


class DummyApp(agl.App):
    @staticmethod
    def process_rest_mesh(model_file, save_dir, joint_pos_txt, scale=1.0):
        if not os.path.exists(joint_pos_txt):
            print(f"Joint position file {os.path.basename(joint_pos_txt)} not found")
            return

        subject_name = os.path.splitext(os.path.basename(model_file))[0]
        save_path = prepare_save_path(save_dir, subject_name)
        if os.path.exists(save_path):
            print("[Subject] ", subject_name, " : already processed")
            return

        print("[Subject] ", subject_name, " : processing")

        model = agl.FBX(model_file, scale=scale).model()
        raw_joint_names, parent_indices, bind_xform_inv, mesh_features = collect_rest_mesh_features(
            model
        )
        template_joint_names = load_template_joint_names(joint_pos_txt)
        (
            mesh_features["skin_w"],
            bind_xform_inv,
            joint_annotations,
        ) = align_joint_data_to_template(
            raw_joint_names,
            parent_indices,
            template_joint_names,
            mesh_features["skin_w"],
            bind_xform_inv,
        )

        vertex_part_indices = compute_vertex_part_indices(
            mesh_features["skin_w"],
            joint_annotations,
        )
        part_face_indices = compute_part_face_indices(
            mesh_features["vp"],
            mesh_features["face"],
            vertex_part_indices,
        )
        save_mesh_features(
            save_path,
            mesh_features,
            bind_xform_inv,
            joint_annotations,
            vertex_part_indices,
            part_face_indices,
        )

        # show_part_mesh(
        #     torch.from_numpy(mesh_features["vp"]),
        #     torch.from_numpy(mesh_features["face"]),
        #     torch.from_numpy(vertex_part_indices),
        #     out_path=save_path + "_part_vis.glb",
        # )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-model", type=str, default=None)
    parser.add_argument("--single-joint-pos", type=str, default=None)
    parser.add_argument("--single-save-dir", type=str, default=None)
    parser.add_argument("--single-scale", type=float, default=1.0)
    return parser.parse_args()


def run_single_model(model_file, joint_pos_txt, save_dir, scale):
    dummy = DummyApp(show_window=False)
    subject_name = os.path.splitext(os.path.basename(model_file))[0]
    print(f"[Child] START :: {subject_name}", flush=True)
    dummy.process_rest_mesh(
        model_file,
        save_dir,
        joint_pos_txt,
        scale=scale,
    )
    print(f"[Child] DONE :: {subject_name}", flush=True)


def run_batch(model_dir, char_dir, save_dir):
    model_files = sorted(
        [file_name for file_name in os.listdir(model_dir) if file_name.endswith(".fbx")]
    )
    print(f"Found {len(model_files)} FBX files", flush=True)

    completed = 0
    failed = 0
    crashed = 0

    for file_idx, file_name in enumerate(model_files, start=1):
        model_file = pjoin(model_dir, file_name)
        joint_pos_txt = pjoin(
            char_dir, "joint_pos", f"{os.path.splitext(file_name)[0]}.txt"
        )
        scale = 100.0 if "rignet" in char_dir else 1.0
        subject_name = os.path.splitext(file_name)[0]
        print(
            f"[{file_idx}/{len(model_files)}] START :: {subject_name}",
            flush=True,
        )

        child_args = [
            sys.executable,
            os.path.abspath(__file__),
            "--single-model",
            model_file,
            "--single-joint-pos",
            joint_pos_txt,
            "--single-save-dir",
            save_dir,
            "--single-scale",
            str(scale),
        ]

        result = subprocess.run(child_args, check=False)
        if result.returncode == 0:
            completed += 1
            print(
                f"[{file_idx}/{len(model_files)}] DONE :: {subject_name}",
                flush=True,
            )
            continue

        if result.returncode < 0:
            crashed += 1
            print(
                f"[{file_idx}/{len(model_files)}] CRASH :: {subject_name} :: returncode={result.returncode}",
                flush=True,
            )
        else:
            failed += 1
            print(
                f"[{file_idx}/{len(model_files)}] FAIL :: {subject_name} :: returncode={result.returncode}",
                flush=True,
            )

    print(
        f"BATCH_DONE :: completed={completed} :: failed={failed} :: crashed={crashed}",
        flush=True,
    )


if __name__ == "__main__":
    args = parse_args()

    DATA_DIR = pjoin(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    char_dir = pjoin(DATA_DIR, "train_mesh", "character")
    model_dir = pjoin(DATA_DIR, "train_mesh", "character", "fbx")
    save_dir = pjoin(DATA_DIR, "train_mesh", "character", "processed")
    try:
        if args.single_model is not None:
            run_single_model(
                args.single_model,
                args.single_joint_pos,
                args.single_save_dir,
                args.single_scale,
            )
        else:
            run_batch(model_dir, char_dir, save_dir)
    except Exception:
        traceback.print_exc()
        raise

    #     vp_mean, vp_std = [], []
    #     vn_mean, vn_std = [], []
    #     for file_name in model_files:
    #         char_name = os.path.splitext(file_name)[0]
    #         if char_name not in TRAIN_CHARS:
    #             continue

    #         print(f"Processing {char_name}")
    #         data = np.load(pjoin(save_dir, char_name + ".npz"))
    #         vp_mean.append(data["vp"].mean(axis=0))
    #         vp_std.append(data["vp"].std(axis=0))
    #         vn_mean.append(data["vn"].mean(axis=0))
    #         vn_std.append(data["vn"].std(axis=0))

    #     vp_mean = torch.from_numpy(np.stack(vp_mean, axis=0).mean(axis=0)).float()
    #     vp_std = torch.from_numpy(np.stack(vp_std, axis=0).mean(axis=0)).float()
    #     vn_mean = torch.from_numpy(np.stack(vn_mean, axis=0).mean(axis=0)).float()
    #     vn_std = torch.from_numpy(np.stack(vn_std, axis=0).mean(axis=0)).float()

    #     torch.save(
    #         {
    #             "vp_m": vp_mean,
    #             "vp_s": vp_std,
    #             "vn_m": vn_mean,
    #             "vn_s": vn_std,
    #         },
    #         pjoin(save_dir, "vtx_ms_dict.pt"),
    #     )
