# -*- coding: utf-8 -*-

# @Time : 2026/3/24 16:48

# @Author : Aumnce
# @Email : 1270888213@qq.com
# @File : nomove_render_relative_sequence.py
# render_relative_sequence.py
import bpy
import csv
import os
import sys
import math
from pathlib import Path
from bpy_extras.object_utils import world_to_camera_view

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config as project_config
except Exception:
    project_config = None

# ---------------------------------
# 解析 Blender 命令行参数
# ---------------------------------
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

def get_arg(name, default=None):
    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default

DEFAULT_OUT_DIR = str(getattr(project_config, "RENDERS_DIR", "renders"))
DEFAULT_SCALE = str(getattr(project_config, "RENDER_SCALE", 50.0))
DEFAULT_RES_X = str(getattr(project_config, "IMG_W", 1024))
DEFAULT_RES_Y = str(getattr(project_config, "IMG_H", 1024))
DEFAULT_FPS = str(getattr(project_config, "RENDER_FPS", 5))
DEFAULT_FOV_DEG = str(getattr(project_config, "RENDER_FOV_DEG", getattr(project_config, "FOV_X_DEG", 15.0)))
DEFAULT_EMISSION = str(getattr(project_config, "RENDER_EMISSION", 8.0))

CSV_PATH = get_arg("--csv")
OBJ_PATH = get_arg("--obj")
OUT_DIR = get_arg("--out", DEFAULT_OUT_DIR)
MODEL_SCALE = float(get_arg("--scale", DEFAULT_SCALE))
RES_X = int(get_arg("--resx", DEFAULT_RES_X))
RES_Y = int(get_arg("--resy", DEFAULT_RES_Y))
FPS = int(get_arg("--fps", DEFAULT_FPS))
FOV_DEG = float(get_arg("--fov_deg", DEFAULT_FOV_DEG))
EMISSION_STRENGTH = float(get_arg("--emission", DEFAULT_EMISSION))

if CSV_PATH is None or OBJ_PATH is None:
    raise ValueError("需要提供 --csv 和 --obj 参数")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------
# 清空场景
# ---------------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

for block in list(bpy.data.meshes):
    if block.users == 0:
        bpy.data.meshes.remove(block)

for block in list(bpy.data.materials):
    if block.users == 0:
        bpy.data.materials.remove(block)

# ---------------------------------
# 导入目标模型
# ---------------------------------
ext = os.path.splitext(OBJ_PATH)[1].lower()
before_objs = set(bpy.data.objects)

if ext == ".obj":
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=OBJ_PATH)
    elif hasattr(bpy.ops.import_scene, "obj"):
        bpy.ops.import_scene.obj(filepath=OBJ_PATH)
    else:
        raise RuntimeError("当前 Blender 版本不支持 OBJ 导入操作符")
elif ext in [".glb", ".gltf"]:
    bpy.ops.import_scene.gltf(filepath=OBJ_PATH)
else:
    raise ValueError(f"暂不支持的模型格式: {ext}")

after_objs = set(bpy.data.objects)
imported_objs = list(after_objs - before_objs)

if len(imported_objs) == 0:
    raise RuntimeError("没有导入任何对象，请检查模型路径")

# ---------------------------------
# 根节点
# ---------------------------------
root = bpy.data.objects.new("TargetRoot", None)
bpy.context.collection.objects.link(root)

for obj in imported_objs:
    obj.parent = root

root.scale = (MODEL_SCALE, MODEL_SCALE, MODEL_SCALE)
root.location = (0.0, 0.0, -100.0)

# ---------------------------------
# 发光材质
# ---------------------------------
mat = bpy.data.materials.new(name="DebugEmission")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

for n in list(nodes):
    nodes.remove(n)

output = nodes.new(type="ShaderNodeOutputMaterial")
emission = nodes.new(type="ShaderNodeEmission")
emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
emission.inputs["Strength"].default_value = EMISSION_STRENGTH
links.new(emission.outputs["Emission"], output.inputs["Surface"])

for obj in imported_objs:
    if obj.type == "MESH":
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

# ---------------------------------
# 相机
# ---------------------------------
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam

cam.location = (0.0, 0.0, 0.0)
cam.rotation_euler = (0.0, 0.0, 0.0)

cam_data.angle = math.radians(FOV_DEG)
cam_data.clip_start = 0.01
cam_data.clip_end = 100000.0

# ---------------------------------
# 黑背景
# ---------------------------------
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg is not None:
    bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs[1].default_value = 0.0

# ---------------------------------
# 渲染设置
# ---------------------------------
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.render.film_transparent = False

# ---------------------------------
# 读取 CSV
# ---------------------------------
rows = []
with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

if len(rows) == 0:
    raise RuntimeError("CSV 为空")

scene.frame_start = 1
scene.frame_end = len(rows)

# ---------------------------------
# 写入位置关键帧
# RTN -> Blender:
# x = R, y = N, z = -T
# ---------------------------------
for i, row in enumerate(rows, start=1):
    R = float(row["rel_R_m"])
    T = float(row["rel_T_m"])
    N = float(row["rel_N_m"])

    x_b = R
    y_b = N
    z_b = -T

    root.location = (x_b, y_b, z_b)
    root.keyframe_insert(data_path="location", frame=i)

scene.render.filepath = os.path.join(OUT_DIR, "frame_")

# ---------------------------------
# 先渲染
# ---------------------------------
bpy.ops.render.render(animation=True)

# ---------------------------------
# 计算每帧 bbox / visible
# ---------------------------------
def collect_mesh_vertices_world(obj):
    verts_world = []
    if obj.type == "MESH":
        for v in obj.data.vertices:
            verts_world.append(obj.matrix_world @ v.co)
    for child in obj.children:
        verts_world.extend(collect_mesh_vertices_world(child))
    return verts_world


def write_invisible_row(writer, frame_idx):
    writer.writerow([
        frame_idx, 0,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    ])


label_path = os.path.join(OUT_DIR, "labels.csv")
with open(label_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "visible",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "cx",
        "cy",
        "w",
        "h",
        "xmin_float",
        "ymin_float",
        "xmax_float",
        "ymax_float",
        "cx_float",
        "cy_float",
        "w_float",
        "h_float"
    ])

    for frame_idx in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame_idx)

        verts_world = collect_mesh_vertices_world(root)

        if len(verts_world) == 0:
            write_invisible_row(writer, frame_idx)
            continue

        coords_2d = []
        in_front = False

        for v in verts_world:
            co_ndc = world_to_camera_view(scene, cam, v)

            # co_ndc.z > 0 表示在相机前方
            if co_ndc.z > 0:
                in_front = True

            coords_2d.append((co_ndc.x, co_ndc.y, co_ndc.z))

        if not in_front:
            write_invisible_row(writer, frame_idx)
            continue

        xs = [c[0] for c in coords_2d]
        ys = [c[1] for c in coords_2d]

        xmin_ndc = min(xs)
        xmax_ndc = max(xs)
        ymin_ndc = min(ys)
        ymax_ndc = max(ys)

        # 完全在屏幕外
        if xmax_ndc < 0 or xmin_ndc > 1 or ymax_ndc < 0 or ymin_ndc > 1:
            write_invisible_row(writer, frame_idx)
            continue

        # 裁剪到图像内
        xmin_ndc = max(0.0, xmin_ndc)
        xmax_ndc = min(1.0, xmax_ndc)
        ymin_ndc = max(0.0, ymin_ndc)
        ymax_ndc = min(1.0, ymax_ndc)

        # ---------- 浮点框（更适合主动感知微小差异） ----------
        xmin_f = xmin_ndc * RES_X
        xmax_f = xmax_ndc * RES_X
        ymin_f = (1.0 - ymax_ndc) * RES_Y
        ymax_f = (1.0 - ymin_ndc) * RES_Y

        w_f = xmax_f - xmin_f
        h_f = ymax_f - ymin_f
        cx_f = xmin_f + w_f / 2.0
        cy_f = ymin_f + h_f / 2.0

        # ---------- 整数框（兼容你现有流程） ----------
        xmin = int(xmin_f)
        xmax = int(xmax_f)
        ymin = int(ymin_f)
        ymax = int(ymax_f)

        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w / 2.0
        cy = ymin + h / 2.0

        visible = 1 if (w > 0 and h > 0) else 0

        if visible == 0:
            write_invisible_row(writer, frame_idx)
        else:
            writer.writerow([
                frame_idx, 1,
                xmin, ymin, xmax, ymax, cx, cy, w, h,
                xmin_f, ymin_f, xmax_f, ymax_f, cx_f, cy_f, w_f, h_f
            ])

print("渲染完成:", OUT_DIR)
print("标签已保存:", label_path)