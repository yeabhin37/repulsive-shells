import os
import numpy as np
from pathlib import Path


# ============================================================
# Output setting
# ============================================================

OUT_DIR = Path("/home/yebhin/repulsive-shells/data/myBoundary/MobiusTwoTurn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mesh resolution
# 처음 실험은 너무 촘촘하지 않게 시작하는 것을 추천
N_U = 96     # longitudinal direction
N_V = 13     # width direction

# Shape parameters
R = 2.0              # center circle radius
HALF_WIDTH = 0.35    # strip half-width

# Debug files
SAVE_COLORED_DEBUG = True
SAVE_PREVIEW_PATH = False   # True로 바꾸면 원하는 0~2turn path preview도 저장됨


# ============================================================
# Möbius strip parameterization
# ============================================================
#
# P(u, v) = (
#   (R + v cos(u/2)) cos u,
#   (R + v cos(u/2)) sin u,
#    v sin(u/2)
# )
#
# 한 바퀴 이동:
#   P(u + 2pi, v) = P(u, -v)
#
# 두 바퀴 이동:
#   P(u + 4pi, v) = P(u, v)
#
# 그래서
#   start = P(u, v)
#   init  = P(u + 2pi, v)
#   end   = P(u + 4pi, v)
#
# 로 만들면, geometry는 같은 Möbius strip이지만
# vertex correspondence 관점에서는 one-turn / two-turn 상태가 됨.
# ============================================================

def mobius_point(u, v, R=2.0):
    x = (R + v * np.cos(u / 2.0)) * np.cos(u)
    y = (R + v * np.cos(u / 2.0)) * np.sin(u)
    z = v * np.sin(u / 2.0)
    return np.array([x, y, z], dtype=float)


def vid(i, j, n_v):
    return i * n_v + j


def build_vertices(theta_offset, n_u=N_U, n_v=N_V, R=R, half_width=HALF_WIDTH):
    """
    theta_offset:
      0       -> start
      2*pi    -> one-turn state
      4*pi    -> two-turn state = start
    """
    vertices = []

    u_values = 2.0 * np.pi * np.arange(n_u) / n_u
    v_values = np.linspace(-half_width, half_width, n_v)

    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            p = mobius_point(u + theta_offset, v, R)
            vertices.append(p)

    return np.array(vertices, dtype=float)


def build_faces(n_u=N_U, n_v=N_V):
    """
    True Möbius topology.
    The seam is stitched with width direction flipped:
      (u = 2pi, v) == (u = 0, -v)
    """
    faces = []

    for i in range(n_u):
        ip = (i + 1) % n_u

        for j in range(n_v - 1):
            if i < n_u - 1:
                a = vid(i, j, n_v)
                b = vid(ip, j, n_v)
                c = vid(ip, j + 1, n_v)
                d = vid(i, j + 1, n_v)
            else:
                # Möbius seam: flip width index
                a = vid(i, j, n_v)
                b = vid(0, n_v - 1 - j, n_v)
                c = vid(0, n_v - 2 - j, n_v)
                d = vid(i, j + 1, n_v)

            faces.append([a, b, c])
            faces.append([a, c, d])

    return np.array(faces, dtype=int)


def build_material_colors(n_u=N_U, n_v=N_V):
    """
    Material-side color for visualization.
    이 색은 geometry가 아니라 vertex index/material coordinate에 붙어 있음.

    start/end에서는 한쪽이 white, 반대쪽이 gray.
    init에서는 vertex positions가 one-turn 이동되어 있으므로
    같은 material color가 반대쪽으로 넘어가 보이게 됨.
    """
    colors = []

    white = np.array([235, 235, 235], dtype=np.uint8)
    gray = np.array([120, 120, 120], dtype=np.uint8)

    center_j = (n_v - 1) / 2.0

    for i in range(n_u):
        for j in range(n_v):
            if j <= center_j:
                colors.append(white)
            else:
                colors.append(gray)

    return np.array(colors, dtype=np.uint8)


def write_ply(path, vertices, faces, colors=None):
    path = Path(path)

    has_color = colors is not None

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        if has_color:
            for p, c in zip(vertices, colors):
                f.write(
                    f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )
        else:
            for p in vertices:
                f.write(f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")

        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def write_boundary_info(path, n_u=N_U, n_v=N_V):
    """
    Möbius strip의 boundary는 하나의 loop.
    필요할 때 확인용으로 boundary vertex index를 저장.
    """
    boundary_loop = []

    # one side
    for i in range(n_u):
        boundary_loop.append(vid(i, 0, n_v))

    # the other side
    for i in range(n_u):
        boundary_loop.append(vid(i, n_v - 1, n_v))

    with open(path, "w") as f:
        f.write("# Boundary vertex loop for the Mobius strip\n")
        f.write("# This is for debugging / visualization.\n")
        f.write("# The solver does not necessarily need this file.\n")
        for v in boundary_loop:
            f.write(f"{v}\n")


def main():
    faces = build_faces()
    colors = build_material_colors()

    # Main solver input files
    start_vertices = build_vertices(theta_offset=0.0)
    init_vertices = build_vertices(theta_offset=2.0 * np.pi)
    end_vertices = build_vertices(theta_offset=4.0 * np.pi)

    write_ply(OUT_DIR / "start.ply", start_vertices, faces)
    write_ply(OUT_DIR / "init.ply", init_vertices, faces)
    write_ply(OUT_DIR / "end.ply", end_vertices, faces)

    # Colored debug files
    if SAVE_COLORED_DEBUG:
        write_ply(OUT_DIR / "start_color.ply", start_vertices, faces, colors=colors)
        write_ply(OUT_DIR / "init_color.ply", init_vertices, faces, colors=colors)
        write_ply(OUT_DIR / "end_color.ply", end_vertices, faces, colors=colors)

    # Optional desired path preview
    if SAVE_PREVIEW_PATH:
        preview_dir = OUT_DIR / "preview_path"
        preview_dir.mkdir(parents=True, exist_ok=True)

        num_preview = 9
        for k in range(num_preview):
            t = k / (num_preview - 1)
            theta = 4.0 * np.pi * t
            verts = build_vertices(theta_offset=theta)
            write_ply(preview_dir / f"path_{k:02d}.ply", verts, faces, colors=colors)

    write_boundary_info(OUT_DIR / "boundary_vertices.txt")

    print("Saved files:")
    print(f"  {OUT_DIR / 'start.ply'}")
    print(f"  {OUT_DIR / 'init.ply'}")
    print(f"  {OUT_DIR / 'end.ply'}")

    if SAVE_COLORED_DEBUG:
        print()
        print("Saved colored debug files:")
        print(f"  {OUT_DIR / 'start_color.ply'}")
        print(f"  {OUT_DIR / 'init_color.ply'}")
        print(f"  {OUT_DIR / 'end_color.ply'}")

    print()
    print("Mesh info:")
    print(f"  vertices: {N_U * N_V}")
    print(f"  faces:    {len(faces)}")
    print()
    print("Recommended YAML:")
    print("Data:")
    print("  startFile: ./data/myBoundary/MobiusTwoTurn/start.ply")
    print("  endFile: ./data/myBoundary/MobiusTwoTurn/end.ply")
    print("  initFile: ./data/myBoundary/MobiusTwoTurn/init.ply")
    print("  dirichletVertices: []")


if __name__ == "__main__":
    main()