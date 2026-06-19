import os
import numpy as np


def write_ply(path, vertices, faces):
    """
    ASCII PLY 저장
    vertices: (N, 3)
    faces: list of [i, j, k]
    """
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def add_triangle_oriented(vertices, faces, i0, i1, i2, desired_normal):
    """
    삼각형 방향을 desired_normal 쪽으로 맞춤.
    """
    p0 = vertices[i0]
    p1 = vertices[i1]
    p2 = vertices[i2]

    n = np.cross(p1 - p0, p2 - p0)

    if np.dot(n, desired_normal) < 0:
        faces.append([i0, i2, i1])
    else:
        faces.append([i0, i1, i2])


def make_open_box_shell(
    center=(0.0, 0.0, 0.0),
    size=(4.0, 2.0, 1.0),
    open_side="top",
    nx=24,
    ny=14,
    nz=10,
):
    """
    얇은 open-box shell 생성.

    open_side="top":
        아래 판 + 옆면 4개
        위쪽이 열린 상자, 즉 ∪ 형태

    open_side="bottom":
        위 판 + 옆면 4개
        아래쪽이 열린 상자, 즉 ∩ 형태

    반환:
        vertices, faces
    """
    cx, cy, cz = center
    sx, sy, sz = size

    xmin, xmax = cx - sx / 2, cx + sx / 2
    ymin, ymax = cy - sy / 2, cy + sy / 2
    zmin, zmax = cz - sz / 2, cz + sz / 2

    vertices = []
    faces = []
    vmap = {}

    def get_vertex(p):
        key = tuple(np.round(p, 10))
        if key not in vmap:
            vmap[key] = len(vertices)
            vertices.append(np.array(p, dtype=float))
        return vmap[key]

    def add_quad(p00, p10, p11, p01, desired_normal):
        i00 = get_vertex(p00)
        i10 = get_vertex(p10)
        i11 = get_vertex(p11)
        i01 = get_vertex(p01)

        add_triangle_oriented(vertices, faces, i00, i10, i11, desired_normal)
        add_triangle_oriented(vertices, faces, i00, i11, i01, desired_normal)

    # ------------------------------------------------------------
    # 1. 바닥 또는 윗판
    # ------------------------------------------------------------
    if open_side == "top":
        cap_z = zmin
        cap_normal = np.array([0.0, 0.0, -1.0])
    elif open_side == "bottom":
        cap_z = zmax
        cap_normal = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError("open_side must be either 'top' or 'bottom'")

    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    zs = np.linspace(zmin, zmax, nz + 1)

    for ix in range(nx):
        for iy in range(ny):
            p00 = (xs[ix],     ys[iy],     cap_z)
            p10 = (xs[ix + 1], ys[iy],     cap_z)
            p11 = (xs[ix + 1], ys[iy + 1], cap_z)
            p01 = (xs[ix],     ys[iy + 1], cap_z)
            add_quad(p00, p10, p11, p01, cap_normal)

    # ------------------------------------------------------------
    # 2. x = xmin 면
    # ------------------------------------------------------------
    normal = np.array([-1.0, 0.0, 0.0])
    for iy in range(ny):
        for iz in range(nz):
            p00 = (xmin, ys[iy],     zs[iz])
            p10 = (xmin, ys[iy + 1], zs[iz])
            p11 = (xmin, ys[iy + 1], zs[iz + 1])
            p01 = (xmin, ys[iy],     zs[iz + 1])
            add_quad(p00, p10, p11, p01, normal)

    # ------------------------------------------------------------
    # 3. x = xmax 면
    # ------------------------------------------------------------
    normal = np.array([1.0, 0.0, 0.0])
    for iy in range(ny):
        for iz in range(nz):
            p00 = (xmax, ys[iy],     zs[iz])
            p10 = (xmax, ys[iy + 1], zs[iz])
            p11 = (xmax, ys[iy + 1], zs[iz + 1])
            p01 = (xmax, ys[iy],     zs[iz + 1])
            add_quad(p00, p10, p11, p01, normal)

    # ------------------------------------------------------------
    # 4. y = ymin 면
    # ------------------------------------------------------------
    normal = np.array([0.0, -1.0, 0.0])
    for ix in range(nx):
        for iz in range(nz):
            p00 = (xs[ix],     ymin, zs[iz])
            p10 = (xs[ix + 1], ymin, zs[iz])
            p11 = (xs[ix + 1], ymin, zs[iz + 1])
            p01 = (xs[ix],     ymin, zs[iz + 1])
            add_quad(p00, p10, p11, p01, normal)

    # ------------------------------------------------------------
    # 5. y = ymax 면
    # ------------------------------------------------------------
    normal = np.array([0.0, 1.0, 0.0])
    for ix in range(nx):
        for iz in range(nz):
            p00 = (xs[ix],     ymax, zs[iz])
            p10 = (xs[ix + 1], ymax, zs[iz])
            p11 = (xs[ix + 1], ymax, zs[iz + 1])
            p01 = (xs[ix],     ymax, zs[iz + 1])
            add_quad(p00, p10, p11, p01, normal)

    return np.array(vertices), faces


def combine_meshes(meshes):
    """
    여러 mesh component를 하나의 mesh로 합침.
    """
    all_vertices = []
    all_faces = []
    offset = 0

    for vertices, faces in meshes:
        all_vertices.append(vertices)

        for face in faces:
            all_faces.append([idx + offset for idx in face])

        offset += len(vertices)

    return np.vstack(all_vertices), all_faces


def make_scene(upper_center_z):
    """
    아래 상자는 고정.
    위 상자의 z 위치만 바꿔서 start / second를 만듦.
    """

    # 아래 상자: 위쪽이 열린 ∪ 형태
    lower_vertices, lower_faces = make_open_box_shell(
        center=(0.0, 0.0, 0.6),
        size=(4.0, 2.2, 1.2),
        open_side="top",
        nx=28,
        ny=16,
        nz=10,
    )

    # 위 상자: 아래쪽이 열린 ∩ 형태
    # 크기를 아래 상자와 동일하게 만들어야 함. 
    upper_vertices, upper_faces = make_open_box_shell(
        center=(0.0, 0.0, upper_center_z),
        size=(4.0, 2.2, 1.2),
        open_side="bottom",
        nx=28,
        ny=16,
        nz=10,
    )

    vertices, faces = combine_meshes([
        (lower_vertices, lower_faces),
        (upper_vertices, upper_faces),
    ])

    return vertices, faces


if __name__ == "__main__":
    out_dir = "./OpenBoxStacking"
    os.makedirs(out_dir, exist_ok=True)

    # 아래 상자의 z 범위: 0.0 ~ 1.2
    # 위 상자의 높이: 1.0

    # start:
    # 위 상자의 아래 rim이 아래 상자보다 위에 있음.
    # upper z 범위: 1.45 ~ 2.45
    start_upper_center_z = 3.00

    # second:
    # 위 상자가 아래쪽으로 이동하여 아래 상자 안쪽으로 들어간 상태.
    # upper z 범위: 0.20 ~ 1.20
    second_upper_center_z = 2.70

    start_vertices, start_faces = make_scene(start_upper_center_z)
    second_vertices, second_faces = make_scene(second_upper_center_z)

    # correspondence 확인
    assert start_vertices.shape == second_vertices.shape
    assert len(start_faces) == len(second_faces)
    assert start_faces == second_faces

    write_ply(
        os.path.join(out_dir, "open_box_start.ply"),
        start_vertices,
        start_faces,
    )

    write_ply(
        os.path.join(out_dir, "open_box_second.ply"),
        second_vertices,
        second_faces,
    )

    print("Saved:")
    print(os.path.join(out_dir, "open_box_start.ply"))
    print(os.path.join(out_dir, "open_box_second.ply"))
    print()
    print(f"num vertices = {len(start_vertices)}")
    print(f"num faces    = {len(start_faces)}")
    
    #%% 
import numpy as np


def read_ply(path):
    with open(path, "r") as f:
        lines = f.readlines()

    nV = None
    header_end = None

    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            nV = int(line.split()[-1])
        if line.strip() == "end_header":
            header_end = i
            break

    V = []
    for line in lines[header_end + 1 : header_end + 1 + nV]:
        V.append(list(map(float, line.split()[:3])))

    return np.array(V)


V0 = read_ply("./OpenBoxStacking/open_box_start.ply")
V1 = read_ply("./OpenBoxStacking/open_box_second.ply")

D = V1 - V0

print("num vertices:", len(V0))
print("max displacement norm:", np.linalg.norm(D, axis=1).max())
print("mean displacement:", D.mean(axis=0))

# 두 component가 같은 vertex 수라면
half = len(V0) // 2

D_lower = D[:half]
D_upper = D[half:]

print("lower max displacement:", np.linalg.norm(D_lower, axis=1).max())
print("upper min displacement:", D_upper.min(axis=0))
print("upper max displacement:", D_upper.max(axis=0))
print("upper mean displacement:", D_upper.mean(axis=0))