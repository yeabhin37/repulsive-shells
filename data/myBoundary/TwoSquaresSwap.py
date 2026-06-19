import os
import argparse
import math


def make_square_patch(cx, cy, cz, side=1.0, n=16):
    """
    하나의 square patch 생성.
    xy-plane 위에 놓인 n x n grid square.
    """
    verts = []
    faces = []

    half = side / 2.0

    for j in range(n + 1):
        y = cy - half + side * j / n
        for i in range(n + 1):
            x = cx - half + side * i / n
            verts.append((x, y, cz))

    def vid(i, j):
        return j * (n + 1) + i

    for j in range(n):
        for i in range(n):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)

            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    return verts, faces


def merge_meshes(meshes):
    """
    여러 connected component를 하나의 PLY mesh로 병합.
    component 순서가 vertex correspondence를 결정하므로 중요함.

    여기서는 항상 [swap, fixed] 순서로 저장한다.
    """
    all_verts = []
    all_faces = []

    offset = 0
    for verts, faces in meshes:
        all_verts.extend(verts)
        all_faces.extend([
            (a + offset, b + offset, c + offset)
            for a, b, c in faces
        ])
        offset += len(verts)

    return all_verts, all_faces


def write_ply(path, verts, faces):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for x, y, z in verts:
            f.write(f"{x} {y} {z}\n")

        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


def point_on_upper_arc(center_x, center_y, radius, t):
    """
    fixed mesh를 중심으로 swap mesh가 위쪽 반원 경로를 따라 이동하도록 하는 함수.

    t = 0.0 -> 왼쪽
    t = 0.5 -> 위쪽
    t = 1.0 -> 오른쪽
    """
    theta = math.pi * (1.0 - t)

    x = center_x + radius * math.cos(theta)
    y = center_y + radius * math.sin(theta)

    return x, y


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        type=str,
        default="/home/yebhin/repulsive-shells/data/myBoundary/TwoSquares_n16",
        help="output directory",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=16,
        help="subdivision count per square side",
    )

    parser.add_argument(
        "--side",
        type=float,
        default=1.0,
        help="square side length",
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=0.3,
        help="gap between neighboring squares",
    )

    parser.add_argument(
        "--init_t",
        type=float,
        default=0.1,
        help="initial movement amount of swap mesh along upper arc",
    )

    parser.add_argument(
        "--z",
        type=float,
        default=0.0,
        help="z coordinate of both square patches",
    )

    args = parser.parse_args()

    if args.n < 1:
        raise ValueError("--n must be >= 1")

    if not (0.0 <= args.init_t <= 1.0):
        raise ValueError("--init_t must be between 0 and 1")

    os.makedirs(args.out, exist_ok=True)

    print("Current working directory:")
    print(os.getcwd())
    print()

    print("Output directory:")
    print(os.path.abspath(args.out))
    print()

    # 두 square가 서로 닿지 않도록 중심 간 거리를 side + gap으로 설정
    d = args.side + args.gap

    # fixed mesh는 모든 파일에서 같은 위치에 고정
    fixed_x = 0.0
    fixed_y = 0.0

    # swap mesh가 fixed mesh 주위를 위쪽 반원으로 이동
    swap_start_x, swap_start_y = point_on_upper_arc(
        fixed_x, fixed_y, d, 0.0
    )

    swap_init_x, swap_init_y = point_on_upper_arc(
        fixed_x, fixed_y, d, args.init_t
    )

    swap_init_mid_x, swap_init_mid_y = point_on_upper_arc(
        fixed_x, fixed_y, d, 0.5
    )

    swap_end_x, swap_end_y = point_on_upper_arc(
        fixed_x, fixed_y, d, 1.0
    )

    # start: swap 왼쪽, fixed 오른쪽
    swap_start = make_square_patch(
        swap_start_x, swap_start_y, args.z, args.side, args.n
    )
    fixed_start = make_square_patch(
        fixed_x, fixed_y, args.z, args.side, args.n
    )

    # init: swap이 위쪽 반원 경로를 따라 약간 이동
    swap_init = make_square_patch(
        swap_init_x, swap_init_y, args.z, args.side, args.n
    )
    fixed_init = make_square_patch(
        fixed_x, fixed_y, args.z, args.side, args.n
    )

    # init_mid: swap이 fixed 위쪽
    swap_init_mid = make_square_patch(
        swap_init_mid_x, swap_init_mid_y, args.z, args.side, args.n
    )
    fixed_init_mid = make_square_patch(
        fixed_x, fixed_y, args.z, args.side, args.n
    )

    # end: fixed 왼쪽, swap 오른쪽
    swap_end = make_square_patch(
        swap_end_x, swap_end_y, args.z, args.side, args.n
    )
    fixed_end = make_square_patch(
        fixed_x, fixed_y, args.z, args.side, args.n
    )

    # vertex correspondence를 유지하기 위해 모든 파일에서 component 순서는 항상 [swap, fixed]
    start_verts, start_faces = merge_meshes([swap_start, fixed_start])
    init_verts, init_faces = merge_meshes([swap_init, fixed_init])
    init_mid_verts, init_mid_faces = merge_meshes([swap_init_mid, fixed_init_mid])
    end_verts, end_faces = merge_meshes([swap_end, fixed_end])

    start_path = os.path.join(args.out, "ab_start.ply")
    init_path = os.path.join(args.out, "ab_init.ply")
    init_mid_path = os.path.join(args.out, "ab_init_mid.ply")
    end_path = os.path.join(args.out, "ab_end.ply")

    write_ply(start_path, start_verts, start_faces)
    write_ply(init_path, init_verts, init_faces)
    write_ply(init_mid_path, init_mid_verts, init_mid_faces)
    write_ply(end_path, end_verts, end_faces)

    print("Saved:")
    print(f"  start    : {start_path}")
    print(f"  init     : {init_path}")
    print(f"  init_mid : {init_mid_path}")
    print(f"  end      : {end_path}")
    print()

    print("Geometry:")
    print(f"  fixed    : x={fixed_x:.4f}, y={fixed_y:.4f}")
    print(f"  start    : swap at x={swap_start_x:.4f}, y={swap_start_y:.4f}")
    print(f"  init     : swap at x={swap_init_x:.4f}, y={swap_init_y:.4f}")
    print(f"  init_mid : swap at x={swap_init_mid_x:.4f}, y={swap_init_mid_y:.4f}")
    print(f"  end      : swap at x={swap_end_x:.4f}, y={swap_end_y:.4f}")
    print()

    print("Vertex/component order in all files:")
    print("  component 0: swap")
    print("  component 1: fixed")


if __name__ == "__main__":
    main()