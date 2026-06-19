import os
import argparse


def parse_vec3(s):
    """
    "x,y,z" 형태의 문자열을 3D 좌표 tuple로 변환.
    예: "-1,0,0" -> (-1.0, 0.0, 0.0)
    """
    vals = s.split(",")
    if len(vals) != 3:
        raise ValueError("좌표는 반드시 'x,y,z' 형식이어야 합니다.")
    return tuple(float(v) for v in vals)


def lerp(p0, p1, t):
    """
    p0에서 p1까지 t 비율만큼 선형 보간한 위치.
    t = 0.0이면 p0, t = 1.0이면 p1, t = 0.5이면 중간점.
    """
    return (
        (1.0 - t) * p0[0] + t * p1[0],
        (1.0 - t) * p0[1] + t * p1[1],
        (1.0 - t) * p0[2] + t * p1[2],
    )


def make_square_patch(center, side=1.0, n=10):
    """
    하나의 connected square patch 생성.

    center: 사각형 중심 좌표 (cx, cy, cz)
    side: 사각형 한 변의 길이
    n: 한 변을 몇 등분할지 결정
       n=1이면 삼각형 2개짜리 사각형
       n=10이면 10x10 grid, 삼각형 200개

    반환:
    vertices: [(x,y,z), ...]
    faces: [(i,j,k), ...]
    """
    cx, cy, cz = center
    half = side / 2.0

    vertices = []
    faces = []

    # xy-plane 위의 정사각형 생성
    # z = cz로 고정
    for j in range(n + 1):
        y = cy - half + side * j / n
        for i in range(n + 1):
            x = cx - half + side * i / n
            vertices.append((x, y, cz))

    def vid(i, j):
        return j * (n + 1) + i

    # 각 grid cell을 삼각형 2개로 분할
    for j in range(n):
        for i in range(n):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)

            # diagonal 방향은 모든 cell에서 동일하게 유지
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    return vertices, faces


def write_ply(path, vertices, faces):
    """
    ASCII PLY 파일 저장.
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

        for x, y, z in vertices:
            f.write(f"{x} {y} {z}\n")

        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        type=str,
        default="./data/myBoundary/MovingSquare",
        help="PLY 파일을 저장할 폴더",
    )

    parser.add_argument(
        "--p0",
        type=str,
        default="-1.0,0.0,0.0",
        help="start square 중심 좌표. 형식: x,y,z",
    )

    parser.add_argument(
        "--p1",
        type=str,
        default="1.0,0.0,0.0",
        help="end square 중심 좌표. 형식: x,y,z",
    )

    parser.add_argument(
        "--init_t",
        type=float,
        default=0.5,
        help="init 위치. p0에서 p1까지의 보간 비율. 0.5면 중간점",
    )

    parser.add_argument(
        "--side",
        type=float,
        default=1.0,
        help="사각형 한 변의 길이",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="사각형 한 변의 subdivision 개수",
    )

    args = parser.parse_args()

    if args.n < 1:
        raise ValueError("--n은 1 이상이어야 합니다.")

    if not (0.0 <= args.init_t <= 1.0):
        raise ValueError("--init_t는 0과 1 사이 값이어야 합니다.")

    os.makedirs(args.out, exist_ok=True)

    p0 = parse_vec3(args.p0)
    p1 = parse_vec3(args.p1)
    p_init = lerp(p0, p1, args.init_t)

    start_vertices, faces = make_square_patch(p0, side=args.side, n=args.n)
    init_vertices, _ = make_square_patch(p_init, side=args.side, n=args.n)
    end_vertices, _ = make_square_patch(p1, side=args.side, n=args.n)

    start_path = os.path.join(args.out, "square_start.ply")
    init_path = os.path.join(args.out, "square_init.ply")
    end_path = os.path.join(args.out, "square_end.ply")

    write_ply(start_path, start_vertices, faces)
    write_ply(init_path, init_vertices, faces)
    write_ply(end_path, end_vertices, faces)

    print("Saved files:")
    print(f"  start: {start_path}")
    print(f"  init : {init_path}")
    print(f"  end  : {end_path}")
    print()
    print("Mesh info:")
    print(f"  vertices: {len(start_vertices)}")
    print(f"  faces   : {len(faces)}")
    print()
    print("Centers:")
    print(f"  p0     = {p0}")
    print(f"  p_init = {p_init}")
    print(f"  p1     = {p1}")


if __name__ == "__main__":
    main()