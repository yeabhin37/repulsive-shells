import numpy as np
from pathlib import Path


def vertex_index(i, j, n_width):
    """
    i : length direction index
    j : width direction index
    """
    return i * (n_width + 1) + j


def make_twisted_strip(
    length=4.0,
    width=1.0,
    n_length=80,
    n_width=12,
    total_twist_angle=0.0,
):
    """
    Make a rectangular strip mesh.

    left boundary  : fixed
    right boundary : twisted by total_twist_angle

    total_twist_angle:
        0          -> start
        pi / 2     -> init, half-twisted
        pi         -> end, twisted / flipped
    """

    vertices = []

    for i in range(n_length + 1):
        u = i / n_length
        x = length * u

        # twist angle changes gradually from left to right
        theta = total_twist_angle * u

        for j in range(n_width + 1):
            v = j / n_width

            # width coordinate
            # j = 0       -> top side
            # j = n_width -> bottom side
            w = width * (0.5 - v)

            # width direction is rotated around x-axis
            y = w * np.cos(theta)
            z = w * np.sin(theta)

            vertices.append([x, y, z])

    vertices = np.array(vertices)

    faces = []

    for i in range(n_length):
        for j in range(n_width):
            v00 = vertex_index(i, j, n_width)
            v10 = vertex_index(i + 1, j, n_width)
            v11 = vertex_index(i + 1, j + 1, n_width)
            v01 = vertex_index(i, j + 1, n_width)

            # split one quad into two triangles
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    faces = np.array(faces)

    return vertices, faces


def save_ply(filename, vertices, faces):
    """
    Save triangular mesh as ASCII PLY.
    """

    with open(filename, "w") as f:
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


def print_corner_info(vertices, n_length, n_width, name):
    """
    Print positions of corner vertices.

    Numbering:
        1 ---- 3
        |      |
        2 ---- 4
    """

    idx_1 = vertex_index(0, 0, n_width)
    idx_2 = vertex_index(0, n_width, n_width)
    idx_3 = vertex_index(n_length, 0, n_width)
    idx_4 = vertex_index(n_length, n_width, n_width)

    print(f"\n[{name}]")
    print("1:", vertices[idx_1])
    print("2:", vertices[idx_2])
    print("3:", vertices[idx_3])
    print("4:", vertices[idx_4])


def main():
    output_dir = Path.cwd() / "twisted_strip"
    output_dir.mkdir(exist_ok=True)

    length = 4.0
    width = 1.0

    n_length = 80
    n_width = 12

    configs = [
        {
            "name": "start",
            "angle": 0.0,
            "filename": "strip_start.ply",
        },
        {
            "name": "init_half_twisted",
            "angle": np.pi / 2,
            "filename": "strip_init_half_twisted.ply",
        },
        {
            "name": "end_twisted",
            "angle": np.pi,
            "filename": "strip_end_twisted.ply",
        },
    ]

    for config in configs:
        vertices, faces = make_twisted_strip(
            length=length,
            width=width,
            n_length=n_length,
            n_width=n_width,
            total_twist_angle=config["angle"],
        )

        save_path = output_dir / config["filename"]
        save_ply(save_path, vertices, faces)

        print(f"Saved: {save_path}")
        print_corner_info(vertices, n_length, n_width, config["name"])


if __name__ == "__main__":
    main()