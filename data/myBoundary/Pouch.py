import os
import json
import numpy as np


# ============================================================
# Pouch-like open shell mesh generator
# - x direction: pouch length
# - y-z plane : side-view cross section
# - top gap   : zipper/opening region
# ============================================================

OUT_DIR = "./newPouch"

NX = 16          # length direction resolution
NC = 18          # cross-section resolution

LENGTH = 3.0
HALF_WIDTH = 0.78
HALF_HEIGHT = 0.78

# superellipse exponent
# smaller than 1.0 -> rounded-rectangle-like shape
SQUARISH = 0.55

# opening size
START_GAP = 0.70     # more open
SECOND_GAP = 0.32    # more closed
INIT_GAP = 0.60      # intermediate

# subtle cloth-like waviness
# set to 0.0 if you want a perfectly smooth mesh
WAVE_AMP = 0.018


def superellipse_point(theta, half_width, half_height, squarish):
    """
    Rounded-rectangle-like cross-section.

    y = horizontal direction in side view
    z = vertical direction
    """
    c = np.cos(theta)
    s = np.sin(theta)

    y = half_width * np.sign(c) * (np.abs(c) ** squarish)
    z = half_height + half_height * np.sign(s) * (np.abs(s) ** squarish)

    return y, z


def alpha_from_gap(gap, half_width, squarish):
    """
    Convert desired top opening gap to angle cut amount.
    """
    ratio = np.clip(gap / (2.0 * half_width), 1e-6, 0.95)
    sin_alpha = ratio ** (1.0 / squarish)
    sin_alpha = np.clip(sin_alpha, 1e-6, 0.95)
    return np.arcsin(sin_alpha)


def build_pouch_vertices(gap):
    """
    Build vertices for one pouch state.

    Cross-section is an open rounded rectangle:
        left top lip -> left wall -> bottom -> right wall -> right top lip

    Then it is extruded along x.
    """
    alpha = alpha_from_gap(gap, HALF_WIDTH, SQUARISH)

    # Start at left top lip, go around bottom, end at right top lip.
    theta_left_lip = np.pi / 2.0 + alpha
    theta_right_lip = np.pi / 2.0 - alpha + 2.0 * np.pi

    thetas = np.linspace(theta_left_lip, theta_right_lip, NC)
    xs = np.linspace(-LENGTH / 2.0, LENGTH / 2.0, NX)

    vertices = []

    for i, x in enumerate(xs):
        u = i / (NX - 1)

        # small taper near both ends to make it look less like a perfect extrusion
        end_taper = 1.0 - 0.035 * (
            np.exp(-((u - 0.0) / 0.12) ** 2)
            + np.exp(-((u - 1.0) / 0.12) ** 2)
        )

        for j, theta in enumerate(thetas):
            v = j / (NC - 1)

            y, z = superellipse_point(theta, HALF_WIDTH, HALF_HEIGHT, SQUARISH)

            y *= end_taper

            # cloth-like subtle wave, strongest near upper part
            upper_factor = np.clip((z - HALF_HEIGHT) / HALF_HEIGHT, 0.0, 1.0)
            wave = WAVE_AMP * upper_factor * (
                0.65 * np.sin(2.0 * np.pi * 2.5 * u)
                + 0.35 * np.sin(2.0 * np.pi * 5.0 * u + 3.0 * v)
            )

            z = z + wave

            vertices.append([x, y, z])

    return np.asarray(vertices, dtype=float)


def idx(i, j):
    return i * NC + j


def build_faces():
    """
    Build identical topology for all states.

    The main surface is a rectangular grid.
    The two x-end side panels are capped with fan triangulation.
    The top opening is left open, including both end-cap top triangles.
    """
    faces = []

    # Main pouch shell
    for i in range(NX - 1):
        for j in range(NC - 1):
            v00 = idx(i, j)
            v10 = idx(i + 1, j)
            v01 = idx(i, j + 1)
            v11 = idx(i + 1, j + 1)

            faces.append([v00, v11, v10])
            faces.append([v00, v01, v11])

    # Add two center vertices for side caps
    left_center = NX * NC
    right_center = NX * NC + 1

    # Left cap, x = -L/2
    for j in range(NC - 1):
        faces.append([left_center, idx(0, j + 1), idx(0, j)])

    # Removed:
    # faces.append([left_center, idx(0, 0), idx(0, NC - 1)])

    # Right cap, x = +L/2
    for j in range(NC - 1):
        faces.append([right_center, idx(NX - 1, j), idx(NX - 1, j + 1)])

    # Removed:
    # faces.append([right_center, idx(NX - 1, NC - 1), idx(NX - 1, 0)])

    return faces

def add_cap_centers(vertices):
    """
    Add center vertices for the two end caps.
    """
    left_cross_section = vertices[0:NC]
    right_cross_section = vertices[(NX - 1) * NC:NX * NC]

    left_center = left_cross_section.mean(axis=0)
    right_center = right_cross_section.mean(axis=0)

    vertices_with_centers = np.vstack([
        vertices,
        left_center,
        right_center
    ])

    return vertices_with_centers


def write_ply(path, vertices, faces):
    """
    Write ASCII PLY file with triangular faces.
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
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def make_state(gap, filename):
    vertices = build_pouch_vertices(gap)
    vertices = add_cap_centers(vertices)
    faces = build_faces()

    path = os.path.join(OUT_DIR, filename)
    write_ply(path, vertices, faces)

    print(f"saved: {path}")
    print(f"  vertices: {len(vertices)}")
    print(f"  faces   : {len(faces)}")
    print(f"  gap     : {gap}")


def write_metadata():
    """
    Useful boundary index information.

    left_lip_indices  : one side of the top opening
    right_lip_indices : the other side of the top opening

    These may be useful for debugging, visualization, or selecting Dirichlet vertices.
    """
    left_lip_indices = [idx(i, 0) for i in range(NX)]
    right_lip_indices = [idx(i, NC - 1) for i in range(NX)]

    metadata = {
        "description": "Pouch-like open shell mesh. Top boundary represents zipper/opening.",
        "NX": NX,
        "NC": NC,
        "num_vertices_without_cap_centers": NX * NC,
        "left_cap_center_index": NX * NC,
        "right_cap_center_index": NX * NC + 1,
        "left_lip_indices": left_lip_indices,
        "right_lip_indices": right_lip_indices,
        "suggested_dirichlet_vertices": [
            idx(0, 0),
            idx(0, NC - 1),
            idx(NX - 1, 0),
            idx(NX - 1, NC - 1),
            NX * NC,
            NX * NC + 1
        ]
    }

    path = os.path.join(OUT_DIR, "pouch_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved: {path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    postfix = 'new'
    make_state(START_GAP, f"pouch_start_{postfix}.ply")          # 구조상 start 
    make_state(SECOND_GAP, f"pouch_third_{postfix}.ply")         # 구조상 third 
    make_state(INIT_GAP, f"pouch_second_{postfix}.ply")          # 구조상 second  

    # write_metadata()