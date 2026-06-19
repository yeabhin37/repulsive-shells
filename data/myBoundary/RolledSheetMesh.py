import os
import numpy as np


def write_ply(filename, V, F):
    """
    ASCII PLY writer.
    V: (num_vertices, 3)
    F: (num_faces, 3)
    """
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in V:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for face in F:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def save_indices(filename, indices):
    with open(filename, "w") as f:
        for idx in indices:
            f.write(f"{idx}\n")


def make_overlap_swap_rolled_sheet(
    nz=81,
    nu=121,
    height=2.4,
    radius=1.0,
    layer_gap=0.12,
    overlap_deg=20.0,
    theta0=np.deg2rad(40.0),
    order_sign=1,
):
    """
    Create a rolled sheet whose top-view cross section is almost a circle
    with a small overlap angle.

    start/end have the same angular shape.
    The only difference is radial layer order.

    order_sign = +1:
        j = 0      is inner layer
        j = nu - 1 is outer layer

    order_sign = -1:
        j = 0      is outer layer
        j = nu - 1 is inner layer

    Vertex indexing:
        idx(i, j) = i * nu + j

    i : vertical direction
    j : rolling/material direction
    """

    V = []
    F = []

    def idx(i, j):
        return i * nu + j

    overlap_angle = np.deg2rad(overlap_deg)
    total_angle = 2.0 * np.pi + overlap_angle

    for i in range(nz):
        v = i / (nz - 1)
        z = (v - 0.5) * height

        for j in range(nu):
            u = j / (nu - 1)

            # Same angular position for start and end.
            # This makes both meshes have the same overlap angle.
            theta = theta0 + total_angle * u

            # Layer order control.
            # order_sign = +1: radius increases with u
            # order_sign = -1: radius decreases with u
            #
            # The radius difference is intentionally small,
            # so the cross section looks like a rolled sheet,
            # not a large spiral.
            r = radius + order_sign * layer_gap * (u - 0.5)

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            V.append([x, y, z])

    # Triangulate rectangular grid
    for i in range(nz - 1):
        for j in range(nu - 1):
            v00 = idx(i, j)
            v01 = idx(i, j + 1)
            v10 = idx(i + 1, j)
            v11 = idx(i + 1, j + 1)

            F.append([v00, v10, v11])
            F.append([v00, v11, v01])

    return np.array(V), np.array(F), idx


def make_boundary_indices(nz, nu, idx):
    """
    Boundary consists of:
    - bottom rim: i = 0
    - top rim: i = nz - 1
    - one vertical seam: j = 0
    - the other vertical seam: j = nu - 1
    """
    boundary = []

    # bottom rim
    for j in range(nu):
        boundary.append(idx(0, j))

    # outer/right vertical seam
    for i in range(1, nz):
        boundary.append(idx(i, nu - 1))

    # top rim
    for j in range(nu - 2, -1, -1):
        boundary.append(idx(nz - 1, j))

    # inner/left vertical seam
    for i in range(nz - 2, 0, -1):
        boundary.append(idx(i, 0))

    return boundary


def make_centerline_dirichlet_indices(nz, nu, idx, num_samples=10):
    """
    Choose Dirichlet indices from the center material line j = nu // 2.

    Why centerline?
    In this mesh, start/end swap layer order.
    Therefore j = 0 and j = nu - 1 are not fixed in space.
    But the center material line u = 0.5 has the same radius in start/end,
    so it is a stable place to put a small number of Dirichlet vertices.
    """
    j_mid = nu // 2
    i_values = np.linspace(0, nz - 1, num_samples, dtype=int)
    return [idx(int(i), j_mid) for i in i_values]


if __name__ == "__main__":
    # Save path is unchanged
    out_dir = "rolled_sheet_mesh"
    os.makedirs(out_dir, exist_ok=True)

    # Mesh resolution
    nz = 81
    nu = 121

    # Geometry parameters
    height = 2.4

    # Top-view cross section parameters
    radius = 1.0

    # Difference between the two overlapping layers.
    # Larger value makes the layer order visually clearer.
    layer_gap = 0.12

    # This is the amount of overlap in top view.
    # Keep this below 30 degrees as intended.
    overlap_deg = 20.0

    # Rotates the whole cross section in top view.
    # Change this only for visualization.
    theta0 = np.deg2rad(40.0)

    # Start:
    # j = 0 is inner layer, j = nu - 1 is outer layer.
    V_start, F, idx = make_overlap_swap_rolled_sheet(
        nz=nz,
        nu=nu,
        height=height,
        radius=radius,
        layer_gap=layer_gap,
        overlap_deg=overlap_deg,
        theta0=theta0,
        order_sign=1,
    )

    # End:
    # j = 0 is outer layer, j = nu - 1 is inner layer.
    # The overlap angle is the same as start.
    V_end, _, _ = make_overlap_swap_rolled_sheet(
        nz=nz,
        nu=nu,
        height=height,
        radius=radius,
        layer_gap=layer_gap,
        overlap_deg=overlap_deg,
        theta0=theta0,
        order_sign=-1,
    )

    # Linear midpoint preview.
    # This should be useful because the two layers tend to pass through
    # each other around the overlap region.
    V_mid = 0.5 * V_start + 0.5 * V_end

    # Save paths are unchanged
    write_ply(os.path.join(out_dir, "rolled_sheet_start.ply"), V_start, F)
    write_ply(os.path.join(out_dir, "rolled_sheet_end.ply"), V_end, F)
    write_ply(os.path.join(out_dir, "rolled_sheet_mid_linear_preview.ply"), V_mid, F)

    # Dirichlet indices:
    # Use center material line, because it is unchanged between start/end.
    dirichlet_indices = make_centerline_dirichlet_indices(
        nz=nz,
        nu=nu,
        idx=idx,
        num_samples=10,
    )

    save_indices(
        os.path.join(out_dir, "rolled_sheet_dirichlet_inner_seam.txt"),
        dirichlet_indices,
    )

    # For reference: all boundary vertex indices
    boundary_indices = make_boundary_indices(nz, nu, idx)
    save_indices(
        os.path.join(out_dir, "rolled_sheet_boundary_indices.txt"),
        boundary_indices,
    )

    print("Generated files:")
    print("  rolled_sheet_mesh/rolled_sheet_start.ply")
    print("  rolled_sheet_mesh/rolled_sheet_end.ply")
    print("  rolled_sheet_mesh/rolled_sheet_mid_linear_preview.ply")
    print("  rolled_sheet_mesh/rolled_sheet_dirichlet_inner_seam.txt")
    print("  rolled_sheet_mesh/rolled_sheet_boundary_indices.txt")

    print()
    print("Mesh info:")
    print(f"  num vertices = {len(V_start)}")
    print(f"  num faces    = {len(F)}")
    print(f"  boundary vertices = {len(boundary_indices)}")
    print(f"  dirichlet vertices = {len(dirichlet_indices)}")

    print()
    print("Top-view cross section:")
    print(f"  overlap_deg = {overlap_deg}")
    print(f"  layer_gap   = {layer_gap}")
    print("  start: j = 0 inner, j = nu-1 outer")
    print("  end:   j = 0 outer, j = nu-1 inner")

    print()
    print("Vertex indexing:")
    print("  idx(i, j) = i * nu + j")
    print("  i: vertical direction")
    print("  j: rolling/material direction")

    print()
    print("Recommended Dirichlet indices:")
    print("  center material line j = nu // 2")
    print(f"  j_mid = {nu // 2}")
    print("  indices =", dirichlet_indices)

    print()
    print("Layer order check:")
    print("  start radius at j=0      =", radius - 0.5 * layer_gap)
    print("  start radius at j=nu-1   =", radius + 0.5 * layer_gap)
    print("  end radius at j=0        =", radius + 0.5 * layer_gap)
    print("  end radius at j=nu-1     =", radius - 0.5 * layer_gap)