import numpy as np


def write_ply(filename, V, F):
    """
    ASCII PLY writer
    V: (n_vertices, 3)
    F: (n_faces, 3)
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


def make_twisted_ribbon(
    nu=81,
    nv=13,
    length=6.0,
    width=1.0,
    twist_angle=np.pi,
):
    """
    Create a rectangular ribbon surface twisted along the x-axis.

    twist_angle = +pi  -> +180 degree clockwise-like twist
    twist_angle = -pi  -> -180 degree counter-clockwise-like twist

    Vertex indexing:
        idx(i, j) = i * nv + j

    i: length direction
    j: width direction
    """

    V = []
    F = []

    def idx(i, j):
        return i * nv + j

    for i in range(nu):
        s = i / (nu - 1)          # 0 to 1
        x = (s - 0.5) * length   # -L/2 to L/2

        theta = twist_angle * s

        for j in range(nv):
            t = j / (nv - 1)          # 0 to 1
            y0 = (t - 0.5) * width   # -W/2 to W/2

            # Rotate width direction around x-axis
            y = y0 * np.cos(theta)
            z = y0 * np.sin(theta)

            V.append([x, y, z])

    # Triangulate rectangular grid
    for i in range(nu - 1):
        for j in range(nv - 1):
            v00 = idx(i, j)
            v01 = idx(i, j + 1)
            v10 = idx(i + 1, j)
            v11 = idx(i + 1, j + 1)

            # Consistent triangulation
            F.append([v00, v10, v11])
            F.append([v00, v11, v01])

    return np.array(V), np.array(F), idx


def save_indices(filename, indices):
    with open(filename, "w") as f:
        for k in indices:
            f.write(f"{k}\n")


if __name__ == "__main__":
    nu = 81
    nv = 13

    # Start mesh: +180 degree twist
    V_start, F, idx = make_twisted_ribbon(
        nu=nu,
        nv=nv,
        length=6.0,
        width=1.0,
        twist_angle=np.pi,
    )

    # End mesh: -180 degree twist
    V_end, _, _ = make_twisted_ribbon(
        nu=nu,
        nv=nv,
        length=6.0,
        width=1.0,
        twist_angle=-np.pi,
    )

    write_ply("twisted_ribbon_start_plus180.ply", V_start, F)
    write_ply("twisted_ribbon_end_minus180.ply", V_end, F)

    # Recommended Dirichlet indices:
    # Fix both short end boundaries.
    # These vertices are geometrically identical in start and end.
    dirichlet_ends = []

    # left end: i = 0
    for j in range(nv):
        dirichlet_ends.append(idx(0, j))

    # right end: i = nu - 1
    for j in range(nv):
        dirichlet_ends.append(idx(nu - 1, j))

    save_indices("twisted_ribbon_dirichlet_ends.txt", dirichlet_ends)

    # Minimal alternative:
    # Fix only four corners.
    dirichlet_corners = [
        idx(0, 0),
        idx(0, nv - 1),
        idx(nu - 1, 0),
        idx(nu - 1, nv - 1),
    ]

    save_indices("twisted_ribbon_dirichlet_corners.txt", dirichlet_corners)

    print("Generated files:")
    print("  twisted_ribbon_start_plus180.ply")
    print("  twisted_ribbon_end_minus180.ply")
    print("  twisted_ribbon_dirichlet_ends.txt")
    print("  twisted_ribbon_dirichlet_corners.txt")

    print()
    print("Mesh info:")
    print(f"  num vertices = {len(V_start)}")
    print(f"  num faces    = {len(F)}")

    print()
    print("Dirichlet indices, recommended end-boundary version:")
    print(dirichlet_ends)

    print()
    print("Dirichlet indices, minimal corner version:")
    print(dirichlet_corners)