import numpy as np
import os

def make_disk(n_r=16, n_theta=64):
    vertices = []
    faces = []

    # center vertex
    vertices.append([0.0, 0.0, 0.0])

    def idx(r, j):
        # r: 1 ~ n_r
        # j: 0 ~ n_theta-1
        return 1 + (r - 1) * n_theta + (j % n_theta)

    # rings
    for r in range(1, n_r + 1):
        radius = r / n_r
        for j in range(n_theta):
            theta = 2 * np.pi * j / n_theta
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vertices.append([x, y, 0.0])

    # center fan
    for j in range(n_theta):
        faces.append([0, idx(1, j), idx(1, j + 1)])

    # ring faces
    for r in range(2, n_r + 1):
        for j in range(n_theta):
            a = idx(r - 1, j)
            b = idx(r - 1, j + 1)
            c = idx(r, j)
            d = idx(r, j + 1)

            faces.append([a, c, d])
            faces.append([a, d, b])

    return np.array(vertices, dtype=float), np.array(faces, dtype=int)

def taco_fold(vertices, amount):
    """
    amount = 0.0 : flat disk
    amount = 0.5 : half folded disk
    amount = 1.0 : strongly folded disk
    """
    V = vertices.copy()
    x = V[:, 0]
    y = V[:, 1]

    # bend around y-axis
    R = 0.40
    theta = x / R

    bent_x = R * np.sin(theta)
    bent_z = R * (1.0 - np.cos(theta))

    V[:, 0] = (1.0 - amount) * x + amount * bent_x
    V[:, 1] = y
    V[:, 2] = amount * bent_z

    return V

def write_ply(filename, vertices, faces):
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

os.makedirs("./data/myBoundary/Disk", exist_ok=True)

V, F = make_disk(n_r=16, n_theta=64)

V_flat = taco_fold(V, 0.0)
V_half = taco_fold(V, 0.5)
V_full = taco_fold(V, 1.0)

write_ply("./data/myBoundary/Disk/disk_flat.ply", V_flat, F)
write_ply("./data/myBoundary/Disk/disk_taco_50.ply", V_half, F)
write_ply("./data/myBoundary/Disk/disk_taco_100.ply", V_full, F)

print("num vertices:", len(V))
print("num faces:", len(F))
print("outer boundary vertex range:", 1 + (16 - 1) * 64, "to", 1 + 16 * 64 - 1)