import os
import numpy as np


# ============================================================
# Common utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_ply(filename, vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)

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

        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def nearest_vertex(vertices, target):
    vertices = np.asarray(vertices)
    target = np.asarray(target, dtype=float)
    return int(np.argmin(np.linalg.norm(vertices - target, axis=1)))


def print_anchor_suggestions(name, vertices, targets):
    print(f"\n[{name}] suggested dirichletVertices")
    anchors = []
    for t in targets:
        idx = nearest_vertex(vertices, t)
        anchors.append(idx)
        print(f"  target {t} -> vertex {idx}, coord {vertices[idx]}")
    print("  yaml:", anchors)
    return anchors


# ============================================================
# 1. Thin strip: flat strip -> folded strip
# ============================================================

def make_thin_strip(nx=80, ny=8, length=2.4, width=0.18):
    """
    Rectangular thin strip.
    x direction: long direction
    y direction: narrow width direction
    """
    vertices = []
    faces = []

    def idx(i, j):
        return i * (ny + 1) + j

    for i in range(nx + 1):
        u = i / nx
        x = (u - 0.5) * length

        for j in range(ny + 1):
            v = j / ny
            y = (v - 0.5) * width
            vertices.append([x, y, 0.0])

    for i in range(nx):
        for j in range(ny):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j + 1)
            d = idx(i, j + 1)

            faces.append([a, b, c])
            faces.append([a, c, d])

    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def fold_thin_strip(vertices, amount, length=2.4, gap=0.055):
    """
    amount = 0.0 : flat strip
    amount = 0.5 : half folded strip
    amount = 1.0 : folded strip

    오른쪽 절반을 왼쪽 절반 위로 접는 구조.
    최종 상태에서는 두 layer가 가까워지도록 gap을 작게 둔다.
    """
    V = vertices.copy()
    half_length = length / 2.0

    def smoothstep(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    angle = np.pi * amount

    for k in range(len(V)):
        x, y, z = V[k]

        if x > 0.0:
            t = x / half_length
            s = smoothstep(t)

            # rotate right half around hinge x=0
            x_new = x * np.cos(angle)
            z_new = x * np.sin(angle)

            # small separation at the fully folded state
            z_new += amount * gap * s

            V[k] = [x_new, y, z_new]

    return V


def generate_thin_strip():
    out_dir = "./data/myBoundary/ThinStrip"
    ensure_dir(out_dir)

    V, F = make_thin_strip(nx=80, ny=8, length=2.4, width=0.18)

    V0 = fold_thin_strip(V, 0.0)
    V1 = fold_thin_strip(V, 0.5)
    V2 = fold_thin_strip(V, 1.0)

    write_ply(f"{out_dir}/strip_flat.ply", V0, F)
    write_ply(f"{out_dir}/strip_fold_50.ply", V1, F)
    write_ply(f"{out_dir}/strip_fold_100.ply", V2, F)

    print("\nGenerated ThinStrip")
    print("  vertices:", len(V))
    print("  faces:", len(F))

    print_anchor_suggestions(
        "ThinStrip",
        V0,
        targets=[
            [-1.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [-1.2, 0.09, 0.0],
            [-1.2, -0.09, 0.0],
        ],
    )


# ============================================================
# 2. Slit disk: wide slit -> narrow slit
# ============================================================

def make_slit_disk_grid(
    n=56,
    radius=1.0,
    slit_x0=-0.15,
    slit_x1=0.95,
    slit_half_width=0.055,
):
    """
    Disk에서 오른쪽 방향으로 slit을 뚫은 형태.
    grid 기반이라 boundary가 완벽히 smooth하지는 않지만,
    boundary proximity 실험용으로는 충분히 유용하다.
    """
    xs = np.linspace(-radius, radius, n + 1)
    ys = np.linspace(-radius, radius, n + 1)

    kept_cells = []

    for i in range(n):
        for j in range(n):
            cx = 0.5 * (xs[i] + xs[i + 1])
            cy = 0.5 * (ys[j] + ys[j + 1])

            inside_disk = (cx * cx + cy * cy) <= radius * radius
            inside_slit = (
                slit_x0 <= cx <= slit_x1
                and abs(cy) <= slit_half_width
            )

            if inside_disk and not inside_slit:
                kept_cells.append((i, j))

    vertex_map = {}
    vertices = []
    faces = []

    def get_vertex(i, j):
        key = (i, j)
        if key not in vertex_map:
            x = xs[i]
            y = ys[j]
            vertex_map[key] = len(vertices)
            vertices.append([x, y, 0.0])
        return vertex_map[key]

    for i, j in kept_cells:
        a = get_vertex(i, j)
        b = get_vertex(i + 1, j)
        c = get_vertex(i + 1, j + 1)
        d = get_vertex(i, j + 1)

        faces.append([a, b, c])
        faces.append([a, c, d])

    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def close_slit_lips(
    vertices,
    amount,
    slit_x0=-0.15,
    slit_x1=0.95,
    initial_half_width=0.055,
    target_half_width=0.008,
    influence_width=0.32,
):
    """
    amount = 0.0 : original wide slit
    amount = 0.5 : partially closed slit
    amount = 1.0 : narrow slit

    slit 양쪽 boundary lip을 y=0 방향으로 당겨서
    서로 매우 가까운 boundary pair를 만든다.
    """
    V = vertices.copy()

    for k in range(len(V)):
        x, y, z = V[k]
        ay = abs(y)

        if slit_x0 <= x <= slit_x1 and initial_half_width <= ay <= influence_width:
            sign = 1.0 if y >= 0.0 else -1.0

            # keep y unchanged at influence_width,
            # move lip near initial_half_width toward target_half_width
            compressed_ay = target_half_width + (
                (ay - initial_half_width)
                * (influence_width - target_half_width)
                / (influence_width - initial_half_width)
            )

            new_ay = (1.0 - amount) * ay + amount * compressed_ay
            V[k, 1] = sign * new_ay

    return V


def generate_slit_disk():
    out_dir = "./data/myBoundary/SlitDisk"
    ensure_dir(out_dir)

    V, F = make_slit_disk_grid(
        n=56,
        radius=1.0,
        slit_x0=-0.15,
        slit_x1=0.95,
        slit_half_width=0.055,
    )

    V0 = close_slit_lips(V, 0.0)
    V1 = close_slit_lips(V, 0.5)
    V2 = close_slit_lips(V, 1.0)

    write_ply(f"{out_dir}/slit_disk_wide.ply", V0, F)
    write_ply(f"{out_dir}/slit_disk_mid.ply", V1, F)
    write_ply(f"{out_dir}/slit_disk_narrow.ply", V2, F)

    print("\nGenerated SlitDisk")
    print("  vertices:", len(V))
    print("  faces:", len(F))

    print_anchor_suggestions(
        "SlitDisk",
        V0,
        targets=[
            [-0.8, 0.0, 0.0],
            [0.0, 0.45, 0.0],
            [0.0, -0.45, 0.0],
            [0.75, 0.16, 0.0],
            [0.75, -0.16, 0.0],
        ],
    )


# ============================================================
# 3. Annulus: inner boundary moves close to outer boundary
# ============================================================

def make_annulus(n_r=12, n_theta=80, r_inner=0.35, r_outer=1.0):
    vertices = []
    faces = []

    def idx(i, j):
        return i * n_theta + (j % n_theta)

    for i in range(n_r + 1):
        t = i / n_r
        r = (1.0 - t) * r_inner + t * r_outer

        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            vertices.append([x, y, 0.0])

    for i in range(n_r):
        for j in range(n_theta):
            a = idx(i, j)
            b = idx(i, j + 1)
            c = idx(i + 1, j)
            d = idx(i + 1, j + 1)

            faces.append([a, c, d])
            faces.append([a, d, b])

    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def deform_annulus_inner_to_outer(
    vertices,
    amount,
    r_inner=0.35,
    r_outer=1.0,
    max_shift=0.54,
    sigma=0.42,
):
    """
    amount = 0.0 : flat annulus
    amount = 0.5 : inner boundary partially moves outward
    amount = 1.0 : inner boundary locally close to outer boundary

    theta = 0 방향, 즉 +x 방향에서 inner boundary를 outer boundary 쪽으로 당긴다.
    """
    V = vertices.copy()

    for k in range(len(V)):
        x, y, z = V[k]
        r = np.sqrt(x * x + y * y)

        if r < 1e-12:
            continue

        theta = np.arctan2(y, x)

        # wrapped angle distance from theta=0
        dtheta = np.arctan2(np.sin(theta), np.cos(theta))

        bump = np.exp(-(dtheta * dtheta) / (2.0 * sigma * sigma))

        # inner boundary gets stronger deformation, outer boundary fixed
        radial_weight = (r_outer - r) / (r_outer - r_inner)
        radial_weight = np.clip(radial_weight, 0.0, 1.0)

        delta_r = amount * max_shift * bump * radial_weight
        r_new = r + delta_r

        V[k, 0] = r_new * np.cos(theta)
        V[k, 1] = r_new * np.sin(theta)

        # small height variation for visibility, not essential
        V[k, 2] = amount * 0.04 * bump * radial_weight

    return V


def generate_annulus():
    out_dir = "./data/myBoundary/Annulus"
    ensure_dir(out_dir)

    n_r = 12
    n_theta = 80

    V, F = make_annulus(
        n_r=n_r,
        n_theta=n_theta,
        r_inner=0.35,
        r_outer=1.0,
    )

    V0 = deform_annulus_inner_to_outer(V, 0.0)
    V1 = deform_annulus_inner_to_outer(V, 0.5)
    V2 = deform_annulus_inner_to_outer(V, 1.0)

    write_ply(f"{out_dir}/annulus_flat.ply", V0, F)
    write_ply(f"{out_dir}/annulus_inner_close_50.ply", V1, F)
    write_ply(f"{out_dir}/annulus_inner_close_100.ply", V2, F)

    print("\nGenerated Annulus")
    print("  vertices:", len(V))
    print("  faces:", len(F))

    outer0 = n_r * n_theta + 0
    outer90 = n_r * n_theta + n_theta // 4
    outer180 = n_r * n_theta + n_theta // 2
    outer270 = n_r * n_theta + 3 * n_theta // 4
    inner0 = 0

    print("\n[Annulus] suggested dirichletVertices")
    print("  yaml:", [outer0, outer90, outer180, outer270, inner0])

def generate_annulus_extrapolation():
    out_dir = "./data/myBoundary/AnnulusExtrapolation"
    ensure_dir(out_dir)

    n_r = 12
    n_theta = 80

    V, F = make_annulus(
        n_r=n_r,
        n_theta=n_theta,
        r_inner=0.35,
        r_outer=1.0,
    )

    # Extrapolation용 단계들
    # 0.00 : 완전한 flat annulus
    # 0.05 : 아주 약간 이동
    # 0.10 : 약간 이동
    # 0.20 : 조금 더 이동
    # 0.30 : extrapolation init 후보
    # 0.50 : 더 강한 deformation 후보
    # 1.00 : 기존 strong deformation
    amounts = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

    for a in amounts:
        Va = deform_annulus_inner_to_outer(V, a)
        name = int(round(a * 100))
        write_ply(
            f"{out_dir}/annulus_inner_close_{name:03d}.ply",
            Va,
            F
        )

    # 보기 편하게 flat 파일 이름도 따로 저장
    write_ply(f"{out_dir}/annulus_flat.ply", deform_annulus_inner_to_outer(V, 0.0), F)

    print("\nGenerated AnnulusExtrapolation")
    print("  vertices:", len(V))
    print("  faces:", len(F))
    print("  files written under:", out_dir)

    outer0 = n_r * n_theta + 0
    outer90 = n_r * n_theta + n_theta // 4
    outer180 = n_r * n_theta + n_theta // 2
    outer270 = n_r * n_theta + 3 * n_theta // 4
    inner0 = 0

    print("\n[AnnulusExtrapolation] suggested dirichletVertices")
    print("  yaml:", [outer0, outer90, outer180, outer270, inner0])
    
# ============================================================
# 4. Open cylinder: straight cylinder -> bent cylinder
# ============================================================

def make_open_cylinder(n_z=36, n_theta=64, radius=0.12, height=2.0):
    vertices = []
    faces = []

    def idx(i, j):
        return i * n_theta + (j % n_theta)

    for i in range(n_z + 1):
        t = i / n_z
        z = (t - 0.5) * height

        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vertices.append([x, y, z])

    for i in range(n_z):
        for j in range(n_theta):
            a = idx(i, j)
            b = idx(i, j + 1)
            c = idx(i + 1, j)
            d = idx(i + 1, j + 1)

            faces.append([a, c, d])
            faces.append([a, d, b])

    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def bend_open_cylinder(
    vertices,
    amount,
    cylinder_radius=0.12,
    height=2.0,
    bend_angle=1.75 * np.pi,
):
    """
    amount = 0.0 : straight open cylinder
    amount = 0.5 : partially bent cylinder
    amount = 1.0 : strongly bent cylinder

    위/아래 boundary rim이 서로 가까워지는 C-shape 원통을 만든다.
    """
    V = vertices.copy()

    R = height / bend_angle

    for k in range(len(V)):
        x, y, z = vertices[k]

        theta = np.arctan2(y, x)
        s = z

        phi = (s / height) * bend_angle

        center = np.array([
            R * np.sin(phi),
            0.0,
            R * np.cos(phi) - R,
        ])

        normal = np.array([
            np.cos(phi),
            0.0,
            -np.sin(phi),
        ])

        binormal = np.array([0.0, 1.0, 0.0])

        bent = center + cylinder_radius * np.cos(theta) * normal + cylinder_radius * np.sin(theta) * binormal
        original = np.array([x, y, z])

        V[k] = (1.0 - amount) * original + amount * bent

    return V


def generate_open_cylinder():
    out_dir = "./data/myBoundary/OpenCylinder"
    ensure_dir(out_dir)

    n_z = 36
    n_theta = 64

    V, F = make_open_cylinder(
        n_z=n_z,
        n_theta=n_theta,
        radius=0.12,
        height=2.0,
    )

    V0 = bend_open_cylinder(V, 0.0)
    V1 = bend_open_cylinder(V, 0.5)
    V2 = bend_open_cylinder(V, 1.0)

    write_ply(f"{out_dir}/cylinder_straight.ply", V0, F)
    write_ply(f"{out_dir}/cylinder_bent_50.ply", V1, F)
    write_ply(f"{out_dir}/cylinder_bent_100.ply", V2, F)

    print("\nGenerated OpenCylinder")
    print("  vertices:", len(V))
    print("  faces:", len(F))

    bottom0 = 0
    bottom90 = n_theta // 4
    bottom180 = n_theta // 2
    bottom270 = 3 * n_theta // 4
    top0 = n_z * n_theta

    print("\n[OpenCylinder] suggested dirichletVertices")
    print("  yaml:", [bottom0, bottom90, bottom180, bottom270, top0])


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # generate_thin_strip()
    # generate_slit_disk()
    # generate_annulus()
    # generate_open_cylinder()
    
    generate_annulus_extrapolation()

    print("\nDone.")
    print("Generated files are under ./data/myBoundary/")