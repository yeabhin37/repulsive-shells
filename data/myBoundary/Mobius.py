import os
import math
import numpy as np


def mobius_point(u, v, R=2.0):
    """
    Standard Mobius strip parametrization.

    Parametrs:
    u: np.ndarray[nu, 1]
        around direction
    v: np.ndarray[nv] # (nv) -> (1, nv)
        width direction
    R: radius of center circle
    
    
    Return
    ------
    return: np.ndarray[nu, nv, 3]
    """
    if False:
        x = (R + v * math.cos(u / 2.0)) * math.cos(u)
        y = (R + v * math.cos(u / 2.0)) * math.sin(u)
        z = v * math.sin(u / 2.0)
        return [x, y, z]
    else:
        x = (R + v * math.cos(u / 2.0)) * math.cos(u)
        y = (R + v * math.cos(u / 2.0)) * math.sin(u)
        z = v * math.sin(u / 2.0)
        return np.stack([x, y, z], -1) # [nu, nv, 3]


def vertex_id(i, j, nv):
    return i * nv + j


def make_mobius_vertices(nu: int, nv: int, width: float, R: float, shift: float
                         ) -> np.ndarray:
    """
    Generate vertices of Mobius strip.

    shift = 0        -> start
    shift = 0.25*2pi -> quarter material shift
    shift = 1.00*2pi -> end, same shape but reversed width correspondence
    """
    vertices = []

    v_values = np.linspace(-width, width, nv)

    if False:
        for i in range(nu):
            u = 2.0 * math.pi * i / nu
    
            for j in range(nv):
                v = v_values[j]
                vertices.append(mobius_point(u + shift, v, R))
    else:
        I = np.arange(nu) # np.ndarray[nu]
        u = 2.0 * math.pi * I / nu # np.ndarray[nu]
        v = v_values # np.ndarray[nv]
        
        # np.ndarray[nu, nv].rehspae(nu*nv)
        # if `arr.shape == (n,)`, `arr[:, None].shape == (n, 1)`,
        # `arr.shape[Non e, :].shape == (1, n)`
        # `(u + shift)[:, None]).shape == (nu, 1)`
        mobius_point((u + shift)[:, None], v, R).reshape(-1, 3) # [nu,nv,3] -> [nu*nv, 3]
        
    if False:
        # Note for NumPy
        A = np.zeros((2, 3))
        B = np.zeros((2, 3))
        
        if False:
            "Not preferrable"
            C = np.zeros((2, 3))
            for i in range(2):
                for j in range(3):
                    C[i,j] = A[i,j] + B[i,j]
        else:
            "preferrable"
            """
            See numpy documentation
            'indexting', 'broacasting'
            Ask AI "change my NumPy code to be vectorized"
            """
            C = A + B

    return vertices


def make_mobius_faces(nu, nv):
    """
    Triangulate Mobius strip.

    Important:
    The seam connects u = 2pi back to u = 0 with reversed width index.
    This realizes the Mobius identification:
        (u + 2pi, v) ~ (u, -v)
    """
    faces = []

    for i in range(nu):
        i_next = (i + 1) % nu

        for j in range(nv - 1):
            if i < nu - 1:
                a = vertex_id(i, j, nv)
                b = vertex_id(i_next, j, nv)
                c = vertex_id(i_next, j + 1, nv)
                d = vertex_id(i, j + 1, nv)
            else:
                # Mobius seam: reverse width index
                a = vertex_id(i, j, nv)
                b = vertex_id(0, nv - 1 - j, nv)
                c = vertex_id(0, nv - 2 - j, nv)
                d = vertex_id(i, j + 1, nv)

            # split quad into two triangles
            faces.append([a, b, c])
            faces.append([a, c, d])

    return faces


def write_ply(path, vertices, faces):
    """
    ASCII PLY writer.
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
            f.write(f"{x:.10f} {y:.10f} {z:.10f}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def save_mobius_sequence(
    output_dir="./data/myBoundary/Mobius",
    nu=60,
    nv=9,
    width=0.35,
    R=2.0,
):
    """
    Generate start, several init meshes, and end mesh.

    nu: number of samples around the strip
    nv: number of samples across the width
    width: half-width of the strip
    R: center radius
    """

    os.makedirs(output_dir, exist_ok=True)

    faces = make_mobius_faces(nu, nv)

    sequence = {
        "mobius_start.ply": 0.0,
        "mobius_init_025.ply": 0.25 * 2.0 * math.pi,
        "mobius_init_050.ply": 0.50 * 2.0 * math.pi,
        "mobius_init_075.ply": 0.75 * 2.0 * math.pi,
        "mobius_end_shift_2pi.ply": 1.00 * 2.0 * math.pi,
    }

    for filename, shift in sequence.items():
        vertices = make_mobius_vertices(
            nu=nu,
            nv=nv,
            width=width,
            R=R,
            shift=shift,
        )

        path = os.path.join(output_dir, filename)
        write_ply(path, vertices, faces)
        print(f"saved: {path}")

    print()
    print("Mesh info")
    print(f"  vertices: {nu * nv}")
    print(f"  faces:    {len(faces)}")
    print(f"  nu:       {nu}")
    print(f"  nv:       {nv}")
    print(f"  width:    {width}")
    print(f"  R:        {R}")


if __name__ == "__main__":
    save_mobius_sequence(
        output_dir="./data/myBoundary/Mobius",
        nu=60,
        nv=9,
        width=0.35,
        R=2.0,
    )