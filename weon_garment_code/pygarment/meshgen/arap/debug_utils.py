"""Debug visualization utilities for ARAP."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def save_colored_obj(
    output_dir: Path,
    filename: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
) -> Path:
    """Save mesh with vertex colors derived from face labels (Red=Front, Blue=Back).

    Args:
        output_dir: Directory to save the file
        filename: Name of the output file (without path)
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        face_labels: (M,) labels per face (0=FRONT, 1=BACK)

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    num_verts = len(vertices)
    vert_colors = np.zeros((num_verts, 3), dtype=np.float32)
    vert_counts = np.zeros(num_verts, dtype=np.float32)

    red = np.array([1.0, 0.0, 0.0])
    blue = np.array([0.0, 0.0, 1.0])

    for i, face in enumerate(faces):
        c = red if face_labels[i] == 0 else blue
        for vid in face:
            vert_colors[vid] += c
            vert_counts[vid] += 1

    mask = vert_counts > 0
    vert_colors[mask] /= vert_counts[mask][:, None]

    with open(path, "w") as f:
        f.write(f"# Colored Debug Mesh: {filename}\n")
        f.write("# Red=Front, Blue=Back, Purple=Seam\n")

        for i, v in enumerate(vertices):
            c = vert_colors[i]
            f.write(
                f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n"
            )

        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    logger.info(f"Saved colored OBJ to {path}")
    return path


def save_debug_mesh(
    output_dir: Path,
    filename: str,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Path:
    """Save a simple OBJ file for debugging.

    Args:
        output_dir: Directory to save the file
        filename: Name of the output file (without path)
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    logger.debug(f"Saved debug mesh to {path}")
    return path


def save_seam_association_debug(
    output_dir: Path,
    filename: str,
    seam_mappings: dict,
    garment_verts: np.ndarray,
    body_verts_3d: np.ndarray,
) -> Path:
    """Plot flat garment mesh overlap with projected body seams.

    Args:
        output_dir: Directory to save the file
        filename: Name of the output file (without path)
        seam_mappings: Dict of seam type -> SeamMapping
        garment_verts: (N, 2 or 3) garment vertex positions
        body_verts_3d: (M, 3) body vertex positions

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    plt.figure(figsize=(10, 10))

    # Plot garment vertices (Background)
    plt.scatter(
        garment_verts[:, 0],
        garment_verts[:, 1],
        c="gray",
        s=1,
        alpha=0.3,
        label="Garment Mesh",
    )

    # Plot Seams
    cmap = plt.get_cmap("tab10")
    for i, (_, mapping) in enumerate(seam_mappings.items()):
        # Garment Seam (2D)
        g_indices = mapping.garment_indices
        g_pts = garment_verts[g_indices]

        # Body Seam (3D -> 2D XY)
        b_pts_3d = mapping.body_positions
        b_pts_2d = b_pts_3d[:, :2]

        color = cmap(i % 10)
        plt.plot(
            g_pts[:, 0],
            g_pts[:, 1],
            "o-",
            color=color,
            markersize=3,
            label=f"G-Seam {i}",
        )
        plt.plot(
            b_pts_2d[:, 0],
            b_pts_2d[:, 1],
            "x--",
            color=color,
            markersize=3,
            alpha=0.7,
            label=f"B-Seam {i}",
        )

        # Start/End Markers
        plt.plot(g_pts[0, 0], g_pts[0, 1], "^", color="green", markersize=6)
        plt.plot(g_pts[-1, 0], g_pts[-1, 1], "v", color="red", markersize=6)
        plt.plot(b_pts_2d[0, 0], b_pts_2d[0, 1], "^", color="green", markersize=6)
        plt.plot(b_pts_2d[-1, 0], b_pts_2d[-1, 1], "v", color="red", markersize=6)

        # Draw connecting lines
        for j in range(min(len(g_pts), len(b_pts_2d))):
            plt.plot(
                [g_pts[j, 0], b_pts_2d[j, 0]],
                [g_pts[j, 1], b_pts_2d[j, 1]],
                "-",
                color=color,
                alpha=0.2,
                linewidth=0.5,
            )

    plt.axis("equal")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Seam Associations (Garment vs Body XY)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    logger.info(f"Saved seam association debug image to {path}")
    return path
