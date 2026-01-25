from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weon_garment_code.pygarment.meshgen.box_mesh_gen import BoxMesh


def extract_mesh_from_boxmesh(box_mesh: BoxMesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract vertices and faces from a BoxMesh object.

    Parameters
    ----------
    box_mesh : BoxMesh
        The BoxMesh object to extract data from.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - vertices: (N, 3) numpy array of vertex positions.
        - faces: (M, 3) numpy array of face indices.
    """
    vertices = np.array(box_mesh.vertices, dtype=np.float64)

    global_faces = []

    for panel_name in box_mesh.panel_names:
        panel = box_mesh.panels[panel_name]
        for face in panel.panel_faces:
            # face is list of local indices
            g_face = box_mesh._get_glob_ids(panel, face)
            global_faces.append(g_face)

    return vertices, np.array(global_faces, dtype=np.int32)


def update_boxmesh_vertices(box_mesh: BoxMesh, new_vertices: np.ndarray) -> None:
    """
    Update the vertices of a BoxMesh object with new positions.

    This updates the global vertices list. Note that this does NOT automatically
    update the local panel vertices (panel.panel_vertices) or the 2D patterns.
    It only updates the 3D mesh representation used for simulation.

    Parameters
    ----------
    box_mesh : BoxMesh
        The BoxMesh object to update.
    new_vertices : np.ndarray
        (N, 3) numpy array of new vertex positions.
    """
    if len(new_vertices) != len(box_mesh.vertices):
        raise ValueError(
            f"Shape mismatch: {len(new_vertices)} vs {len(box_mesh.vertices)}"
        )

    # box_mesh.vertices is a list, convert to list
    box_mesh.vertices = new_vertices.tolist()


def extract_arap_rest_data(box_mesh: BoxMesh) -> dict[tuple[int, int], np.ndarray]:
    """
    Extract rest vectors from the 2D panels for ARAP.

    For every edge (i, j) in the global mesh, we find the corresponding 2D edge
    in the panels and compute the rest vector (u_i - u_j).

    If an edge is shared (e.g. stitch), we take the first occurrence.
    This assumes that sewn edges have compatible lengths in the pattern.

    Returns
    -------
    dict[tuple[int, int], np.ndarray]
        Mapping from global edge (i, j) to 2D rest vector (3D array with z=0).
        Includes mappings for both (i, j) and (j, i).
    """
    rest_data = {}

    for panel_name in box_mesh.panel_names:
        panel = box_mesh.panels[panel_name]

    # Safer iteration using _get_glob_ids
    for panel_name in box_mesh.panel_names:
        panel = box_mesh.panels[panel_name]

        for face in panel.panel_faces:
            # Get global IDs
            # face is [l1, l2, l3]
            try:
                g_indices = box_mesh._get_glob_ids(panel, face)
            except Exception:
                # Fallback or skip
                continue

            # Get 2D vertices
            # panel.panel_vertices consists of numpy arrays (2,), so we concatenate
            v2d = []
            for lid in face:
                p = panel.panel_vertices[lid]
                # If p is list, use + [0.0]. If array, use concatenate.
                if isinstance(p, np.ndarray):
                    v3 = np.concatenate([p, [0.0]])
                else:
                    v3 = np.array(list(p) + [0.0])
                v2d.append(v3)

            # Edges: (0,1), (1,2), (2,0)
            local_edges = [(0, 1), (1, 2), (2, 0)]

            for i, j in local_edges:
                gi, gj = g_indices[i], g_indices[j]

                # Vector from gi to gj
                rest_vec = v2d[i] - v2d[j]

                # Store if not present
                if (gi, gj) not in rest_data:
                    rest_data[(gi, gj)] = rest_vec

                # We typically need both directions for the solver logic,
                # but solver can invert if needed.
                # Let's populate symmetric for O(1) lookup.
                if (gj, gi) not in rest_data:
                    rest_data[(gj, gi)] = -rest_vec

    return rest_data
