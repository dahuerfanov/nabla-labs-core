import numpy as np
from collections import defaultdict


def identify_rings(vertices: np.ndarray, faces: np.ndarray) -> list[list[int]]:
    """
    Identify rings (boundary loops) in the mesh.
    A ring is a consecutive sequence of mesh edges that belong to only one mesh triangle.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) array of vertices.
    faces : np.ndarray
        (M, 3) array of face indices.

    Returns
    -------
    list[list[int]]
        A list of rings, where each ring is a list of vertex indices in order.
    """
    # 1. Build edge usage count
    # Keys: tuple(sorted((u, v))), Value: count
    edge_counts = defaultdict(int)

    # Store adjacency for boundary edges: u -> v (directed)
    # Since boundary edges have only 1 face, we want to traverse them in a consistent winding order.
    # Standard winding (CCW) implies that boundary edges are traversed such that the face is on the left.
    # An edge (u, v) is a boundary edge if it appears in exactly one face.
    # To chain them, we need directed edges.
    # If (u, v) is in a face, it goes u->v. If it's a boundary, there is no corresponding v->u from another face.

    # Let's track directed edges.
    # If (u, v) exists and (v, u) does not exist, then (u, v) is a boundary edge.

    directed_edges = set()
    for face in faces:
        i, j, k = face
        directed_edges.add((i, j))
        directed_edges.add((j, k))
        directed_edges.add((k, i))

    boundary_edges = {}  # start_node -> end_node

    for u, v in list(directed_edges):
        if (v, u) not in directed_edges:
            # (u, v) is a boundary edge
            # Check for manifoldness/validity: should handle vertices with > 1 boundary edge?
            # Ideally each boundary vertex has 1 incoming and 1 outgoing boundary edge.
            boundary_edges[u] = v

    # 2. Chain edges into loops
    rings = []
    visited_edges = set()  # Set of start nodes

    # Collect all start nodes
    start_nodes = list(boundary_edges.keys())

    used_nodes = set()

    for start_node in start_nodes:
        if start_node in used_nodes:
            continue

        # Start a new loop
        current_ring = []
        curr = start_node

        # Traverse
        is_loop = False
        while curr in boundary_edges:
            if curr in used_nodes:
                # We hit a visited node.
                # If it's the start_node, we closed the loop.
                if curr == start_node:
                    is_loop = True
                break

            used_nodes.add(curr)
            current_ring.append(curr)
            next_node = boundary_edges[curr]
            curr = next_node

        if is_loop and len(current_ring) >= 3:
            rings.append(current_ring)

    return rings
