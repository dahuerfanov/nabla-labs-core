"""Base class for ring-based mesh partitioning."""

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from loguru import logger


class MeshPartitioner(ABC):
    """
    Abstract base class for mesh partitioning using ring-based seam detection.

    The algorithm uses a Sequential Greedy Path Selection approach:
    1. Define ring sequences (connectivity chains).
    2. For each ring, identify 2 opposite connector vertices.
    3. Process pairs in sequence: pick shortest path among unused vertices, mark as used.
    4. Partition mesh based on detected seams.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Initialize the partitioner.

        Args:
            vertices: (N, 3) vertex positions
            faces: (M, 3) face indices
        """
        self.vertices = vertices
        self.faces = faces
        self.adj = self._build_adjacency()
        self.G = self._build_graph()

    def _build_adjacency(self) -> list[set[int]]:
        """Build vertex adjacency list from faces."""
        adj: list[set[int]] = [set() for _ in range(len(self.vertices))]
        for face in self.faces:
            for i in range(3):
                u, v = int(face[i]), int(face[(i + 1) % 3])
                adj[u].add(v)
                adj[v].add(u)
        return adj

    def _build_graph(self) -> nx.Graph:
        """Build weighted graph for shortest path computation."""
        G = nx.Graph()
        for i, neighbors in enumerate(self.adj):
            for n in neighbors:
                dist = float(np.linalg.norm(self.vertices[i] - self.vertices[n]))
                G.add_edge(i, n, weight=dist)
        return G

    @abstractmethod
    def get_ring_definitions(self) -> dict:
        """
        Get ring definitions mapping ring types to vertex lists.

        Returns:
            Dict mapping RingType to list of vertex indices forming the ring.
        """
        ...

    @abstractmethod
    def get_ring_sequences(self) -> list[list]:
        """
        Get ring connection sequences for seam finding.

        Each sequence is a list of RingTypes to connect in order.

        Returns:
            List of ring sequences.
        """
        ...

    @abstractmethod
    def get_ring_connectors(self, ring: list[int]) -> list[int]:
        """
        Get connector vertices (opposite points) on a ring.

        Args:
            ring: List of vertex indices forming the ring.

        Returns:
            List of exactly 2 vertex indices.
        """
        ...

    def get_ring_opposite(self, ring: list[int], start_v: int) -> int:
        """Find the vertex opposite to start_v on the ring (by geodesic distance)."""
        if start_v not in ring:
            return ring[len(ring) // 2]

        idx = ring.index(start_v)
        ordered_ring = ring[idx:] + ring[:idx]

        dists = [0.0]
        cum_dist = 0.0
        for i in range(len(ordered_ring)):
            u = ordered_ring[i]
            v = ordered_ring[(i + 1) % len(ordered_ring)]
            d = float(np.linalg.norm(self.vertices[u] - self.vertices[v]))
            cum_dist += d
            dists.append(cum_dist)

        total_len = dists[-1]
        target = total_len / 2.0
        diffs = np.abs(np.array(dists) - target)
        opp_idx_local = int(np.argmin(diffs)) % len(ordered_ring)
        return ordered_ring[opp_idx_local]

    def partition(self) -> tuple[np.ndarray, dict]:
        """
        Partition the mesh into front/back using sequential greedy path selection.

        Returns:
            face_labels: (M,) array, 0 for FRONT, 1 for BACK
            seams: Dict of detected seam paths.
        """
        seams: dict = {}
        ring_defs = self.get_ring_definitions()
        ring_sequences = self.get_ring_sequences()

        # Collect connectors for each ring
        ring_connectors: dict = {}
        for r_type, ring in ring_defs.items():
            connectors = self.get_ring_connectors(ring)

            if len(connectors) == 1:
                p1 = connectors[0]
                p2 = self.get_ring_opposite(ring, p1)
                connectors.append(p2)
                logger.info(f"Ring {r_type}: Inferred opposite point {p2}.")

            if len(connectors) == 0:
                connectors = [ring[0], ring[len(ring) // 2]]
                logger.warning(
                    f"Ring {r_type}: No connectors found, using geometric fallback."
                )

            ring_connectors[r_type] = connectors[:2]

        # Track used vertices per ring
        used_vertices: dict = {r: set() for r in ring_defs}

        # Process each sequence
        for seq in ring_sequences:
            valid_seq = [r for r in seq if r in ring_defs]
            if len(valid_seq) < 2:
                continue

            for i in range(len(valid_seq) - 1):
                r1, r2 = valid_seq[i], valid_seq[i + 1]

                avail_r1 = [
                    v for v in ring_connectors[r1] if v not in used_vertices[r1]
                ]
                avail_r2 = [
                    v for v in ring_connectors[r2] if v not in used_vertices[r2]
                ]

                if not avail_r1 or not avail_r2:
                    logger.warning(f"Skipping {r1}-{r2}: No unused connectors.")
                    continue

                best_path = None
                best_len = float("inf")
                best_pair = (None, None)

                for v1 in avail_r1:
                    for v2 in avail_r2:
                        try:
                            path = nx.shortest_path(
                                self.G, source=v1, target=v2, weight="weight"
                            )
                            path_len = sum(
                                self.G[path[j]][path[j + 1]]["weight"]
                                for j in range(len(path) - 1)
                            )
                            if path_len < best_len:
                                best_len = path_len
                                best_path = path
                                best_pair = (v1, v2)
                        except nx.NetworkXNoPath:
                            continue

                if best_path is not None:
                    v1, v2 = best_pair
                    used_vertices[r1].add(v1)
                    used_vertices[r2].add(v2)
                    seams[(r1, r2, 0)] = best_path
                    logger.debug(f"Seam {r1}->{r2}: {v1} -> {v2}, len={best_len:.2f}")
                else:
                    logger.warning(f"No valid path found for {r1}-{r2}.")

        # Partition faces based on seams
        face_labels = self._partition_faces(ring_defs, seams)
        return face_labels, seams

    def _partition_faces(self, ring_defs: dict, seams: dict) -> np.ndarray:
        """
        Partition faces into front/back based on seam paths.

        This base implementation uses flood fill from seam vertices.
        Subclasses may override for specific partitioning logic.
        """
        # Collect all seam vertices
        seam_vertices = set()
        for path in seams.values():
            seam_vertices.update(path)

        # Get a reference ring (first available)
        ref_ring = None
        for ring in ring_defs.values():
            if ring:
                ref_ring = ring
                break

        if ref_ring is None:
            return np.zeros(len(self.faces), dtype=int)

        # Find split points on reference ring
        split_points = [v for v in ref_ring if v in seam_vertices]
        if len(split_points) < 2:
            return np.zeros(len(self.faces), dtype=int)

        # Split ring into two halves
        s_indices = sorted([ref_ring.index(v) for v in split_points])
        s1, s2 = s_indices[0], s_indices[-1]

        front_half = set(ref_ring[s1 : s2 + 1])
        back_half = set(ref_ring[s2:] + ref_ring[: s1 + 1])

        # BFS flood fill from each half
        front_verts = self._flood_fill(front_half, seam_vertices)
        back_verts = self._flood_fill(back_half, seam_vertices)

        # Label faces
        face_labels = np.zeros(len(self.faces), dtype=int)
        for i, face in enumerate(self.faces):
            in_front = sum(1 for v in face if v in front_verts)
            in_back = sum(1 for v in face if v in back_verts)
            if in_back > in_front:
                face_labels[i] = 1

        return face_labels

    def _flood_fill(self, start_verts: set[int], boundary: set[int]) -> set[int]:
        """BFS flood fill from start vertices, stopping at boundary."""
        visited = set(start_verts)
        queue = list(start_verts)

        while queue:
            v = queue.pop(0)
            for n in self.adj[v]:
                if n not in visited and n not in boundary:
                    visited.add(n)
                    queue.append(n)

        return visited
