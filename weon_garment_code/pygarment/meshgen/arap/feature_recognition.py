"""Feature recognition for garment and body meshes.

This module provides extractors for identifying semantic features on meshes:
- BodyFeatureExtractor: Identifies body rings (collar, cuffs, ankles, waist)
- GarmentFeatureExtractor: Identifies garment rings and partitions mesh
"""

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
from loguru import logger

from weon_garment_code.pygarment.meshgen.arap.category_strategies import get_strategy
from weon_garment_code.pygarment.meshgen.arap.core_types import (
    BodyRingType,
    GarmentCategory,
    GarmentMetadata,
    GarmentRingType,
)


def load_smplx_segmentation() -> dict[str, list[int]]:
    """Load SMPL-X vertex segmentation from standard locations."""
    current_dir = Path(os.path.dirname(__file__))

    candidates = [
        current_dir / "data" / "smplx_vert_segmentation.json",
        current_dir.parents[3] / "assets" / "smplx_vert_segmentation.json",
        Path.cwd() / "assets" / "smplx_vert_segmentation.json",
        Path.cwd() / "weon_garment_code" / "assets" / "smplx_vert_segmentation.json",
    ]

    for p in candidates:
        if p.exists():
            logger.info(f"Loading segmentation from {p}")
            with open(p) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Segmentation file not found. Checked: {[str(p) for p in candidates]}"
    )


class BodyFeatureExtractor:
    """Extract features from body mesh."""

    # Define boundary pairs for ring detection
    RING_PAIRS: dict[BodyRingType, tuple[str, str]] = {
        BodyRingType.COLLAR: ("head", "neck"),
        BodyRingType.LEFT_CUFF: ("leftHand", "leftForeArm"),
        BodyRingType.RIGHT_CUFF: ("rightHand", "rightForeArm"),
        BodyRingType.LEFT_ANKLE: ("leftFoot", "leftLeg"),
        BodyRingType.RIGHT_ANKLE: ("rightFoot", "rightLeg"),
        BodyRingType.WAIST: ("spine1", "hips"),
    }

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = vertices
        self.faces = faces
        self.segmentation = load_smplx_segmentation()
        self.adj = self._build_adjacency()

    def _build_adjacency(self) -> list[set[int]]:
        adj: list[set[int]] = [set() for _ in range(len(self.vertices))]
        for face in self.faces:
            for i in range(3):
                u, v = face[i], face[(i + 1) % 3]
                adj[u].add(v)
                adj[v].add(u)
        return adj

    def get_body_rings(self) -> dict[BodyRingType, list[int]]:
        """Extract key body rings (collar, cuffs, ankles, waist)."""
        rings: dict[BodyRingType, list[int]] = {}

        # Invert segmentation for fast lookup
        vert_to_part: dict[int, str] = {}
        for part, indices in self.segmentation.items():
            for idx in indices:
                vert_to_part[idx] = part

        for ring_type, (part_a, part_b) in self.RING_PAIRS.items():
            boundary_edges: list[tuple[int, int]] = []
            part_a_indices = self.segmentation.get(part_a, [])

            for u in part_a_indices:
                if u >= len(self.adj):
                    continue
                for v in self.adj[u]:
                    part_v = vert_to_part.get(v)
                    if part_v == part_b:
                        edge = tuple(sorted((u, v)))
                        boundary_edges.append(edge)  # type: ignore

            rings[ring_type] = self._order_edges_to_ring(boundary_edges)

        return rings

    def _order_edges_to_ring(self, edges: list[tuple[int, int]]) -> list[int]:
        if not edges:
            return []

        G = nx.Graph()
        G.add_edges_from(edges)

        try:
            cycles = list(nx.cycle_basis(G))
            if cycles:
                return max(cycles, key=len)
            else:
                comps = list(nx.connected_components(G))
                largest_comp = max(comps, key=len)
                subg = G.subgraph(largest_comp)
                try:
                    cycle_edges = list(nx.find_cycle(subg, orientation="ignore"))
                    return [e[0] for e in cycle_edges]
                except nx.NetworkXNoCycle:
                    return list(largest_comp)
        except Exception as e:
            logger.warning(f"Failed to order ring: {e}")
            nodes: set[int] = set()
            for u, v in edges:
                nodes.add(u)
                nodes.add(v)
            return list(nodes)


class GarmentFeatureExtractor:
    """Extract features from garment mesh.

    This extractor supports two modes of operation:
    1. Deterministic: Using GarmentMetadata from garment creation (preferred)
    2. Fallback: Using vertex_labels and heuristics (legacy support)
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        category: GarmentCategory,
        vertex_panels: list[set[str]] | None = None,
        vertex_labels: dict[str, list[int]] | None = None,
        garment_metadata: GarmentMetadata | None = None,
    ) -> None:
        """Initialize the garment feature extractor.

        Parameters
        ----------
        vertices : np.ndarray
            Mesh vertices (N, 3).
        faces : np.ndarray
            Mesh faces (M, 3).
        category : GarmentCategory
            The garment category.
        vertex_panels : list[set[str]] | None
            Panel membership for each vertex (for seam detection).
        vertex_labels : dict[str, list[int]] | None
            Ring labels from BoxMesh (enables deterministic mode).
        garment_metadata : GarmentMetadata | None
            Deterministic metadata from garment creation (preferred).
        """
        self.vertices = vertices
        self.faces = faces
        self.category = category
        self.strategy = get_strategy(category)
        self.vertex_panels = (
            vertex_panels
            if vertex_panels is not None
            else [set() for _ in range(len(vertices))]
        )
        self.vertex_labels = vertex_labels
        self.garment_metadata = garment_metadata
        self.adj = self._build_adjacency()
        self.G = self._build_graph()

    def _build_adjacency(self) -> list[set[int]]:
        adj: list[set[int]] = [set() for _ in range(len(self.vertices))]
        for face in self.faces:
            for i in range(3):
                u, v = face[i], face[(i + 1) % 3]
                adj[u].add(v)
                adj[v].add(u)
        return adj

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for i, neighbors in enumerate(self.adj):
            for n in neighbors:
                dist = float(np.linalg.norm(self.vertices[i] - self.vertices[n]))
                G.add_edge(i, n, weight=dist)
        return G

    def categorize_rings(self) -> dict[GarmentRingType, list[int]]:
        """Categorize rings using deterministic data if available.

        Priority order:
        1. mesh_rings from GarmentMetadata (ARAPInitializable)
        2. vertex_labels with strategy categorization (legacy)

        Returns
        -------
        dict[GarmentRingType, list[int]]
            Mapping from ring type to vertex indices.
            Raises ValueError if no valid data is available.
        """
        # Priority 1: Use mesh_rings from GarmentMetadata (ARAPInitializable)
        if (
            self.garment_metadata is not None
            and self.garment_metadata.mesh_rings is not None
            and len(self.garment_metadata.mesh_rings) > 0
        ):
            logger.info(
                f"Using mesh_rings from ARAPInitializable: "
                f"{list(self.garment_metadata.mesh_rings.keys())}"
            )
            return self.garment_metadata.mesh_rings

        # Priority 2: Use vertex_labels with strategy categorization (legacy)
        if self.vertex_labels:
            categorized = self.strategy.categorize_rings_from_labels(self.vertex_labels)
            if categorized:
                logger.info(
                    f"Using deterministic ring categorization from labels: "
                    f"{list(categorized.keys())}"
                )
                return categorized

        raise ValueError("No valid ring data available for categorization.")

    def partition_mesh(
        self,
        categorized_rings: dict[GarmentRingType, list[int]],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        dict[
            tuple[GarmentRingType, GarmentRingType, int],
            list[int] | tuple[list[int], list[set[str]]],
        ],
    ]:
        """Partition mesh into front/back and find seams."""
        seams: dict[
            tuple[GarmentRingType, GarmentRingType, int],
            list[int] | tuple[list[int], list[set[str]]],
        ] = {}

        # Get ring sequences from strategy
        ring_sequences = self.strategy.get_ring_sequences()

        # Collect connectors - prefer upstream garment_metadata if available
        ring_connectors: dict[GarmentRingType, list[int]] = {}

        # Check if we have pre-computed data from BoxMesh/ARAPInitializable
        use_upstream = self.garment_metadata is not None and (
            self.garment_metadata.ring_connectors
            or self.garment_metadata.mesh_ring_connections is not None
        )

        # Check for mesh_ring_connections (ARAPInitializable format)
        has_mesh_connections = (
            self.garment_metadata is not None
            and self.garment_metadata.mesh_ring_connections is not None
            and len(self.garment_metadata.mesh_ring_connections) > 0
        )

        for r_type, ring in categorized_rings.items():
            # Try to use upstream ring connectors first
            if use_upstream and r_type in self.garment_metadata.ring_connectors:  # type: ignore[union-attr]
                assert self.garment_metadata is not None  # for mypy
                upstream_pair = self.garment_metadata.ring_connectors[r_type]
                connectors = [upstream_pair.connector_1, upstream_pair.connector_2]
                logger.debug(f"Using upstream ring connectors for {r_type.value}")
            else:
                # Fallback to mesh-based computation
                connectors = self._get_ring_seam_connectors(ring)

                if len(connectors) == 1:
                    p1 = connectors[0]
                    p2 = self._get_ring_opposite(ring, p1)
                    connectors.append(p2)

                if len(connectors) == 0:
                    connectors = [ring[0], ring[len(ring) // 2]]

            ring_connectors[r_type] = connectors[:2]

        used_vertices: dict[GarmentRingType, set[int]] = {
            r: set() for r in categorized_rings
        }

        # Process each sequence
        for seq in ring_sequences:
            valid_seq = [r for r in seq if r in categorized_rings]
            if len(valid_seq) < 2:
                continue

            for i in range(len(valid_seq) - 1):
                r1, r2 = valid_seq[i], valid_seq[i + 1]

                # Priority 1: Use mesh_ring_connections (ARAPInitializable format)
                if has_mesh_connections:
                    assert self.garment_metadata is not None
                    assert self.garment_metadata.mesh_ring_connections is not None
                    matching_conns = []
                    for conn in self.garment_metadata.mesh_ring_connections:
                        if (conn.ring_1 == r1 and conn.ring_2 == r2) or (
                            conn.ring_1 == r2 and conn.ring_2 == r1
                        ):
                            matching_conns.append(conn)

                    if matching_conns:
                        for idx, conn in enumerate(matching_conns):
                            # Orient path correctly (r1 -> r2)
                            if conn.ring_1 == r1:
                                path_indices = conn.path_vertex_indices
                                path_panels = conn.panel_ids
                            else:
                                path_indices = conn.path_vertex_indices[::-1]
                                path_panels = (
                                    conn.panel_ids[::-1] if conn.panel_ids else None
                                )

                            # Return seams with panel info if available
                            if path_panels:
                                seams[(r1, r2, idx)] = (path_indices, path_panels)
                            else:
                                seams[(r1, r2, idx)] = path_indices

                            if path_indices:
                                used_vertices[r1].add(path_indices[0])
                                used_vertices[r2].add(path_indices[-1])
                        continue

                # Priority 2: Use seam_paths (legacy format)
                if use_upstream and self.garment_metadata.seam_paths:  # type: ignore[union-attr]
                    assert self.garment_metadata is not None
                    matching_seams = [
                        s
                        for s in self.garment_metadata.seam_paths
                        if (s.ring_type_1 == r1 and s.ring_type_2 == r2)
                        or (s.ring_type_1 == r2 and s.ring_type_2 == r1)
                    ]

                    if matching_seams:
                        for idx, seam in enumerate(matching_seams):
                            # Orient path correctly (r1 -> r2)
                            if seam.ring_type_1 == r1:
                                path_indices = seam.path
                            else:
                                path_indices = seam.path[::-1]

                            seams[(r1, r2, idx)] = path_indices
                            used_vertices[r1].add(path_indices[0])
                            used_vertices[r2].add(path_indices[-1])
                        continue

                avail_r1 = [
                    v for v in ring_connectors[r1] if v not in used_vertices[r1]
                ]
                avail_r2 = [
                    v for v in ring_connectors[r2] if v not in used_vertices[r2]
                ]

                if not avail_r1 or not avail_r2:
                    continue

                best_path: list[int] | None = None
                best_len = float("inf")
                best_pair: tuple[int | None, int | None] = (None, None)

                # Find best pair based on Euclidean distance to avoid twisting
                # (Shortest path on manifold might favor wrong side due to curvature)
                for v1 in avail_r1:
                    for v2 in avail_r2:
                        dist = float(
                            np.linalg.norm(self.vertices[v1] - self.vertices[v2])
                        )
                        if dist < best_len:
                            best_len = dist
                            best_pair = (v1, v2)

                # Now compute actual path for the best pair
                if best_pair[0] is not None and best_pair[1] is not None:
                    try:
                        best_path = nx.shortest_path(
                            self.G,
                            source=best_pair[0],
                            target=best_pair[1],
                            weight="weight",
                        )
                    except nx.NetworkXNoPath:
                        logger.warning(
                            f"No path found between {best_pair[0]} and {best_pair[1]}"
                        )
                        continue

                if (
                    best_path is not None
                    and best_pair[0] is not None
                    and best_pair[1] is not None
                ):
                    v1, v2 = best_pair[0], best_pair[1]
                    used_vertices[r1].add(v1)
                    used_vertices[r2].add(v2)
                    seams[(r1, r2, 0)] = best_path

        # Partitioning
        face_labels, vertex_labels = self._partition_by_seams(categorized_rings, seams)

        return face_labels, vertex_labels, seams

    def _partition_by_seams(
        self,
        categorized_rings: dict[GarmentRingType, list[int]],
        seams: dict[
            tuple[GarmentRingType, GarmentRingType, int],
            list[int] | tuple[list[int], list[set[str]]],
        ],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partition mesh into front/back based on seams."""
        hem_ring = categorized_rings.get(GarmentRingType.HEM)
        if not hem_ring:
            return np.zeros(len(self.faces)), np.zeros(len(self.vertices), dtype=int)

        hem_set = set(hem_ring)
        split_points: list[int] = []

        for key, path_data in seams.items():
            # Handle potential (indices, panel_ids) tuple
            if isinstance(path_data, tuple):
                path = path_data[0]
            else:
                path = path_data

            t1, t2, _ = key
            if t1 == GarmentRingType.HEM:
                split_points.append(path[0])
            if t2 == GarmentRingType.HEM:
                split_points.append(path[-1])

        split_points = list(set([v for v in split_points if v in hem_set]))

        if len(split_points) < 2:
            return np.zeros(len(self.faces)), np.zeros(len(self.vertices), dtype=int)

        s_indices = sorted([hem_ring.index(v) for v in split_points])
        s1, s2 = s_indices[0], s_indices[-1]

        if s1 == s2 and len(s_indices) > 1:
            s2 = s_indices[-2]
            if s1 > s2:
                s1, s2 = s2, s1

        arc1 = hem_ring[s1 : s2 + 1]
        arc2 = hem_ring[s2:] + hem_ring[: s1 + 1]

        mid1 = arc1[len(arc1) // 2]
        mid2 = arc2[len(arc2) // 2]

        seam_vertices: set[int] = set()
        for path_data in seams.values():
            if isinstance(path_data, tuple):
                p_indices = path_data[0]
            else:
                p_indices = path_data
            seam_vertices.update(p_indices)
        cut_nodes = seam_vertices.union(*[set(r) for r in categorized_rings.values()])

        G_cut = self.G.copy()
        G_cut.remove_nodes_from(cut_nodes)

        vertex_labels: dict[int, int] = {}

        seed_f = next((n for n in self.G.neighbors(mid1) if n not in cut_nodes), None)
        seed_b = next((n for n in self.G.neighbors(mid2) if n not in cut_nodes), None)

        if seed_f is not None:
            try:
                for n in nx.node_connected_component(G_cut, seed_f):
                    vertex_labels[n] = 0
            except Exception:
                pass

        if seed_b is not None:
            try:
                for n in nx.node_connected_component(G_cut, seed_b):
                    vertex_labels[n] = 1
            except Exception:
                pass

        # Propagate labels to cut nodes
        for node in cut_nodes:
            nbs = [n for n in self.G.neighbors(node) if n in vertex_labels]
            if not nbs:
                continue
            votes = [vertex_labels[n] for n in nbs]
            cnt0 = votes.count(0)
            cnt1 = votes.count(1)
            vertex_labels[node] = 1 if cnt1 > cnt0 else 0

        # Output arrays
        vertex_labels_arr = np.zeros(len(self.vertices), dtype=int)
        for i in range(len(self.vertices)):
            vertex_labels_arr[i] = vertex_labels.get(i, 0)

        face_labels_arr = np.zeros(len(self.faces), dtype=int)
        for i, face in enumerate(self.faces):
            votes = [vertex_labels.get(int(v), -1) for v in face]
            cnt0 = votes.count(0)
            cnt1 = votes.count(1)
            face_labels_arr[i] = 1 if cnt1 > cnt0 else 0

        return face_labels_arr, vertex_labels_arr

    def _get_ring_seam_connectors(self, ring: list[int]) -> list[int]:
        """Identify ring vertices connecting FRONT and BACK panels.

        Uses garment_metadata.panel_positions if available (deterministic),
        otherwise raises ValueError.
        """
        candidates_indices: list[int] = []
        for i, v in enumerate(ring):
            if v < len(self.vertex_panels):
                panels = self.vertex_panels[v]
                if len(panels) > 1:
                    tags: set[str] = set()
                    for pname in panels:
                        # Use deterministic panel positions if available
                        if (
                            self.garment_metadata
                            and pname in self.garment_metadata.panel_positions
                        ):
                            pos = self.garment_metadata.panel_positions[pname]
                            tags.add(pos.value)
                        else:
                            raise ValueError(
                                f"Panel {pname} not found in garment_metadata"
                            )
                    if "front" in tags and "back" in tags:
                        candidates_indices.append(i)

        if not candidates_indices:
            return []

        # Group adjacent candidates
        groups: list[list[int]] = []
        if candidates_indices:
            current_group = [candidates_indices[0]]
            for x in candidates_indices[1:]:
                if x == current_group[-1] + 1:
                    current_group.append(x)
                else:
                    groups.append(current_group)
                    current_group = [x]
            groups.append(current_group)

        # Handle wrap-around
        if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == len(ring) - 1:
            merged = groups[-1] + groups[0]
            groups[0] = merged
            groups.pop()

        if len(groups) != 2:
            logger.warning(
                f"Ring topology warning: Found {len(groups)} transition groups, expected 2."
            )

        connectors: list[int] = []
        for g in groups:
            mid_idx = len(g) // 2
            idx_in_ring = g[mid_idx]
            connectors.append(ring[idx_in_ring])

        return connectors

    def _get_ring_opposite(self, ring: list[int], start_v: int) -> int:
        """Find the vertex opposite to start_v on the ring."""
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
