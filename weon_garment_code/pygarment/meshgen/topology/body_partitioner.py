"""SMPLX body mesh partitioner implementation."""

import json
from pathlib import Path

import numpy as np
from loguru import logger

from weon_garment_code.pygarment.meshgen.topology.base import MeshPartitioner
from weon_garment_code.pygarment.meshgen.topology.ring_types import BodyRingType


class BodyPartitioner(MeshPartitioner):
    """
    Partitioner for SMPLX body meshes.

    Uses body segmentation data to identify ring vertices
    and partitions into front/back regions.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        segmentation_path: str | Path | None = None,
    ):
        """
        Initialize the body partitioner.

        Args:
            vertices: (N, 3) vertex positions
            faces: (M, 3) face indices
            segmentation_path: Path to smplx_vert_segmentation.json
        """
        super().__init__(vertices, faces)
        self.segmentation = self._load_segmentation(segmentation_path)
        self._ring_cache: dict[BodyRingType, list[int]] = {}

    def _load_segmentation(self, path: str | Path | None) -> dict[str, list[int]]:
        """Load SMPLX vertex segmentation data."""
        search_paths = [
            path,
            Path("assets/smplx_vert_segmentation.json"),
            Path.cwd() / "assets" / "smplx_vert_segmentation.json",
        ]

        for p in search_paths:
            if p is None:
                continue
            p = Path(p)
            if p.exists():
                logger.info(f"Loading body segmentation from {p}")
                with open(p) as f:
                    return json.load(f)

        logger.warning("Body segmentation file not found, using empty segmentation")
        return {}

    def get_ring_definitions(self) -> dict[BodyRingType, list[int]]:
        """Get ring definitions from body segmentation."""
        if self._ring_cache:
            return self._ring_cache

        # Map segmentation keys to ring types and their expected neighbor segment
        # This allows distinguishing Ankle (Leg-Foot) from Knee (Leg-UpLeg)
        mapping = {
            BodyRingType.COLLAR: ("head", "neck"),
            BodyRingType.LEFT_CUFF: ("leftForeArm", "leftHand"),
            BodyRingType.RIGHT_CUFF: ("rightForeArm", "rightHand"),
            BodyRingType.LEFT_ANKLE: ("leftLeg", "leftFoot"),
            BodyRingType.RIGHT_ANKLE: ("rightLeg", "rightFoot"),
        }

        for ring_type, (seg_key, neighbor_key) in mapping.items():
            if seg_key in self.segmentation:
                seg_verts = set(self.segmentation[seg_key])

                neighbor_verts = None
                if neighbor_key and neighbor_key in self.segmentation:
                    neighbor_verts = set(self.segmentation[neighbor_key])

                ring = self._extract_boundary(seg_verts, neighbor_verts)
                self._ring_cache[ring_type] = ring

        return self._ring_cache

    def _extract_boundary(
        self, segment_verts: set[int], neighbor_verts: set[int] | None = None
    ) -> list[int]:
        """Extract boundary ring from a segment, optionally filtering by neighbor."""
        # Find vertices that have neighbors outside the segment
        boundary_verts = []
        for v in segment_verts:
            if v < len(self.adj):
                for n in self.adj[v]:
                    if n not in segment_verts:
                        # If neighbor filter is active, only accept if neighbor is in target set
                        if neighbor_verts is not None:
                            if n in neighbor_verts:
                                boundary_verts.append(v)
                                break
                        else:
                            boundary_verts.append(v)
                            break

        if not boundary_verts:
            return []

        # Order into a ring
        return self._order_to_ring(boundary_verts)

    def _order_to_ring(self, verts: list[int]) -> list[int]:
        """Order boundary vertices into a connected ring."""
        if len(verts) < 3:
            return verts

        ordered = [verts[0]]
        remaining = set(verts[1:])

        while remaining:
            current = ordered[-1]
            # Find next adjacent vertex
            found = False
            for n in self.adj[current]:
                if n in remaining:
                    ordered.append(n)
                    remaining.remove(n)
                    found = True
                    break
            if not found:
                # Can't continue, take closest
                if remaining:
                    ordered.append(remaining.pop())

        return ordered

    def get_ring_sequences(self) -> list[list[BodyRingType]]:
        """
        Get ring sequences for body front/back partitioning.

        Traces a loop: Collar -> L_Cuff -> L_Ankle -> R_Ankle -> R_Cuff -> (loop)
        """
        return [
            [
                BodyRingType.COLLAR,
                BodyRingType.LEFT_CUFF,
                BodyRingType.LEFT_ANKLE,
                BodyRingType.RIGHT_ANKLE,
                BodyRingType.RIGHT_CUFF,
                BodyRingType.COLLAR,  # Close the loop
            ]
        ]

    def _compute_segment_centers(self) -> dict[str, np.ndarray]:
        """Compute centers of mass for each body segment."""
        centers = {}
        for seg_name, indices in self.segmentation.items():
            if indices:
                seg_verts = self.vertices[indices]
                centers[seg_name] = np.mean(seg_verts, axis=0)
        return centers

    def get_ring_connectors(self, ring: list[int]) -> list[int]:
        """
        Get connector points on body rings using geometric Z-plane intersection.

        This ensures the seam endpoints align with the local Front/Back partition
        boundary defined by the segment's center Z.
        """
        if len(ring) < 2:
            return ring

        # Identify ring type and associated segment
        ring_type = None
        seg_key = None

        # ring_defs mapping is now ring_type -> list[int]
        # internal mapping in get_ring_definitions is ring_type -> (seg, neighbor)
        # We need to access the seg_key used for creation.
        # But get_ring_definitions returns the cached dict.
        # We can reconstruct the lookup.

        mapping = {
            BodyRingType.COLLAR: "head",
            BodyRingType.LEFT_CUFF: "leftForeArm",
            BodyRingType.RIGHT_CUFF: "rightForeArm",
            BodyRingType.LEFT_ANKLE: "leftLeg",
            BodyRingType.RIGHT_ANKLE: "rightLeg",
        }

        # Inverse lookup to find type
        ring_set = set(ring)
        for rt, rt_verts in self.get_ring_definitions().items():
            if set(rt_verts) == ring_set:
                ring_type = rt
                break

        if ring_type and ring_type in mapping:
            seg_key = mapping[ring_type]

        if seg_key is None:
            # Fallback to Z-heuristic based on ring center
            return self._get_z_heuristic_connectors(ring)

        # Get segment center
        centers = self._compute_segment_centers()
        if seg_key not in centers:
            return self._get_z_heuristic_connectors(ring)

        local_center_z = centers[seg_key][2]

        # Find zero crossings of (v.z - local_center_z) along the ring
        connectors = []
        ring_z = self.vertices[ring, 2]

        # We walk the ring and look for sign changes
        for i in range(len(ring)):
            z1 = ring_z[i] - local_center_z
            z2 = ring_z[(i + 1) % len(ring)] - local_center_z

            if z1 * z2 <= 0:
                # Sign change: crossing found. Pick vertex closer to plane.
                if abs(z1) < abs(z2):
                    connectors.append(ring[i])
                else:
                    connectors.append(ring[(i + 1) % len(ring)])

        # Dedup
        connectors = list(dict.fromkeys(connectors))

        if len(connectors) == 2:
            return connectors
        elif len(connectors) > 2:
            # Too many crossings (wobbly ring). Pick 2 furthest apart (geodesic or euclidean)
            # Simple heuristic: Max Euclidean distance pair
            best_pair = connectors[:2]
            max_dist = -1.0

            for i in range(len(connectors)):
                for j in range(i + 1, len(connectors)):
                    d = np.linalg.norm(
                        self.vertices[connectors[i]] - self.vertices[connectors[j]]
                    )
                    if d > max_dist:
                        max_dist = d
                        best_pair = [connectors[i], connectors[j]]
            return best_pair

        # Fallback if no crossings (ring entirely on one side?)
        return self._get_z_heuristic_connectors(ring)

    def _get_z_heuristic_connectors(self, ring: list[int]) -> list[int]:
        """Original Z-coordinate based heuristic."""
        ring_verts = self.vertices[ring]
        z_coords = ring_verts[:, 2]

        front_idx = int(np.argmax(z_coords))
        back_idx = int(np.argmin(z_coords))

        if front_idx == back_idx:
            back_idx = (front_idx + len(ring) // 2) % len(ring)

        return [ring[front_idx], ring[back_idx]]

    def _partition_faces(self, ring_defs: dict, seams: dict) -> np.ndarray:
        """
        Override partitioning for body mesh.

        Uses per-segment local centering for front/back classification:
        1. For each body segment, compute its local center
        2. Classify each vertex based on its Z position relative to local center
        3. Z > local_center.z → front, else → back

        This matches the smplx_uv.py approach and handles limbs correctly.
        """
        # Build vertex-to-segment mapping
        vertex_segment: dict[int, str] = {}
        for seg_name, indices in self.segmentation.items():
            for idx in indices:
                vertex_segment[idx] = seg_name

        # Compute per-segment centers
        segment_centers: dict[str, np.ndarray] = {}
        for seg_name, indices in self.segmentation.items():
            if indices:
                seg_verts = self.vertices[indices]
                segment_centers[seg_name] = np.mean(seg_verts, axis=0)

        # Classify each vertex relative to its segment's local center
        vertex_labels = np.zeros(len(self.vertices), dtype=int)
        for v_idx in range(len(self.vertices)):
            seg = vertex_segment.get(v_idx)
            if seg and seg in segment_centers:
                local_center = segment_centers[seg]
                # Relative Z: positive = front (0), negative = back (1)
                rel_z = self.vertices[v_idx, 2] - local_center[2]
                vertex_labels[v_idx] = 0 if rel_z >= 0 else 1
            else:
                # Fallback to global Z
                vertex_labels[v_idx] = 0 if self.vertices[v_idx, 2] >= 0 else 1

        # Label faces by majority vote
        face_labels = np.zeros(len(self.faces), dtype=int)
        for i, face in enumerate(self.faces):
            votes = vertex_labels[face]
            face_labels[i] = 1 if np.sum(votes) >= 2 else 0

        logger.info(
            f"Body partition: {np.sum(face_labels == 0)} front, "
            f"{np.sum(face_labels == 1)} back"
        )
        return face_labels
