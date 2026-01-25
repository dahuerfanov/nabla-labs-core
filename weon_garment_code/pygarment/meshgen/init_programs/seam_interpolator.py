"""Core seam interpolation logic."""

import numpy as np
from loguru import logger

from weon_garment_code.pygarment.meshgen.init_programs.seam_types import (
    EdgeDefinition,
    InitPreference,
    SeamMapping,
)


class SeamInterpolator:
    """Interpolates garment seam vertices onto body seam positions."""

    @staticmethod
    def compute_cumulative_distances(
        points: np.ndarray, panel_ids: list[set[str]] | None = None
    ) -> np.ndarray:
        """Compute cumulative arc-length distances along a path.

        Args:
            points: (N, 3) ordered vertex positions
            panel_ids: Optional (N,) list of panel sets for each vertex.
                       If provided, distance is 0 when transitioning between
                       disjoint panel sets (i.e., no shared panel).

        Returns:
            (N,) cumulative distances, starting at 0
        """
        if len(points) < 2:
            return np.zeros(len(points))

        diffs = np.diff(points, axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)

        # Zero out lengths for panel transitions
        if panel_ids is not None and len(panel_ids) == len(points):
            for i in range(len(edge_lengths)):
                # If no common panels between v[i] and v[i+1], treat as 0 distance
                if not (panel_ids[i] & panel_ids[i + 1]):
                    edge_lengths[i] = 0.0

        cumulative = np.zeros(len(points))
        cumulative[1:] = np.cumsum(edge_lengths)
        return cumulative

    @staticmethod
    def interpolate_position(
        body_points: np.ndarray,
        body_distances: np.ndarray,
        target_dist: float,
    ) -> np.ndarray:
        """Interpolate position on body seam at given distance.

        Args:
            body_points: (N, 3) body seam vertices
            body_distances: (N,) cumulative distances
            target_dist: Target distance along seam

        Returns:
            (3,) interpolated 3D position
        """
        # Clamp to valid range
        target_dist = np.clip(target_dist, 0, body_distances[-1])

        # Find segment
        idx = np.searchsorted(body_distances, target_dist)

        if idx == 0:
            return body_points[0].copy()
        if idx >= len(body_distances):
            return body_points[-1].copy()

        # Interpolate within segment
        d0 = body_distances[idx - 1]
        d1 = body_distances[idx]
        t = (target_dist - d0) / (d1 - d0) if d1 > d0 else 0.0

        p0 = body_points[idx - 1]
        p1 = body_points[idx]
        return p0 + t * (p1 - p0)

    def interpolate_seam(
        self,
        garment_points: np.ndarray,
        body_points: np.ndarray,
        preference: InitPreference,
        start_offset: float = 0.0,
        garment_indices: list[int] | None = None,
        garment_panel_ids: list[set[str]] | None = None,
    ) -> SeamMapping:
        """Interpolate garment seam vertices onto body seam.

        Args:
            garment_points: (N, 3) ordered garment seam vertices
            body_points: (M, 3) ordered body seam vertices
            preference: How to align when garment is shorter
            start_offset: Padding from previous seam (for START/MID calculation)
            garment_indices: Optional actual mesh vertex indices. If None, uses [0, 1, 2...]
            garment_panel_ids: Optional panel sets for garment vertices to detect transitions

        Returns:
            SeamMapping with interpolated positions
        """
        # Use actual indices if provided, otherwise sequential
        actual_indices = (
            garment_indices
            if garment_indices is not None
            else list(range(len(garment_points)))
        )

        garment_dists = self.compute_cumulative_distances(
            garment_points, garment_panel_ids
        )
        body_dists = self.compute_cumulative_distances(body_points)

        L_garment = garment_dists[-1] if len(garment_dists) > 0 else 0.0
        L_body = body_dists[-1] if len(body_dists) > 0 else 0.0

        if L_body < 1e-8:
            logger.warning("Body seam has zero length")
            return SeamMapping(
                garment_indices=actual_indices,
                body_positions=np.zeros_like(garment_points),
                remaining_dist=0.0,
                scale_factor=0.0,
            )

        # Determine scale factor and offset
        logger.debug(
            f"Seam lengths: L_garment={L_garment:.2f}, L_body={L_body:.2f}, "
            f"garment_pts={len(garment_points)}, body_pts={len(body_points)}"
        )
        if L_garment > L_body:
            # Garment longer: compress to fit
            scale_factor = L_body / L_garment
            offset = 0.0
            logger.debug(f"Compressing garment seam by {scale_factor:.2f}")
        else:
            # Garment shorter or equal: apply preference
            scale_factor = 1.0
            padding = L_body - L_garment

            if preference == InitPreference.START:
                offset = start_offset
            elif preference == InitPreference.END:
                offset = padding - start_offset
            else:  # MID
                offset = padding / 2.0

            # Clamp offset to ensure seam fits
            offset = np.clip(offset, 0, padding)

        # Interpolate each garment vertex
        body_positions = np.zeros_like(garment_points)
        last_mapped_dist = 0.0

        for i, g_dist in enumerate(garment_dists):
            target_dist = g_dist * scale_factor + offset
            body_positions[i] = self.interpolate_position(
                body_points, body_dists, target_dist
            )
            last_mapped_dist = target_dist

        remaining_dist = L_body - last_mapped_dist

        return SeamMapping(
            garment_indices=actual_indices,
            body_positions=body_positions,
            remaining_dist=remaining_dist,
            scale_factor=scale_factor,
        )

    def process_edge_sequence(
        self,
        edges: list[EdgeDefinition],
        garment_seams: dict[
            tuple, tuple[list[int], np.ndarray]
        ],  # Now (indices, coords)
        body_seams: dict[tuple, np.ndarray],
    ) -> dict[tuple, SeamMapping]:
        """Process a sequence of edges, propagating offsets.

        Args:
            edges: Ordered list of edge definitions
            garment_seams: Dict (ring_from, ring_to) -> (vertex_indices, coordinates)
            body_seams: Dict (ring_from, ring_to) -> (M, 3) vertices

        Returns:
            Dict (ring_from, ring_to) -> SeamMapping
        """
        mappings: dict[tuple, SeamMapping] = {}
        carry_offset = 0.0

        last_to_ring = None

        for edge in edges:
            # Reset carry_offset if not continuing from previous edge OR explicit reset
            if edge.reset_offset or (
                last_to_ring is not None and edge.ring_from != last_to_ring
            ):
                carry_offset = 0.0

            last_to_ring = edge.ring_to

            key = (edge.ring_from, edge.ring_to)

            garment_data = garment_seams.get(key)
            body_pts = body_seams.get(key)

            if garment_data is None or body_pts is None:
                # Try reversed
                rev_key = (edge.ring_to, edge.ring_from)
                if rev_key in garment_seams and rev_key in body_seams:
                    rev_data = garment_seams[rev_key]
                    # Unpack potentially 3 elements
                    if len(rev_data) == 3:
                        garment_indices, garment_pts, garment_panel_ids = (
                            rev_data[0][::-1],
                            rev_data[1][::-1],
                            rev_data[2][::-1],
                        )
                    else:
                        garment_indices, garment_pts = (
                            rev_data[0][::-1],
                            rev_data[1][::-1],
                        )
                        garment_panel_ids = None

                    body_pts = body_seams[rev_key][::-1]
                else:
                    logger.warning(f"Missing seam data for edge {key}")
                    continue
            else:
                # Unpack potentially 3 elements
                if len(garment_data) == 3:
                    garment_indices, garment_pts, garment_panel_ids = (
                        garment_data[0],
                        garment_data[1],
                        garment_data[2],
                    )
                else:
                    garment_indices, garment_pts = garment_data[0], garment_data[1]
                    garment_panel_ids = None

            mapping = self.interpolate_seam(
                garment_pts,
                body_pts,
                edge.preference,
                start_offset=carry_offset,
                garment_indices=garment_indices,
                garment_panel_ids=garment_panel_ids,
            )

            mappings[key] = mapping
            carry_offset = mapping.remaining_dist

            logger.info(
                f"Seam {edge.ring_from.value}->{edge.ring_to.value}: "
                f"scale={mapping.scale_factor:.2f}, remaining={mapping.remaining_dist:.3f}"
            )

        return mappings
