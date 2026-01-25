"""High-level garment initialization orchestration."""

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from weon_garment_code.pygarment.meshgen.init_programs.seam_interpolator import (
    SeamInterpolator,
)
from weon_garment_code.pygarment.meshgen.init_programs.seam_types import (
    EdgeDefinition,
    InitPreference,
    RingType,
    SeamMapping,
)


class GarmentInitializer(ABC):
    """Abstract base class for garment-to-body seam initialization."""

    def __init__(self):
        self.interpolator = SeamInterpolator()

    @abstractmethod
    def get_edge_sequence(self) -> list[EdgeDefinition]:
        """Get ordered edge definitions with preferences."""
        ...

    @abstractmethod
    def get_ring_mapping(self) -> dict[RingType, RingType]:
        """Map garment rings to body rings (for virtual HEM etc.)."""
        ...

    def initialize(
        self,
        garment_seams: dict[tuple, tuple[list[int], np.ndarray, list[set[str]] | None]],
        body_seams: dict[tuple, np.ndarray],
        garment_vertices: np.ndarray,
        body_vertices: np.ndarray,
    ) -> dict[tuple[RingType, RingType], SeamMapping]:
        """Initialize garment seams on body.

        Args:
            garment_seams: (ring_from, ring_to) -> vertex positions
            body_seams: (ring_from, ring_to) -> vertex positions
            garment_vertices: Full garment mesh vertices
            body_vertices: Full body mesh vertices

        Returns:
            Seam mappings for each edge
        """
        edges = self.get_edge_sequence()
        ring_map = self.get_ring_mapping()

        # Remap body seams using virtual mapping
        mapped_body_seams = {}
        for (r1, r2), pts in body_seams.items():
            # Map to garment ring names
            g1 = ring_map.get(r1, r1)
            g2 = ring_map.get(r2, r2)
            mapped_body_seams[(g1, g2)] = pts

        return self.interpolator.process_edge_sequence(
            edges, garment_seams, mapped_body_seams
        )

    def save_debug_visualization(
        self,
        mappings: dict[tuple[RingType, RingType], SeamMapping],
        body_vertices: np.ndarray,
        body_faces: np.ndarray,
        output_path: str | Path,
    ) -> None:
        """Save debug visualization of seam mappings.

        Creates:
        1. OBJ with colored seam points
        2. 2D projection image with seam markings
        """
        output_path = Path(output_path)

        # Collect all mapped points
        all_body_positions = []
        edge_colors = {}
        color_palette = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]

        for i, (edge_key, mapping) in enumerate(mappings.items()):
            color = color_palette[i % len(color_palette)]
            edge_colors[edge_key] = color
            for pos in mapping.body_positions:
                all_body_positions.append((pos, color))

        # Save OBJ with seam points
        obj_path = output_path.with_suffix(".obj")
        self._save_seam_obj(obj_path, body_vertices, body_faces, all_body_positions)

        # Save 2D projection
        img_path = output_path.with_suffix(".png")
        self._save_seam_projection(
            img_path, body_vertices, all_body_positions, edge_colors, mappings
        )

        logger.info(f"Saved debug visualization to {obj_path} and {img_path}")

    def _save_seam_obj(
        self,
        path: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        seam_points: list[tuple[np.ndarray, tuple[int, int, int]]],
    ) -> None:
        """Save OBJ with body mesh and colored seam point markers."""
        with open(path, "w") as f:
            # Write body vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write seam points as additional vertices (offset by body vertex count)
            seam_start = len(vertices)
            for pos, color in seam_points:
                r, g, b = color[0] / 255, color[1] / 255, color[2] / 255
                f.write(f"v {pos[0]} {pos[1]} {pos[2]} {r} {g} {b}\n")

            # Write faces
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

            # Write point group for seams
            f.write("g seam_points\n")
            for i in range(len(seam_points)):
                f.write(f"p {seam_start + i + 1}\n")

    def _save_seam_projection(
        self,
        path: Path,
        body_vertices: np.ndarray,
        seam_points: list[tuple[np.ndarray, tuple[int, int, int]]],
        edge_colors: dict,
        mappings: dict,
        size: tuple[int, int] = (800, 1200),
    ) -> None:
        """Save 2D front projection with seam markings."""
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

        # Project body vertices (front view: X, Y)
        x_coords = body_vertices[:, 0]
        y_coords = body_vertices[:, 1]

        # Normalize to image coordinates
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        margin = 50
        scale_x = (size[0] - 2 * margin) / (x_max - x_min)
        scale_y = (size[1] - 2 * margin) / (y_max - y_min)
        scale = min(scale_x, scale_y)

        def to_img(pt: np.ndarray) -> tuple[int, int]:
            x = int((pt[0] - x_min) * scale + margin)
            y = int(size[1] - (pt[1] - y_min) * scale - margin)
            return (x, y)

        # Draw body silhouette (projected boundary)
        body_proj = np.array([to_img(v) for v in body_vertices])
        cv2.polylines(
            img, [body_proj], isClosed=False, color=(200, 200, 200), thickness=1
        )

        # Draw seam points
        for pos, color in seam_points:
            pt = to_img(pos)
            cv2.circle(img, pt, 4, color, -1)

        # Draw legend
        y_legend = 30
        for edge_key, color in edge_colors.items():
            label = f"{edge_key[0].value} -> {edge_key[1].value}"
            cv2.putText(
                img, label, (10, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            y_legend += 20

        cv2.imwrite(str(path), img)


class ShirtInitializer(GarmentInitializer):
    """Initializer for shirt garments."""

    def get_edge_sequence(self) -> list[EdgeDefinition]:
        """Shirt seam sequence."""
        return [
            EdgeDefinition(RingType.COLLAR, RingType.LEFT_CUFF, InitPreference.START),
            EdgeDefinition(RingType.LEFT_CUFF, RingType.HEM, InitPreference.START),
            EdgeDefinition(RingType.COLLAR, RingType.RIGHT_CUFF, InitPreference.START),
            EdgeDefinition(RingType.RIGHT_CUFF, RingType.HEM, InitPreference.START),
        ]

    def get_ring_mapping(self) -> dict[RingType, RingType]:
        """Map body rings to garment ring names.

        Virtual HEM: LEFT_CUFF-HEM uses LEFT_ANKLE, RIGHT_CUFF-HEM uses RIGHT_ANKLE
        """
        return {
            # Body LEFT_ANKLE becomes garment HEM for left arm path
            RingType.LEFT_ANKLE: RingType.HEM,
            # Body RIGHT_ANKLE becomes garment HEM for right arm path
            RingType.RIGHT_ANKLE: RingType.HEM,
        }


class PantsInitializer(GarmentInitializer):
    """Initializer for pants garments."""

    def get_edge_sequence(self) -> list[EdgeDefinition]:
        """Pants seam sequence."""
        return [
            EdgeDefinition(
                RingType.LEFT_ANKLE, RingType.RIGHT_ANKLE, InitPreference.MID
            ),
            EdgeDefinition(RingType.RIGHT_ANKLE, RingType.HEM, InitPreference.START),
            EdgeDefinition(
                RingType.LEFT_ANKLE,
                RingType.HEM,
                InitPreference.START,
                reset_offset=True,
            ),
        ]

    def get_ring_mapping(self) -> dict[RingType, RingType]:
        """Map body cuffs to garment waist (HEM).

        The body side seam goes from Ankle to Cuff (Armpit).
        Pants side seam goes from Ankle to Waist (HEM).
        So we map the body's 'upper' boundary (Cuff) to the garment's upper (HEM).
        """
        return {
            RingType.LEFT_CUFF: RingType.HEM,
            RingType.RIGHT_CUFF: RingType.HEM,
        }
