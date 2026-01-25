"""Protocol definitions for structural typing."""

from typing import Any, Protocol

import numpy as np


class StitchSegmentationEntry(Protocol):
    """Protocol for stitch segmentation entry."""

    panel_name: str | None
    stitch_id: int | None


class StitchInfo(Protocol):
    """Protocol for stitch information."""

    panel_1: str
    panel_2: str


class BoxMeshProtocol(Protocol):
    """Protocol for BoxMesh-like objects."""

    name: str
    panel_names: list[str]
    stitch_segmentation: list[list[StitchSegmentationEntry]] | None
    stitches: list[StitchInfo]
    vertex_labels: list[int] | None
    attachment_constraints: list[Any]

    def load(self) -> None: ...
    def process_attachment_constraints(self, callback: Any) -> None: ...


class PatternPieceProtocol(Protocol):
    """Protocol for pattern piece objects."""

    def assembly(self) -> Any: ...
    def get_vertex_processor_callback(self) -> Any | None: ...


class PatternDataProtocol(Protocol):
    """Protocol for pattern data objects."""

    piece: PatternPieceProtocol
    body_def: Any


class IntermediateMeshProtocol(Protocol):
    """Protocol for intermediate mesh data."""

    vertices: np.ndarray
    faces: np.ndarray


class TryOnPersonProtocol(Protocol):
    """Protocol for TryOnPerson-like objects."""

    intermediate_meshes: list[IntermediateMeshProtocol]
    segmentation_mapping: dict[str, Any]
