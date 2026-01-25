"""Data structures for ARAP initialization.

This module defines strongly-typed data structures for deterministic ARAP
initialization. These structures are computed at garment creation time and
translated to mesh-level data during box mesh generation.

Key structures:
- RingConnector: A point on a ring edge where seams attach
- GarmentRing: A closed ring of edges from garment interfaces
- RingConnection: A connection between two rings via panel boundary edges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from weon_garment_code.pygarment.meshgen.arap.core_types import GarmentRingType

if TYPE_CHECKING:
    from weon_garment_code.pygarment.garmentcode.edge import Edge, EdgeSequence


class RingValidationError(Exception):
    """Raised when ring validation fails."""


class RingConnectionValidationError(Exception):
    """Raised when ring connection validation fails."""


@dataclass
class RingConnector:
    """A point on a ring edge where seams attach.

    Connectors define the attachment points for seam paths on each ring.
    Each ring has exactly two connectors (front/back boundary points).

    Attributes
    ----------
    edge : Edge
        The pygarment Edge this connector lies on
    parameter : float
        Position along the edge in [0, 1], where 0 is edge start, 1 is edge end
    ring_type : GarmentRingType
        Which ring this connector belongs to
    """

    edge: Edge
    parameter: float
    ring_type: GarmentRingType

    def __post_init__(self) -> None:
        if not 0.0 <= self.parameter <= 1.0:
            raise ValueError(f"parameter must be in [0, 1], got {self.parameter}")

    def get_point(self) -> tuple[float, float]:
        """Get the 2D coordinates of this connector on the edge."""
        curve = self.edge.as_curve()
        # svgpathtools returns complex number; convert to (x, y)
        complex_point = curve.point(self.parameter)
        return (complex_point.real, complex_point.imag)


@dataclass
class GarmentRing:
    """A closed ring of edges from garment interfaces.

    Rings represent the boundary curves of the garment (collar, cuffs, hem, etc).
    Each ring is defined by an ordered sequence of edges that form a closed loop.

    Note: Dart edges are explicitly EXCLUDED from rings as they are internal
    stitches, not boundary curves.

    Attributes
    ----------
    ring_type : GarmentRingType
        The semantic type of this ring (COLLAR, LEFT_CUFF, etc)
    edges : EdgeSequence
        Ordered edges forming a closed loop
    connectors : tuple[RingConnector, RingConnector]
        The two attachment points for seam paths (front/back boundaries)
    """

    ring_type: GarmentRingType
    edges: EdgeSequence
    connectors: tuple[RingConnector, RingConnector]

    def validate(self) -> None:
        """Validate that edges form a closed curve.

        Raises
        ------
        RingValidationError
            If edges don't form a closed curve or connectors are invalid
        """
        if len(self.edges) == 0:
            raise RingValidationError(f"Ring {self.ring_type.value} has no edges")

        # Check edges form closed loop (last edge end == first edge start)
        first_start = self.edges[0].start
        last_end = self.edges[-1].end

        # Tolerance for floating point comparison
        tol = 1e-6
        if (
            abs(first_start[0] - last_end[0]) > tol
            or abs(first_start[1] - last_end[1]) > tol
        ):
            raise RingValidationError(
                f"Ring {self.ring_type.value} is not closed: "
                f"first edge starts at {first_start}, last edge ends at {last_end}"
            )

        # Validate connectors belong to this ring
        for i, conn in enumerate(self.connectors):
            if conn.ring_type != self.ring_type:
                raise RingValidationError(
                    f"Connector {i} has ring_type {conn.ring_type.value}, "
                    f"expected {self.ring_type.value}"
                )
            if conn.edge not in self.edges:
                raise RingValidationError(
                    f"Connector {i} edge is not part of ring {self.ring_type.value}"
                )


@dataclass
class RingConnection:
    """Connection between two rings via panel boundary edges.

    Ring connections define the seam paths that connect different rings.
    The order is significant: path goes from ring_1 → ring_2.

    Attributes
    ----------
    ring_1 : GarmentRingType
        Source ring (path starts here)
    ring_2 : GarmentRingType
        Target ring (path ends here)
    connector_1 : RingConnector
        Connector on ring_1 where path starts
    connector_2 : RingConnector
        Connector on ring_2 where path ends
    path_edges : EdgeSequence
        Ordered edges from connector_1 to connector_2
    """

    ring_1: GarmentRingType
    ring_2: GarmentRingType
    connector_1: RingConnector
    connector_2: RingConnector
    path_edges: EdgeSequence

    def validate(self) -> None:
        """Validate that path connects the connectors in order.

        Raises
        ------
        RingConnectionValidationError
            If path doesn't properly connect the two connectors
        """
        if len(self.path_edges) == 0:
            raise RingConnectionValidationError(
                f"Connection {self.ring_1.value}→{self.ring_2.value} has no edges"
            )

        # Check connector ring types match
        if self.connector_1.ring_type != self.ring_1:
            raise RingConnectionValidationError(
                f"connector_1 ring_type {self.connector_1.ring_type.value} "
                f"doesn't match ring_1 {self.ring_1.value}"
            )
        if self.connector_2.ring_type != self.ring_2:
            raise RingConnectionValidationError(
                f"connector_2 ring_type {self.connector_2.ring_type.value} "
                f"doesn't match ring_2 {self.ring_2.value}"
            )

        # Check path starts at connector_1 and ends at connector_2
        tol = 1e-6
        path_start = self.path_edges[0].start
        path_end = self.path_edges[-1].end

        conn1_point = self.connector_1.get_point()
        conn2_point = self.connector_2.get_point()

        if (
            abs(path_start[0] - conn1_point[0]) > tol
            or abs(path_start[1] - conn1_point[1]) > tol
        ):
            raise RingConnectionValidationError(
                f"Path start {path_start} doesn't match connector_1 at {conn1_point}"
            )

        if (
            abs(path_end[0] - conn2_point[0]) > tol
            or abs(path_end[1] - conn2_point[1]) > tol
        ):
            raise RingConnectionValidationError(
                f"Path end {path_end} doesn't match connector_2 at {conn2_point}"
            )


@dataclass
class MeshRing:
    """Mesh-level representation of a ring after stitching.

    This is the translated version of GarmentRing with vertex indices.

    Attributes
    ----------
    ring_type : GarmentRingType
        The semantic type of this ring
    vertex_indices : list[int]
        Ordered vertex indices forming a closed loop
    connector_indices : tuple[int, int]
        Vertex indices of the two connectors
    """

    ring_type: GarmentRingType
    vertex_indices: list[int]
    connector_indices: tuple[int, int]

    def validate(self) -> None:
        """Validate that vertex sequence forms a closed ring.

        Raises
        ------
        RingValidationError
            If vertices don't form a valid closed ring
        """
        if len(self.vertex_indices) < 3:
            raise RingValidationError(
                f"MeshRing {self.ring_type.value} has fewer than 3 vertices"
            )

        # Check connectors are in the ring
        for i, conn_idx in enumerate(self.connector_indices):
            if conn_idx not in self.vertex_indices:
                raise RingValidationError(
                    f"Connector {i} (vertex {conn_idx}) not in ring vertices"
                )


@dataclass
class MeshRingConnection:
    """Mesh-level representation of a ring connection after stitching.

    This is the translated version of RingConnection with vertex indices.

    Attributes
    ----------
    ring_1 : GarmentRingType
        Source ring
    ring_2 : GarmentRingType
        Target ring
    connector_1_idx : int
        Vertex index of connector on ring_1
    connector_2_idx : int
        Vertex index of connector on ring_2
    path_vertex_indices : list[int]
        Ordered vertex indices from connector_1 to connector_2
    """

    ring_1: GarmentRingType
    ring_2: GarmentRingType
    connector_1_idx: int
    connector_2_idx: int
    path_vertex_indices: list[int]
    panel_ids: list[set[str]] | None = None

    def validate(self) -> None:
        """Validate that path connects the intended connectors.

        Raises
        ------
        RingConnectionValidationError
            If path doesn't connect the connectors correctly
        """
        if len(self.path_vertex_indices) < 2:
            raise RingConnectionValidationError(
                f"MeshRingConnection {self.ring_1.value}→{self.ring_2.value} "
                f"has fewer than 2 vertices"
            )

        if self.path_vertex_indices[0] != self.connector_1_idx:
            raise RingConnectionValidationError(
                f"Path starts at vertex {self.path_vertex_indices[0]}, "
                f"expected connector_1 at {self.connector_1_idx}"
            )

        if self.path_vertex_indices[-1] != self.connector_2_idx:
            raise RingConnectionValidationError(
                f"Path ends at vertex {self.path_vertex_indices[-1]}, "
                f"expected connector_2 at {self.connector_2_idx}"
            )
