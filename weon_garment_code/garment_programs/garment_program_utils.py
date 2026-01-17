"""Utility functions for garment program attachment constraint processing."""

from collections.abc import Callable
from typing import Any

import weon_garment_code.pygarment.garmentcode as pyg
from loguru import logger
from weon_garment_code.pygarment.meshgen.box_mesh_gen.stitch_types import EdgeLabel as BoxEdgeLabel


def link_symmetric_components(comp_a: pyg.Component, comp_b: pyg.Component, prefix_a: str, prefix_b: str) -> None:
    """Link symmetric components by identifying mirrored pairs of panels.

    We assume that panels are named with a prefix (e.g. "left_front_panel" and
    "right_front_panel"). This function iterates recursively over all panels in
    both components, strips the given prefixes, and links panels that have matching
    base names.

    Parameters
    ----------
    comp_a : pyg.Component
        First component (e.g. self.left)
    comp_b : pyg.Component
        Second component (e.g. self.right)
    prefix_a : str
        Prefix for the first component
    prefix_b : str
        Prefix for the second component
    """

    def _collect_panels(comp: Any) -> list[pyg.Panel]:
        """Recursively collect all Panel instances from a component tree."""
        panels: list[pyg.Panel] = []
        if isinstance(comp, pyg.Panel):
            panels.append(comp)
        elif hasattr(comp, "_get_subcomponents"):
            for sub in comp._get_subcomponents():
                panels.extend(_collect_panels(sub))
        return panels

    panels_a = _collect_panels(comp_a)
    panels_b = _collect_panels(comp_b)

    # Map base name to panel for component A
    # Handle both prefix (l_pant_f) and suffix (pant_f_l) naming conventions
    map_a: dict[str, pyg.Panel] = {}
    for p in panels_a:
        # Try suffix first (more common: pant_f_l)
        if p.name.endswith(f"_{prefix_a}"):
            base = p.name[: -len(prefix_a) - 1]  # Remove _suffix
            map_a[base] = p
        elif p.name.startswith(f"{prefix_a}_"):
            base = p.name[len(prefix_a) + 1 :]  # Remove prefix_
            map_a[base] = p

    # Find matches in component B
    for p_b in panels_b:
        # Try suffix first
        if p_b.name.endswith(f"_{prefix_b}"):
            base = p_b.name[: -len(prefix_b) - 1]
            if base in map_a:
                p_a = map_a[base]
                p_a.symmetry_partner = p_b.name
                p_b.symmetry_partner = p_a.name
        elif p_b.name.startswith(f"{prefix_b}_"):
            base = p_b.name[len(prefix_b) + 1 :]
            if base in map_a:
                p_a = map_a[base]
                p_a.symmetry_partner = p_b.name
                p_b.symmetry_partner = p_a.name


class AttachmentHandler:
    """Static class for handling attachment constraint processing."""

    @staticmethod
    def process_lower_interface_constraint(constraint: Any, box_mesh: Any, label_value: str) -> list[int] | None:
        """
        Process LOWER_INTERFACE constraint by using vertices already labeled from edges.

        Parameters
        ----------
        constraint : AttachmentConstraint
            The attachment constraint to process.
        box_mesh : BoxMesh
            The BoxMesh instance containing the vertices.
        label_value : str
            The string value of the constraint label.

        Returns
        -------
        list[int] | None
            List of vertex IDs if found, None otherwise.
        """
        if label_value in box_mesh.vertex_labels:
            vertices = box_mesh.vertex_labels[label_value]
            logger.debug(f"Using {len(vertices)} already-labeled vertices for LOWER_INTERFACE constraint")
            return vertices
        else:
            logger.warning(f"No vertices found with label {label_value} for LOWER_INTERFACE constraint")
            return None

    @staticmethod
    def process_crotch_constraint(constraint: Any, box_mesh: Any) -> list[int] | None:
        """
        Process CROTCH constraint by finding edges, then finding all vertices below in Y direction.

        Parameters
        ----------
        constraint : AttachmentConstraint
            The attachment constraint to process.
        box_mesh : BoxMesh
            The BoxMesh instance containing the vertices.

        Returns
        -------
        list[int] | None
            List of vertex IDs below the crotch point, None if no reference vertices found.
        """
        if not constraint.vertex_labels_to_find:
            logger.warning("CROTCH constraint has no vertex_labels_to_find")
            return None

        # Find reference vertices from CROTCH_POINT_SEAM edges
        labels_to_find = constraint.vertex_labels_to_find_enums
        reference_vertices = box_mesh.find_vertices_by_edge_labels(labels_to_find)

        if not reference_vertices:
            logger.warning(
                f"No reference vertices found for CROTCH constraint "
                f"with labels {[label.value for label in labels_to_find]}"
            )
            return None

        # Find all vertices below the crotch point in Y direction
        crotch_y_values = [box_mesh.vertices[v_id][1] for v_id in reference_vertices]
        min_crotch_y = min(crotch_y_values)

        vertices = [i for i, v in enumerate(box_mesh.vertices) if v[1] < min_crotch_y]

        logger.debug(
            f"Processed CROTCH constraint: found {len(vertices)} vertices below crotch point (y < {min_crotch_y:.2f})"
        )

        return vertices

    @staticmethod
    def process_default_constraint(constraint: Any, box_mesh: Any, label_value: str) -> list[int] | None:
        """
        Process constraint with default behavior: find vertices from edge labels.

        Parameters
        ----------
        constraint : AttachmentConstraint
            The attachment constraint to process.
        box_mesh : BoxMesh
            The BoxMesh instance containing the vertices.
        label_value : str
            The string value of the constraint label.

        Returns
        -------
        list[int] | None
            List of vertex IDs found from edge labels, None if no labels to find.
        """
        if not constraint.vertex_labels_to_find:
            return None

        logger.info(f"Processing constraint {label_value} with default behavior: finding vertices from edge labels")
        labels_to_find = constraint.vertex_labels_to_find_enums
        vertices = box_mesh.find_vertices_by_edge_labels(labels_to_find)
        return vertices

    @staticmethod
    def label_vertices(box_mesh: Any, label_value: str, vertices: list[int]) -> None:
        """
        Label vertices in the box mesh, removing duplicates.

        Parameters
        ----------
        box_mesh : BoxMesh
            The BoxMesh instance to label vertices in.
        label_value : str
            The label to apply to the vertices.
        vertices : list[int]
            List of vertex IDs to label.
        """
        if not vertices:
            logger.warning(f"No final vertices to label for constraint {label_value}")
            return

        # Remove duplicates while preserving order
        unique_vertices = list(dict.fromkeys(vertices))
        box_mesh.vertex_labels.setdefault(label_value, []).extend(unique_vertices)
        logger.debug(f"Labeled {len(unique_vertices)} vertices with label {label_value}")

    @staticmethod
    def process_attachment_constraints_generic(
        garment_class: type[Any],
        box_mesh: Any,
        constraint_processors: dict[BoxEdgeLabel, Callable[[Any, Any], list[int] | None]] | None = None,
    ) -> None:
        """
        Generic function to process attachment constraints for any garment class.

        This function handles the common flow:
        1. Get constraints from the garment class
        2. Store constraints in BoxMesh
        3. Process each constraint using provided processors or default behavior
        4. Label vertices

        Parameters
        ----------
        garment_class : Type
            The garment class to get constraints from (must have get_attachment_constraints static method).
        box_mesh : BoxMesh
            The BoxMesh instance to process constraints for.
        constraint_processors : dict[BoxEdgeLabel, Callable], optional
            Dictionary mapping constraint labels to custom processor functions.
            If a constraint label is not in this dict, default behavior is used.
            Processor functions should take (constraint, box_mesh) and return list[int] | None.
        """
        # Get constraints from garment class
        constraints = garment_class.get_attachment_constraints()

        if not constraints:
            logger.debug(f"No attachment constraints found for {garment_class.__name__}")
            return

        # Store constraints in BoxMesh for later use
        box_mesh.attachment_constraints = constraints

        # Default processors for common constraint types
        default_processors = {
            BoxEdgeLabel.LOWER_INTERFACE: lambda c, bm: AttachmentHandler.process_lower_interface_constraint(
                c, bm, c.label_enum.value
            ),
            BoxEdgeLabel.CROTCH: AttachmentHandler.process_crotch_constraint,
        }

        # Merge with provided processors (provided ones take precedence)
        if constraint_processors:
            processors = {**default_processors, **constraint_processors}
        else:
            processors = default_processors

        # Process each constraint
        for constraint in constraints:
            constraint_label = constraint.label_enum
            label_value = constraint_label.value

            if not constraint.vertex_labels_to_find:
                logger.warning(f"Constraint with label {label_value} has no vertex_labels_to_find. Skipping.")
                continue

            # Get processor for this constraint type
            if constraint_label in processors:
                vertices = processors[constraint_label](constraint, box_mesh)
                if vertices is None:
                    continue
            else:
                # Use default behavior
                vertices = AttachmentHandler.process_default_constraint(constraint, box_mesh, label_value)
                if vertices is None:
                    continue

            # Label the vertices
            AttachmentHandler.label_vertices(box_mesh, label_value, vertices)

    @staticmethod
    def create_default_vertex_processor(garment_class: type[Any]) -> Callable[[Any], None]:
        """
        Create a default vertex processor callback for garments with no special handling.

        Parameters
        ----------
        garment_class : Type
            The garment class to create a processor for.

        Returns
        -------
        Callable[[BoxMesh], None]
            A vertex processor callback function.
        """

        def processor(box_mesh: Any) -> None:
            """Default vertex processor that uses generic constraint processing."""
            AttachmentHandler.process_attachment_constraints_generic(garment_class, box_mesh)

        return processor
