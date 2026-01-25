"""Routines to run cloth simulation."""

import multiprocessing
import platform
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from loguru import logger

import weon_garment_code.pygarment.meshgen.box_mesh_gen as bmg
from weon_garment_code.pygarment.garmentcode.component import Component
from weon_garment_code.pygarment.meshgen.arap import (
    ARAPSolver,
    extract_mesh_from_boxmesh,
    update_boxmesh_vertices,
)
from weon_garment_code.pygarment.meshgen.arap.debug_utils import (
    save_colored_obj,
    save_seam_association_debug,
)
from weon_garment_code.pygarment.meshgen.arap.feature_recognition import (
    GarmentFeatureExtractor,
)
from weon_garment_code.pygarment.meshgen.arap.ring_identifier import identify_rings
from weon_garment_code.pygarment.meshgen.arap.types import GarmentCategory
from weon_garment_code.pygarment.meshgen.box_mesh_gen import BoxMesh
from weon_garment_code.pygarment.meshgen.constants import Z_OFFSET_ARAP
from weon_garment_code.pygarment.meshgen.init_programs import (
    PantsInitializer,
    ShirtInitializer,
)
from weon_garment_code.pygarment.meshgen.init_programs.seam_types import RingType
from weon_garment_code.pygarment.meshgen.pattern_data import (
    PatternData,
    SimulationConfig,
)
from weon_garment_code.pygarment.meshgen.sim_data import GarmentData, GCTryOnPerson
from weon_garment_code.pygarment.meshgen.simulation import run_sim
from weon_garment_code.pygarment.meshgen.topology import BodyPartitioner


def batch_simulate_garment(
    pattern_data: dict[str, PatternData],
    sim_config: SimulationConfig,
    try_on_person: GCTryOnPerson,
    num_samples: int | None = None,
    output_dir: Path | None = None,
) -> bool:
    """Perform pattern simulation using in-memory pattern data."""
    if not pattern_data:
        logger.error("No pattern data provided")
        return False

    first_pattern = next(iter(pattern_data.values()))
    if not isinstance(first_pattern, PatternData):
        logger.error("Pattern data must contain PatternData objects")
        return False

    pattern_names = list(pattern_data.keys())
    total_patterns = len(pattern_names)
    logger.info(
        f"Starting batch simulation with in-memory data: {total_patterns} patterns found"
    )

    if num_samples is not None:
        logger.info(f"Processing {num_samples} samples in this batch")

    count = 0
    for pattern_name in pattern_names:
        try:
            pattern_info = pattern_data[pattern_name]
            if not isinstance(pattern_info, PatternData):
                logger.error(f"Pattern {pattern_name} is not a PatternData object")
                continue

            simulate_garment(
                pattern_info,
                sim_config,
                try_on_person,
                output_dir=output_dir,
            )
        except BaseException as e:
            logger.error(f"Pattern simulation failed for {pattern_name}: {e}")
            logger.exception("Exception details:")

        count += 1
        if count % 10 == 0:
            logger.info(f"Processed {count} patterns...")

        if num_samples is not None and count >= num_samples:
            logger.info(f"Batch limit reached ({num_samples} samples processed)")
            break

    logger.info(
        f"Finished batch processing: {count}/{total_patterns} patterns processed."
    )
    return count == total_patterns


def simulate_garment(
    pattern_data: PatternData,
    sim_config: SimulationConfig,
    try_on_person: GCTryOnPerson,
    output_dir: Path | None = None,
) -> Any:
    """Simulate a garment template using in-memory pattern data.

    Args:
        pattern_data: Pattern data with piece and body definition
        sim_config: Simulation configuration
        try_on_person: Body mesh data
        output_dir: Directory for debug output files. If None, uses CWD.
    """
    # Default output directory
    debug_output_dir = output_dir if output_dir is not None else Path.cwd()
    debug_paths: dict[str, Path] = {}

    piece: Component = pattern_data.piece
    body_def = pattern_data.body_def

    pattern = piece.assembly()

    config = sim_config.sim_config
    res = config.resolution_scale

    garment_metadata = piece.get_garment_metadata()
    garment_box_mesh = BoxMesh(pattern, garment_metadata, res, program=piece)

    timeout_after = int(config._props.get("max_meshgen_time", 20))

    try:
        _load_boxmesh_timeout(garment_box_mesh, timeout_after)
    except TimeoutError as e:
        logger.error(f"Mesh generation timeout for {garment_box_mesh.name}: {e}")
        return None, None, debug_paths
    except bmg.PatternLoadingError as e:
        logger.error(f"Pattern loading error for {garment_box_mesh.name}: {e}")
        return None, None, debug_paths
    except BaseException as e:
        logger.error(f"Pattern loading failed for {garment_box_mesh.name}: {e}")
        logger.exception("Exception details:")
        return None, None, debug_paths

    # --- ARAP Initialization & Seam Interpolation ---
    try:
        logger.info(f"Running ARAP Initialization for {garment_box_mesh.name}...")

        # 1. Extract Flat Mesh
        verts, faces = extract_mesh_from_boxmesh(garment_box_mesh)

        # 2. Detect Garment Category - prefer garment_metadata if available
        if garment_box_mesh.garment_metadata is not None:
            category = garment_box_mesh.garment_metadata.category
            logger.info(f"Using GarmentMetadata category: {category.value}")
        else:
            raise ValueError(
                "GarmentMetadata is required for ARAP initialization. Please provide it."
            )

        # 3. Extract vertex panel info
        vertex_panels_sets = _extract_vertex_panels(garment_box_mesh, len(verts))

        # Pass vertex_labels and garment_metadata for deterministic processing
        garment_extractor = GarmentFeatureExtractor(
            verts,
            faces,
            category,
            vertex_panels=vertex_panels_sets,
            vertex_labels=garment_box_mesh.vertex_labels,
            garment_metadata=garment_box_mesh.garment_metadata,
        )

        # 4. Body Feature Extraction
        body_verts = try_on_person.intermediate_meshes[0].vertices.astype(np.float64)
        body_faces = try_on_person.intermediate_meshes[0].faces.astype(np.int32)

        body_h = body_verts[:, 1].max() - body_verts[:, 1].min()
        body_scale_factor = 100.0 if body_h < 3.0 else 1.0
        if body_scale_factor > 1.0:
            logger.info(
                f"Body height {body_h:.2f} < 3.0, assuming meters. Scaling by 100."
            )

        scaled_body_verts = body_verts * body_scale_factor

        body_partitioner = BodyPartitioner(scaled_body_verts, body_faces)
        body_face_labels, body_seams = body_partitioner.partition()
        logger.info(
            f"Body Partition: {len(body_seams)} seams, {len(body_face_labels)} faces labelled"
        )

        # 5. Garment Ring Detection
        raw_garment_rings = identify_rings(verts, faces)
        logger.info(f"Identified {len(raw_garment_rings)} boundary loops on garment.")

        categorized_rings = garment_extractor.categorize_rings()

        garment_face_labels, garment_vertex_labels, garment_seams_path = (
            garment_extractor.partition_mesh(categorized_rings)
        )
        logger.info(f"Garment Partitioned: {len(garment_seams_path)} seams found.")

        # 6. Seam Interpolation
        garment_seams_dict = _build_garment_seams_dict(
            garment_seams_path, verts, vertex_panels=vertex_panels_sets
        )
        body_seams_dict = _build_body_seams_dict(body_seams, scaled_body_verts)

        initializer = (
            ShirtInitializer()
            if category == GarmentCategory.SHIRT
            else PantsInitializer()
        )
        seam_mappings = initializer.initialize(
            garment_seams_dict,  # type: ignore
            body_seams_dict,
            verts,
            scaled_body_verts,
        )
        logger.info(
            f"Seam Interpolation: Mapping generated for {len(seam_mappings)} seam segments."
        )

        # Debug Visualization
        png_path = save_seam_association_debug(
            debug_output_dir,
            f"debug_seam_assoc_{garment_box_mesh.name}.png",
            seam_mappings,
            verts,
            scaled_body_verts,
        )
        debug_paths["seam_association"] = png_path

        # 7. ARAP Solve
        solver = ARAPSolver(verts, faces, stiffness=5.0)

        anchor_map: dict[int, np.ndarray] = {}
        for mapping in seam_mappings.values():
            indices = mapping.garment_indices
            positions = mapping.body_positions
            targets_2d = positions[:, :2]

            for idx, tgt in zip(indices, targets_2d, strict=False):
                if idx not in anchor_map:
                    anchor_map[idx] = tgt
                else:
                    anchor_map[idx] = (anchor_map[idx] + tgt) * 0.5

        anchor_indices = np.array(list(anchor_map.keys()), dtype=np.int32)
        anchor_targets = np.array(list(anchor_map.values()), dtype=np.float32)

        logger.info(f"Solving ARAP with {len(anchor_indices)} anchors...")
        relaxed_2d = solver.solve(
            initial_guess=verts[:, :2],
            anchor_targets=anchor_targets,
            anchor_indices=anchor_indices,
            anchor_weights=np.full(len(anchor_indices), 10.0, dtype=np.float32),
            iterations=20,
        )

        # Reconstruct 3D
        relaxed_verts_3d = np.zeros_like(verts)
        relaxed_verts_3d[:, :2] = relaxed_2d[0]

        # 8. Z-Projection
        logger.info("Projecting to Z...")
        wp_body_pts = wp.array(scaled_body_verts, dtype=wp.vec3, device=solver.device)
        wp_body_idx = wp.array(body_faces.flatten(), dtype=int, device=solver.device)

        body_mesh = wp.Mesh(points=wp_body_pts, indices=wp_body_idx)

        panel_types = (garment_vertex_labels + 1).astype(np.int32)

        final_verts = solver.project_z(
            garment_vertices=relaxed_verts_3d[None, ...],
            body_mesh_ids=[body_mesh.id],
            panel_types=panel_types,
            body_face_labels_list=[body_face_labels],
            z_offset=Z_OFFSET_ARAP,
        )[0]

        update_boxmesh_vertices(garment_box_mesh, final_verts)

        obj_path = save_colored_obj(
            debug_output_dir,
            f"debug_arap_final_{garment_box_mesh.name}.obj",
            final_verts,
            faces,
            garment_face_labels,
        )
        debug_paths["arap_debug_mesh"] = obj_path
        logger.info("ARAP Complete.")

    except Exception as e:
        logger.error(f"ARAP Initialization failed: {e}")
        logger.exception("Traceback:")

    # Process attachment constraints
    vertex_processor = piece.get_vertex_processor_callback()
    if vertex_processor is not None:
        garment_box_mesh.process_attachment_constraints(vertex_processor)
        logger.debug(
            f"Processed attachment constraints for {garment_box_mesh.name} "
            f"({len(garment_box_mesh.attachment_constraints)} constraints)"
        )

    render_config = sim_config.render_config

    body_vertices = try_on_person.intermediate_meshes[0].vertices.astype(np.float64)
    body_faces_arr = try_on_person.intermediate_meshes[0].faces.astype(np.int32)
    body_indices = body_faces_arr.flatten()

    body_segmentation = try_on_person.segmentation_mapping

    vertex_labels = (
        garment_box_mesh.vertex_labels if garment_box_mesh.vertex_labels else None
    )

    garment_data = GarmentData(
        box_mesh=garment_box_mesh,
        body_definition=body_def,
        body_vertices=body_vertices,
        body_indices=body_indices,
        body_faces=body_faces_arr,
        body_segmentation=body_segmentation,
        pattern_file=pattern,
        vertex_labels=vertex_labels,
    )

    if garment_box_mesh.attachment_constraints:
        constraints_dicts = [
            c.model_dump() for c in garment_box_mesh.attachment_constraints
        ]

        if "options" not in config._props:
            config._props["options"] = {}
        config._props["options"]["attachment_constraints"] = constraints_dicts
        config._props["options"]["enable_attachment_constraint"] = True

    sim_dict = {
        "sim": {"config": config._props, "stats": {}},
        "render": {"config": render_config.model_dump(), "stats": {}},
    }

    logger.debug(
        f"Running simulation for {garment_box_mesh.name} with in-memory data..."
    )

    draped_mesh, images = run_sim(
        garment_box_mesh.name,
        sim_dict,
        try_on_person=try_on_person,
        garment_data=garment_data,
    )

    return draped_mesh, images, debug_paths


def _extract_vertex_panels(box_mesh: BoxMesh, num_verts: int) -> list[set[str]]:
    """Extract vertex-to-panel mapping from box mesh."""
    vertex_panels_sets: list[set[str]] = []

    stitch_seg = box_mesh.stitch_segmentation
    if stitch_seg is not None:
        for entry_list in stitch_seg:
            panels: set[str] = set()
            for entry in entry_list:
                panels.update(entry.get_panels(box_mesh.stitches))
            vertex_panels_sets.append(panels)

    # Pad to match vertex count
    while len(vertex_panels_sets) < num_verts:
        vertex_panels_sets.append(set())

    return vertex_panels_sets


def _build_garment_seams_dict(
    garment_seams_path: dict,
    verts: np.ndarray,
    vertex_panels: list[set[str]] | None = None,
) -> dict[tuple, tuple[list[int], np.ndarray, list[set[str]] | None]]:
    """Build garment seams dict with (indices, coordinates, panel_ids) tuples."""
    garment_seams_dict: dict[
        tuple, tuple[list[int], np.ndarray, list[set[str]] | None]
    ] = {}

    for (r1, r2, _), path_data in garment_seams_path.items():
        key_fwd = (RingType(r1.value), RingType(r2.value))
        key_rev = (RingType(r2.value), RingType(r1.value))

        # Handle potential (indices, panel_ids) tuple
        path_panels: list[set[str]] | None = None
        if isinstance(path_data, tuple):
            path, path_panels = path_data
        else:
            path = path_data

        pts = verts[path]
        path_list = list(path)

        # Extract panels if available
        panels: list[set[str]] | None = None
        if path_panels:
            panels = path_panels
        elif vertex_panels:
            panels = [vertex_panels[i] for i in path_list]

        garment_seams_dict[key_fwd] = (path_list, pts, panels)

        # Reversed
        if panels:
            panels_rev = panels[::-1]
        else:
            panels_rev = None

        garment_seams_dict[key_rev] = (
            path_list[::-1],
            pts[::-1],
            panels_rev,
        )

    return garment_seams_dict


def _build_body_seams_dict(
    body_seams: dict,
    scaled_body_verts: np.ndarray,
) -> dict[tuple, np.ndarray]:
    """Build body seams dict with coordinates."""
    body_seams_dict: dict[tuple, np.ndarray] = {}

    for (r1, r2, _), path in body_seams.items():
        key_fwd = (RingType(r1.value), RingType(r2.value))
        key_rev = (RingType(r2.value), RingType(r1.value))
        pts = scaled_body_verts[path]
        body_seams_dict[key_fwd] = pts
        body_seams_dict[key_rev] = pts[::-1]

    return body_seams_dict


def _load_boxmesh_timeout(garment: BoxMesh, timeout_after: int) -> None:
    """Load boxmesh with timeout protection."""
    if platform.system() == "Windows":

        def load_garment() -> None:
            garment.load()

        p = multiprocessing.Process(target=load_garment, name="GarmentGeneration")
        p.start()
        time.sleep(timeout_after)

        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError(f"Garment loading exceeded timeout of {timeout_after}s")

    elif platform.system() in ["Linux", "OSX"]:

        def alarm_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Garment loading exceeded timeout of {timeout_after}s")

        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout_after)
        try:
            garment.load()
        except TimeoutError:
            raise
        finally:
            signal.alarm(0)
