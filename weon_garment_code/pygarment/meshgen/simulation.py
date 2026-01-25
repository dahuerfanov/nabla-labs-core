import time
import traceback
from typing import Any

import numpy as np
import trimesh
import warp as wp
from loguru import logger
from tqdm import tqdm  # type: ignore[import-untyped]

from weon_garment_code.config import PathConfig
from weon_garment_code.config.dataset_properties import DatasetProperties
from weon_garment_code.config.sim_config import SimConfig
from weon_garment_code.pygarment.meshgen.garment import SimulationGarment
from weon_garment_code.pygarment.meshgen.render.pythonrender import render_images
from weon_garment_code.pygarment.meshgen.sim_data import GarmentData, GCTryOnPerson

wp.init()


class SimulationError(BaseException):
    """Exception raised when panel stitching cannot be executed correctly."""

    pass


def optimize_garment_storage(paths: PathConfig) -> None:
    """Prepare the data element for compact storage.

    Converts OBJ meshes to PLY format and removes texture files to reduce storage size.

    Parameters
    ----------
    paths : PathConfig
        Path configuration object containing paths to garment mesh files.

    Note
    ----
    This function silently handles any errors during conversion, continuing with
    the next file if one fails.
    """
    # Objs to ply
    try:
        boxmesh = trimesh.load(paths.g_box_mesh)
        boxmesh.export(paths.g_box_mesh_compressed)
        paths.g_box_mesh.unlink()
        logger.info(
            f"Converted {paths.g_box_mesh} to {paths.g_box_mesh_compressed} and removed original OBJ."
        )
    except BaseException as e:
        logger.warning(f"Failed to convert {paths.g_box_mesh} to PLY: {e}")

    try:
        simmesh = trimesh.load(paths.g_sim)
        simmesh.export(paths.g_sim_compressed)
        paths.g_sim.unlink()
        logger.info(
            f"Converted {paths.g_sim} to {paths.g_sim_compressed} and removed original OBJ."
        )
    except BaseException as e:
        logger.warning(f"Failed to convert {paths.g_sim} to PLY: {e}")

    # Remove large texture file and mtl -- not so necessary
    try:
        paths.g_texture_fabric.unlink(missing_ok=True)
        logger.info(f"Removed texture file: {paths.g_texture_fabric}")
    except Exception as e:
        logger.warning(f"Failed to remove texture file {paths.g_texture_fabric}: {e}")

    try:
        paths.g_mtl.unlink(missing_ok=True)
        logger.info(f"Removed material file: {paths.g_mtl}")
    except Exception as e:
        logger.warning(f"Failed to remove material file {paths.g_mtl}: {e}")


def sim_frame_sequence(
    garment: SimulationGarment, config: SimConfig, verbose: bool = False
) -> None:
    """Run the simulation frame sequence.

    Executes the simulation for the specified number of frames, checking for
    static equilibrium after the minimum number of steps and zero gravity steps.

    Parameters
    ----------
    garment : SimulationGarment
        The garment simulation object to run frames on.
    config : SimConfig
        Simulation configuration containing frame limits and step counts.
    store_usd : bool, optional
        Whether to store USD frames. If True, saves the initial state as USD.
        Default is False.
    verbose : bool, optional
        Whether to print verbose output including frame-by-frame information
        and self-intersection counts. Default is False.

    Note
    ----
    The simulation will stop early if static equilibrium is achieved after
    both `zero_gravity_steps` and `min_sim_steps` have been completed.
    """
    # Use tqdm for progress bar if not verbose
    frame_iter = (
        range(0, config.max_sim_steps)
        if verbose
        else tqdm(
            range(0, config.max_sim_steps), desc="Simulation frames", unit="frame"
        )
    )
    # non_static_len_list = []
    for frame in frame_iter:
        if verbose:
            logger.info(f"------ Frame {frame + 1} ------")

        garment.frame = frame
        garment.run_frame()

        if verbose:
            num_cloth_cloth_contacts = garment.count_self_intersections()
            logger.debug(f"Self-Intersection: {num_cloth_cloth_contacts}")

        if (
            frame >= config.control.zero_gravity_steps
            and frame >= config.control.min_sim_steps
        ):
            static, non_static_len = garment.is_static()
            # non_static_len_list.append(non_static_len)
            if static:
                logger.info(
                    f"Simulation stopped early at frame {frame + 1} due to static equilibrium."
                )
                break


def run_sim(
    cloth_name: str,
    props: dict[str, Any],
    try_on_person: GCTryOnPerson,
    garment_data: GarmentData | None = None,
) -> tuple[trimesh.Trimesh, dict[str, np.ndarray]]:
    """Initialize and run the simulation.

    This function sets up the simulation environment, runs the frame sequence,
    performs quality checks, saves the results, and optionally renders images.

    Parameters
    ----------
    cloth_name : str
        Name of the cloth/garment being simulated.
    props : Any
        Properties object (dict-like) containing simulation and render properties.
        Will be converted to DatasetProperties internally. Must have keys:
        - 'sim': Dictionary with simulation configuration and stats
        - 'render': Dictionary with rendering configuration and stats
    paths : PathConfig
        Path configuration object containing all necessary file paths.
    save_v_norms : bool, optional
        Whether to save vertex normals in the output mesh. Default is False.
    store_usd : bool, optional
        Whether to store USD frames. Slows down simulation significantly
        due to CPU-GPU copies and file writes. Use only for debugging.
        Default is False.
    optimize_storage : bool, optional
        Whether to optimize storage by converting OBJ files to PLY format
        and removing texture files. Default is False.
    verbose : bool, optional
        Whether to print verbose output during simulation. Default is False.
    garment_data : GarmentData, optional
        Optional in-memory garment data. If provided, the simulation will
        use this data directly instead of reading from disk. This improves
        performance by avoiding redundant I/O operations. Default is None
        (read from disk).

    Raises
    ------
    SimulationError
        If the simulation fails due to a known error condition.
    BaseException
        If the simulation crashes due to an unexpected error.

    Note
    ----
    After simulation, quality checks are performed and logged as warnings
    if issues are detected. The simulation continues regardless of quality
    check results.
    """
    start_time = time.time()

    # Use DatasetProperties class instead of dictionary access
    dataset_props = DatasetProperties.from_props(props)
    config = dataset_props.sim.get_sim_config()
    garment = SimulationGarment(cloth_name, config, garment_data=garment_data)

    try:
        logger.info("Simulation started...")
        sim_frame_sequence(garment, config, verbose=False)
    except SimulationError as e:
        logger.error(f"Simulation failed for {cloth_name} with error: {e}")
        logger.error(traceback.format_exc())
        raise
    except BaseException as e:
        logger.error(f"Simulation crashed for {cloth_name} with error: {e}")
        logger.error(traceback.format_exc())
        raise

    # --- Retarget to target posed mesh with smooth interpolation ---
    logger.info("--- Starting Retargeting Sequence ---")

    intermediate_obj_list = try_on_person.intermediate_meshes

    config.control.zero_gravity_steps = 0  # Gravity should already be on

    transition_frames = (
        config.control.transition_frame_steps
    )  # Number of frames to interpolate between each pose

    # Get the starting pose from the garment object
    previous_verts = garment.v_body.copy()

    for idx, intermediate_obj in enumerate(intermediate_obj_list):
        if idx == 0:
            continue

        # Load the next target pose
        target_verts = intermediate_obj.vertices.copy().astype(np.float64)

        # Apply the same transformations as the initial body
        target_verts = target_verts * garment.b_scale
        if garment.shift_y:
            target_verts[:, 1] = target_verts[:, 1] + garment.shift_y

        # Inner loop for smooth interpolation
        for i in range(transition_frames):
            # Calculate interpolation factor (alpha) from 0.0 to 1.0
            alpha = (i + 1) / float(transition_frames)

            # Linearly interpolate between previous and target vertices
            interpolated_verts = (1.0 - alpha) * previous_verts + alpha * target_verts

            # Update the body pose with the smoothly changing mesh
            garment.update_body_mesh(interpolated_verts)

            # Run just ONE frame of simulation
            # We use garment.run_frame() directly for single-step control
            garment.frame += 1  # Manually increment frame counter
            garment.run_frame()

        # The transition is done, the new 'previous' is the pose we just reached
        previous_verts = target_verts.copy()

    logger.info("--- Retargeting Sequence Finished ---")

    # Quality checks (log warnings and continue)
    frame = garment.frame
    logger.info(f"Simulation completed with {frame + 1} frames")

    if garment.frame == config.max_sim_steps - 1:
        _, non_st_count = garment.is_static()
        logger.warning(
            f"Failed to achieve static equilibrium for {cloth_name}. {non_st_count} non-static vertices out of {len(garment.current_verts)}"
        )

    if time.time() - start_time < 0.5:  # 0.5 sec -- finished suspiciously fast
        logger.warning(
            f"Simulation finished suspiciously fast for {cloth_name}. Time taken: < 0.5s"
        )

    # 3D penetrations
    num_body_collisions = garment.count_body_intersections()
    logger.info(f"Body-Cloth intersections: {num_body_collisions}.")
    num_self_collisions = garment.count_self_intersections()

    if num_body_collisions > config.max_body_collisions:
        logger.warning(
            f"Excessive body-cloth intersections for {cloth_name}. {num_body_collisions} > {config.max_body_collisions}"
        )

    if num_self_collisions:
        logger.info(f"Self-intersecting with {num_self_collisions}.")
        if num_self_collisions > config.max_self_collisions:
            logger.warning(
                f"Excessive self-intersections for {cloth_name}. {num_self_collisions} > {config.max_self_collisions}"
            )
    else:
        logger.info("Mesh is intersection-free.")

    draped_garment_mesh = trimesh.Trimesh(
        vertices=garment.current_verts / 100, faces=garment.f_cloth, process=False
    )

    # Render images
    images_dict = render_images(
        draped_garment_mesh,
        garment.v_body,
        garment.f_body,
        dataset_props.render.config,
    )

    # Final info output
    sec = round(time.time() - start_time, 3)
    min_ = int(sec / 60)
    logger.info(
        f"Simulation pipeline completed. Time taken: {min_} m {sec - min_ * 60} s."
    )

    return draped_garment_mesh, images_dict
