import warp as wp


@wp.func
def polar_decomp_r(A: wp.mat22) -> wp.mat22:
    """Compute Rotation matrix R from Polar Decomposition A = RS."""
    # Standard formula: R = A * (A^T A)^{-1/2}

    t = A[0, 0] + A[1, 1]
    u = A[1, 0] - A[0, 1]

    inv_norm = 1.0 / wp.sqrt(t * t + u * u + 1e-9)
    c = t * inv_norm
    s = u * inv_norm

    return wp.mat22(c, -s, s, c)


@wp.kernel
def compute_edge_covariance(
    vertices: wp.array(dtype=wp.vec2),
    rest_pose_edges: wp.array(dtype=wp.vec2),  # (E, 2) edge vectors in rest pose
    edge_indices: wp.array(dtype=wp.int32, ndim=2),  # (E, 2) vertex indices per edge
    edge_weights: wp.array(dtype=float),  # (E,) Cotan weights
    neighbors_start: wp.array(dtype=int),  # (N,) Start index in adjacency
    neighbors_count: wp.array(dtype=int),  # (N,) Count of neighbors
    neighbor_edge_indices: wp.array(
        dtype=int
    ),  # (M,) Mapping from flat neighbor list to edge index
    # Output
    covariance_matrices: wp.array(dtype=wp.mat22),  # (N,) Covariance matrix per vertex
):
    """Compute covariance matrix S_i = sum_j w_ij * (pi - pj) * (pi' - pj')^T"""
    i = wp.tid()

    start = neighbors_start[i]
    count = neighbors_count[i]

    Si = wp.mat22(0.0, 0.0, 0.0, 0.0)

    p_i = vertices[i]

    for k in range(count):
        edge_idx = neighbor_edge_indices[start + k]

        # Get edge data (ndim=2 access)
        v0 = edge_indices[edge_idx, 0]
        v1 = edge_indices[edge_idx, 1]

        w = edge_weights[edge_idx]
        rest_vec = rest_pose_edges[edge_idx]

        sign = 1.0
        j = v1
        if i == v1:
            sign = -1.0
            j = v0

        p_j = vertices[j]
        e_current = p_i - p_j

        e_rest = rest_vec * sign

        term = wp.mat22(
            e_current[0] * e_rest[0],
            e_current[0] * e_rest[1],
            e_current[1] * e_rest[0],
            e_current[1] * e_rest[1],
        )

        Si += term * w

    covariance_matrices[i] = Si


@wp.kernel
def fit_rotations(
    covariance_matrices: wp.array(dtype=wp.mat22), rotations: wp.array(dtype=wp.mat22)
):
    i = wp.tid()
    Si = covariance_matrices[i]

    Ri = polar_decomp_r(Si)
    rotations[i] = Ri


@wp.kernel
def compute_rhs_batched(
    vertices: wp.array(dtype=wp.vec2),
    rotations: wp.array(dtype=wp.mat22),
    rest_pose_edges: wp.array(dtype=wp.vec2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_weights: wp.array(dtype=float),
    neighbors_start: wp.array(dtype=int),
    neighbors_count: wp.array(dtype=int),
    neighbor_edge_indices: wp.array(dtype=int),
    # Output
    rhs: wp.array(dtype=wp.vec2),
):
    i = wp.tid()

    start = neighbors_start[i]
    count = neighbors_count[i]

    b_i = wp.vec2(0.0, 0.0)

    Ri = rotations[i]

    for k in range(count):
        edge_idx = neighbor_edge_indices[start + k]

        v0 = edge_indices[edge_idx, 0]
        v1 = edge_indices[edge_idx, 1]

        w = edge_weights[edge_idx]
        rest_vec = rest_pose_edges[edge_idx]

        sign = 1.0
        j = v1
        if i == v1:
            sign = -1.0
            j = v0

        Rj = rotations[j]

        R_sum = Ri + Rj
        e_rest = rest_vec * sign

        rotated_rest = R_sum * e_rest

        b_i += rotated_rest * (0.5 * w)

    rhs[i] = b_i


# =========================================================================
# Z-Projection Kernels
# =========================================================================


@wp.kernel(enable_backward=False)
def project_z_batched_kernel(
    garment_vertices: wp.array(dtype=wp.vec3),  # (N_total, 3) XYZ
    batch_indices: wp.array(dtype=int),  # (N_total,) Which batch this vertex belongs to
    body_mesh_ids: wp.array(dtype=wp.uint64),  # (BatchSize,) mesh handles
    panel_types: wp.array(dtype=int),  # (N_total,) 1=Front, 2=Back, 0=None
    axis0: int,
    axis1: int,
    z_offset: float,
    body_face_labels: wp.array(dtype=int),  # (F_total,) 0=Front, 1=Back, -1=Unknown
    body_face_offsets: wp.array(dtype=int),  # (B,) Start index of faces for each batch
    # Outputs
    projected_z: wp.array(dtype=float),  # (N_total,) Result Z
    valid_mask: wp.array(dtype=int),  # (N_total,) 1 if hit found, 0 otherwise
):
    i = wp.tid()

    batch_idx = batch_indices[i]
    mesh_id = body_mesh_ids[batch_idx]

    p_orig = garment_vertices[i]
    panel_type = panel_types[i]

    z_axis = 3 - axis0 - axis1

    # Define ray direction
    ray_dir = wp.vec3(0.0, 0.0, 0.0)
    if z_axis == 0:
        ray_dir[0] = 1.0
    elif z_axis == 1:
        ray_dir[1] = 1.0
    else:
        ray_dir[2] = 1.0

    # Extract XY
    val_a0 = float(0.0)
    val_a1 = float(0.0)

    if axis0 == 0:
        val_a0 = p_orig[0]
    elif axis0 == 1:
        val_a0 = p_orig[1]
    else:
        val_a0 = p_orig[2]

    if axis1 == 0:
        val_a1 = p_orig[0]
    elif axis1 == 1:
        val_a1 = p_orig[1]
    else:
        val_a1 = p_orig[2]

    start_point = wp.vec3(0.0, 0.0, 0.0)
    if axis0 == 0:
        start_point[0] = val_a0
    elif axis0 == 1:
        start_point[1] = val_a0
    else:
        start_point[2] = val_a0

    if axis1 == 0:
        start_point[0] = val_a1
    elif axis1 == 1:
        start_point[1] = val_a1
    else:
        start_point[2] = val_a1

    if z_axis == 0:
        start_point[0] = -100.0
    elif z_axis == 1:
        start_point[1] = -100.0
    else:
        start_point[2] = -100.0

    min_z_hit = float(10000.0)
    max_z_hit = float(-10000.0)
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3(0.0, 0.0, 0.0)
    face_index = wp.int32(0)

    offset = float(0.01)
    t_start = float(0.0)
    hit_count = wp.int32(0)

    # Hit loop (max 4 hits)
    for _ in range(4):
        curr_start = start_point + ray_dir * t_start
        remaining = 200.0 - t_start
        if remaining <= 0.0:
            break

        if wp.mesh_query_ray(
            mesh_id, curr_start, ray_dir, remaining, t, u, v, sign, n, face_index
        ):
            total_t = t_start + t

            # 1. Normal Check
            valid_hit = True
            nz = float(0.0)
            if z_axis == 0:
                nz = n[0]
            elif z_axis == 1:
                nz = n[1]
            else:
                nz = n[2]

            if (panel_type == 1 and nz < -0.2) or (panel_type == 2 and nz > 0.2):
                valid_hit = False

            # 2. Partition Check
            if valid_hit:
                global_f_idx = body_face_offsets[batch_idx] + face_index
                b_label = body_face_labels[global_f_idx]

                # Panel Front(1) -> Needs Body Front(0)
                if panel_type == 1 and b_label != 0:
                    valid_hit = False
                # Panel Back(2) -> Needs Body Back(1)
                elif panel_type == 2 and b_label != 1:
                    valid_hit = False

            if valid_hit:
                z_hit = -100.0 + total_t
                min_z_hit = min(min_z_hit, z_hit)
                max_z_hit = max(max_z_hit, z_hit)
                hit_count += 1

            t_start = total_t + offset
        else:
            break

    if hit_count > 0:
        if panel_type == 1:  # Front uses MAX Z (outermost)
            projected_z[i] = max_z_hit + z_offset
        elif panel_type == 2:  # Back uses MIN Z (outermost)
            projected_z[i] = min_z_hit - z_offset
        else:
            projected_z[i] = (min_z_hit + max_z_hit) * 0.5
        valid_mask[i] = 1
    else:
        # Fallback: Closest Point
        # Check CP partition + normal
        if wp.mesh_query_point(mesh_id, p_orig, 200.0, sign, face_index, u, v):
            cp = wp.mesh_eval_position(mesh_id, face_index, u, v)

            normal = wp.mesh_eval_face_normal(mesh_id, face_index)
            nz = float(0.0)
            if z_axis == 0:
                nz = normal[0]
            elif z_axis == 1:
                nz = normal[1]
            else:
                nz = normal[2]

            consistent = True
            if (panel_type == 1 and nz < -0.2) or (panel_type == 2 and nz > 0.2):
                consistent = False

            if consistent:
                global_f_idx = body_face_offsets[batch_idx] + face_index
                b_label = body_face_labels[global_f_idx]
                if panel_type == 1 and b_label != 0:
                    consistent = False
                elif panel_type == 2 and b_label != 1:
                    consistent = False

            if consistent:
                z_val = float(0.0)
                if z_axis == 0:
                    z_val = cp[0]
                elif z_axis == 1:
                    z_val = cp[1]
                else:
                    z_val = cp[2]

                if panel_type == 1:
                    projected_z[i] = z_val + z_offset
                elif panel_type == 2:
                    projected_z[i] = z_val - z_offset
                else:
                    projected_z[i] = z_val
                valid_mask[i] = 1
            else:
                valid_mask[i] = 0
        else:
            valid_mask[i] = 0
