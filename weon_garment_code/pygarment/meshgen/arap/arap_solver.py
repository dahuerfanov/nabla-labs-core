from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warp as wp

from .ops import (
    compute_edge_covariance,
    compute_rhs_batched,
    fit_rotations,
    project_z_batched_kernel,
)


class ARAPSolver:
    """ARAP solver using Scipy (Global) and Warp (Local).

    Optimizes 2D layout to match seam constraints, then projects to 3D Z.
    """

    def __init__(
        self,
        vertices: np.ndarray,  # (V, 2) or (V, 3) rest pose
        faces: np.ndarray,  # (F, 3)
        stiffness: float = 1.0,
        device: str = "cuda",
    ):
        self.device = device
        self.num_vertices = len(vertices)
        self.stiffness = stiffness

        # 1. Build Topology
        self._build_topology(vertices, faces)

        # 2. Build Laplacian
        self._build_laplacian()

        self.factorization: spla.LinearOperator
        self.current_anchor_indices: np.ndarray | None = None
        self.current_batch_size: int | None = None

        # Warp data structures for Local Step
        self._init_warp_structures(vertices)

    def _build_topology(self, vertices: np.ndarray, faces: np.ndarray):
        """Precompute edges, cotan weights, and adjacency."""
        edges = set()
        for f in faces:
            for i in range(3):
                u, v = f[i], f[(i + 1) % 3]
                if u > v:
                    u, v = v, u
                edges.add((u, v))

        self.edges = np.array(list(edges), dtype=np.int32)
        if self.edges.size == 0:
            self.edges = np.zeros((0, 2), dtype=np.int32)
        self.num_edges = len(self.edges)

        # Adjacency
        self.adjacency_list: list[list[int]] = [[] for _ in range(self.num_vertices)]
        for u, v in self.edges:
            self.adjacency_list[u].append(v)
            self.adjacency_list[v].append(u)

        # Rest edge vectors (assume 2D optimization primarily)
        v_rest = vertices[:, :2]
        e_vecs = v_rest[self.edges[:, 0]] - v_rest[self.edges[:, 1]]
        self.rest_edge_vectors = e_vecs.astype(np.float32)

        # Edge weights (Uniform stiffness for now)
        self.edge_weights = np.full(self.num_edges, self.stiffness, dtype=np.float32)

    def _build_laplacian(self):
        """Construct the topological Laplacian matrix L."""
        rows = []
        cols = []
        data = []
        diag = np.zeros(self.num_vertices, dtype=np.float32)

        for idx, (u, v) in enumerate(self.edges):
            w = self.edge_weights[idx]
            # Off-diagonal
            rows.append(u)
            cols.append(v)
            data.append(-w)

            rows.append(v)
            cols.append(u)
            data.append(-w)

            # Diagonal accum
            diag[u] += w
            diag[v] += w

        # Add diagonal
        rows.extend(range(self.num_vertices))
        cols.extend(range(self.num_vertices))
        data.extend(diag)

        self.L = sp.csc_matrix(
            (data, (rows, cols)), shape=(self.num_vertices, self.num_vertices)
        )

    def _init_warp_structures(self, vertices: np.ndarray):
        """Initialize Warp arrays for Local Step."""
        self.wp_rest_edges = wp.array(
            self.rest_edge_vectors, dtype=wp.vec2, device=self.device
        )
        self.wp_edge_indices = wp.array(self.edges, dtype=wp.int32, device=self.device)
        self.wp_edge_weights = wp.array(
            self.edge_weights, dtype=float, device=self.device
        )

        adj: list[list[int]] = [[] for _ in range(self.num_vertices)]
        for e_idx, (u, v) in enumerate(self.edges):
            adj[u].append(e_idx)
            adj[v].append(e_idx)

        neighbors_start = []
        neighbors_count = []
        neighbor_edge_indices = []

        current_start = 0
        for i in range(self.num_vertices):
            neighbors_start.append(current_start)
            count = len(adj[i])
            neighbors_count.append(count)
            neighbor_edge_indices.extend(adj[i])
            current_start += count

        self.wp_neighbors_start = wp.array(
            np.array(neighbors_start, dtype=np.int32), dtype=int, device=self.device
        )
        self.wp_neighbors_count = wp.array(
            np.array(neighbors_count, dtype=np.int32), dtype=int, device=self.device
        )
        self.wp_neighbor_edge_indices = wp.array(
            np.array(neighbor_edge_indices, dtype=np.int32),
            dtype=int,
            device=self.device,
        )

    def update_system(
        self,
        anchor_indices: np.ndarray,
        anchor_weights: np.ndarray,
        batch_size: int = 1,
    ):
        """Update system matrix with new anchors and factorize."""
        is_same = False
        if (
            self.current_anchor_indices is not None
            and self.current_batch_size == batch_size
            and np.array_equal(self.current_anchor_indices, anchor_indices)
        ):
            is_same = True

        if is_same and self.factorization is not None:
            return

        self.current_anchor_indices = anchor_indices.copy()
        self.current_batch_size = batch_size

        if batch_size > 1:
            L_batched = sp.block_diag([self.L] * batch_size, format="csc")
        else:
            L_batched = self.L

        M = L_batched.copy()

        # Add anchor weights
        V = self.num_vertices
        if batch_size > 1:
            offsets = np.repeat(np.arange(batch_size) * V, len(anchor_indices))
            anchor_rows = np.tile(anchor_indices, batch_size) + offsets
            anchor_data = np.tile(anchor_weights, batch_size)
        else:
            anchor_rows = anchor_indices
            anchor_data = anchor_weights

        # Sparse matrix addition for diagonal weights
        W = sp.csc_matrix((anchor_data, (anchor_rows, anchor_rows)), shape=M.shape)
        M = M + W
        self.factorization = spla.factorized(M)

    def solve(
        self,
        initial_guess: np.ndarray,  # (B, V, 2)
        anchor_targets: np.ndarray,  # (B, NumAnchors, 2)
        anchor_indices: np.ndarray,  # (NumAnchors,)
        anchor_weights: np.ndarray,  # (NumAnchors,)
        iterations: int = 15,
    ) -> np.ndarray:
        """Run batched ARAP solve."""
        if initial_guess.ndim == 2:
            initial_guess = initial_guess[None, ...]
            anchor_targets = anchor_targets[None, ...]

        B, V, _ = initial_guess.shape

        self.update_system(anchor_indices, anchor_weights, batch_size=B)

        current_pos_host = initial_guess.reshape(-1, 2).astype(np.float32)
        wp_pos = wp.array(current_pos_host, dtype=wp.vec2, device=self.device)

        # Temp arrays
        wp_cov = wp.zeros(B * V, dtype=wp.mat22, device=self.device)
        wp_rot = wp.zeros(B * V, dtype=wp.mat22, device=self.device)
        wp_rhs = wp.zeros(B * V, dtype=wp.vec2, device=self.device)

        # Batched topology preparation
        # (Handling B > 1 requires tiling pointers)

        # Helper to tile generic arrays
        def tile_arr(arr, n, dtype):
            return wp.array(np.tile(arr, n), dtype=dtype, device=self.device)

        wp_neighbors_start_batched = self.wp_neighbors_start
        wp_neighbors_count_batched = self.wp_neighbors_count
        wp_neighbor_edge_indices_batched = self.wp_neighbor_edge_indices
        wp_edge_indices_batched = self.wp_edge_indices
        wp_edge_weights_batched = self.wp_edge_weights
        wp_rest_edges_batched = self.wp_rest_edges

        if B > 1:
            # Complex tiling logic for B > 1 (indices need offsets)
            neighbors_start_host = self.wp_neighbors_start.numpy()
            total_neighbor_entries = len(self.wp_neighbor_edge_indices)
            start_offsets = np.repeat(np.arange(B) * total_neighbor_entries, V).astype(
                int
            )
            wp_neighbors_start_batched = wp.array(
                np.tile(neighbors_start_host, B) + start_offsets,
                dtype=int,
                device=self.device,
            )
            wp_neighbors_count_batched = tile_arr(
                self.wp_neighbors_count.numpy(), B, int
            )

            edges_host = self.edges
            batched_edges_list = [edges_host + b * V for b in range(B)]
            wp_edge_indices_batched = wp.array(
                np.vstack(batched_edges_list), dtype=wp.int32, device=self.device
            )

            wp_edge_weights_batched = tile_arr(self.edge_weights, B, float)
            wp_rest_edges_batched = tile_arr(self.rest_edge_vectors, B, wp.vec2)

            E = self.num_edges
            neighbor_indices_host = self.wp_neighbor_edge_indices.numpy()
            offsets = np.repeat(np.arange(B) * E, len(neighbor_indices_host))
            wp_neighbor_edge_indices_batched = wp.array(
                (np.tile(neighbor_indices_host, B) + offsets).astype(np.int32),
                dtype=wp.int32,
                device=self.device,
            )

        # Prepare RHS Host Anchors
        rhs_anchor_term_flat = np.zeros((B * V, 2), dtype=np.float32)

        offsets = np.repeat(np.arange(B) * V, len(anchor_indices))
        batched_anchor_idx = np.tile(anchor_indices, B) + offsets

        batched_weights = np.tile(anchor_weights, B)[:, None]  # (B*A, 1)
        targets_flat = anchor_targets.reshape(-1, 2)

        rhs_anchor_term_flat[batched_anchor_idx] = batched_weights * targets_flat

        for _ in range(iterations):
            wp.launch(
                kernel=compute_edge_covariance,
                dim=B * V,
                inputs=[
                    wp_pos,
                    wp_rest_edges_batched,
                    wp_edge_indices_batched,
                    wp_edge_weights_batched,
                    wp_neighbors_start_batched,
                    wp_neighbors_count_batched,
                    wp_neighbor_edge_indices_batched,
                    wp_cov,
                ],
                device=self.device,
            )

            wp.launch(
                kernel=fit_rotations,
                dim=B * V,
                inputs=[wp_cov, wp_rot],
                device=self.device,
            )

            wp.launch(
                kernel=compute_rhs_batched,
                dim=B * V,
                inputs=[
                    wp_pos,
                    wp_rot,
                    wp_rest_edges_batched,
                    wp_edge_indices_batched,
                    wp_edge_weights_batched,
                    wp_neighbors_start_batched,
                    wp_neighbors_count_batched,
                    wp_neighbor_edge_indices_batched,
                    wp_rhs,
                ],
                device=self.device,
            )

            rhs_flat = wp_rhs.numpy().reshape(B * V, 2)
            rhs_flat += rhs_anchor_term_flat

            sol_x = self.factorization(rhs_flat[:, 0])
            sol_y = self.factorization(rhs_flat[:, 1])
            new_pos = np.stack([sol_x, sol_y], axis=1)

            wp_pos = wp.array(new_pos, dtype=wp.vec2, device=self.device)

        return wp_pos.numpy().reshape(B, V, 2)

    def project_z(
        self,
        garment_vertices: np.ndarray,  # (B, V, 3)
        body_mesh_ids: list[int],
        panel_types: np.ndarray,  # (V,) or (B, V)  1=Front, 2=Back
        body_face_labels_list: list[
            np.ndarray
        ],  # List of (F, ) int arrays, 0=Front, 1=Back
        z_offset: float = 0.5,
        xy_axes: tuple[int, int] = (0, 1),
    ) -> np.ndarray:
        """Project vertices to Z based on panel type and body partition."""
        B, V, _ = garment_vertices.shape
        wp_garment = wp.array(
            garment_vertices.reshape(-1, 3), dtype=wp.vec3, device=self.device
        )

        batch_indices = np.repeat(np.arange(B), V).astype(np.int32)
        wp_batch = wp.array(batch_indices, dtype=int, device=self.device)

        wp_bodies = wp.array(
            np.array(body_mesh_ids, dtype=np.uint64),
            dtype=wp.uint64,
            device=self.device,
        )

        # Tile panel types
        if panel_types.ndim == 1:
            batched_panels = np.tile(panel_types, B)
        else:
            batched_panels = panel_types.flatten()
        wp_panels = wp.array(batched_panels, dtype=int, device=self.device)

        # Prepare Body Face Labels
        concatenated_labels = np.concatenate(body_face_labels_list).astype(np.int32)
        wp_body_face_labels = wp.array(
            concatenated_labels, dtype=int, device=self.device
        )

        offsets = []
        curr = 0
        for labels in body_face_labels_list:
            offsets.append(curr)
            curr += len(labels)
        wp_offsets = wp.array(
            np.array(offsets, dtype=np.int32), dtype=int, device=self.device
        )

        wp_z = wp.zeros(B * V, dtype=float, device=self.device)
        wp_valid = wp.zeros(B * V, dtype=int, device=self.device)

        wp.launch(
            kernel=project_z_batched_kernel,
            dim=B * V,
            inputs=[
                wp_garment,
                wp_batch,
                wp_bodies,
                wp_panels,
                xy_axes[0],
                xy_axes[1],
                z_offset,
                wp_body_face_labels,
                wp_offsets,
                wp_z,
                wp_valid,
            ],
            device=self.device,
        )

        z_vals = wp_z.numpy().reshape(B, V)
        valid_vals = wp_valid.numpy().reshape(B, V)

        # Post-processing
        for b in range(B):
            self._fill_invalid_z(z_vals[b], valid_vals[b])
            self._smooth_z(z_vals[b], iterations=3)

        result = garment_vertices.copy()
        z_axis = 3 - xy_axes[0] - xy_axes[1]
        result[:, :, z_axis] = z_vals
        return result

    def _fill_invalid_z(self, z_vals, valid_mask):
        mask = valid_mask.astype(bool)
        # Iteratively fill holes
        for _ in range(20):
            if mask.all():
                break
            invalid = np.where(~mask)[0]
            filled = 0
            for i in invalid:
                nbs = self.adjacency_list[i]
                valid_nbs = [z_vals[n] for n in nbs if mask[n]]
                if valid_nbs:
                    z_vals[i] = sum(valid_nbs) / len(valid_nbs)
                    mask[i] = True
                    filled += 1
            if filled == 0:
                break

        if not mask.all():
            mean = np.mean(z_vals[mask]) if mask.any() else 0.0
            z_vals[~mask] = mean

    def _smooth_z(self, z_vals, iterations=2):
        for _ in range(iterations):
            new_z = z_vals.copy()
            for i in range(len(z_vals)):
                nbs = self.adjacency_list[i]
                if nbs:
                    avg = sum(z_vals[n] for n in nbs) / len(nbs)
                    new_z[i] = 0.5 * z_vals[i] + 0.5 * avg
            z_vals[:] = new_z[:]
