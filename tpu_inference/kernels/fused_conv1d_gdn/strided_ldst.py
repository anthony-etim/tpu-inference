# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_conv1d_gdn import configs


def load_as_qkv_large(
        qkv_vmem_ref: jax.Ref,
        cfgs: configs.GDNConfigs) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Use strided LDST to split qkv along last dim and perform transpose."""

    num_lanes = pltpu.get_tpu_info().num_lanes
    lanes_per_col = qkv_vmem_ref.shape[-1] // num_lanes
    kq_lanes_per_head = cfgs.kq_head_dim // num_lanes
    k_offset = cfgs.num_kq_heads * kq_lanes_per_head

    q_large_list = []
    k_large_list = []
    v_large_list = []

    for s_idx in range(cfgs.seq_tile_size):
        q_seq_list = []
        k_seq_list = []
        v_seq_list = []

        qkv_slot_flat_ref = qkv_vmem_ref.at[s_idx].reshape(-1, num_lanes)
        for kq_head in range(cfgs.num_kq_heads):
            q_head_list = []
            k_head_list = []
            for lane in range(kq_lanes_per_head):
                q_lane = kq_head * kq_lanes_per_head + lane
                k_lane = k_offset + q_lane

                q_head_list.append(qkv_slot_flat_ref[q_lane::lanes_per_col])
                k_head_list.append(qkv_slot_flat_ref[k_lane::lanes_per_col])
            q_seq_list.append(jnp.concat(q_head_list, axis=-1))
            k_seq_list.append(jnp.concat(k_head_list, axis=-1))
        v_offset = kq_lanes_per_head * cfgs.num_kq_heads * 2
        v_lanes_per_head = cfgs.v_head_dim // num_lanes
        for v_head in range(cfgs.num_v_heads):
            v_head_list = []
            for lane in range(v_lanes_per_head):
                v_lane = v_offset + v_head * v_lanes_per_head + lane
                v_head_list.append(qkv_slot_flat_ref[v_lane::lanes_per_col])
            v_seq_list.append(jnp.concat(v_head_list, axis=-1))

        q_large_list.append(jnp.stack(q_seq_list, axis=0))
        k_large_list.append(jnp.stack(k_seq_list, axis=0))
        v_large_list.append(jnp.stack(v_seq_list, axis=0))

    q_large = jnp.stack(q_large_list, axis=0)
    k_large = jnp.stack(k_large_list, axis=0)
    v_large = jnp.stack(v_large_list, axis=0)

    return q_large, k_large, v_large


def load_as_qkv_compact(
    qkv_vmem_ref: jax.Ref,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array]:

    # (seqs, chunk, 1, num_kq_heads * kq_head_dim * 2 + num_v_heads * v_head_dim)
    k_offset = cfgs.num_kq_heads * cfgs.kq_head_dim
    v_offset = cfgs.num_kq_heads * 2 * cfgs.kq_head_dim

    q_compact_list = []
    k_compact_list = []
    v_compact_list = []

    for kq_head in range(cfgs.num_kq_heads):
        q_start = kq_head * cfgs.kq_head_dim
        q_end = q_start + cfgs.kq_head_dim
        k_start = k_offset + q_start
        k_end = k_start + cfgs.kq_head_dim
        q_compact_list.append(qkv_vmem_ref[..., q_start:q_end])
        k_compact_list.append(qkv_vmem_ref[..., k_start:k_end])
    for v_head in range(cfgs.num_v_heads):
        v_start = v_offset + v_head * cfgs.v_head_dim
        v_end = v_start + cfgs.v_head_dim
        v_compact_list.append(qkv_vmem_ref[..., v_start:v_end])

    q_compact = jnp.stack(q_compact_list, axis=1)
    k_compact = jnp.stack(k_compact_list, axis=1)
    v_compact = jnp.stack(v_compact_list, axis=1)

    return q_compact, k_compact, v_compact


def load_compact_to_large(vmem_ref) -> jax.Array:
    # NOTE: Only support 32-bits for now.
    assert vmem_ref.dtype.itemsize == 4
    assert vmem_ref.shape[-2] == 1
    col_size = vmem_ref.shape[-1]
    new_shape = vmem_ref.shape[:-2] + (col_size, )
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes

    vreg_list = []
    vmem_ref = vmem_ref.reshape(-1, col_size)
    for col_start in range(0, col_size, num_lanes):
        col_end = min(col_start + num_lanes, col_size)
        vreg = vmem_ref[..., col_start:col_end]
        vreg_list.append(vreg)
    return jnp.concat(vreg_list, axis=-1).reshape(new_shape)
