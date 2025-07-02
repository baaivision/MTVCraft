# coding=utf-8
# rewritten, Copyright (c) 2021, Ming Ding.  All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from sat import mpu
from sat.mpu import get_model_parallel_world_size, ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, \
    gather_from_model_parallel_region, copy_to_model_parallel_region, checkpoint

from sat.mpu.utils import divide, sqrt, scaled_init_method, unscaled_init_method, gelu
from sat.ops.layernorm import LayerNorm

from .transformer_defaults import HOOKS_DEFAULT, standard_attention, split_tensor_along_last_dim

from einops import rearrange
from .resampler import Resampler
from icecream import ic


def zero_module(module):
    for p in module.parameters():
        torch.nn.init.zeros_(p)

    return module


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True,
                 qkv_bias=False, num_multi_query_heads=0, row_parallel_linear_final_bias=True,
                 hooks={}, transformer_pointer=None, params_dtype=torch.float, skip_init=False,
                 device=torch.device('cpu')):
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_multi_query_heads = num_multi_query_heads
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.num_multi_query_heads_per_partition = divide(num_multi_query_heads, world_size)
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition

        # Strided linear layer.
        if num_multi_query_heads == 0:
            qkv_size = 3 * self.inner_hidden_size
            self.stride = 3
        else:  # multi-query
            qkv_size = self.inner_hidden_size + self.hidden_size_per_attention_head * self.num_multi_query_heads * 2
            self.stride = [self.num_attention_heads_per_partition, self.num_multi_query_heads_per_partition,
                           self.num_multi_query_heads_per_partition]
        self.query_key_value = ColumnParallelLinear(
            hidden_size,  # 3072
            qkv_size,
            stride=self.stride,
            gather_output=False,
            init_method=init_method,
            bias=bias or qkv_bias,
            params_dtype=params_dtype,
            module=self,
            name="query_key_value",
            skip_init=skip_init,
            device=device
        )
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.dense = RowParallelLinear(
            self.inner_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense",
            skip_init=skip_init,
            device=device,
            final_bias=row_parallel_linear_final_bias
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (-1,  # flexible for multi-query
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask, *args, **kw_args):
        if 'attention_forward' in self.hooks:
            return self.hooks['attention_forward'](hidden_states, mask, **kw_args)
        else:
            return HOOKS_DEFAULT['attention_forward'](self, hidden_states, mask, **kw_args)

    def repartition(self):
        world_size = get_model_parallel_world_size()
        self.num_attention_heads_per_partition = divide(self.num_attention_heads, world_size)
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition


class CrossAttention(torch.nn.Module):
    """Parallel cross-attention layer for Transformer"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method,
                 layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True,
                 cross_num_multi_query_heads=0, row_parallel_linear_final_bias=True, hooks={},
                 cross_attn_hidden_size=None, transformer_pointer=None, params_dtype=torch.float, skip_init=False,
                 device=torch.device('cpu')):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        self.cross_num_multi_query_heads = cross_num_multi_query_heads
        # Strided linear layer.
        if cross_num_multi_query_heads == 0:
            kv_size = 2 * self.inner_hidden_size
        else:  # multi-query
            kv_size = self.hidden_size_per_attention_head * self.cross_num_multi_query_heads * 2

        self.query = ColumnParallelLinear(hidden_size,
                                          self.inner_hidden_size,
                                          gather_output=False,
                                          init_method=init_method,
                                          bias=bias,
                                          params_dtype=params_dtype,
                                          module=self,
                                          name="query",
                                          skip_init=skip_init,
                                          device=device)
        if cross_attn_hidden_size is None:
            cross_attn_hidden_size = hidden_size
        self.cross_attn_hidden_size = cross_attn_hidden_size
        self.key_value = ColumnParallelLinear(cross_attn_hidden_size,
                                              kv_size,
                                              stride=2,
                                              gather_output=False,
                                              init_method=init_method,
                                              bias=bias,
                                              params_dtype=params_dtype,
                                              module=self,
                                              name="key_value",
                                              skip_init=skip_init,
                                              device=device)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(
            self.inner_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense",
            skip_init=skip_init,
            device=device,
            final_bias=row_parallel_linear_final_bias)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        # self.dense = zero_module(self.dense)

        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (-1,  # flexible for multi-query
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
        # hidden_states: [b, s, h]
        if 'cross_attention_forward' in self.hooks:
            return self.hooks['cross_attention_forward'](hidden_states, cross_attention_mask, encoder_outputs,
                                                         **kw_args)
        else:
            return HOOKS_DEFAULT['cross_attention_forward'](self, hidden_states, cross_attention_mask, encoder_outputs,
                                                            **kw_args)

    def repartition(self):
        world_size = get_model_parallel_world_size()
        self.num_attention_heads_per_partition = divide(self.num_attention_heads, world_size)
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition


class MLP(torch.nn.Module):
    def __init__(self, hidden_size, output_dropout_prob, init_method, inner_hidden_size=None,
                 output_layer_init_method=None, layer_id=None, row_parallel_linear_final_bias=True, hooks={}, bias=True,
                 activation_func=gelu, transformer_pointer=None, is_gated_mlp=False, num_experts=1,
                 params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super(MLP, self).__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        # Project to 4h.
        self.hidden_size = hidden_size  # 3270
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(
            self.hidden_size,
            self.inner_hidden_size,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense_h_to_4h",
            skip_init=skip_init,
            device=device
        )
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            self.inner_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense_4h_to_h",
            skip_init=skip_init,
            device=device,
            final_bias=row_parallel_linear_final_bias
        )
        self.is_gated_mlp = is_gated_mlp
        if is_gated_mlp:
            self.dense_h_to_4h_gate = ColumnParallelLinear(
                self.hidden_size,
                self.inner_hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=False,
                params_dtype=params_dtype,
                module=self,
                name="dense_h_to_4h_gate",
                skip_init=skip_init,
                device=device
            )
        self.num_experts = num_experts
        for i in range(1, num_experts):
            self.register_module(f"dense_h_to_4h_{i}", ColumnParallelLinear(
                self.hidden_size,
                self.inner_hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=bias,
                params_dtype=params_dtype,
                module=self,
                name=f"dense_h_to_4h_{i}",
                skip_init=skip_init,
                device=device
            ))
            # Project back to h.
            self.register_module(f"dense_4h_to_h_{i}", RowParallelLinear(
                self.inner_hidden_size,
                self.hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                bias=bias,
                params_dtype=params_dtype,
                module=self,
                name=f"dense_4h_to_h_{i}",
                skip_init=skip_init,
                device=device,
                final_bias=row_parallel_linear_final_bias
            ))
            if is_gated_mlp:
                self.register_module(f"dense_h_to_4h_gate_{i}", ColumnParallelLinear(
                    self.hidden_size,
                    self.inner_hidden_size,
                    gather_output=False,
                    init_method=init_method,
                    bias=False,
                    params_dtype=params_dtype,
                    module=self,
                    name=f"dense_h_to_4h_gate_{i}",
                    skip_init=skip_init,
                    device=device
                ))
        self.dropout = torch.nn.Dropout(output_dropout_prob)
        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

    def forward(self, hidden_states, **kw_args):
        if 'mlp_forward' in self.hooks:
            output = self.hooks['mlp_forward'](hidden_states, **kw_args)
        else:
            output = HOOKS_DEFAULT['mlp_forward'](self, hidden_states, **kw_args)

        if self.training:
            output = self.dropout(output)
        return output


class MLP_Audio(torch.nn.Module):
    def __init__(self, hidden_size, output_dropout_prob, init_method, inner_hidden_size=None,
                 output_layer_init_method=None, layer_id=None, row_parallel_linear_final_bias=True, hooks={}, bias=True,
                 activation_func=gelu, transformer_pointer=None, is_gated_mlp=False, num_experts=1,
                 params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super(MLP_Audio, self).__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        # Project to 4h.
        self.hidden_size = hidden_size  # 3270
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(
            self.hidden_size,
            self.inner_hidden_size,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense_h_to_4h",
            skip_init=skip_init,
            device=device
        )
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            self.inner_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense_4h_to_h",
            skip_init=skip_init,
            device=device,
            final_bias=row_parallel_linear_final_bias
        )

    def forward(self, hidden_states):
        # import pdb; pdb.set_trace()
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class AudioProjModel(torch.nn.Module):
    def __init__(
            self,
            seq_len=5,
            blocks=12,  # add a new parameter blocks
            channels=768,  # add a new parameter channels
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
                seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = torch.nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = torch.nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = torch.nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = torch.nn.LayerNorm(output_dim)

        self.conv1 = torch.nn.Conv1d(in_channels=context_tokens * output_dim,
                                     out_channels=context_tokens * output_dim,
                                     kernel_size=2,
                                     stride=2,
                                     padding=0)

    def forward(self, audio_embeds):
        # merge
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        # context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f (m c)", f=video_length
        )

        b, f, c = context_tokens.shape
        for _ in range(2):
            context_tokens = context_tokens.permute(0, 2, 1)
            if context_tokens.shape[-1] % 2 == 1:
                x_first, x_rest = context_tokens[..., 0], context_tokens[..., 1:]
                if x_rest.shape[-1] > 0:
                    x_rest = self.conv1(x_rest)

                context_tokens = torch.cat([x_first[..., None], x_rest], dim=-1)
                context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)
            else:
                context_tokens = self.conv1(context_tokens)
                context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)

        context_tokens = rearrange(context_tokens, "b f (m c) -> b f m c", m=self.context_tokens)
        context_tokens = self.norm(context_tokens)

        return context_tokens


def modulate(x, shift, scale):
    if (shift.shape[0] != 1):
        shift = shift[:1]
    if (scale.shape[0] != 1):
        scale = scale[:1]
    if x.dim() == 4:
        return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def c_dim0_2_to_1(x):
    if (x.shape[0] != 1):
        return x[:1]
    return x


class SelfAttention_Audio(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True,
                 qkv_bias=False, num_multi_query_heads=0, row_parallel_linear_final_bias=True,
                 hooks={}, transformer_pointer=None, params_dtype=torch.float, skip_init=False,
                 device=torch.device('cpu')):
        super(SelfAttention_Audio, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_multi_query_heads = num_multi_query_heads
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.num_multi_query_heads_per_partition = divide(num_multi_query_heads, world_size)
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition

        # Strided linear layer.
        if num_multi_query_heads == 0:
            qkv_size = 3 * self.inner_hidden_size
            self.stride = 3
        else:  # multi-query
            qkv_size = self.inner_hidden_size + self.hidden_size_per_attention_head * self.num_multi_query_heads * 2
            self.stride = [self.num_attention_heads_per_partition, self.num_multi_query_heads_per_partition,
                           self.num_multi_query_heads_per_partition]
        self.query_key_value = ColumnParallelLinear(
            hidden_size,
            qkv_size,
            stride=self.stride,
            gather_output=False,
            init_method=init_method,
            bias=bias or qkv_bias,
            params_dtype=params_dtype,
            module=self,
            name="query_key_value",
            skip_init=skip_init,
            device=device
        )
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.dense = RowParallelLinear(
            self.inner_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense",
            skip_init=skip_init,
            device=device,
            final_bias=row_parallel_linear_final_bias
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (-1,  # flexible for multi-query
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # import pdb; pdb.set_trace()
        mixed_raw_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, self.stride)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        batch_size, num_query_heads = query_layer.shape[:2]  # [b, np, s, hn]
        num_kv_heads = key_layer.shape[1]  # [b, np, s, hn]
        key_layer = key_layer.unsqueeze(2).expand(-1, -1, num_query_heads // num_kv_heads, -1, -1).contiguous().view(
            batch_size, num_query_heads, *key_layer.shape[2:])
        value_layer = value_layer.unsqueeze(2).expand(-1, -1, num_query_heads // num_kv_heads, -1,
                                                      -1).contiguous().view(batch_size, num_query_heads,
                                                                            *value_layer.shape[2:])

        context_layer = standard_attention(query_layer, key_layer, value_layer,
                                           torch.ones((1, 1)).to(query_layer.dtype))

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output


class Audio_Dual_Block(torch.nn.Module):
    def __init__(
        self,
        hooks,
        transformer_pointer,
        params_dtype,
        skip_init,
        is_multi_person,
        device,
        layernorm_epsilon=1.0e-5,
    ):
        super().__init__()
        self.vocal_layernorm = LayerNorm(768, eps=layernorm_epsilon)
        self.accm_layernorm = LayerNorm(768, eps=layernorm_epsilon)

        self.post_attention_layernorm = LayerNorm(768, eps=layernorm_epsilon)
        self.is_multi_person = is_multi_person
        self.adaLN_modulations = nn.Sequential(nn.SiLU(), nn.Linear(512, 12 * 768))

        self.attention = SelfAttention_Audio(
            768,
            48,
            0,
            0,
            unscaled_init_method(0.02),
            0,
            hidden_size_per_attention_head=None,
            output_layer_init_method=scaled_init_method(0.02, 1),
            bias=True,
            qkv_bias=False,
            num_multi_query_heads=0,
            row_parallel_linear_final_bias=True,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device
        )
        self.mlp = MLP_Audio(
            768,
            0,
            unscaled_init_method(0.02),
            inner_hidden_size=None,
            output_layer_init_method=scaled_init_method(0.02, 1),
            bias=True,
            layer_id=0,
            row_parallel_linear_final_bias=True,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            is_gated_mlp=False,
            num_experts=1,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device
        )
    def forward(self, audio_embed_vocal, audio_embed_vocal_1, audio_embed_accm, emb):

        (
            vocal_shift_msa,
            vocal_scale_msa,
            vocal_gate_msa,
            vocal_shift_mlp,
            vocal_scale_mlp,
            vocal_gate_mlp,
            accm_shift_msa, 
            accm_scale_msa,
            accm_gate_msa,
            accm_shift_mlp,
            accm_scale_mlp,
            accm_gate_mlp,
        ) = self.adaLN_modulations(emb).chunk(12, dim=1)
        # import pdb; pdb.set_trace()
        audio_embed_vocal_input = self.vocal_layernorm(audio_embed_vocal)
        audio_embed_accm_input = self.accm_layernorm(audio_embed_accm)
        if (self.is_multi_person):
            audio_embed_vocal_input_1 = self.vocal_layernorm(audio_embed_vocal_1)

        audio_embed_vocal_input = modulate(audio_embed_vocal_input, vocal_shift_msa, vocal_scale_msa)
        audio_embed_accm_input = modulate(audio_embed_accm_input, accm_shift_msa, accm_scale_msa)
        if (self.is_multi_person):
            audio_embed_vocal_input_1 = modulate(audio_embed_vocal_input_1, vocal_shift_msa, vocal_scale_msa)
        
        B, T, C = audio_embed_vocal_input.shape
        if (self.is_multi_person):
            attention_input = torch.cat((audio_embed_vocal_input, audio_embed_vocal_input_1, audio_embed_accm_input), dim=1)
        else:
            attention_input = torch.cat((audio_embed_vocal_input, audio_embed_accm_input), dim=1)
        attention_output = self.attention(attention_input)
        if (self.is_multi_person):
            audio_embed_vocal_output = attention_output[:, : T]
            audio_embed_vocal_output_1 = attention_output[:, T: 2 * T]
            audio_embed_accm_output = attention_output[:, 2 * T : ]
        else:
            audio_embed_vocal_output = attention_output[:, : T]
            audio_embed_accm_output = attention_output[:, T:]
        # import pdb; pdb.set_trace()
        vocal_gate_msa = c_dim0_2_to_1(vocal_gate_msa)
        accm_gate_msa = c_dim0_2_to_1(accm_gate_msa)
        
        audio_embed_vocal = audio_embed_vocal + vocal_gate_msa * audio_embed_vocal_output  # (b,(t n),d)
        if (self.is_multi_person):
            audio_embed_vocal_1 = audio_embed_vocal_1 + vocal_gate_msa * audio_embed_vocal_output_1  # (b,(t n),d)
        audio_embed_accm = audio_embed_accm + accm_gate_msa * audio_embed_accm_output  # (b,n,d)

        vocal_mlp_input = self.post_attention_layernorm(audio_embed_vocal)
        if (self.is_multi_person):
            vocal_mlp_input_1 = self.post_attention_layernorm(audio_embed_vocal_1)
        accm_mlp_input = self.post_attention_layernorm(audio_embed_accm)

        vocal_mlp_input = modulate(vocal_mlp_input, vocal_shift_mlp, vocal_scale_mlp)
        if (self.is_multi_person):
            vocal_mlp_input_1 = modulate(vocal_mlp_input_1, vocal_shift_mlp, vocal_scale_mlp)
        accm_mlp_input = modulate(accm_mlp_input, accm_shift_mlp, accm_scale_mlp)
        # import pdb; pdb.set_trace()
        if (self.is_multi_person):
            mlp_input = torch.cat((vocal_mlp_input, vocal_mlp_input_1, accm_mlp_input), dim=1)
        else:
            mlp_input = torch.cat((vocal_mlp_input, accm_mlp_input), dim=1)
        mlp_output = self.mlp(mlp_input)

        if (self.is_multi_person):
            vocal_mlp_output = mlp_output[:, :T]
            vocal_mlp_output_1 = mlp_output[:, T : 2 * T]
            accm_mlp_output = mlp_output[:, 2 * T : ]
        else:
            vocal_mlp_output = mlp_output[:, :T]
            accm_mlp_output = mlp_output[:, T:]

        vocal_gate_mlp = c_dim0_2_to_1(vocal_gate_mlp)
        accm_gate_mlp = c_dim0_2_to_1(accm_gate_mlp)
        
        audio_embed_vocal =  audio_embed_vocal + vocal_gate_mlp * vocal_mlp_output
        if (self.is_multi_person):
            audio_embed_vocal_1 = audio_embed_vocal_1 + vocal_gate_mlp * vocal_mlp_output_1
        audio_embed_accm = audio_embed_accm + accm_gate_mlp * accm_mlp_output


        return audio_embed_vocal, audio_embed_vocal_1, audio_embed_accm


class AudioCombineModel(torch.nn.Module):
    def __init__(
        self,
        hooks,
        transformer_pointer,
        params_dtype,
        skip_init,
        is_multi_person,
        num_layers_dual,
        device,
        layernorm_epsilon=1.0e-5,
    ):
        super().__init__()


        num_layers_dual = num_layers_dual
        self.is_multi_person = is_multi_person
        self.dual_blocks = torch.nn.ModuleList(
            [Audio_Dual_Block(
                hooks,
                transformer_pointer,
                params_dtype,
                skip_init,
                is_multi_person,
                device,
                layernorm_epsilon=layernorm_epsilon,
            ) for _ in range(num_layers_dual)])
        


    def forward(self, audio_embed_vocal, audio_embed_vocal_1, audio_embed_accm, emb):
        # import pdb; pdb.set_trace()
        for block in self.dual_blocks:
            audio_embed_vocal, audio_embed_vocal_1, audio_embed_accm = block(audio_embed_vocal, audio_embed_vocal_1, audio_embed_accm, emb)  # [B, T, 768]
        
        if (self.is_multi_person):
            combine_audio = torch.cat((audio_embed_vocal, audio_embed_vocal_1, audio_embed_accm), dim=1)
        else:
            combine_audio = torch.cat((audio_embed_vocal, audio_embed_accm), dim=1)

        return combine_audio






class BaseTransformerLayer(torch.nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            layernorm_epsilon,
            init_method,
            layer_id,
            inner_hidden_size=None,
            hidden_size_per_attention_head=None,
            cross_hidden_size_per_attention_head=None,
            output_layer_init_method=None,
            layernorm_order='pre',
            layernorm=LayerNorm,
            is_decoder=False,
            cross_attn_hidden_size=None,
            use_bias=True,
            use_qkv_bias=False,
            num_multi_query_heads=0,
            cross_num_multi_query_heads=0,
            row_parallel_linear_final_bias=True,
            drop_path=0,
            activation_func=gelu,
            is_gated_mlp=False,
            num_experts=1,
            hooks={},
            transformer_pointer=None,
            params_dtype=torch.float,
            skip_init=False,
            device=torch.device('cpu'),
            add_audio_module=True,
            is_multi_person=False
    ):
        super(BaseTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        self.is_decoder = is_decoder[layer_id] if type(is_decoder) is list else is_decoder
        self.layernorm_order = layernorm_order
        self.drop_path = drop_path
        self.hooks = hooks
        self.add_audio_module = add_audio_module
        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            qkv_bias=use_qkv_bias,
            num_multi_query_heads=num_multi_query_heads,
            row_parallel_linear_final_bias=row_parallel_linear_final_bias,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device
        )

        # Layernorm after self-attn.
        self.post_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        if self.layernorm_order == 'sandwich':
            self.third_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
            self.fourth_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)



        if self.add_audio_module:
            self.audio_input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
            self.audio_attn = CrossAttention(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                init_method,
                layer_id,
                hidden_size_per_attention_head=cross_hidden_size_per_attention_head,
                output_layer_init_method=output_layer_init_method,
                cross_attn_hidden_size=cross_attn_hidden_size,
                bias=use_bias,
                cross_num_multi_query_heads=cross_num_multi_query_heads,
                row_parallel_linear_final_bias=row_parallel_linear_final_bias,
                hooks=hooks,
                transformer_pointer=transformer_pointer,
                params_dtype=params_dtype
            )
            if (is_multi_person):
                self.audio_input_layernorm_second = layernorm(hidden_size, eps=layernorm_epsilon)
                self.audio_attn_second = CrossAttention(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    init_method,
                    layer_id,
                    hidden_size_per_attention_head=cross_hidden_size_per_attention_head,
                    output_layer_init_method=output_layer_init_method,
                    cross_attn_hidden_size=cross_attn_hidden_size,
                    bias=use_bias,
                    cross_num_multi_query_heads=cross_num_multi_query_heads,
                    row_parallel_linear_final_bias=row_parallel_linear_final_bias,
                    hooks=hooks,
                    transformer_pointer=transformer_pointer,
                    params_dtype=params_dtype
                )

            self.audio_input_layernorm_effect = layernorm(hidden_size, eps=layernorm_epsilon)
            self.audio_attn_effect = CrossAttention(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                init_method,
                layer_id,
                hidden_size_per_attention_head=cross_hidden_size_per_attention_head,
                output_layer_init_method=output_layer_init_method,
                cross_attn_hidden_size=cross_attn_hidden_size,
                bias=use_bias,
                cross_num_multi_query_heads=cross_num_multi_query_heads,
                row_parallel_linear_final_bias=row_parallel_linear_final_bias,
                hooks=hooks,
                transformer_pointer=transformer_pointer,
                params_dtype=params_dtype
            )
            self.audio_music_adaLn = zero_module(torch.nn.Linear(768, hidden_size * 2))

        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            inner_hidden_size=inner_hidden_size,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            layer_id=layer_id,
            activation_func=activation_func,
            row_parallel_linear_final_bias=row_parallel_linear_final_bias,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            is_gated_mlp=is_gated_mlp,
            num_experts=num_experts,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device
        )

        self.bank = []

    def forward(self, hidden_states, mask, *args, **kw_args):
        return HOOKS_DEFAULT['layer_forward'](self, hidden_states, mask, *args, **kw_args)


class BaseTransformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 embedding_dropout_prob=0,
                 attention_dropout_prob=0,
                 output_dropout_prob=0,
                 drop_path=0,
                 checkpoint_activations=False,
                 checkpoint_num_layers=1,
                 checkpoint_skip_layers=0,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 inner_hidden_size=None,
                 hidden_size_per_attention_head=None,
                 cross_hidden_size_per_attention_head=None,
                 layernorm_order='pre',
                 parallel_output=False,
                 is_decoder=False,
                 cross_attn_hidden_size=None,
                 use_bias=True,
                 use_qkv_bias=False,
                 num_multi_query_heads=0,
                 cross_num_multi_query_heads=0,
                 row_parallel_linear_final_bias=True,
                 activation_func=gelu,
                 is_gated_mlp=False,
                 is_rotary_emb=False,
                 num_experts=1,
                 layernorm=LayerNorm,
                 init_method=None,
                 use_final_layernorm=True,
                 hooks={},
                 params_dtype=torch.float,
                 skip_init=False,
                 device=torch.device('cpu'),
                 add_audio_module=True,
                 is_multi_person=False,
                 num_layers_dual=4,
                 ):
        super(BaseTransformer, self).__init__()
        # recording parameters
        self.hidden_size = hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.cross_hidden_size_per_attention_head = cross_hidden_size_per_attention_head
        self.is_decoder = is_decoder
        self.cross_attn_hidden_size = cross_attn_hidden_size
        self.cross_num_multi_query_heads = cross_num_multi_query_heads
        # if not is_decoder and cross_attn_hidden_size is not None:
        #     print('warning: cross_attn_hidden_size is set but is_decoder is False')
        self.use_bias = use_bias
        self.use_qkv_bias = use_qkv_bias
        self.num_multi_query_heads = num_multi_query_heads
        self.is_gated_mlp = is_gated_mlp
        self.is_rotary_emb = is_rotary_emb
        self.num_experts = num_experts
        self.use_final_layernorm = use_final_layernorm
        self.layernorm_epsilon = layernorm_epsilon
        self.parallel_output = parallel_output
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.is_multi_person = is_multi_person
        self.checkpoint_skip_layers = checkpoint_skip_layers
        assert checkpoint_skip_layers <= num_layers - checkpoint_num_layers, f'checkpoint_skip_layers too large. Please consider remove checkpoint_activations.'
        self.max_sequence_length = max_sequence_length
        self.layernorm_order = layernorm_order
        self.row_parallel_linear_final_bias = row_parallel_linear_final_bias
        self.hooks = copy.copy(hooks)  # hooks will be updated each forward
        object.__setattr__(self, 'transformer', self)  # to give the default hooks the same api as outer hooks

        # create embedding parameters
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        if vocab_size < 1000:
            self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, dtype=params_dtype, device=device)
            torch.nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=init_method_std)
        else:
            self.word_embeddings = VocabParallelEmbedding(
                num_embeddings=vocab_size, embedding_dim=hidden_size,
                params_dtype=params_dtype, skip_init=skip_init, device=device)

        if self.is_rotary_emb:  # False
            from sat.model.position_embedding.triton_rotary_embeddings import FastRotaryEmbedding
            self.position_embeddings = FastRotaryEmbedding(hidden_size // num_attention_heads)
        else:
            self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
            torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        # create all layers
        if init_method is None:
            self.output_layer_init_method = scaled_init_method(init_method_std, num_layers)
            self.init_method = unscaled_init_method(init_method_std)
        else:
            self.output_layer_init_method = init_method
            self.init_method = init_method

        def get_layer(layer_id):
            return BaseTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                self.init_method,
                layer_id,
                inner_hidden_size=inner_hidden_size,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                cross_hidden_size_per_attention_head=cross_hidden_size_per_attention_head,
                output_layer_init_method=self.output_layer_init_method,
                is_decoder=self.is_decoder,
                cross_attn_hidden_size=cross_attn_hidden_size,
                layernorm_order=layernorm_order,
                layernorm=layernorm,
                use_bias=use_bias,
                use_qkv_bias=use_qkv_bias,
                num_multi_query_heads=num_multi_query_heads,
                cross_num_multi_query_heads=cross_num_multi_query_heads,
                row_parallel_linear_final_bias=row_parallel_linear_final_bias,
                drop_path=drop_path,
                activation_func=activation_func,
                is_gated_mlp=is_gated_mlp,
                num_experts=num_experts,
                hooks=self.hooks,
                transformer_pointer=self,
                params_dtype=params_dtype,
                skip_init=skip_init,
                device=device,
                add_audio_module=add_audio_module,
                is_multi_person=is_multi_person
            )

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)])

        # Final layer norm before output.
        if use_final_layernorm:
            self.final_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        if add_audio_module:
            if (is_multi_person):
                self.audio_vocal_embedding = torch.nn.Parameter(torch.zeros(768))
                self.audio_vocal_embedding_1 = torch.nn.Parameter(torch.zeros(768))
            self.audio_proj = AudioProjModel()
            self.audio_proj_effect = AudioProjModel()
            self.audio_proj_music = AudioProjModel()
            self.audio_combine = AudioCombineModel(
                self.hooks,
                self,
                params_dtype,
                skip_init,
                is_multi_person,
                num_layers_dual,
                device
            )

    def forward(self, input_ids, position_ids, attention_mask, audio_emb_vocal, audio_emb_vocal_1, audio_emb_accm, audio_emb_music, *,
                output_hidden_states=False, **kw_args):
        # sanity check
        assert len(input_ids.shape) >= 2
        batch_size, query_length = input_ids.shape[:2]

        if attention_mask is None:
            # Definition: None means full attention
            attention_mask = torch.ones(1, 1, device=input_ids.device)
        elif isinstance(attention_mask, int) and (attention_mask < 0):
            # Definition: -1 means lower triangular attention mask
            attention_mask = torch.ones(query_length, query_length,
                                        device=input_ids.device).tril()

        attention_mask = attention_mask.type_as(
            next(self.parameters())
        )
        assert len(attention_mask.shape) == 2 or \
               len(attention_mask.shape) == 4 and attention_mask.shape[1] == 1

        # initial output_cross_layer might be generated by word/position_embedding_forward
        output_cross_layer = {}

        # embedding part
        if 'word_embedding_forward' in self.hooks:
            hidden_states = self.hooks['word_embedding_forward'](input_ids, output_cross_layer=output_cross_layer,
                                                                 **kw_args)
        else:  # default
            hidden_states = HOOKS_DEFAULT['word_embedding_forward'](self, input_ids,
                                                                    output_cross_layer=output_cross_layer, **kw_args)

        # handle position embedding
        if 'position_embedding_forward' in self.hooks:
            position_embeddings = self.hooks['position_embedding_forward'](position_ids,
                                                                           output_cross_layer=output_cross_layer,
                                                                           **kw_args)
        else:
            assert len(position_ids.shape) <= 2
            assert position_ids.shape[-1] == hidden_states.shape[1], (position_ids.shape, hidden_states.shape)
            position_embeddings = HOOKS_DEFAULT['position_embedding_forward'](self, position_ids,
                                                                              output_cross_layer=output_cross_layer,
                                                                              **kw_args)
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        if audio_emb_vocal != None:
            # import pdb; pdb.set_trace()
            audio_emb_vocal = self.audio_proj(audio_emb_vocal)
            if (self.is_multi_person):
                audio_emb_vocal_1 = self.audio_proj(audio_emb_vocal_1)
            audio_emb_accm = self.audio_proj_effect(audio_emb_accm)
            audio_emb_music = self.audio_proj_music(audio_emb_music)

            if (self.is_multi_person):
                audio_emb_vocal = audio_emb_vocal + self.audio_vocal_embedding
                audio_emb_vocal_1 = audio_emb_vocal_1 + self.audio_vocal_embedding_1

            _, f, m, _ = audio_emb_vocal.shape
            assert f == 13, print(f)

            audio_emb_vocal = rearrange(audio_emb_vocal, "b f m c -> (b f) m c")
            if (self.is_multi_person):
                audio_emb_vocal_1 = rearrange(audio_emb_vocal_1, "b f m c -> (b f) m c")
            audio_emb_accm = rearrange(audio_emb_accm, "b f m c -> (b f) m c")
            audio_emb_music = rearrange(audio_emb_music, "b f m c -> (b f) m c")
            time_emb = kw_args["emb"]
            if (self.is_multi_person):
                audio_emb = self.audio_combine(audio_emb_vocal, audio_emb_vocal_1, audio_emb_accm, time_emb)
            else:
                audio_emb = self.audio_combine(audio_emb_vocal, None, audio_emb_accm, time_emb)
            audio_emb = torch.cat((audio_emb, audio_emb_music), dim=1)


        output_per_layers = []
        if self.checkpoint_activations:
            # define custom_forward for checkpointing
            def custom(start, end, kw_args_index, cross_layer_index):
                def custom_forward(*inputs):
                    layers_ = self.layers[start:end]
                    x_, mask, audio = inputs[0], inputs[1], inputs[2]

                    # recover kw_args and output_cross_layer
                    flat_inputs = inputs[3:]
                    kw_args, output_cross_layer = {}, {}
                    for k, idx in kw_args_index.items():
                        kw_args[k] = flat_inputs[idx]
                    for k, idx in cross_layer_index.items():
                        output_cross_layer[k] = flat_inputs[idx]
                    # -----------------

                    output_per_layers_part = []
                    for i, layer in enumerate(layers_):
                        output_this_layer_obj, output_cross_layer_obj = {}, {}
                        if 'layer_forward' in self.hooks:

                            layer_ret = self.hooks['layer_forward'](
                                x_, mask, audio, layer_id=layer.layer_id,
                                **kw_args, position_ids=position_ids, **output_cross_layer,
                                output_this_layer=output_this_layer_obj,
                                output_cross_layer=output_cross_layer_obj
                            )
                        else:

                            layer_ret = layer(
                                x_, mask, audio, layer_id=layer.layer_id,
                                **kw_args, position_ids=position_ids, **output_cross_layer,
                                output_this_layer=output_this_layer_obj,
                                output_cross_layer=output_cross_layer_obj
                            )
                        if isinstance(layer_ret, tuple):
                            layer_ret = layer_ret[0]  # for legacy API
                        x_, output_this_layer, output_cross_layer = layer_ret, output_this_layer_obj, output_cross_layer_obj
                        if output_hidden_states:
                            output_this_layer['hidden_states'] = x_
                        output_per_layers_part.append(output_this_layer)

                    # flatten for re-aggregate keywords outputs
                    flat_outputs = []
                    for output_this_layer in output_per_layers_part:
                        for k in output_this_layer:
                            # TODO add warning for depth>=2 grad tensors
                            flat_outputs.append(output_this_layer[k])
                            output_this_layer[k] = len(flat_outputs) - 1
                    for k in output_cross_layer:
                        flat_outputs.append(output_cross_layer[k])
                        output_cross_layer[k] = len(flat_outputs) - 1
                    # --------------------

                    return (x_, output_per_layers_part, output_cross_layer, *flat_outputs)

                return custom_forward

            # prevent to lose requires_grad in checkpointing.
            # To save memory when only finetuning the final layers, don't use checkpointing.
            if self.training:
                hidden_states.requires_grad_(True)
                # for layer in self.layers:
                #     layer.bank[0].requires_grad_(True)

            l, num_layers = 0, len(self.layers)
            chunk_length = self.checkpoint_num_layers
            output_this_layer = []
            while l < num_layers:
                args = [hidden_states, attention_mask, audio_emb]
                # flatten kw_args and output_cross_layer
                flat_inputs, kw_args_index, cross_layer_index = [], {}, {}
                for k, v in kw_args.items():
                    flat_inputs.append(v)
                    kw_args_index[k] = len(flat_inputs) - 1
                for k, v in output_cross_layer.items():
                    flat_inputs.append(v)
                    cross_layer_index[k] = len(flat_inputs) - 1
                # --------------------
                if l + self.checkpoint_skip_layers >= num_layers:
                    # no checkpointing
                    hidden_states, output_per_layers_part, output_cross_layer, *flat_outputs = \
                        custom(l, l + chunk_length, kw_args_index, cross_layer_index)(*args, *flat_inputs)
                else:
                    hidden_states, output_per_layers_part, output_cross_layer, *flat_outputs = \
                        checkpoint(custom(l, l + chunk_length, kw_args_index, cross_layer_index), *args, *flat_inputs)

                # recover output_per_layers_part, output_cross_layer
                for output_this_layer in output_per_layers_part:
                    for k in output_this_layer:
                        output_this_layer[k] = flat_outputs[output_this_layer[k]]
                for k in output_cross_layer:
                    output_cross_layer[k] = flat_outputs[output_cross_layer[k]]
                # --------------------

                output_per_layers.extend(output_per_layers_part)
                l += chunk_length
        else:
            output_this_layer = []
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask, audio_emb]

                output_this_layer_obj, output_cross_layer_obj = {}, {}

                if 'layer_forward' in self.hooks:  # customized layer_forward
                    layer_ret = self.hooks['layer_forward'](*args,
                                                            layer_id=torch.tensor(i),
                                                            **kw_args,
                                                            position_ids=position_ids,
                                                            **output_cross_layer,
                                                            output_this_layer=output_this_layer_obj,
                                                            output_cross_layer=output_cross_layer_obj
                                                            )
                else:
                    layer_ret = layer(*args, layer_id=torch.tensor(i), **kw_args, position_ids=position_ids,
                                      **output_cross_layer,
                                      output_this_layer=output_this_layer_obj,
                                      output_cross_layer=output_cross_layer_obj)
                if isinstance(layer_ret, tuple):
                    layer_ret = layer_ret[0]  # for legacy API
                hidden_states, output_this_layer, output_cross_layer = layer_ret, output_this_layer_obj, output_cross_layer_obj

                if output_hidden_states:
                    output_this_layer['hidden_states'] = hidden_states
                output_per_layers.append(output_this_layer)

        # Final layer norm.
        if self.use_final_layernorm:
            logits = self.final_layernorm(hidden_states)
        else:
            logits = hidden_states

        logits = copy_to_model_parallel_region(logits)
        if 'final_forward' in self.hooks:
            logits_parallel = self.hooks['final_forward'](logits, **kw_args, parallel_output=self.parallel_output)
        else:
            logits_parallel = HOOKS_DEFAULT['final_forward'](self, logits, **kw_args,
                                                             parallel_output=self.parallel_output)

        outputs = [logits_parallel]
        outputs.extend(output_per_layers)

        return outputs
