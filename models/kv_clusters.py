import torch
from torch import nn
from torch.nn import functional as F
import math


class SnapKVCluster:
    def __init__(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
    ):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(
                query_states[..., -self.window_size :, :], key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
            mask = torch.full(
                (self.window_size, self.window_size),
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
            )
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[
                :, :, -self.window_size :, -self.window_size :
            ] += attention_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights_sum = attn_weights[
                :, :, -self.window_size :, : -self.window_size
            ].sum(dim=-2)
            if self.pooling == "avgpool":
                attn_cache = F.avg_pool1d(
                    attn_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
            elif self.pooling == "maxpool":
                attn_cache = F.max_pool1d(
                    attn_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
            else:
                raise ValueError("Pooling method not supported")
            indices = attn_cache.topk(
                self.max_capacity_prompt - self.window_size, dim=-1
            ).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


class H2OCluster:
    def __init__(self, window_size=32, max_capacity_prompt=2048):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
    ):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
        mask = torch.full(
            (q_len, q_len),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
        )
        mask = mask.triu(1)
        attention_mask = mask[None, None, :, :]

        attn_weights += attention_mask

        # if attn_weights.shape[-1] > 6000:
        #     attn_weights_sum = torch.zeros(
        #         bsz,
        #         num_heads,
        #         q_len,
        #         dtype=query_states.dtype,
        #         device=query_states.device,
        #     )
        #     chunk_size = 128
        #     chunk_attn_weights = attn_weights.split(chunk_size, 2)
        #     for chunk_aw in chunk_attn_weights:
        #         chunk_aw = (
        #             nn.functional.softmax(chunk_aw, dim=-1, dtype=torch.float32)
        #             .to(query_states.dtype)
        #         )
        #         attn_weights_sum += chunk_aw[:, :, : -self.window_size].sum(dim=-2)
        # else:
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # [b, nh, s]
        attn_weights_sum = attn_weights[:, :, : -self.window_size].sum(dim=-2)
        # [b, nh, k]
        indices = attn_weights_sum.topk(
            self.max_capacity_prompt - self.window_size, dim=-1
        ).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states


class SinkCluster:
    def __init__(self, num_sink_tokens=4, max_capacity_prompt=2048):
        self.num_sink_tokens = num_sink_tokens
        self.max_capacity_prompt = max_capacity_prompt

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
    ):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        key_states = torch.cat(
            [
                key_states[:, :, : self.num_sink_tokens],
                key_states[:, :, -self.max_capacity_prompt + self.num_sink_tokens :],
            ],
            dim=2,
        )
        value_states = torch.cat(
            [
                value_states[:, :, : self.num_sink_tokens],
                value_states[:, :, -self.max_capacity_prompt + self.num_sink_tokens :],
            ],
            dim=2,
        )
        return key_states, value_states
