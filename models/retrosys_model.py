# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from argparse import ArgumentParser, ArgumentTypeError, ArgumentError, Namespace
from dataclasses import dataclass, _MISSING_TYPE, MISSING
from enum import Enum
import inspect
import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from random import uniform
import torch


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embed(x)


class RetroSysModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Initialize encoder and decoder
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        plm_input,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            plm_input=plm_input,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        return decoder_out

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.finetune_plm = getattr(args, "finetune_plm", False)
        self.gradmultiply = getattr(args, "plm_grad", None)
        if self.gradmultiply is None:
            self.gradmultiply = 1 / getattr(args, "gradmultiply", 1.)
        layer = EncoderLayer(args=args)
        self.layers = nn.ModuleList([layer, layer, layer, layer, layer, layer])
        self.padding_idx = 0
        self.layer_norm = nn.LayerNorm(args.encoder_ffn_embed_dim)

    def forward(
        self,
        embedding_input,
        plm_input=None,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        return self.forward_scriptable(
            embedding_input,
            plm_input,
            src_lengths,
            return_all_hiddens,
            token_embeddings,
        )

    def forward_scriptable(
        self,
        embedding_input,
        plm_input: Optional[torch.Tensor],
        src_lengths: Optional[torch.Tensor],
        return_all_hiddens: bool,
        token_embeddings: Optional[torch.Tensor],
    ):
        plm_out = plm_input
        encoder_padding_mask = embedding_input.eq(self.padding_idx)
        has_pads = embedding_input.device.type == "xla" or encoder_padding_mask.any()
        # x, encoder_embedding = self.forward_embedding(embedding_input, token_embeddings)
        x = embedding_input

        # if has_pads:
        #     print(x.shape)
        #     print(encoder_padding_mask.shape)
        #     x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

            # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # if plm_out is not None:
        #     plm_padding_mask = plm_out["encoder_padding_mask"][0]
        #     plm_out = plm_out["encoder_out"][0]
        # plm_has_pads = plm_out.device.type == "xla" or plm_padding_mask.any()
        plm_has_pads = False

        # encoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask= None,
                plm_out=plm_out,
                plm_padding_mask= None,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        # if self.layer_norm is not None:
        #     x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "plm_out": [plm_out],  # T x B x C
            "plm_padding_mask": [],  # B x T
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        layer = DecoderLayer(args=args)
        self.layers = nn.ModuleList([layer, layer, layer, layer, layer, layer])
        self.layer_norm = nn.LayerNorm(args.decoder_ffn_embed_dim)
        self.num_layers = args.decoder_layers
        self.dropout_module = nn.Dropout(args.dropout)

    
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # Call extract_features_scriptable with the required arguments
        x, extra = self.extract_features_scriptable(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        # Optionally, apply layer normalization if defined
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x, extra
    

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        full_context_alignment: bool,
        alignment_layer: Optional[int],
        alignment_heads: Optional[int],
    ):
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        plm: Optional[Tensor] = None
        plm_padding_mask: Optional[Tensor] = None
        if encoder_out is not None:
            enc = encoder_out["encoder_out"][0]
            padding_mask = encoder_out["encoder_padding_mask"][0]
            assert enc.size()[1] == bs, f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
            plm = encoder_out["plm_out"][0]
            # plm_padding_mask = encoder_out["plm_padding_mask"][0]
            # assert plm.size()[1] == bs, f"Expected plm.shape == (t, {bs}, c) got {plm.shape}"

        positions = None
        # if self.embed_positions is not None:
        #     positions = self.embed_positions(
        #         prev_output_tokens, incremental_state=incremental_state
        #     )

        # if incremental_state is not None:
        #     prev_output_tokens = prev_output_tokens[:, -1:]
        #     if positions is not None:
        #         positions = positions[:, -1:]

        # embed tokens and positions
        self.embed_scale = math.sqrt(768)
        # x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = prev_output_tokens

        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)

        # if self.project_in_dim is not None:
        #     x = self.project_in_dim(x)

        # if positions is not None:
        #     x += positions

        # if self.layernorm_embedding is not None:
        #     x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        # if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
        #     self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            # if incremental_state is None and not full_context_alignment:
            #     self_attn_mask = self.buffered_future_mask(x)
            # else:
            #     self_attn_mask = None
            self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                plm,
                plm_padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
        attn = attn.mean(dim=0)

        # if self.layer_norm is not None:
        #     x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # if self.project_out_dim is not None:
        #     x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

def quant_noise(module, p=0.1, block_size=8):
    """
    在给定的模型模块上应用量化噪声。

    Args:
        module (torch.nn.Module): 要应用量化噪声的模型层（例如 nn.Linear）。
        p (float): 噪声应用的概率。
        block_size (int): 量化噪声应用的块大小。

    Returns:
        torch.nn.Module: 带有量化噪声的模型层。
    """
    for param in module.parameters():
        mask = (torch.rand(param.size()) < p).float()  # 随机生成掩码
        noise = (torch.randint(low=-1, high=2, size=param.size()).float())  # [-1, 1] 范围内的离散噪声
        param.data = param.data * (1 - mask) + param.data * mask * noise
    return module

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.outside_attn = self.build_encoder_plm_attention(self.embed_dim, args)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout_module = nn.Dropout(args.dropout)
        self.activation_fn = nn.functional.relu
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = nn.Dropout(float(activation_dropout_p))
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropnet = getattr(args, "dropnet", 0.25)
        self.gradmultiply = getattr(args, "gradmultiply", 1.)
        self.plm_as_encoder = getattr(args, "plm_as_encoder", False)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return nn.MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn
        )

    def build_encoder_plm_attention(self, embed_dim, args):
        return nn.MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            kdim=args.plm_encoder_embed_dim,
            vdim=args.plm_encoder_embed_dim,
            dropout=args.attention_dropout,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def get_ratio(self):
        if self.plm_as_encoder:
            return [0, 1]
        if self.dropnet > 0 and self.training:
            frand = float(uniform(0, 1))
            if frand < self.dropnet:
                return [1, 0]
            elif frand > 1 - self.dropnet:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [0.5, 0.5]

    def forward(
        self,
        x,
        plm_out,
        encoder_padding_mask: Optional[Tensor],
        plm_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x

        x1, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x2, _ = self.outside_attn(
            query=x, key=plm_out, value=plm_out, key_padding_mask=plm_padding_mask, attn_mask=None
        )
        x1 = self.dropout_module(x1)
        x2 = self.dropout_module(x2)
        # x2 = GradMultiply.apply(x2, self.gradmultiply)
        dropnet = self.get_ratio()
        x = residual + dropnet[0] * x1 + dropnet[1] * x2

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = nn.Dropout(args.dropout)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
        )

        self.outside_attn = self.build_decoder_plm_attention(self.embed_dim, args)

        self.activation_fn = nn.functional.relu
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = nn.Dropout(float(activation_dropout_p))

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False
        self.dropnet = getattr(args, "dropnet", 0.25)
        self.gradmultiply = getattr(args, "gradmultiply", 1.)
        self.plm_as_encoder = getattr(args, "plm_as_encoder", False)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return nn.MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn
        )

    def build_encoder_attention(self, embed_dim, args):
        return nn.MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
        )

    def build_decoder_plm_attention(self, embed_dim, args):
        return nn.MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            kdim=args.plm_encoder_embed_dim,
            vdim=args.plm_encoder_embed_dim,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def get_ratio(self):
        if self.plm_as_encoder:
            return [0, 1]
        if self.dropnet > 0 and self.training:
            frand = float(uniform(0, 1))
            if frand < self.dropnet:
                return [1, 0]
            elif frand > 1 - self.dropnet:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [0.5, 0.5]

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        plm_out: Optional[torch.Tensor] = None,
        plm_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        if need_head_weights:
            need_attn = True

        residual = x
        # if self.normalize_before:
        #     x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        # _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        # if self.cross_self_attention:
        #     if self_attn_mask is not None:
        #         assert encoder_out is not None
        #         self_attn_mask = torch.cat(
        #             (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
        #         )
        #     if self_attn_padding_mask is not None:
        #         if encoder_padding_mask is None:
        #             assert encoder_out is not None
        #             encoder_padding_mask = self_attn_padding_mask.new_zeros(
        #                 encoder_out.size(1), encoder_out.size(0)
        #             )
        #         self_attn_padding_mask = torch.cat(
        #             (encoder_padding_mask, self_attn_padding_mask), dim=1
        #         )
        #     assert encoder_out is not None
        #     y = torch.cat((encoder_out, x), dim=0)
        # else:
        #     y = x

        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        # if not self.normalize_before:
        #     x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            # if self.normalize_before:
            #     x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            
            encoder_padding_mask = None
            
            x1, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                need_weights=need_attn or (not self.training and self.need_attn)
            )
            x1 = self.dropout_module(x1)
            x2, _ = self.outside_attn(
                query=x,
                key=plm_out,
                value=plm_out,
                key_padding_mask=plm_padding_mask,
                need_weights=False
            )
            # x2 = GradMultiply.apply(x2, self.gradmultiply)
            x2 = self.dropout_module(x2)
            dropnet = self.get_ratio()
            x = residual + dropnet[0] * x1 + dropnet[1] * x2
            # if not self.normalize_before:
            #     x = self.encoder_attn_layer_norm(x)

        residual = x
        # if self.normalize_before:
        #     x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        # if not self.normalize_before:
        #     x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None