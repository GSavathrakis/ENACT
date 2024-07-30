# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .ENACT_attn.enact import ClustAttn
from .embedding import Embedding
from sklearn.cluster import KMeans
import numpy as np 


class Transformer(nn.Module):

    def __init__(self, device, sigma=3., d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(device, sigma, d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        
        self.embed_layer = Embedding(d_embed=d_model)
        self.q_embed = Embedding(d_embed=d_model)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, h, w, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, h, w,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(self.with_pos_embed(output,pos), self.with_pos_embed(output,pos), output, h, w)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, device, sigma, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = ClustAttn(sigma, d_model, dropout, nhead, device)
        #self.EnClu = ClustAttn(nhead, d_model, dropout)
        #self.self_attn = MHSA_fl_bs(nhead, d_model, dropout)
        # Implementation of Feedforward model

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def forward(self, q, k, v, h, w):
        attn = self.self_attn(q, k, v, h, w)
        v = self.norm1(v + self.dropout1(attn))
        v2 = self.activation(self.linear1(v))
        v2 = self.dropout2(v2)
        v2 = self.linear2(v2)
        v = self.norm2(v + self.dropout3(v2))

        return v


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        #print(memory.shape, mem_with_pos.shape, tgt.shape)
        #tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   #key=self.with_pos_embed(memory, pos),
                                   #value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)




class MHSA_fl_bs(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model//n_heads) 
        self.W_k = nn.Linear(d_model, d_model//n_heads) 
        self.W_v = nn.Linear(d_model, d_model//n_heads)
        self.W_b = nn.Linear(d_model//n_heads, d_model) 

        self.dropout_q = nn.Dropout(dropout)
        self.dropout_k = nn.Dropout(dropout)
        self.dropout_v = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.d_model = d_model
    
    def forward(self, q, k, h, w, v):
        k = k.permute(1,0,2)

        q = self.W_q(q.unsqueeze(1).expand(q.shape[0], self.n_heads, q.shape[1], q.shape[2]))
        k = self.W_k(k.unsqueeze(1).expand(k.shape[0], self.n_heads, k.shape[1], k.shape[2]))
        v = self.W_v(v.unsqueeze(1).expand(v.shape[0], self.n_heads, v.shape[1], v.shape[2]))

        attention = torch.mean(self.W_b(torch.matmul(F.softmax(torch.matmul(k, q.transpose(2,3))/(self.d_model//self.n_heads)), v)), dim=1).permute(1,0,2)
        #attention = torch.mean(self.W_b(torch.matmul(self.non_zero_softmax(mask*torch.matmul(k,q.transpose(2,3))/(self.d_model//self.n_heads)), v)), dim=0).view(spat,bs,-1)
        #print(torch.mean(self.W_b(torch.matmul(self.non_zero_softmax(mask*torch.matmul(k,q.transpose(1,2))/(self.d_model//self.n_heads)), v)), dim=0).shape)
        return attention



class MHSA(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model//n_heads) 
        self.W_k = nn.Linear(d_model, d_model//n_heads) 
        self.W_v = nn.Linear(d_model, d_model//n_heads)
        self.W_b = nn.Linear(d_model//n_heads, d_model) 

        self.dropout_q = nn.Dropout(dropout)
        self.dropout_k = nn.Dropout(dropout)
        self.dropout_v = nn.Dropout(dropout)

        self.n_heads = n_heads
    
    def forward(self, q, k, h, w, v):

        q = self.W_q(q.unsqueeze(1).expand(q.shape[0], self.n_heads, q.shape[1], q.shape[2]))
        k = self.W_k(k.unsqueeze(1).expand(k.shape[0], self.n_heads, k.shape[1], k.shape[2]))
        v = self.W_v(v.unsqueeze(1).expand(v.shape[0], self.n_heads, v.shape[1], v.shape[2]))

        attn = torch.mean(self.W_b(torch.matmul(F.softmax(torch.matmul(q,k.transpose(2,3)), dim=-1), v)), dim=1) #FORGOT SOFTMAX

        return attn





def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        device = args.device,
        sigma = args.smoothing_sigma,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
