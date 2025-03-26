#################################################################################
'''
MIT License

Copyright (c) 2025 Hou Sai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
####################################################################################33

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils_good as utils
import layers

class LowRank(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop):
        super(LowRank, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_net = layers.create(att_type, att_mid_dim, att_mid_drop)

    def forward(self, query, key, mask, value1, value2, precompute=False):
        print(query.shape)
        print(key.shape)
        print(value1.shape)
        print(value2.shape)
        batch_size = query.size()[0]
        B, N_kv, _ = key.shape
        H = self.num_heads
        D = self.head_dim
        query = query.unsqueeze(1)
        Q = self.q_proj(query).view(B, 1, H, D).transpose(1, 2)
        K = self.k_proj(key).view(B, N_kv, H, D).transpose(1, 2)
        V = self.v_proj(key).view(B, N_kv, H, D).transpose(1, 2)
        v1 = value1.view(batch_size, self.num_heads, self.head_dim)
        v2 = self.v_proj(value2).view(B, N_kv, H, D).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attended = torch.matmul(scores, V)
        attn_map = attended.expand(-1, -1, N_kv, -1)
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn

    def forward2(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        query = query.view(-1, query.size()[-1])
        value1 = value1.view(-1, value1.size()[-1])

        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            if self.buffer_keys is not None and self.buffer_value2 is not None:
                self.buffer_keys = torch.cat([self.buffer_keys, k], dim=2)
                self.buffer_value2 = torch.cat([self.buffer_value2, v2], dim=2)
                k = self.buffer_keys
                v2 = self.buffer_value2
        else:
            k = key
            v2 = value2

        #attn_map = q.unsqueeze(-2) * k.unsqueeze(-3)
        # Assuming q.shape is [batch_size, num_heads, seq_length_q, head_dim]
        # and k.shape is [batch_size, num_heads, seq_length_k, head_dim]
        # after unsqueezing
        # q is unsqueezed to add an extra dimension for seq_length_k
        # k is unsqueezed to add an extra dimension for seq_length_q

        q_expanded = q.unsqueeze(-2)  # Add dimension for seq_length_k, preparing for broadcasting
        k_expanded = k.unsqueeze(-3)  # Add dimension for seq_length_q, preparing for broadcasting

        # Compute element-wise mean
        mean_qk = (q_expanded + k_expanded) / 2

        # Compute cross-variance
        cross_variance = ((q_expanded - mean_qk) ** 2 + (k_expanded - mean_qk) ** 2)/2

        attn_map = self.dropout(cross_variance)

        attn = self.attn_net.forward(attn_map, mask, v1, v2).transpose(1, 2).contiguous()
        attn = attn.view(batch_size, -1, self.num_heads * self.head_dim)
        return attn

    def precompute(self, key, value2):
        batch_size = value2.size()[0]
        key = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2
