''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v=None, attn4rel= False, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.attn4rel = attn4rel

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        if not self.attn4rel:
            self.d_v = d_v
            self.w_vs = nn.Linear(d_model, n_head * d_v)
            nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

            self.layer_norm = nn.LayerNorm(d_model)

            self.fc = nn.Linear(n_head * d_v, d_model)
            nn.init.xavier_normal_(self.fc.weight)

            self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v=None, mask=None):

        d_k, n_head = self.d_k, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()


        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk

        if not self.attn4rel:       #when use only attn then no need of v
            d_v = self.d_v
            sz_b, len_v, _ = v.size()
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v=v, attn4rel = self.attn4rel, mask=mask)

        if self.attn4rel:
            return attn
        else:
            output = output.view(n_head, sz_b, len_q, d_v)
            output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

            output = self.dropout(self.fc(output))
            output = self.layer_norm(output +residual)

            return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, use_w2 = True):
        super().__init__()
        self.use_w2 = use_w2
        if self.use_w2:
            self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        else:
            d_hid = d_in
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        if self.use_w2:
            output = self.w_2(F.relu(self.w_1(output)))
        else:
            output =F.relu(self.w_1(output))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
