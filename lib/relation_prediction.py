import torch
from torch import nn
from torch.nn import functional as F
from config import BATCHNORM_MOMENTUM
import numpy as np
from siren_pytorch import Siren
from lib.pytorch_misc import init_weights


class Relation_Prediction(nn.Module):

    def __init__(self, config, num_rels):

        super(Relation_Prediction, self).__init__()
        assert len(config.spo)>0 and  config.nl_edge>0

        self.spo =config.spo
        self.use_gap = config.use_gap #False #to reproduce 163
        gap_dim = 4096 if config.use_mmdet else 512
        self.use_tanh = config.use_tanh
        self.hidden_dim = config.attn_dim
        self.num_rels = num_rels
        self.n_dropout= nn.Dropout(config.dropout)
        self.focal_loss = config.use_FL
        self.use_extra_pos = config.use_extra_pos
        self.nl_edge = config.nl_edge



        if self.use_extra_pos:
            self.e_pos_emb = nn.Sequential(*[
                nn.LayerNorm(4), #nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
                nn.Linear(4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ])

        if self.use_gap:
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=37)  # todo replcae wd correct param

        if self.spo=='sop':
            self.ws = Siren(self.hidden_dim, self.hidden_dim)
            self.wo = Siren(self.hidden_dim, self.hidden_dim)
            self.wp = Siren(self.hidden_dim, self.hidden_dim)
            self.wf = Siren((self.hidden_dim+(128 if self.use_extra_pos else 0) + (gap_dim if self.use_gap else 0)), self.hidden_dim, activation=nn.LeakyReLU())
            #self.ln = nn.LayerNorm(self.hidden_dim)
            # #init weight
            # nn.init.xavier_normal_(self.ws.weight)
            # nn.init.xavier_normal_(self.wo.weight)
            # nn.init.xavier_normal_(self.wp.weight)
            # nn.init.xavier_normal_(self.wf.weight)
        else:

            sop_emb_size = self.hidden_dim * 3 + (128 if self.use_extra_pos else 0) + (gap_dim if self.use_gap else 0)
            if self.nl_edge > 0 and self.spo == 'spo':  # todo for normalize increase dropout/add layernorm, dropout replace with self.dropout
                self.sub_obj_emb = nn.Sequential(*[nn.LayerNorm(sop_emb_size),nn.Linear(sop_emb_size, 2*self.hidden_dim),nn.LayerNorm(2*self.hidden_dim),
                                                nn.Dropout(config.dropout),nn.Linear(2*self.hidden_dim, self.hidden_dim), nn.LeakyReLU()])

            elif self.nl_edge > 0:
                self.sub_obj_emb = nn.Sequential(*[nn.Linear(self.hidden_dim, self.hidden_dim), nn.Dropout(0.3)])
            else:
                self.sub_obj_emb = nn.Sequential(*[nn.LayerNorm(sop_emb_size), nn.Linear(sop_emb_size, 2 * self.hidden_dim),
                                                   nn.LayerNorm(2 * self.hidden_dim), nn.Dropout(config.dropout),
                                                   nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                                   nn.LeakyReLU()])  # + (4 if self.use_extra_pos else 0) , nn.LayerNorm(self.hidden_dim) , nn.Dropout(0.3)nn.LayerNorm(sop_emb_size),

            self.sub_obj_emb.apply(init_weights)

        #final rels
        self.final_rel = nn.Linear(self.hidden_dim, self.num_rels, bias=True)
        nn.init.xavier_normal_(self.final_rel.weight)

        if self.focal_loss:
            #b_weight = np.log(39826/303266)
            b_weight = -0.881657035
            self.final_rel.bias.data.fill_(-0.9670156622)
        else:
            self.final_rel.bias.data.zero_()


    def forward(self, result, obj_ctx, edge_ctx, s_o_rois, o_s_rois, fwd_rels, inv_rels):

        #if use of global feats is there
        if self.use_gap:
            if isinstance(result.fmap, list):
                result.fmap = result.fmap[0]
            gap_feats = self.avg_pool(result.fmap).view(result.fmap.shape[0], -1)[result.obj_comb[:, 0]]
        else:
            gap_feats = torch.Tensor().cuda(obj_ctx.get_device())

        #if use of extra pos is there
        if self.use_extra_pos:
            s_o_emb = self.e_pos_emb(s_o_rois)
            o_s_emb = self.e_pos_emb(o_s_rois)
        else:
            s_o_emb = torch.Tensor().cuda(obj_ctx.get_device())
            o_s_emb = torch.Tensor().cuda(obj_ctx.get_device())

        if self.spo == 'sop':
            fwd_edge_rep = self.ws(obj_ctx[result.obj_comb[:,1]]) * self.wo(obj_ctx[result.obj_comb[:,2]]) * self.wp(edge_ctx)
            inv_edge_rep = self.ws(obj_ctx[result.obj_comb[:, 2]]) * self.wo(obj_ctx[result.obj_comb[:, 1]]) * self.wp(edge_ctx)

            #Now add extra global feats and box feats
            fwd_edge_rep = torch.cat((fwd_edge_rep, gap_feats, s_o_emb), 1)
            inv_edge_rep = torch.cat((inv_edge_rep, gap_feats, o_s_emb), 1)

        elif self.spo =='spo':
            fwd_edge_rep = self.sub_obj_emb(torch.cat((obj_ctx[result.obj_comb[:,1]], edge_ctx, obj_ctx[result.obj_comb[:,2]],gap_feats, s_o_emb), 1))
            inv_edge_rep = self.sub_obj_emb(torch.cat((obj_ctx[result.obj_comb[:,2]], edge_ctx, obj_ctx[result.obj_comb[:,1]],gap_feats, o_s_emb), 1))
        else:
            #add another embedding space to learn sub-object representation
            edge_rep = self.sub_obj_emb(edge_ctx if self.nl_edge>0 else result.rm_obj_dists)

            # Split into subject and object representations
            edge_rep = edge_rep.view(-1, 2 , self.hidden_dim)
            obj1_rep = edge_rep[:,0,:]
            obj2_rep = edge_rep[:,1,:]
            fwd_edge_rep = torch.cat((obj1_rep, obj2_rep), 1)
            inv_edge_rep = torch.cat((obj2_rep, obj1_rep),1)

        result.all_rels = torch.cat((fwd_rels, inv_rels))
        if not fwd_rels.shape[0] == 0:
            fwd_pred = fwd_edge_rep[fwd_rels[:, 1]]
        if not inv_rels.shape[0] == 0:
            inv_pred = inv_edge_rep[inv_rels[:, 1]]

        if fwd_rels.shape[0] == 0:
            all_pred = inv_pred
        elif inv_rels.shape[0] == 0:
            all_pred = fwd_pred
        else:
            all_pred = torch.cat((fwd_pred, inv_pred), 0)

        #for sop
        if self.spo=='sop':
            all_pred = self.wf(all_pred)

        if self.use_tanh:
            all_pred = self.n_dropout(all_pred)
            all_pred = F.tanh(all_pred)
        else:
            all_pred = self.n_dropout(all_pred)

        result.rel_dists = self.final_rel(all_pred)



