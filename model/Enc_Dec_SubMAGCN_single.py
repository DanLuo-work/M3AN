# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from model.Sub_MAGCN_singletask import Sub_MAGCN
import time


class Encoder_GRU(nn.Module):

    def __init__(self, adj_sub_edge, L_tilde_edge, dim_in_edge, range_K, types_accident,
                 device,
                 in_drop=0.0,
                 gcn_drop=0.0, residual=False, share_weight=True):
        super(Encoder_GRU, self).__init__()
        self.DEVICE = device
        self.dim_in_edge = dim_in_edge
        self.gate = Sub_MAGCN( adj_sub_edge, L_tilde_edge,
                              dim_in_edge * 2,
                              dim_in_edge * 2, range_K, types_accident, device, in_drop=in_drop, gcn_drop=gcn_drop,
                              residual=residual, share_weight=share_weight)
        self.update = Sub_MAGCN(adj_sub_edge, L_tilde_edge, dim_in_edge * 2,
                                dim_in_edge, range_K, types_accident=None, device=device, in_drop=in_drop,
                                gcn_drop=gcn_drop, residual=residual, share_weight=share_weight)
        self.W_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, dim_in_edge))
        self.b_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, ))

    def forward(self, inputs_node=None, inputs_edge=None, hidden_state_node=None, hidden_state_edge=None,
                input_sub_edge=None,
                accident=None):
        '''
        :param inputs: (P,B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, seq_len, num_edge, feature_edge = inputs_edge.shape
        output_inner = []
        if hidden_state_edge is None:
            hx_edge = torch.zeros((batch_size, num_edge, feature_edge)).to(self.DEVICE)
        else:
            hx_edge = hidden_state_edge
        # start0 = time.time()
        for index in range(seq_len):

            if inputs_edge is None:
                x_edge = torch.zeros((batch_size, num_edge, feature_edge)).to(self.DEVICE)
            else:
                x_edge = inputs_edge[:, index].squeeze(1)

            combined_edge = torch.cat((x_edge, hx_edge), 2)  # B,N, num_features*2
            start_gate = time.time()
            if accident is not None:
                input_acc = accident[:, index].squeeze(1)
            else:
                input_acc = None
            gates_edge = self.gate( combined_edge, input_sub_edge[:, index].squeeze(1),
                                               input_acc)  # gates: B,N, num_features*4
            # print('gate=', time.time() - start_gate)
            # print(time.time() - start)
            resetgate_edge, updategate_edge = torch.split(gates_edge, self.dim_in_edge, dim=2)
            resetgate_edge = torch.sigmoid(resetgate_edge)
            updategate_edge = torch.sigmoid(updategate_edge)
            # start_update = time.time()
            cy_edge = self.update(torch.cat((x_edge, (resetgate_edge * hx_edge)), 2),
                                           X_sub_edge=None)
            # print(time.time() - start_update)
            cy_edge = torch.tanh(cy_edge)
            hy_edge = updategate_edge * hx_edge + (1.0 - updategate_edge) * cy_edge
            hx_edge = hy_edge
            yt_edge = torch.sigmoid(hy_edge.matmul(self.W_edge) + self.b_edge)
            # print('total=', time.time() - start0)
        #     output_inner.append(yt)
        #     # print(time.time() - start1)
        # output_inner = torch.stack(output_inner, dim=0)
        return yt_edge, hy_edge


class Decoder_GRU(nn.Module):

    def __init__(self, seq_target,adj_sub_edge, L_tilde_edge, dim_in_edge, range_K, device,
                 in_drop=0.0,
                 gcn_drop=0.0, residual=False, share_weight=True):
        super(Decoder_GRU, self).__init__()
        self.seq_target = seq_target
        self.DEVICE = device
        self.dim_in_edge = dim_in_edge
        self.gate = Sub_MAGCN( adj_sub_edge, L_tilde_edge,
                              dim_in_edge * 2,
                              dim_in_edge * 2, range_K, types_accident=None, device=device, in_drop=in_drop,
                              gcn_drop=gcn_drop, residual=residual, share_weight=share_weight)
        self.update = Sub_MAGCN(adj_sub_edge, L_tilde_edge, dim_in_edge * 2,
                                dim_in_edge, range_K, types_accident=None, device=device, in_drop=in_drop,
                                gcn_drop=gcn_drop, residual=residual, share_weight=share_weight)
        self.W_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, dim_in_edge))
        self.b_edge = nn.Parameter(torch.FloatTensor(dim_in_edge, ))

    def forward(self, inputs_node=None, inputs_edge=None, hidden_state_node=None, hidden_state_edge=None):
        '''
        :param inputs: (P,B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, num_edge, feature_edge = inputs_edge.shape
        output_node = []
        output_edge = []

        if hidden_state_edge is None:
            hx_edge = torch.zeros((batch_size, num_edge, feature_edge)).to(self.DEVICE)
        else:
            hx_edge = hidden_state_edge
        for index in range(self.seq_target):
            if inputs_edge is None:
                x_edge = torch.zeros((batch_size, num_edge, feature_edge)).to(self.DEVICE)
            else:
                x_edge = inputs_edge

            combined_edge = torch.cat((x_edge, hx_edge), -1)  # B,N, num_features*2
            gates_edge = self.gate(combined_edge,
                                               X_sub_edge=None)  # gates: B,N, num_features*4
            # print(time.time() - start)
            resetgate_edge, updategate_edge = torch.split(gates_edge, self.dim_in_edge, dim=-1)
            resetgate_edge = torch.sigmoid(resetgate_edge)
            updategate_edge = torch.sigmoid(updategate_edge)
            cy_edge = self.update(torch.cat((x_edge, (resetgate_edge * hx_edge)), -1),
                                           X_sub_edge=None)
            cy_edge = torch.tanh(cy_edge)
            hy_edge = updategate_edge * hx_edge + (1.0 - updategate_edge) * cy_edge
            hx_edge = hy_edge
            yt_edge = torch.sigmoid(hy_edge.matmul(self.W_edge) + self.b_edge)
            output_edge.append(yt_edge)
            # print(time.time() - start1)
        output_edge = torch.stack(output_edge, dim=0)
        return output_edge


class Enc_Dec(nn.Module):
    def __init__(self, seq_target, L_tilde_node, dim_in_node, dim_out_node, adj_sub_edge, L_tilde_edge, dim_in_edge,
                 dim_out_edge, range_K, types_accident, device,
                 in_drop=0.0, gcn_drop=0.0, residual=False, share_weight=True):
        super(Enc_Dec, self).__init__()
        self.linear_in_edge = nn.Linear(1, dim_in_edge)
        self.Encoder = Encoder_GRU( adj_sub_edge, L_tilde_edge, dim_in_edge, range_K,
                                   types_accident,
                                   device,
                                   in_drop=in_drop, gcn_drop=gcn_drop, residual=residual, share_weight=share_weight)
        self.Decoder = Decoder_GRU(seq_target,adj_sub_edge, L_tilde_edge, dim_in_edge,
                                   range_K, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual,
                                   share_weight=share_weight)
        self.linear_out_edge = nn.Linear(dim_in_edge, 1)
        self.linear_out_edge1 = nn.Linear(dim_in_edge, dim_in_edge)
        self.linear_out_edge2 = nn.Linear(dim_in_edge, dim_in_edge)
        self.linear_out_edge3 = nn.Linear(dim_in_edge, dim_in_edge)

    def forward(self, inputs_node=None, hidden_state_node=None, inputs_edge=None, hidden_state_edge=None,
                input_sub_edge=None, accident=None):
        inputs_edge = self.linear_in_edge(inputs_edge)
        yt_edge, hy_edge = self.Encoder(inputs_node=inputs_node, inputs_edge=inputs_edge,
                                                          input_sub_edge=input_sub_edge, accident=accident)
        output_edge = self.Decoder(yt_edge, hy_edge)
        output_edge = self.linear_out_edge1(output_edge.permute(1, 0, 2, 3))
        output_edge = self.linear_out_edge2(output_edge)
        output_edge = self.linear_out_edge3(output_edge)
        output_edge = self.linear_out_edge(output_edge)
        return output_edge
