# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
from model_attn import Attention
from utils import l2norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vid2Vec(nn.Module):  # videoLSTM(GRU)
    def __init__(self, video_size, hidden_size, num_layers, bidirectional, common_size):
        super(Vid2Vec, self).__init__()
        self.batch_size = video_size[0]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.h0, self.c0 = self.init_hidden()
        self.t_attention = Attention(video_size[2])
        self.context_t_attention = Attention(video_size[2])
        self.s_attention = Attention(video_size[1])
        self.context_s_attention = Attention(video_size[1])
        # self.lstm = nn.LSTM(video_size[1], self.hidden_size, num_layers=self.num_layers,
        #                     batch_first=True, dropout=0, bidirectional=self.bidirectional)
        # self.context_lstm = nn.LSTM(video_size[1], self.hidden_size, num_layers=self.num_layers,
        #                     batch_first=True, dropout=0, bidirectional=self.bidirectional)
        self.gru = nn.GRU(video_size[1], self.hidden_size, num_layers=self.num_layers,
                          batch_first=True, dropout=0, bidirectional=self.bidirectional)
        self.context_gru = nn.GRU(video_size[1], self.hidden_size, num_layers=self.num_layers,
                                  batch_first=True, dropout=0, bidirectional=self.bidirectional)
        ### RNNを使うとき(bideirectinal=True) ###
        if self.bidirectional == True:
            self.vid_attention = Attention(self.hidden_size*2)
            self.sen_out_vid_attention = Attention(self.hidden_size*2)
            self.temp_linear = nn.Linear(video_size[2], 1)
            self.cat_linear = nn.Linear(self.hidden_size*4, self.hidden_size*2)
            self.vid2vec_linear = nn.Linear(self.hidden_size*2, common_size)
            init.xavier_uniform_(self.temp_linear.weight)
            init.xavier_uniform_(self.cat_linear.weight)
            init.xavier_uniform_(self.vid2vec_linear.weight)
        ### RNNを使うとき(bideirectinal=False) ###
        elif self.bidirectional == False:
            self.vid_attention = Attention(self.hidden_size)
            self.sen_out_vid_attention = Attention(self.hidden_size)
            self.temp_linear = nn.Linear(video_size[2], 1)
            self.cat_linear = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.sen_out_cat_attention = Attention(self.hidden_size)
            self.vid2vec_linear = nn.Linear(self.hidden_size, common_size)
            init.xavier_uniform_(self.temp_linear.weight)
            init.xavier_uniform_(self.cat_linear.weight)
            init.xavier_uniform_(self.vid2vec_linear.weight)
        ### RNNを使わないとき ###
        else:
            self.linear = nn.Linear(video_size[1], common_size)
            self.context_linear = nn.Linear(video_size[1], common_size)
        # self.vid2vec_attention = Attention(common_size)
        self.vid2vec_dropout = nn.Dropout(p=0.1)

    def init_hidden(self):
        if self.bidirectional == True:
            h0 = torch.zeros(self.batch_size, self.num_layers *
                             2, self.hidden_size).to(device)
            c0 = torch.zeros(self.batch_size, self.num_layers *
                             2, self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.batch_size, self.num_layers,
                             self.hidden_size).to(device)
            c0 = torch.zeros(self.batch_size, self.num_layers,
                             self.hidden_size).to(device)
        return h0, c0

    def forward(self, iter_num, batch_video, batch_context_video, sen_out, sen_h=None):
        # batch_video, batch_video_attn_t = self.t_attention(
        #     batch_video, batch_video)
        # batch_context_video, batch_context_video_attn_t = self.context_t_attention(
        #     batch_context_video, batch_video)

        trim_size = batch_video.sum(dim=1).ne(0).sum(dim=1).to(device)
        context_size = batch_context_video.sum(
            dim=1).ne(0).sum(dim=1).to(device)

        batch_video = batch_video.permute(0, 2, 1)
        batch_context_video = batch_context_video.permute(0, 2, 1)

        # batch_video, batch_video_attn_s = self.s_attention(
        #     batch_video, batch_video)
        # batch_context_video, batch_context_video_attn_s = self.context_s_attention(
        #     batch_context_video, batch_context_video)

        if batch_video.size(0) != sen_out.size(0):
            sen_out = sen_out.repeat_interleave(batch_video.size(0), dim=0)
            if sen_h is not None:
                sen_h = sen_h.repeat_interleave(batch_video.size(0), dim=1)
        if sen_h is None:
            sen_h = self.h0.permute(1, 0, 2).contiguous()

        ### batch_video, (h_n, c_n) = self.lstm(batch_video, (sen_h, self.c0.permute(1, 0, 2).contiguous())) ###
        ### batch_context_video, (context_h_n, context_c_n) = self.context_lstm(batch_video, (sen_h, self.c0.permute(1, 0, 2).contiguous())) ###
        batch_video, h_n = self.gru(
            batch_video, self.h0.permute(1, 0, 2).contiguous())
        batch_context_video, context_h_n = self.context_gru(
            batch_context_video, self.h0.permute(1, 0, 2).contiguous())
        # batch_video, h_n = self.gru(
        #     batch_video, sen_h)
        # batch_context_video, context_h_n = self.context_gru(
        #     batch_context_video, sen_h)

        # self_attention = True
        self_attention = False

        if self_attention == True:
            vid_output, vid_output_attn = self.vid_attention(
                batch_video, batch_context_video)  # 'vid_output' #
            vid_output, sen_out_vid_attn = self.sen_out_vid_attention(
                vid_output, sen_out)  # 'vid_output' , 'sen_out' #
            vid_output = vid_output.permute(0, 2, 1)
            vid_output = self.temp_linear(vid_output)
            vid2vec = vid_output.squeeze(dim=2)
        else:
            batch_video = torch.stack([batch_video[i, j-1, :]
                                       for i, j in enumerate(trim_size)], dim=0)
            batch_context_video = torch.stack([batch_context_video[i, j-1, :]
                                               for i, j in enumerate(context_size)], dim=0)
            vid2vec = torch.cat((batch_video, batch_context_video), dim=1)
            vid2vec = self.cat_linear(vid2vec)

        vid2vec = self.vid2vec_linear(vid2vec)
        # vid2vec = self.vid2vec_attention(vid2vec, vid2vec)
        vid2vec = self.vid2vec_dropout(vid2vec)
        vid2vec = l2norm(vid2vec)
        # print('vid2vec:', vid2vec.size())
        # if iter_num % 100 == 0:
        #     print('vid2vec:', vid2vec)
        return vid2vec

# forward処理の処理過程(nn.linearなど)
