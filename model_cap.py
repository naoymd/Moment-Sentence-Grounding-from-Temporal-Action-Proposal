# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init

from dataset_cap import Vocabulary
from model_attn import Attention
from utils import l2norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, batch_size, hidden_size, num_layers, bidirectional, common_size, max_seq_len, max_verb_len, weights):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sen_embed = nn.Embedding.from_pretrained(
            embeddings=weights, freeze=True)
        self.verb_embed = nn.Embedding.from_pretrained(
            embeddings=weights, freeze=True)
        self.sen_embed_attention = Attention(weights.size(1))
        self.verb_embed_attention = Attention(weights.size(1))
        self.sen_h0, self.sen_c0 = self.init_hidden()
        self.verb_h0, self_verb_c0 = self.init_hidden()
        self.sen_lstm = nn.LSTM(weights.size(1), self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, dropout=0, bidirectional=self.bidirectional)
        self.verb_lstm = nn.LSTM(weights.size(1), self.hidden_size, num_layers=self.num_layers,
                                 batch_first=True, dropout=0, bidirectional=self.bidirectional)
        if self.bidirectional == True:
            self.lstm_attention = Attention(self.hidden_size*2)
            self.cat_linear = nn.Linear(self.hidden_size*4, self.hidden_size*2)
            self.sen2vec_linear = nn.Linear(self.hidden_size*2, common_size)
            self.sen_h_linear = nn.Linear(
                self.hidden_size*2, self.hidden_size)
            self.sen_out_linear = nn.Linear(
                self.hidden_size*2, self.hidden_size*2)
        else:
            self.lstm_attention = Attention(self.hidden_size)
            self.cat_linear = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.sen2vec_linear = nn.Linear(self.hidden_size, common_size)
            self.sen_h_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.sen_out_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.len_linear = nn.Linear(max_seq_len, 1)
        self.sen2vec_attention = Attention(common_size)
        self.sen2vec_dropout = nn.Dropout(p=0.1)
        self.sen_out_dropout = nn.Dropout(p=0.1)
        self.sen_h_dropout = nn.Dropout(p=0.1)
        init.xavier_uniform_(self.len_linear.weight)
        init.xavier_uniform_(self.cat_linear.weight)
        init.xavier_uniform_(self.sen2vec_linear.weight)
        init.xavier_uniform_(self.sen_h_linear.weight)
        init.xavier_uniform_(self.sen_out_linear.weight)

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

    def forward(self, iter_num, batch_sentence, batch_verb, vocab):
        words_num = batch_sentence.ne(vocab.padidx).sum(dim=1).to(device)
        verbs_num = batch_verb.ne(vocab.padidx).sum(dim=1).to(device)
        sen_embed = self.sen_embed(batch_sentence)
        verb_embed = self.verb_embed(batch_verb)

        # sen_embed, sen_embed_attn = self.sen_embed_attention(
        #     sen_embed, sen_embed)  # embed_attention #
        # verb_embed, verb_embed_attn = self.verb_embed_attention(
        #     verb_embed, sen_embed)  # embed_attention #

        sen_output, (sen_h_n, sen_c_n) = self.sen_lstm(sen_embed, self.sen_h0.permute(
            1, 0, 2).contiguous(), self.sen_c0.permute(1, 0, 2).contiguous())
        verb_output, (verb_h_n, verb_c_n) = self.verb_lstm(verb_embed, self.verb_h0.permute(
            1, 0, 2).contiguous(), self.verb_c0.permute(1, 0, 2).contiguous())

        ### sen_output.size() -> [batch_size, max_seq_len, hidden_size*2] ###

        ### sen_output_cat.size() -> [batch_size, max_seq_len+max_verb_len, hidden_size*2] ###
        # output_cat = torch.cat((sen_output, verb_output), dim=1)

        # self_attention = True
        self_attention = False

        if self_attention == True:
            output, output_attn = self.lstm_attention(
                sen_output, verb_output)  # lstm_attention #
            sen2vec = output.permute(0, 2, 1)
            sen2vec = self.len_linear(sen2vec).squeeze(dim=2)
        else:
            sen2vec = torch.stack([sen_output[i, j-1, :]
                                   for i, j in enumerate(words_num)], dim=0)
            verb2vec = torch.stack([verb_output[i, j-1, :]
                                    for i, j in enumerate(verbs_num)], dim=0)
            sen2vec = torch.cat((sen2vec, verb2vec), dim=1)
            sen2vec = self.cat_linear(sen2vec)

        sen2vec = self.sen2vec_linear(sen2vec)
        # sen2vec, sen2vec_attn = self.sen2vec_attention(sen2vec, sen2vec)
        sen2vec = self.sen2vec_dropout(sen2vec)
        sen2vec = l2norm(sen2vec)
        sen_out = torch.cat((sen_output, verb_output), dim=1)
        sen_out = self.sen_out_linear(sen_out)
        sen_out = self.sen_out_dropout(sen_out)
        sen_out = l2norm(sen_out)
        sen_h = torch.cat((sen_h_n, verb_h_n), dim=2)
        sen_h = self.sen_h_linear(sen_h)
        sen_h = self.sen_h_dropout(sen_h)
        sen_h = l2norm(sen_h)
        # print('sen2vec:', sen2vec.size(), sen_out.size(), sen_h.size())
        # if iter_num % 100 == 0:
        #     print('sen2vec:', sen2vec)
        return sen2vec, sen_out, sen_h


class GRU(nn.Module):
    def __init__(self, batch_size, hidden_size, num_layers, bidirectional, common_size, max_seq_len, max_verb_len, weights):
        super(GRU, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sen_embed = nn.Embedding.from_pretrained(
            embeddings=weights, freeze=True)
        self.verb_embed = nn.Embedding.from_pretrained(
            embeddings=weights, freeze=True)
        self.sen_embed_attention = Attention(weights.size(1))
        self.verb_embed_attention = Attention(weights.size(1))
        self.sen_h0 = self.init_hidden()
        self.verb_h0 = self.init_hidden()
        self.sen_gru = nn.GRU(weights.size(1), self.hidden_size, num_layers=self.num_layers,
                              batch_first=True, dropout=0, bidirectional=self.bidirectional)
        self.verb_gru = nn.GRU(weights.size(1), self.hidden_size, num_layers=self.num_layers,
                               batch_first=True, dropout=0, bidirectional=self.bidirectional)
        if self.bidirectional == True:
            self.gru_attention = Attention(self.hidden_size*2)
            self.cat_linear = nn.Linear(self.hidden_size*4, self.hidden_size*2)
            self.sen2vec_linear = nn.Linear(self.hidden_size*2, common_size)
            self.sen_h_linear = nn.Linear(
                self.hidden_size*2, self.hidden_size)
            self.sen_out_linear = nn.Linear(
                self.hidden_size*2, self.hidden_size*2)
        else:
            self.gru_attention = Attention(self.hidden_size)
            self.cat_linear = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.sen2vec_linear = nn.Linear(self.hidden_size, common_size)
            self.sen_h_linear = nn.Linear(self.hidden_size, self.hidden_size)
            self.sen_out_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.len_linear = nn.Linear(max_seq_len, 1)
        self.sen2vec_attention = Attention(common_size)
        self.sen2vec_dropout = nn.Dropout(p=0.1)
        self.sen_out_dropout = nn.Dropout(p=0.1)
        self.sen_h_dropout = nn.Dropout(p=0.1)
        init.xavier_uniform_(self.len_linear.weight)
        init.xavier_uniform_(self.cat_linear.weight)
        init.xavier_uniform_(self.sen2vec_linear.weight)
        init.xavier_uniform_(self.sen_h_linear.weight)
        init.xavier_uniform_(self.sen_out_linear.weight)

    def init_hidden(self):
        if self.bidirectional == True:
            h0 = torch.zeros(self.batch_size, self.num_layers *
                             2, self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.batch_size, self.num_layers,
                             self.hidden_size).to(device)
        return h0

    def forward(self, iter_num, batch_sentence, batch_verb, vocab):
        words_num = batch_sentence.ne(vocab.padidx).sum(dim=1).to(device)
        verbs_num = batch_verb.ne(vocab.padidx).sum(dim=1).to(device)
        sen_embed = self.sen_embed(batch_sentence)
        verb_embed = self.verb_embed(batch_verb)

        # sen_embed, sen_embed_attn = self.sen_embed_attention(
        #     sen_embed, sen_embed)  # embed_attention #
        # verb_embed, verb_embed_attn = self.verb_embed_attention(
        #     verb_embed, sen_embed)  # embed_attention #

        sen_output, sen_h_n = self.sen_gru(
            sen_embed, self.sen_h0.permute(1, 0, 2).contiguous())
        verb_output, verb_h_n = self.verb_gru(
            verb_embed, self.verb_h0.permute(1, 0, 2).contiguous())

        ### sen_output.size() -> [batch_size, max_seq_len, hidden_size*2] ###

        ### sen_output_cat.size() -> [batch_size, max_seq_len+max_verb_len, hidden_size*2] ###
        # output_cat = torch.cat((sen_output, verb_output), dim=1)

        # self_attention = True
        self_attention = False

        if self_attention == True:
            output, output_attn = self.gru_attention(
                sen_output, verb_output)  # gru_attention #
            sen2vec = output.permute(0, 2, 1)
            sen2vec = self.len_linear(sen2vec).squeeze(dim=2)
        else:
            sen2vec = torch.stack([sen_output[i, j-1, :]
                                   for i, j in enumerate(words_num)], dim=0)
            verb2vec = torch.stack([verb_output[i, j-1, :]
                                    for i, j in enumerate(verbs_num)], dim=0)
            sen2vec = torch.cat((sen2vec, verb2vec), dim=1)
            sen2vec = self.cat_linear(sen2vec)

        sen2vec = self.sen2vec_linear(sen2vec)
        # sen2vec, sen2vec_attn = self.sen2vec_attention(sen2vec, sen2vec)
        sen2vec = self.sen2vec_dropout(sen2vec)
        sen2vec = l2norm(sen2vec)
        sen_out = torch.cat((sen_output, verb_output), dim=1)
        sen_out = self.sen_out_linear(sen_out)
        sen_out = self.sen_out_dropout(sen_out)
        sen_out = l2norm(sen_out)
        sen_h = torch.cat((sen_h_n, verb_h_n), dim=2)
        sen_h = self.sen_h_linear(sen_h)
        sen_h = self.sen_h_dropout(sen_h)
        sen_h = l2norm(sen_h)
        # print('sen2vec:', sen2vec.size(), sen_out.size(), sen_h.size())
        # if iter_num % 100 == 0:
        #     print('sen2vec:', sen2vec)
        return sen2vec, sen_out, sen_h

# forward処理のreturnすべき値とその処理過程(nn.linearなど)
# output_sizeなどをopts.pyにまとめる作業


class Transformer(nn.Module):
    def __init__(self, batch_size, common_size, max_seq_len, max_verb_len, weights):
        super(Transformer, self).__init__()
        self.batch_size = batch_size
        self.common_size = common_size
        self.transformer_size = common_size*4
        self.sen_embed = nn.Embedding.from_pretrained(
            embeddings=weights, freeze=True)
        self.verb_embed = nn.Embedding.from_pretrained(
            embeddings=weights, freeze=True)
        self.sen_embed_linear = nn.Linear(
            weights.size(1), self.transformer_size)
        self.sen_embed_dropout = nn.Dropout(p=0.1)
        self.verb_embed_linear = nn.Linear(
            weights.size(1), self.transformer_size)
        self.verb_embed_dropout = nn.Dropout(p=0.1)
        self.cat_embed_linear = nn.Linear(
            self.transformer_size, self.transformer_size)
        self.cat_embed_dropout = nn.Dropout(p=0.1)
        self.transformer = nn.Transformer(self.transformer_size)
        self.output_linear = nn.Linear(max_seq_len+max_verb_len, 1)
        self.output_dropout = nn.Dropout(p=0.1)
        self.sen2vec_linear = nn.Linear(
            self.transformer_size, self.common_size)
        self.sen2vec_dropout = nn.Dropout(p=0.2)
        init.xavier_uniform_(self.sen_embed_linear.weight)
        init.xavier_uniform_(self.verb_embed_linear.weight)
        init.xavier_uniform_(self.output_linear.weight)
        init.xavier_uniform_(self.sen2vec_linear.weight)

    def forward(self, iter_num, batch_sentence, batch_verb, vocab):
        sen_embed = self.sen_embed(batch_sentence)
        verb_embed = self.verb_embed(batch_verb)
        sen_embed = self.sen_embed_linear(sen_embed)
        # sen_embed = self.sen_embed_dropout(sen_embed)
        verb_embed = self.verb_embed_linear(verb_embed)
        # verb_embed = self.verb_embed_dropout(verb_embed)
        embed = torch.cat((sen_embed, verb_embed), dim=1)
        embed = self.cat_embed_linear(embed)
        embed = self.cat_embed_dropout(embed)
        ### embed.size() -> [batch_size, max_seq_len, common_size] ###
        src = embed.permute(1, 0, 2)
        ### scr.size() -> [max_seq_len, batch_size, common_size] ###
        tgt = embed.permute(1, 0, 2)
        ### tgt.size() -> [max_seq_len, batch_size, common_size] ###
        output = self.transformer(src, tgt)
        ### output.size() -> [max_seq_len, batch_size, common_size] ###
        sen2vec = self.output_linear(output.permute(1, 2, 0)).permute(2, 0, 1)
        sen2vec = l2norm(sen2vec)
        ### sen2vec.size() -> [1, batch_size, common_size] ###
        sen2vec = self.sen2vec_linear(sen2vec)
        sen2vec = self.sen2vec_dropout(sen2vec)
        sen2vec = l2norm(sen2vec)
        sen2vec = sen2vec.squeeze(dim=0)
        # print('sen2vec:', sen2vec.size(), src.size(), tgt.size(), output.size(), output.permute(1, 0, 2).size())
        # if iter_num % 100 == 0:
        #     print('sen2vec:', sen2vec)
        return sen2vec, output.permute(1, 0, 2)
