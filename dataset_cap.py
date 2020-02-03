# -*- coding: utf-8 -*-

import os
import time

import torch
import torchtext
import spacy
import pandas as pd

from utils import sec2str

spacy = spacy.load('en_core_web_sm')


class Vocabulary():
    def __init__(self, min_freq=5, max_seq_len=85):
        super(Vocabulary, self).__init__()

        self.annotation_path = './dataset/mydataset/annotation/'
        self.annotation_file = ['train/', 'val/', 'test/']
        self.caption_all_json = ['caption_train.json',
                                 'caption_val.json', 'caption_test.json']

        self.min_freq = min_freq
        self.max_seq_len = max_seq_len
        self.text_proc = torchtext.data.Field(sequential=True, tokenize='spacy', init_token='<bos>', eos_token='<eos>',
                                              lower=True, fix_length=self.max_seq_len, batch_first=True, include_lengths=False)

    def load_json_text(self, json_file):
        print(json_file)
        json_df = pd.read_json(json_file).T
        print(len(json_df))

        max_seq_len_token = 0
        for i in json_df.index:
            sent_json_df = json_df['sentences'][i]
            for j in range(len(sent_json_df)):
                self.text_json_df.append(sent_json_df[j])
                token = sent_json_df[j].split()
                len_token = len(token)
                if max_seq_len_token < len_token:
                    max_seq_len_token = len_token

        print('max_seq_len_token:', max_seq_len_token)
        return self.text_json_df

    def load_vocab(self):
        time_start = time.time()
        print('building vocabulary...', flush=True)

        self.text_json_df = []
        for i in range(len(self.annotation_file)):
            sentences = self.load_json_text(os.path.join(
                self.annotation_path, self.annotation_file[i], self.caption_all_json[i]))
        # print(sentences[:3])

        sent_proc = list(map(self.text_proc.preprocess, sentences))
        # print(sent_proc[:3])
        print('number of sentences:', len(sent_proc))

        # self.text_proc.build_vocab(sent_proc, min_freq=self.min_freq)
        self.text_proc.build_vocab(
            sent_proc, min_freq=self.min_freq, vectors=torchtext.vocab.GloVe(name='840B', dim=300))
        vocab_proc = self.text_proc.vocab
        # print('最頻出単語top10:', self.text_proc.vocab.freqs.most_common(10))

        word_embeddings = self.text_proc.vocab.vectors
        # print('self.text_proc.vocab.vectors.size():',
        #       self.text_proc.vocab.vectors.size())

        self.len = len(self.text_proc.vocab)
        self.padidx = self.text_proc.vocab.stoi['<pad>']
        print("done building vocabulary, minimum frequency is {} times".format(
            self.min_freq), flush=True)
        print("# of words in vocab: {} | {}".format(
            self.len, sec2str(time.time()-time_start)), flush=True)
        print(
            '================================================================================')
        return vocab_proc, word_embeddings

    def return_idx(self, sentence_batch):
        out = []
        preprocessed = list(map(self.text_proc.preprocess, sentence_batch))
        out = self.text_proc.process(preprocessed)
        return out

    # return sentence batch from indexes from torch.LongTensor
    def return_sentences(self, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.tolist()
        out = []
        for idxs in tensor:
            tokenlist = [self.text_proc.vocab.itos[idx] for idx in idxs]
            out.append(" ".join(tokenlist))
        return out

    def __len__(self):
        return self.len


if __name__ == '__main__':
    start = time.time()
    vocab = Vocabulary()
    vocab.load_vocab()
    print('=========================================')
    sentence = ["The cat and the hat sat on a mat in tokyo."]
    ten = vocab.return_idx(sentence)
    print(ten)
    print('=======================================================')
    sent = vocab.return_sentences(ten)
    print(sent)
    print(len(vocab))
    end = time.time() - start
    print('{} s'.format(end))


# gloveのベクトルを初期値としてそこからちゃんと学習できているか
