# -*- coding: utf-8 -*-
import os
import time
import datetime
import torch
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import OrderedDict

from dataset_cap import Vocabulary
from dataset_vid import MyDataset_BMN, MyDataset_GT
from model_cap import LSTM, GRU, Transformer
from model_tf import TransformerClassification
from model_vid import Vid2Vec
from model_grd import Ground
from loss_function import matmul_loss_function, cos_loss_function, PairwiseRankingLoss
from eval import temporal_IoU
from utils import sec2str, model_state_dict, collate_fn

import spacy

spacy = spacy.load('en_core_web_sm')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


start_now = datetime.datetime.now()
print(start_now.strftime('%Y/%m/%d %H:%M:%S'))
### Test ###
print('----- Test -----')
start1 = time.time()

print('Loading test dataset...')
start2 = time.time()
my_test_dataset = MyDataset_BMN(mode='test')
end2 = sec2str(time.time() - start2)
print('Finished loading test dataset. | {}'.format(end2))


test_batch_size = 1
# ['LSTM', 'GRU', 'Transformer', 'TF']
rnn_model = 'TF'
attn_mode = 'multihead'  # ['simple', 'multihead']
cap_hidden_size = vid_hidden_size = 512
cap_num_layers = vid_num_layers = 2
cap_bidirectional = vid_bidirectional = True
common_size = 256
grd_mode = 'simple'  # ['simple', 'multi']
num_workers = 4

test_data_loader = torch.utils.data.DataLoader(
    dataset=my_test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, num_workers=num_workers, collate_fn=collate_fn)

test_video_size = iter(test_data_loader).next()['video']
if len(test_video_size.size()) == 4:
    test_video_size = test_video_size.squeeze(dim=0)
test_video_size = test_video_size.size()
print(test_video_size)

min_freq = 5
max_seq_len = 60
max_verb_len = 15
vocab = Vocabulary(min_freq=min_freq, max_seq_len=max_seq_len)
vocab_proc, word_embeddings = vocab.load_vocab()

if rnn_model == 'LSTM':
    print(rnn_model, grd_mode)
    model_cap = LSTM(batch_size=test_batch_size, hidden_size=cap_hidden_size, num_layers=cap_num_layers, bidirectional=cap_bidirectional,
                     common_size=common_size, max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings).to(device)

elif rnn_model == 'GRU':
    print(rnn_model, grd_mode)
    model_cap = GRU(batch_size=test_batch_size, hidden_size=cap_hidden_size, num_layers=cap_num_layers, bidirectional=cap_bidirectional,
                    common_size=common_size, max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings).to(device)

elif rnn_model == 'Transformer':
    print(rnn_model, grd_mode)
    model_cap = Transformer(batch_size=test_batch_size, common_size=common_size,
                            max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings).to(device)

elif rnn_model == 'TF':
    print(rnn_model, attn_mode, grd_mode)
    model_cap = TransformerClassification(
        batch_size=test_batch_size, max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings, attn_mode=attn_mode).to(device)

model_vid = Vid2Vec(video_size=test_video_size, hidden_size=vid_hidden_size,
                    num_layers=vid_num_layers, bidirectional=vid_bidirectional, common_size=common_size).to(device)
model_grd = Ground(common_size=common_size, mode=grd_mode).to(device)

model_cap = torch.nn.DataParallel(model_cap, device_ids=[0])
model_vid = torch.nn.DataParallel(model_vid, device_ids=[0])
model_grd = torch.nn.DataParallel(model_grd, device_ids=[0])

cap_checkpoint = 'cap_best.pth.tar'
vid_checkpoint = 'vid_best.pth.tar'
grd_checkpoint = 'grd_best.pth.tar'

checkpoint_dir = './checkpoint'
date_dir = '2020-01-31'
# save_dir = os.path.join(checkpoint_dir, date_dir)
save_dir = os.path.join(checkpoint_dir, date_dir, 'best')
print(save_dir)

cap_state_dict = model_state_dict(save_dir, cap_checkpoint)
vid_state_dict = model_state_dict(save_dir, vid_checkpoint)
grd_state_dict = model_state_dict(save_dir, grd_checkpoint)
model_cap.load_state_dict(cap_state_dict)
model_vid.load_state_dict(vid_state_dict)
model_grd.load_state_dict(grd_state_dict)

model_cap.eval()
model_vid.eval()
model_grd.eval()

all_count = 0
gt_count = 0
gt_highest_count = 0
gt_high_count = 0
gt_mid_count = 0
gt_low_count = 0
no_gt_count = 0
true_count = 0
highest_count = 0
highest_counts = 0
high_count = 0
high_counts = 0
mid_count = 0
mid_counts = 0
low_count = 0
low_counts = 0
tIoU_th_highest = 0.7
tIoU_th_high = 0.5
tIoU_th_mid = 0.3
tIoU_th_low = 0.1
tIoU_list = []
true_tIoU_list = []
gt_tIoU_list = []
with torch.no_grad():
    for i, test_data in enumerate(test_data_loader):
        # print('i:', i)
        sentence = test_data['sentence']
        for sen in sentence:
            sen = ''.join(map(str, sen))
            sen = spacy(sen)
            verb_list = []
            for token in sen:
                # verb_list.append(token.head.text)
                if token.pos_ == 'VERB':
                    verb_list.append(token.text)

        sentence = vocab.return_idx(sentence).to(device)
        verb = vocab.return_idx([verb_list]).to(device)
        verb = verb[:, :max_verb_len]

        proposed_videos = test_data['video']
        context_video = test_data['context_video']
        if len(proposed_videos.size()) == 4:
            proposed_videos = proposed_videos.squeeze(dim=0)
            context_video = context_video.squeeze(dim=0)
        proposed_videos = proposed_videos.to(device)
        context_video = context_video.to(device)

        if rnn_model in ['LSTM', 'GRU']:
            sen2vec, sen_out, sen_h = model_cap(i, sentence, verb, vocab)
            vid2vec = model_vid(i, proposed_videos,
                                context_video, sen_out, sen_h)
            # vid2vec = model_vid(i, proposed_videos,
            #                     context_video, sen_out)
        elif rnn_model in ['Transformer', 'TF']:
            sen2vec, sen_out = model_cap(i, sentence, verb, vocab)
            vid2vec = model_vid(i, proposed_videos, context_video, sen_out)

        matmul_sim, cos_sim = model_grd(sen2vec, vid2vec)

        index = torch.argmax(cos_sim).detach().cpu().numpy()
        sentence_timestamp = test_data['sentence_timestamp'][0]
        video_timestamp = test_data['video_timestamp'][0]
        tIoU = temporal_IoU(sentence_timestamp, video_timestamp[index])
        # print('tIoU:', index, tIoU)
        gt_index = None
        gt_tIoU = -1
        num_proposals = test_data['video'].size(1)
        for j in range(num_proposals):
            each_tIoU = temporal_IoU(sentence_timestamp, video_timestamp[j])
            if each_tIoU > gt_tIoU:
                gt_tIoU = each_tIoU
                gt_index = j
        tIoU_list.append(tIoU)
        gt_tIoU_list.append(gt_tIoU)
        # print('gt_tIoU:', gt_index, gt_tIoU)
        all_count += 1
        if gt_tIoU >= tIoU_th_low:
            gt_count += 1
        if gt_index == None or gt_tIoU < tIoU_th_low:
            no_gt_count += 1
        if gt_tIoU >= tIoU_th_highest:
            gt_highest_count += 1
        if gt_tIoU >= tIoU_th_high:
            gt_high_count += 1
        if gt_tIoU >= tIoU_th_mid:
            gt_mid_count += 1
        if gt_tIoU >= tIoU_th_low:
            gt_low_count += 1
        if index == gt_index and gt_tIoU >= tIoU_th_low and tIoU >= tIoU_th_low:
            true_count += 1
            true_tIoU_list.append(tIoU)
        if index == gt_index and tIoU >= tIoU_th_highest:
            highest_count += 1
        if index == gt_index and tIoU >= tIoU_th_high:
            high_count += 1
        if index == gt_index and tIoU >= tIoU_th_mid:
            mid_count += 1
        if index == gt_index and tIoU >= tIoU_th_low:
            low_count += 1
        if tIoU >= tIoU_th_highest:
            highest_counts += 1
        if tIoU >= tIoU_th_high:
            high_counts += 1
        if tIoU >= tIoU_th_mid:
            mid_counts += 1
        if tIoU >= tIoU_th_low:
            low_counts += 1
        # print('=======================================================================================================')
        # if i+1 == 100:
        #     break
        # break
    print('GT Count: gt:{} no_gt:{} / gt_highest:{} gt_high:{} gt_mid:{} gt_low:{} / all{}'.format(
        gt_count, no_gt_count, gt_highest_count, gt_high_count, gt_mid_count, gt_low_count, all_count))
    print('Count: highest@{}: {}, high@{}: {}, mid@{}: {}, low@{}: {}, true:{} / {}'.format(
        tIoU_th_highest, highest_count, tIoU_th_high, high_count, tIoU_th_mid, mid_count, tIoU_th_low, low_count, true_count, gt_count))
    print('Rank: highest@{}: {:.2f}%, high@{}: {:.2f}%, mid@{}: {:.2f}%, low@{}: {:.2f}%, true:{:.2f}%'.format(
        tIoU_th_highest, 100*highest_count /
        gt_highest_count, tIoU_th_high, 100*high_count/gt_high_count,
        tIoU_th_mid, 100*mid_count/gt_mid_count, tIoU_th_low, 100*low_count/gt_low_count, 100*true_count/gt_count))
    print('Counts: highest@{}: {}, high@{}: {}, mid@{}: {}, low@{}: {}, true:{} / {}'.format(
        tIoU_th_highest, highest_counts, tIoU_th_high, high_counts, tIoU_th_mid, mid_counts, tIoU_th_low, low_counts, true_count, gt_count))
    print('Rank: highest@{}: {:.2f}%, high@{}: {:.2f}%, mid@{}: {:.2f}%, low@{}: {:.2f}%, true:{:.2f}%'.format(
        tIoU_th_highest, 100*highest_counts /
        gt_highest_count, tIoU_th_high, 100*high_counts/gt_high_count,
        tIoU_th_mid, 100*mid_counts/gt_mid_count, tIoU_th_low, 100*low_counts/gt_low_count, 100*true_count/gt_count))
    print('Upper Bound: highest@{}: {:.2f}%, high@{}: {:.2f}%, mid@{}: {:.2f}%, low@{}: {:.2f}%, true:{:.2f}%'.format(
        tIoU_th_highest, 100*gt_highest_count /
        all_count, tIoU_th_high, 100*gt_high_count/all_count,
        tIoU_th_mid, 100*gt_mid_count/all_count, tIoU_th_low, 100*gt_low_count/all_count, 100*gt_count/all_count))
    print('Recall: highest@{}: {:.2f}%, high@{}: {:.2f}%, mid@{}: {:.2f}%, low@{}: {:.2f}%, true:{:.2f}%'.format(
        tIoU_th_highest, 100*highest_counts /
        all_count, tIoU_th_high, 100*high_counts/all_count,
        tIoU_th_mid, 100*mid_counts/all_count, tIoU_th_low, 100*low_counts/all_count, 100*true_count/all_count))
    print('mean tIoU:{:.3f}, true mean tIoU:{:.3f}, GT mean tIoU:{:.3f} '.format(np.mean(
        np.array(tIoU_list)), np.mean(np.array(true_tIoU_list)), np.mean(np.array(gt_tIoU_list))))

print('=======================================================================================================')
all_count = 0
gt_count = 0
gt_highest_count = 0
gt_high_count = 0
gt_mid_count = 0
gt_low_count = 0
no_gt_count = 0
true_count = 0
highest_count = 0
highest_counts = 0
high_count = 0
high_counts = 0
mid_count = 0
mid_counts = 0
low_count = 0
low_counts = 0
tIoU_th_highest = 0.7
tIoU_th_high = 0.5
tIoU_th_mid = 0.3
tIoU_th_low = 0.1
random_tIoU_list = []
true_tIoU_list = []
gt_tIoU_list = []
for i, test_data in enumerate(test_data_loader):
    num_proposals = test_data['video'].size(1)
    random_index = random.randrange(0, num_proposals)
    sentence_timestamp = test_data['sentence_timestamp'][0]
    video_timestamp = test_data['video_timestamp'][0]
    random_tIoU = temporal_IoU(
        sentence_timestamp, video_timestamp[random_index])
    random_tIoU_list.append(random_tIoU)
    gt_index = None
    gt_tIoU = -1
    for j in range(num_proposals):
        each_tIoU = temporal_IoU(sentence_timestamp, video_timestamp[j])
        if each_tIoU > gt_tIoU:
            gt_tIoU = each_tIoU
            gt_index = j
    gt_tIoU_list.append(gt_tIoU)
    all_count += 1
    if gt_tIoU >= tIoU_th_low:
        gt_count += 1
    if gt_index == None or gt_tIoU < tIoU_th_low:
        no_gt_count += 1
    if gt_tIoU >= tIoU_th_highest:
        gt_highest_count += 1
    if gt_tIoU >= tIoU_th_high:
        gt_high_count += 1
    if gt_tIoU >= tIoU_th_mid:
        gt_mid_count += 1
    if gt_tIoU >= tIoU_th_low:
        gt_low_count += 1
    if random_index == gt_index and gt_tIoU >= tIoU_th_low and random_tIoU >= tIoU_th_low:
        true_count += 1
        true_tIoU_list.append(random_tIoU)
    if random_index == gt_index and random_tIoU >= tIoU_th_highest:
        highest_count += 1
    if random_index == gt_index and random_tIoU >= tIoU_th_high:
        high_count += 1
    if random_index == gt_index and random_tIoU >= tIoU_th_mid:
        mid_count += 1
    if random_index == gt_index and random_tIoU >= tIoU_th_low:
        low_count += 1
    if random_tIoU >= tIoU_th_highest:
        highest_counts += 1
    if random_tIoU >= tIoU_th_high:
        high_counts += 1
    if random_tIoU >= tIoU_th_mid:
        mid_counts += 1
    if random_tIoU >= tIoU_th_low:
        low_counts += 1
    # print('=======================================================================================================')
    # if i+1 == 100:
    #     break
    # break
print('GT Count: gt:{} no_gt:{} / gt_highest:{} gt_high:{} gt_mid:{} gt_low:{} / all{}'.format(
    gt_count, no_gt_count, gt_highest_count, gt_high_count, gt_mid_count, gt_low_count, all_count))
print('Count: highest@{}: {}, high@{}: {}, mid@{}: {}, low@{}: {}, true:{} / {}'.format(
    tIoU_th_highest, highest_count, tIoU_th_high, high_count, tIoU_th_mid, mid_count, tIoU_th_low, low_count, true_count, gt_count))
print('Rank: highest@{}: {:.2f}%, high@{}: {:.2f}%, mid@{}: {:.2f}%, low@{}: {:.2f}%, true:{:.2f}%'.format(
    tIoU_th_highest, 100*highest_count /
    gt_highest_count, tIoU_th_high, 100*high_count/gt_high_count,
    tIoU_th_mid, 100*mid_count/gt_mid_count, tIoU_th_low, 100*low_count/gt_low_count, 100*true_count/gt_count))
print('Counts: highest@{}: {}, high@{}: {}, mid@{}: {}, low@{}: {}, true:{} / {}'.format(
    tIoU_th_highest, highest_counts, tIoU_th_high, high_counts, tIoU_th_mid, mid_counts, tIoU_th_low, low_counts, true_count, gt_count))
print('Rank: highest@{}: {:.2f}%, high@{}: {:.2f}%, mid@{}: {:.2f}%, low@{}: {:.2f}%, true:{:.2f}%'.format(
    tIoU_th_highest, 100*highest_counts /
    gt_highest_count, tIoU_th_high, 100*high_counts/gt_high_count,
    tIoU_th_mid, 100*mid_counts/gt_mid_count, tIoU_th_low, 100*low_counts/gt_low_count, 100*true_count/gt_count))
print('Upper Bound: highest@{}: {:.2f}%, high@{}: {:.2f}%, mid@{}: {:.2f}%, low@{}: {:.2f}%, true:{:.2f}%'.format(
    tIoU_th_highest, 100*gt_highest_count /
    all_count, tIoU_th_high, 100*gt_high_count/all_count,
    tIoU_th_mid, 100*gt_mid_count/all_count, tIoU_th_low, 100*gt_low_count/all_count, 100*gt_count/all_count))
print('Recall: highest@{}: {:.2f}%, high@{}: {:.2f}%, mid@{}: {:.2f}%, low@{}: {:.2f}%, true:{:.2f}%'.format(
    tIoU_th_highest, 100*highest_counts /
    all_count, tIoU_th_high, 100*high_counts/all_count,
    tIoU_th_mid, 100*mid_counts/all_count, tIoU_th_low, 100*low_counts/all_count, 100*true_count/all_count))
print('mean tIoU:{:.3f}, true mean tIoU:{:.3f}, GT mean tIoU:{:.3f} '.format(np.mean(
    np.array(random_tIoU_list)), np.mean(np.array(true_tIoU_list)), np.mean(np.array(gt_tIoU_list))))


end1 = sec2str(time.time() - start1)
end_now = datetime.datetime.now()
print('Finished test.py | {} | {}'.format(
    end1, end_now.strftime('%Y/%m/%d %H:%M:%S')))
