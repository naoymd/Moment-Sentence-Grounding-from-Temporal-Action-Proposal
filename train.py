# -*- coding: utf-8 -*-
import os
import time
import datetime
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

from make_myfile import MyFile, MyTestFile
from make_dataset import MakeDataset
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
start0 = time.time()
print('Setting dataset...')
MyFile()
MyTestFile()
MakeDataset()
end0 = sec2str(time.time() - start0)
print('Finished setting dataset. | {}'.format(end0))
print('================================================================================')

print('----- Train & Validation -----')
start1 = time.time()

print('Loading train dataset...')
start2 = time.time()
train_dataset = MyDataset_BMN(mode='train')
end2 = sec2str(time.time() - start2)
print('Finished loading train dataset. | {}'.format(end2))

print('Loading validation dataset...')
start3 = time.time()
val_dataset = MyDataset_BMN(mode='val')
end3 = sec2str(time.time() - start3)
print('Finished loading validation dataset. | {}'.format(end3))
print('================================================================================')


train_batch_size = 256
val_batch_size = 1
epoch_num = 150  # 10, 30, 50, 100, 150, 200, 400
# ['LSTM', 'GRU', 'Transformer', 'TF']
rnn_model = 'TF'
attn_mode = 'multihead'  # ['simple', 'multihead']
cap_hidden_size = vid_hidden_size = 512
cap_num_layers = vid_num_layers = 2
cap_bidirectional = vid_bidirectional = True
common_size = 256
grd_mode = 'simple'  # ['simple', 'multi']
num_workers = 4
learning_rate = math.sqrt(1e-7)  # lr=1e-3, 1e-4, (1e-5)
min_learning_rate = math.sqrt(1e-12)
patience = 30  # 2, 3, 4, 5, 10, 20, 25, 30, 50
cooldown = 0
rank_loss_method = 'sum'  # 'sum', 'max'
alpha_cos = torch.tensor(math.exp(-2)).to(device)
alpha_rank = torch.tensor(math.exp(0)).to(device)

train_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=num_workers, collate_fn=collate_fn)
val_data_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=True, num_workers=num_workers, collate_fn=collate_fn)


train_video_size = iter(train_data_loader).next()['video']
if len(train_video_size.size()) == 4:
    train_video_size = train_video_size.squeeze(dim=0)
train_video_size = train_video_size.size()
print('train_video_size:', train_video_size)
val_video_size = iter(val_data_loader).next()['video']
if len(val_video_size.size()) == 4:
    val_video_size = val_video_size.squeeze(dim=0)
val_video_size = val_video_size.size()
print('val_video_size:', val_video_size)

min_freq = 5
max_seq_len = 60
max_verb_len = 15
vocab = Vocabulary(min_freq=min_freq, max_seq_len=max_seq_len)
vocab_proc, word_embeddings = vocab.load_vocab()


if rnn_model == 'LSTM':
    print(rnn_model, grd_mode, rank_loss_method)
    model_cap = LSTM(batch_size=train_batch_size, hidden_size=cap_hidden_size, num_layers=cap_num_layers, bidirectional=cap_bidirectional,
                     common_size=common_size, max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings).to(device)
    model_cap_val = LSTM(batch_size=val_batch_size, hidden_size=cap_hidden_size, num_layers=cap_num_layers, bidirectional=cap_bidirectional,
                         common_size=common_size, max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings).to(device)

elif rnn_model == 'GRU':
    print(rnn_model, grd_mode, rank_loss_method)
    model_cap = GRU(batch_size=train_batch_size, hidden_size=cap_hidden_size, num_layers=cap_num_layers, bidirectional=cap_bidirectional,
                    common_size=common_size, max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings).to(device)
    model_cap_val = GRU(batch_size=val_batch_size, hidden_size=cap_hidden_size, num_layers=cap_num_layers, bidirectional=cap_bidirectional,
                        common_size=common_size, max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings).to(device)

elif rnn_model == 'Transformer':
    print(rnn_model, grd_mode, rank_loss_method)
    model_cap = Transformer(batch_size=train_batch_size, common_size=common_size,
                            max_seq_len=max_seq_len,  max_verb_len=max_verb_len, weights=word_embeddings).to(device)
    model_cap_val = Transformer(batch_size=val_batch_size, common_size=common_size,
                                max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings).to(device)

elif rnn_model == 'TF':
    print(rnn_model, attn_mode, grd_mode, rank_loss_method)
    model_cap = TransformerClassification(
        batch_size=train_batch_size, max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings, attn_mode=attn_mode).to(device)
    model_cap_val = TransformerClassification(
        batch_size=val_batch_size, max_seq_len=max_seq_len, max_verb_len=max_verb_len, weights=word_embeddings, attn_mode=attn_mode).to(device)


model_vid = Vid2Vec(video_size=train_video_size, hidden_size=vid_hidden_size,
                    num_layers=vid_num_layers, bidirectional=vid_bidirectional, common_size=common_size).to(device)
model_vid_val = Vid2Vec(video_size=val_video_size, hidden_size=vid_hidden_size,
                        num_layers=vid_num_layers, bidirectional=vid_bidirectional, common_size=common_size).to(device)
model_grd = Ground(common_size=common_size, mode=grd_mode).to(device)
model_grd_val = Ground(common_size=common_size, mode=grd_mode).to(device)

model_cap = torch.nn.DataParallel(model_cap, device_ids=None)
model_vid = torch.nn.DataParallel(model_vid, device_ids=None)
model_grd = torch.nn.DataParallel(model_grd, device_ids=[0])
model_cap_val = torch.nn.DataParallel(model_cap_val, device_ids=[0])
model_vid_val = torch.nn.DataParallel(model_vid_val, device_ids=[0])
model_grd_val = torch.nn.DataParallel(model_grd_val, device_ids=[0])

param_list = [
    {'params': model_cap.parameters()},
    {'params': model_vid.parameters()},
    {'params': model_grd.parameters()}
]
optimizer = torch.optim.Adam(
    param_list, lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=math.sqrt(0.1), patience=patience, verbose=True, cooldown=cooldown, min_lr=min_learning_rate)


cap_checkpoint = 'cap_best.pth.tar'
vid_checkpoint = 'vid_best.pth.tar'
grd_checkpoint = 'grd_best.pth.tar'

rec_loss = []
rec_acc = []
rec_mean_tIoU = []
rec_val_acc = []
rec_val_loss = []
save_date = datetime.date.today().strftime('%Y-%m-%d')
save_dir = os.path.join('./checkpoint', save_date)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
mvp_count = 0
best_loss = 1e10
best_val_acc = 0
best_save_dir = os.path.join('./checkpoint', save_date, 'best')
if not os.path.exists(best_save_dir):
    os.makedirs(best_save_dir)
for epoch in range(epoch_num):
    ### Train ###
    start_train = time.time()
    # print('epoch:', epoch)
    best_count = 0
    worst_count = train_batch_size
    model_cap.train()
    model_vid.train()
    model_grd.train()
    print('--- Train ---')
    for iter_num, batch_data in enumerate(train_data_loader):
        # print('epoch: {}, iter: {}'.format(epoch, iter_num))
        batch_sentence = batch_data['sentence']
        batch_verb_list = []
        for sentence in batch_sentence:
            sentence = ''.join(map(str, sentence))
            sentence = spacy(sentence)
            verb_list = []
            for token in sentence:
                # verb_list.append(token.head.text)
                if token.pos_ == 'VERB':
                    verb_list.append(token.text)
            batch_verb_list.append(verb_list)
        batch_sentence = vocab.return_idx(batch_sentence).to(device)
        batch_verb = vocab.return_idx(batch_verb_list).to(device)
        batch_verb = batch_verb[:, :max_verb_len]

        batch_video = batch_data['video']
        batch_context_video = batch_data['context_video']
        if isinstance(batch_video, list):
            batch_video = torch.cat([batch_video[i]
                                     for i in range(len(batch_video))], dim=0)
            batch_context_video = torch.cat([batch_context_video[i]
                                             for i in range(len(batch_video))], dim=0)
        batch_video = batch_video.to(device)
        batch_context_video = batch_context_video.to(device)

        optimizer.zero_grad()

        if rnn_model in ['LSTM', 'GRU']:
            sen2vec, sen_out, sen_h = model_cap(
                iter_num, batch_sentence, batch_verb, vocab)
            vid2vec = model_vid(iter_num, batch_video,
                                batch_context_video, sen_out, sen_h)
            # vid2vec = model_vid(iter_num, batch_video,
            #                     batch_context_video, sen_out)
        elif rnn_model in ['Transformer', 'TF']:
            sen2vec, sen_out = model_cap(
                iter_num, batch_sentence, batch_verb, vocab)
            vid2vec = model_vid(iter_num, batch_video,
                                batch_context_video, sen_out)

        matmul_sim, cos_sim = model_grd(sen2vec, vid2vec)

        count = 0
        for i in range(train_batch_size):
            index = torch.argmax(matmul_sim[i])
            if i == index:
                count += 1
        if count >= best_count:
            # print('epoch: {}, iter: {}, Accuracy: {} / {}'.format(epoch,
            #                                                       iter_num, count, train_batch_size))
            best_count = count
        if count < worst_count:
            worst_count = count

        # cos_loss = cos_loss_function(cos_sim).to(device)
        cos_loss = alpha_cos * cos_loss_function(cos_sim).to(device)
        ranking_loss_function = PairwiseRankingLoss(
            method=rank_loss_method).to(device)
        # rank_loss = ranking_loss_function(matmul_sim)
        rank_loss = alpha_rank * ranking_loss_function(matmul_sim)
        loss = cos_loss + rank_loss

        loss.backward()
        optimizer.step()

        if iter_num % 25 == 0:
            rec_loss.append(loss.detach().cpu().numpy())
            rec_acc.append(count)
            print('-------------------------------------------------------------------')
            print('epoch: {}, iter: {}'.format(epoch, iter_num))
            print('Worst Accuracy: {} / {}'.format(worst_count, train_batch_size))
            print('Best Accuracy: {} / {}'.format(best_count, train_batch_size))
            print('cos_loss:', cos_loss)
            print('rank_loss:', rank_loss)
            print('total_loss:', loss)
            # print('sen2vec')
            # print(sen2vec)
            # print('vid2vec')
            # print(vid2vec)
            print('-------------------------------------------------------------------')

        if loss < best_loss:
            best_loss = loss
            torch.save(model_cap.state_dict(), os.path.join(
                save_dir, 'cap_best.pth.tar'))
            torch.save(model_vid.state_dict(), os.path.join(
                save_dir, 'vid_best.pth.tar'))
            torch.save(model_grd.state_dict(), os.path.join(
                save_dir, 'grd_best.pth.tar'))
        if best_count >= mvp_count:
            mvp_count = best_count
            torch.save(model_cap.state_dict(), os.path.join(
                save_dir, 'cap_best.pth.tar'))
            torch.save(model_vid.state_dict(), os.path.join(
                save_dir, 'vid_best.pth.tar'))
            torch.save(model_grd.state_dict(), os.path.join(
                save_dir, 'grd_best.pth.tar'))
        # break

    lr_scheduler.step(loss)
    end_train = sec2str(time.time() - start_train)
    print('Epoch: {} |Train: {}'.format(epoch, end_train))
    print('================================================================================')

    ### Validation ###
    if (epoch+1) % 5 == 1 or (epoch+1) == epoch_num:
        print('--- Validation ---')
        start_val = time.time()
        cap_state_dict = model_state_dict(save_dir, cap_checkpoint)
        vid_state_dict = model_state_dict(save_dir, vid_checkpoint)
        grd_state_dict = model_state_dict(save_dir, grd_checkpoint)
        model_cap_val.load_state_dict(cap_state_dict)
        model_vid_val.load_state_dict(vid_state_dict)
        model_grd_val.load_state_dict(grd_state_dict)
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
        val_loss_list = []
        model_cap_val.eval()
        model_vid_val.eval()
        model_grd_val.eval()
        with torch.no_grad():
            for i, val_data in enumerate(val_data_loader):
                # print('i:', i)
                sentence = val_data['sentence']
                for sen in sentence:
                    sen = ''.join(map(str, sen))
                    sen = spacy(sen)
                    verb_list = []
                    for token in sen:
                        if token.pos_ == 'VERB':
                            verb_list.append(token.text)

                sentence = vocab.return_idx(sentence).to(device)
                verb = vocab.return_idx([verb_list]).to(device)
                verb = verb[:, :max_verb_len]

                proposed_videos = val_data['video']
                context_video = val_data['context_video']
                if len(proposed_videos.size()) == 4:
                    proposed_videos = proposed_videos.squeeze(dim=0)
                    context_video = context_video.squeeze(dim=0)
                proposed_videos = proposed_videos.to(device)
                context_video = context_video.to(device)

                if rnn_model in ['LSTM', 'GRU']:
                    sen2vec, sen_out, sen_h = model_cap_val(
                        iter_num, sentence, verb, vocab)
                    vid2vec = model_vid_val(
                        iter_num, proposed_videos, context_video, sen_out, sen_h)
                    # vid2vec = model_vid_val(
                    #     iter_num, proposed_videos, context_video, sen_out)
                elif rnn_model in ['Transformer', 'TF']:
                    sen2vec, sen_out = model_cap_val(
                        iter_num, sentence, verb, vocab)
                    vid2vec = model_vid_val(
                        iter_num, proposed_videos, context_video, sen_out)

                matmul_sim, cos_sim = model_grd_val(sen2vec, vid2vec)

                # cos_loss = cos_loss_function(cos_sim).to(device)
                cos_loss = alpha_cos * cos_loss_function(cos_sim).to(device)
                ranking_loss_function = PairwiseRankingLoss(
                    method=rank_loss_method).to(device)
                # rank_loss = ranking_loss_function(matmul_sim)
                rank_loss = alpha_rank * ranking_loss_function(matmul_sim)
                val_loss = cos_loss + rank_loss
                val_loss_list.append(val_loss.detach().cpu().numpy())

                index = torch.argmax(cos_sim).detach().cpu().numpy()
                sentence_timestamp = val_data['sentence_timestamp'][0]
                video_timestamp = val_data['video_timestamp'][0]
                tIoU = temporal_IoU(sentence_timestamp, video_timestamp[index])
                # print('tIoU:', index, tIoU)
                gt_index = None
                gt_tIoU = -1
                num_proposals = val_data['video'].size(1)
                for j in range(num_proposals):
                    each_tIoU = temporal_IoU(
                        sentence_timestamp, video_timestamp[j])
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
                if i+1 == 1000:
                    break
            print('-------------------------------------------------------------------')
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
            print('mean val loss: {:.5f}'.format(
                np.mean(np.array(val_loss_list))))
            print('-------------------------------------------------------------------')
            mean_tIoU = 100 * np.mean(np.array(tIoU_list))
            rec_mean_tIoU.append(mean_tIoU)
            rec_val_acc.append(100*true_count/gt_count)
            mean_val_loss = np.mean(np.array(val_loss_list))
            rec_val_loss.append(mean_val_loss)
        val_acc = tIoU_th_highest*highest_count + tIoU_th_high * \
            high_count + tIoU_th_mid*mid_count + tIoU_th_low*low_count
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model_cap.state_dict(), os.path.join(
                best_save_dir, cap_checkpoint))
            torch.save(model_vid.state_dict(), os.path.join(
                best_save_dir, vid_checkpoint))
            torch.save(model_grd.state_dict(), os.path.join(
                best_save_dir, grd_checkpoint))
        end_val = sec2str(time.time() - start_val)
        print('Epoch: {} |Validation: {}'.format(epoch, end_val))
    end_train_val = sec2str(time.time() - start_train)
    print('Epoch: {} |Train & Validation: {}'.format(epoch, end_train_val))
    print('================================================================================')


result_dir = os.path.join('./result', save_date)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
plt.figure()
plt.plot(rec_loss)
plt.savefig(os.path.join(result_dir, 'loss.png'))
plt.figure()
plt.hist(rec_acc, stacked=True)
l = [i for i in range(len(rec_acc))]
plt.savefig(os.path.join(result_dir, 'hist.png'))
plt.figure()
plt.scatter(l, rec_acc)
plt.savefig(os.path.join(result_dir, 'scatter.png'))
plt.figure()
plt.plot(rec_mean_tIoU)
plt.savefig(os.path.join(result_dir, 'mean_tIoU.png'))
plt.figure()
plt.plot(rec_val_acc)
plt.savefig(os.path.join(result_dir, 'true_count.png'))
plt.figure()
plt.plot(rec_val_loss)
plt.savefig(os.path.join(result_dir, 'val_loss.png'))
print('max_true_count:', max(rec_val_acc))
end1 = sec2str(time.time() - start1)
end_now = datetime.datetime.now()
print('Finished train.py. | {} | {}'.format(
    end1, end_now.strftime('%Y/%m/%d %H:%M:%S')))
