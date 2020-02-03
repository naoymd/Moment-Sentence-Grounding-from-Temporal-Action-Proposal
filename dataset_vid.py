# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import glob
import time
import math
import ast

from utils import collate_fn


class MyDataset_BMN(torch.utils.data.Dataset):
    ### pooling_size = [400, 1], [400, 2], [400, 4], [400, 5], o[400, 10], [400, 20], [400, 25] [400, 100]###
    def __init__(self, mode='train', transform=None, pooling_size=(400, 100)):
        super(MyDataset_BMN, self).__init__()
        self.temporal_scale = 100
        if mode in ['train', 'val', 'test', 'train-val']:
            self.mode = mode
        else:
            print(
                "mode must be specified. ['train', 'val', 'test', 'train-val]")

        self.bmn_video_path = './dataset/mydataset/video/'
        self.train_dataset_path = './dataset/mydataset/train'
        self.val_dataset_path = './dataset/mydataset/val'
        self.test_dataset_path = './dataset/mydataset/test'
        self.train_val_dataset_path = './dataset/mydataset/train-val/'

        self.c3d_video_path = './dataset/mydataset/C3D/c3d-feat'
        self.train_c3d_path = './dataset/mydataset/C3D/train/'
        self.val_c3d_path = './dataset/mydataset/C3D/val/'
        self.test_c3d_path = './dataset/mydataset/C3D/test/'
        self.train_val_c3d_path = './dataset/mydataset/C3D/train-val/'

        self.bmn_results_path = './dataset/mydataset/BMN_results'
        self.bmn_results_all_json = ['bmn_result_train.json',
                                     'bmn_result_val.json', 'bmn_result_test.json']

        self.proposals_threshold = 5
        self.context_sampling = 20

        self.transform = transform
        self.pooling_size = pooling_size

        if self.mode == 'train':
            self.dataset_list = self._get_dataset_list(self.train_dataset_path)
            self.dataset_list = self._get_dataset_list(self.train_dataset_path)

        elif self.mode == 'val':
            self.dataset_list = self._get_dataset_list(self.val_dataset_path)
            # self.dataset_list = self._get_dataset_list(self.train_dataset_path)

        elif self.mode == 'test':
            self.dataset_list = self._get_dataset_list(self.test_dataset_path)

        else:
            self.dataset_list = self._get_dataset_list(
                self.train_val_dataset_path)

    def _get_dataset_list(self, datasets_path):
        dataset_list = []
        count = 0
        for dataset_path in glob.glob(os.path.join(datasets_path, '*.csv')):
            # print(count, dataset_path)
            count += 1
            df = pd.read_csv(dataset_path, index_col=0)
            for i in range(len(df['number'])):
                df_dict = df.iloc[i].to_dict()
                dataset_list.append(df_dict)
            # if count == 100:
            #     break
        return dataset_list

    def _trim_video(self, video_data, duration, timestamp):
        if isinstance(timestamp, str):
            timestamp = ast.literal_eval(timestamp)
        start = math.floor(self.temporal_scale * timestamp[0] / duration)
        end = math.ceil(self.temporal_scale * timestamp[1] / duration)
        if start > end:
            start, end = end, start
        if start < 0:
            start = 0
        if end > self.temporal_scale:
            end = self.temporal_scale
        video_data_part = video_data[start: (end+1)]
        video_data_part = torch.t(video_data_part)
        video_data_part = video_data_part.unsqueeze(
            dim=0).unsqueeze(dim=0)
        # print(start, end)
        if end - start >= self.pooling_size[1]:
            # print('GAP')
            avg_pool = torch.nn.AdaptiveAvgPool2d(
                self.pooling_size)  # [400, n]->pooling_size
            trim_video_data = avg_pool(video_data_part)
            # print(trim_video_data.size())
        else:
            # print('ZeroPad')
            zero_pad = torch.nn.ZeroPad2d(
                (0, self.pooling_size[1]-video_data_part.size()[3], 0, 0))  # [400, n]->pooling_size
            trim_video_data = zero_pad(video_data_part)
            # print(trim_video_data.size())
        trim_video_data = trim_video_data.squeeze(dim=0).squeeze(dim=0)
        assert trim_video_data.size()[1] == self.pooling_size[1]
        return trim_video_data

    def _context_video(self, video_data, duration, timestamp):
        if isinstance(timestamp, str):
            timestamp = ast.literal_eval(timestamp)
        start = self.context_sampling * \
            math.floor(
                ((self.temporal_scale * timestamp[0] / duration) - 1)/self.context_sampling)
        end = self.context_sampling * \
            math.ceil(
                ((self.temporal_scale * timestamp[1] / duration) + 1)/self.context_sampling)
        # start = self.context_sampling * \
        #     math.floor(
        #         ((self.temporal_scale * timestamp[0] / duration) - (self.temporal_scale/self.context_sampling))/self.context_sampling)
        # end = self.context_sampling * math.ceil(((self.temporal_scale * timestamp[1] / duration) + (
        #     self.temporal_scale/self.context_sampling))/self.context_sampling)
        if start > end:
            start, end = end, start
        if start < 0:
            start = 0
        if end > self.temporal_scale:
            end = self.temporal_scale

        video_data_part = video_data[start: (end+1)]
        video_data_part = torch.t(video_data_part)
        video_data_part = video_data_part.unsqueeze(
            dim=0).unsqueeze(dim=0)
        # print(start, end)
        if end - start >= self.pooling_size[1]:
            # print('GAP')
            avg_pool = torch.nn.AdaptiveAvgPool2d(
                self.pooling_size)  # [400, n]->pooling_size
            trim_video_data = avg_pool(video_data_part)
            # print(trim_video_data.size())
        else:
            # print('ZeroPad')
            zero_pad = torch.nn.ZeroPad2d(
                (0, self.pooling_size[1]-video_data_part.size()[3], 0, 0))  # [400, n]->pooling_size
            trim_video_data = zero_pad(video_data_part)
            # print(trim_video_data.size())
        trim_video_data = trim_video_data.squeeze(dim=0).squeeze(dim=0)
        assert trim_video_data.size()[1] == self.pooling_size[1]
        return trim_video_data

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'train-val':
            dataset = self.dataset_list[index]
            # number = dataset['number']
            video_id = dataset['video_name']
            video_path = dataset['video_path']
            sentence = dataset['sentence']
            timestamp = dataset['timestamp']
            if isinstance(timestamp, str):
                timestamp = ast.literal_eval(timestamp)
            duration = dataset['duration']
            video_data_df = pd.read_csv(video_path)
            full_video = video_data_df.values[:, :]
            full_video = torch.tensor(full_video, dtype=torch.float32)
            trim_video = self._trim_video(full_video, duration, timestamp)
            context_video = self._context_video(
                full_video, duration, timestamp)
            if self.transform:
                context_video = self.transform(context_video)
                trim_video = self.transform(trim_video)
            dataset = {'video_id': video_id, 'context_video': context_video, 'video': trim_video, 'video_timestamp': [timestamp], 'sentence': sentence,
                       'sentence_timestamp': timestamp}
            return dataset

        else:
            dataset = self.dataset_list[index]
            video_id = dataset['video_name']
            video_path = dataset['video_path']
            sentence = dataset['sentence']
            sentence_timestamp = dataset['timestamp']
            if isinstance(sentence_timestamp, str):
                sentence_timestamp = ast.literal_eval(sentence_timestamp)

            if self.mode == 'val':
                bmn_results_df = pd.read_json(os.path.join(
                    self.bmn_results_path, self.bmn_results_all_json[1]))
                # bmn_results_df = pd.read_json(os.path.join(
                #     self.bmn_results_path, self.bmn_results_all_json[0]))
            else:
                bmn_results_df = pd.read_json(os.path.join(
                    self.bmn_results_path, self.bmn_results_all_json[2]))

            bmn_result_df = bmn_results_df.loc[video_id]['results'][: self.proposals_threshold]
            video_timestamp = [bmn_result_df[i]['segment']
                               for i in range(len(bmn_result_df))]

            duration = dataset['duration']
            video_data_df = pd.read_csv(video_path)
            full_video = video_data_df.values[:, :]
            full_video = torch.tensor(full_video, dtype=torch.float32)
            video = torch.stack([self._trim_video(full_video, duration, video_timestamp[i])
                                 for i in range(self.proposals_threshold)], dim=0)
            context_videos = torch.stack([self._context_video(full_video, duration, video_timestamp[i])
                                          for i in range(self.proposals_threshold)], dim=0)

            if self.transform:
                context_videos = self.transform(context_videos)
                video = self.transform(video)
            dataset = {'video_id': video_id, 'context_video': context_videos, 'video': video, 'video_timestamp': video_timestamp,
                       'sentence': sentence, 'sentence_timestamp': sentence_timestamp}
            return dataset

    def __len__(self):
        return len(self.dataset_list)


class MyDataset_GT(torch.utils.data.Dataset):
    ### pooling_size = [400, 1], [400, 2], [400, 4], [400, 5], o[400, 10], [400, 20], [400, 25] [400, 100]###
    def __init__(self, mode='train', transform=None, pooling_size=(400, 100)):
        super(MyDataset_GT, self).__init__()
        self.temporal_scale = 100
        if mode in ['train', 'val', 'test', 'train-val']:
            self.mode = mode
        else:
            print(
                "mode must be specified. ['train', 'val', 'test', 'train-val]")

        self.bmn_video_path = './dataset/mydataset/video/'
        self.train_dataset_path = './dataset/mydataset/train'
        self.val_dataset_path = './dataset/mydataset/val'
        self.test_dataset_path = './dataset/mydataset/test'
        self.train_val_dataset_path = './dataset/mydataset/train-val/'

        self.bmn_results_path = './dataset/mydataset/BMN_results'
        self.bmn_results_all_json = ['bmn_result_train.json',
                                     'bmn_result_val.json', 'bmn_result_test.json']
        self.proposals_threshold = 30
        self.context_sampling = 20

        self.transform = transform
        self.pooling_size = pooling_size

        if self.mode == 'train':
            self.dataset_list = self._get_dataset_list(self.train_dataset_path)

        elif self.mode == 'val':
            self.dataset_list = self._get_dataset_list(self.train_dataset_path)
#             self.dataset_list = self._get_dataset_list(self.val_dataset_path)

        elif self.mode == 'test':
            self.dataset_list = self._get_dataset_list(self.test_dataset_path)

        else:
            self.dataset_list = self._get_dataset_list(
                self.train_val_dataset_path)

    def _get_dataset_list(self, datasets_path):
        dataset_list = []
        count = 0
        for dataset_path in glob.glob(os.path.join(datasets_path, '*.csv')):
            # print(count, dataset_path)
            count += 1
            df = pd.read_csv(dataset_path, index_col=0)
            if self.mode == 'train' or self.mode == 'train-val':
                for i in range(len(df['number'])):
                    df_dict = df.iloc[i].to_dict()
                    dataset_list.append(df_dict)
            else:
                for i in range(len(df['number'])):
                    df_sentence = df['sentence'][i]
                    df_sentence_timestamp = df['timestamp'][i]
                    df_dict = df.to_dict()
                    df_dict['sentence'] = df_sentence
                    df_dict['sentence_timestamp'] = df_sentence_timestamp
                    dataset_list.append(df_dict)
            # if count == 100:
            #     break
        return dataset_list

    def _trim_video(self, video_data, duration, timestamp):
        if isinstance(timestamp, str):
            timestamp = ast.literal_eval(timestamp)
        start = math.floor(self.temporal_scale * timestamp[0] / duration)
        end = math.ceil(self.temporal_scale * timestamp[1] / duration)
        if start > end:
            start, end = end, start
        if start < 0:
            start = 0
        if end > self.temporal_scale:
            end = self.temporal_scale
        video_data_tensor_part = video_data[start: (end+1)]
        video_data_tensor_part = torch.t(video_data_tensor_part)
        video_data_tensor_part = video_data_tensor_part.unsqueeze(
            dim=0).unsqueeze(dim=0)
        # print(start, end)
        if end - start >= self.pooling_size[1]:
            # print('GAP')
            avg_pool = torch.nn.AdaptiveAvgPool2d(
                self.pooling_size)  # [400, n]->pooling_size
            trim_video_data = avg_pool(video_data_tensor_part)
            # print(trim_video_data.size())
        else:
            # print('ZeroPad')
            zero_pad = torch.nn.ZeroPad2d(
                (0, self.pooling_size[1]-video_data_tensor_part.size()[3], 0, 0))  # [400, n]->pooling_size
            trim_video_data = zero_pad(video_data_tensor_part)
            # print(trim_video_data.size())
        trim_video_data = trim_video_data.squeeze(dim=0).squeeze(dim=0)
        assert trim_video_data.size()[1] == self.pooling_size[1]
        return trim_video_data

    def _context_video(self, video_data, duration, timestamp):
        if isinstance(timestamp, str):
            timestamp = ast.literal_eval(timestamp)
        start = self.context_sampling * \
            math.floor(
                ((self.temporal_scale * timestamp[0] / duration) - 1)/self.context_sampling)
        end = self.context_sampling * \
            math.ceil(
                ((self.temporal_scale * timestamp[1] / duration) + 1)/self.context_sampling)
        # start = self.context_sampling * \
        #     math.floor(
        #         ((self.temporal_scale * timestamp[0] / duration) - (self.temporal_scale/self.context_sampling))/self.context_sampling)
        # end = self.context_sampling * math.ceil(((self.temporal_scale * timestamp[1] / duration) + (
        #     self.temporal_scale/self.context_sampling))/self.context_sampling)
        if start > end:
            start, end = end, start
        if start < 0:
            start = 0
        if end > self.temporal_scale:
            end = self.temporal_scale

        video_data_part = video_data[start: (end+1)]
        video_data_part = torch.t(video_data_part)
        video_data_part = video_data_part.unsqueeze(
            dim=0).unsqueeze(dim=0)
        # print(start, end)
        if end - start >= self.pooling_size[1]:
            # print('GAP')
            avg_pool = torch.nn.AdaptiveAvgPool2d(
                self.pooling_size)  # [400, n]->pooling_size
            trim_video_data = avg_pool(video_data_part)
            # print(trim_video_data.size())
        else:
            # print('ZeroPad')
            zero_pad = torch.nn.ZeroPad2d(
                (0, self.pooling_size[1]-video_data_part.size()[3], 0, 0))  # [400, n]->pooling_size
            trim_video_data = zero_pad(video_data_part)
            # print(trim_video_data.size())
        trim_video_data = trim_video_data.squeeze(dim=0).squeeze(dim=0)
        assert trim_video_data.size()[1] == self.pooling_size[1]
        return trim_video_data

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'train-val':
            dataset = self.dataset_list[index]
            # number = dataset['number']
            video_id = dataset['video_name']
            video_path = dataset['video_path']
            sentence = dataset['sentence']
            timestamp = dataset['timestamp']
            if isinstance(timestamp, str):
                timestamp = ast.literal_eval(timestamp)
            duration = dataset['duration']
            video_data_df = pd.read_csv(video_path)
            full_video = video_data_df.values[:, :]
            full_video = torch.tensor(full_video, dtype=torch.float32)
            trim_video = self._trim_video(full_video, duration, timestamp)
            context_video = self._context_video(
                full_video, duration, timestamp)
            if self.transform:
                context_video = self.transform(context_video)
                trim_video = self.transform(trim_video)
            dataset = {'video_id': video_id, 'context_video': context_video, 'video': trim_video, 'video_timestamp': [timestamp], 'sentence': sentence,
                       'sentence_timestamp': timestamp}
            return dataset

        else:
            dataset = self.dataset_list[index]
            video_id = dataset['video_name'][0]
            video_path = dataset['video_path'][0]
            sentence = dataset['sentence']
            video_timestamp = dataset['timestamp']
            sentence_timestamp = dataset['sentence_timestamp']
            if isinstance(sentence_timestamp, str):
                sentence_timestamp = ast.literal_eval(sentence_timestamp)
            if len(video_timestamp) <= self.proposals_threshold:
                for i in range(len(video_timestamp)):
                    video_timestamp[i] = ast.literal_eval(video_timestamp[i])
                for i in range(len(video_timestamp), self.proposals_threshold):
                    video_timestamp[i] = list([0, 0])
            video_timestamp_list = [video_timestamp[i]
                                    for i in range(len(video_timestamp))]

            duration = dataset['duration'][0]
            video_data_df = pd.read_csv(video_path)
            full_video = video_data_df.values[:, :]
            full_video = torch.tensor(full_video, dtype=torch.float32)
            video = torch.stack([self._trim_video(full_video, duration, video_timestamp[i])
                                 for i in range(len(video_timestamp))], dim=0)
            context_videos = torch.stack([self._context_video(full_video, duration, video_timestamp[i])
                                          for i in range(len(video_timestamp))], dim=0)

            if self.transform:
                context_video = self.transform(context_video)
                video = self.transform(video)
            dataset = {'video_id': video_id, 'context_video': context_videos, 'video': video, 'video_timestamp': video_timestamp_list,
                       'sentence': sentence, 'sentence_timestamp': sentence_timestamp}
            return dataset

    def __len__(self):
        return len(self.dataset_list)


if __name__ == '__main__':
    start = time.time()
    my_dataset = MyDataset_BMN()
    my_test_dataset = MyDataset_BMN(mode='test')
    print('~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=')
    end = time.time() - start
    print('{} s'.format(end))
    print('--------------------------------------------------------')
    # print(my_dataset[0])
    print(len(my_dataset))
    print(len(my_test_dataset))
    print('=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~')
    train_data_loader = torch.utils.data.DataLoader(
        my_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=2, collate_fn=collate_fn)
    for i, data in enumerate(train_data_loader):
        print(i, data['video_id'], data['context_video'].size(), data['video'].size(
        ), data['video_timestamp'], data['sentence'], data['sentence_timestamp'])
        break
    print('=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~')
    test_data_loader = torch.utils.data.DataLoader(
        my_test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=2, collate_fn=collate_fn)
    for i, data in enumerate(test_data_loader):
        print(i, data['video_id'], data['context_video'].size(), data['video'].size(
        ), data['video_timestamp'], data['sentence'], data['sentence_timestamp'])
        print(len(data['video'].size()))
        break

# __getitem__のreturnでvideo_dataのpoolingしたTensorを返しているが,trim_videoはtorch.utils.data.Dataloaderのあとにやるほうがメモリは食わない気もするけどいいんかね
# pooling_sizeなどをopts.pyにまとめる作業
