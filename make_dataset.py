# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MakeDataset():
    def __init__(self):
        super(MakeDataset, self).__init__()

        self.bmn_video_path = './dataset/mydataset/video/'
        self.annotation_path = './dataset/mydataset/annotation/'
        self.annotation_file = ['train/', 'val/', 'test/']

        self.train_dataset_path = './dataset/mydataset/train/'
        self.val_dataset_path = './dataset/mydataset/val/'
        self.test_dataset_path = './dataset/mydataset/test/'
        self.train_val_dataset_path = './dataset/mydataset/train-val/'

        self.make_dataset()

    def make_dataset(self):
        caption_json_train_df = pd.read_json(os.path.join(
            self.annotation_path, self.annotation_file[0], 'caption_train.json')).T
        caption_json_val_df = pd.read_json(os.path.join(
            self.annotation_path, self.annotation_file[1], 'caption_val.json')).T
        caption_json_test_df = pd.read_json(os.path.join(
            self.annotation_path, self.annotation_file[2], 'caption_test.json')).T

        if not os.path.isdir(self.train_dataset_path):
            os.makedirs(self.train_dataset_path)
            print('Making train datasets.')
            self._make_csv_file(caption_json_train_df, self.train_dataset_path)
        else:
            print('You may already have train datasets.')
            pass
        if not os.path.isdir(self.val_dataset_path):
            os.makedirs(self.val_dataset_path)
            print('Maiking val datasets.')
            self._make_csv_file(caption_json_val_df, self.val_dataset_path)
        else:
            print('You may already have val datasets.')
            pass
        if not os.path.isdir(self.test_dataset_path):
            os.makedirs(self.test_dataset_path)
            print('Making test datasets.')
            self._make_csv_file(caption_json_test_df, self.test_dataset_path)
        else:
            print('You may already have test datasets.')
            pass
        if not os.path.isdir(self.train_val_dataset_path):
            os.makedirs(self.train_val_dataset_path)
            print('Maiking train-val datasets.')
            self._make_csv_file(caption_json_train_df,
                                self.train_val_dataset_path)
            self._make_csv_file(caption_json_val_df,
                                self.train_val_dataset_path)
        else:
            print('You may already have train-val datasets.')
            pass

    def _make_csv_file(self, caption_json_df, dataset_path):
        for i in range(len(caption_json_df)):
            video_name, video_path, sentences, timestamps, duration = self._load_video_sentence(
                caption_json_df, i)

            trim_list = self._get_trim_list(sentences, timestamps)

            for j in range(len(trim_list)):
                csv_dict = {'number': j, 'video_name': video_name, 'video_path': video_path,
                            'sentence': trim_list[j][0], 'timestamp': trim_list[j][1], 'duration': duration}
                csv_df = pd.DataFrame.from_dict(
                    csv_dict, orient='index', columns=[j]).T
                path = os.path.join(dataset_path, video_name + '.csv')
                if os.path.exists(path):
                    csv_df.to_csv(path, mode='a', header=None)
                else:
                    csv_df.to_csv(path)
                csv_dict.clear()
            print('{}: {}'.format(i, path))
#             break
        print('==================================================================================================================')

    def _load_video_sentence(self, caption_json_df, i):
        video_dict = {}
        video_name = caption_json_df.index.values[i]
        # print('video_name:', video_name)
        video_info = caption_json_df.loc[video_name]
        video_dict[video_name] = video_info

        video_path = os.path.join(self.bmn_video_path, video_name + '.csv')

        duration = video_dict[video_name]['duration']
        timestamps = video_dict[video_name]['timestamps']
        sentences = video_dict[video_name]['sentences']
        return video_name, video_path, sentences, timestamps, duration

    def _get_trim_list(self, sentences, timestamps):
        trim_sentence_list = []
        trim_timestamp_list = []
        trim_list = []
        for timestamp, sentence in zip(timestamps, sentences):
            trim_sentence_list.append(sentence)
            trim_timestamp_list.append(timestamp)

        for p in range(len(trim_sentence_list)):
            for q in range(len(trim_timestamp_list)):
                if p == q:
                    trim_list += [[trim_sentence_list[q],
                                   trim_timestamp_list[p]]]
                else:
                    pass
        return trim_list


if __name__ == '__main__':
    start = time.time()
    MakeDataset()
    time = time.time() - start
    print('{} s'.format(time))
