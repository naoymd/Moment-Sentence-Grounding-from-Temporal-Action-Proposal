# -*- coding: utf-8 -*-
import os
import pandas as pd
import shutil
import time


class MyFile():
    def __init__(self):
        super(MyFile, self).__init__()

        self.bmn_video_path = './dataset/activitynet_feature_cuhk/'
        self.bmn_csv_path = './dataset/activitynet_annotations/video_info_new.csv'
        self.bmn_json_path = './dataset/activitynet_annotations/anet_anno_action.json'

        self.caption_path = './dataset/ActivityNet_Captions/'
        # ['train.json', 'val_1.json']
        self.caption_all_json = ['train.json', 'val_1.json', 'val_2.json']

        self.bmn_video = 'csv_mean_100'

        self.save_path = './dataset/mydataset/annotation/'

        self.video_path = './dataset/mydataset/video'

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
            self.make_file()

        else:
            print('you may already have annotation files.')

        if not os.path.isdir(self.video_path):
            print('copy bmn video directory...')
            shutil.copytree(os.path.join(self.bmn_video_path,
                                         self.bmn_video), self.video_path)

        else:
            pass

    def make_file(self):
        bmn_csv_df = pd.read_csv(self.bmn_csv_path)
        bmn_csv_train_df = bmn_csv_df[bmn_csv_df['subset'] == 'training']
        bmn_csv_val_df = bmn_csv_df[bmn_csv_df['subset'] == 'validation']
        bmn_csv_test_df = bmn_csv_df[bmn_csv_df['subset'] == 'testing']
        bmn_json_df = pd.read_json(self.bmn_json_path).T

        for caption_json_file in (os.path.join(self.caption_path, json_path) for json_path in self.caption_all_json):
            caption_json_df = pd.read_json(caption_json_file).T

            if 'train.json' in caption_json_file:
                print(caption_json_file)
                self.caption_json_train_df = caption_json_df[caption_json_df.index.isin(
                    bmn_csv_train_df['video'])]
                self.bmn_csv_train_df = bmn_csv_train_df[bmn_csv_train_df['video'].isin(
                    self.caption_json_train_df.index)]
                self.bmn_json_train_df = bmn_json_df[bmn_json_df.index.isin(
                    self.caption_json_train_df.index)]

                save_train_path = os.path.join(self.save_path, 'train')
                if not os.path.isdir(save_train_path):
                    os.makedirs(save_train_path)

                self.caption_json_train_df.T.to_json(
                    os.path.join(save_train_path, 'caption_train.json'))
                self.bmn_csv_train_df.to_csv(
                    os.path.join(save_train_path, 'bmn_train.csv'))
                self.bmn_json_train_df.T.to_json(
                    os.path.join(save_train_path, 'bmn_train.json'))

            elif 'val_1.json' in caption_json_file:
                print(caption_json_file)
                self.caption_json_val_df = caption_json_df[caption_json_df.index.isin(
                    bmn_csv_val_df['video'])]
                self.bmn_csv_val_df = bmn_csv_val_df[bmn_csv_val_df['video'].isin(
                    self.caption_json_val_df.index)]
                self.bmn_json_val_df = bmn_json_df[bmn_json_df.index.isin(
                    self.caption_json_val_df.index)]

                save_val_path = os.path.join(self.save_path, 'val')
                if not os.path.isdir(save_val_path):
                    os.makedirs(save_val_path)

                self.caption_json_val_df.T.to_json(
                    os.path.join(save_val_path, 'caption_val.json'))
                self.bmn_csv_val_df.to_csv(
                    os.path.join(save_val_path, 'bmn_val.csv'))
                self.bmn_json_val_df.T.to_json(
                    os.path.join(save_val_path, 'bmn_val.json'))

            elif 'val_2.json' in caption_json_file:
                print(caption_json_file)
                self.caption_json_test_df = caption_json_df[caption_json_df.index.isin(
                    bmn_csv_val_df['video'])]
                self.bmn_csv_test_df = bmn_csv_val_df[bmn_csv_val_df['video'].isin(
                    self.caption_json_test_df.index)]
                self.bmn_json_test_df = bmn_json_df[bmn_json_df.index.isin(
                    self.caption_json_test_df.index)]

                save_test_path = os.path.join(self.save_path, 'test')
                if not os.path.isdir(save_test_path):
                    os.makedirs(save_test_path)

                self.caption_json_test_df.T.to_json(
                    os.path.join(save_test_path, 'caption_test.json'))
                self.bmn_csv_test_df.to_csv(
                    os.path.join(save_test_path, 'bmn_test.csv'))
                self.bmn_json_test_df.T.to_json(
                    os.path.join(save_test_path, 'bmn_test.json'))


class MyTestFile():
    def __init__(self):
        super(MyTestFile, self).__init__()

        self.annotation_path = './dataset/mydataset/annotation/'
        self.annotation_all_files = [
            'train/caption_train.json', 'val/caption_val.json', 'test/caption_test.json']

        self.bmn_results_path = './dataset/BMN_results'
        self.bmn_results_all_json = [
            'bmn_result_train.json', 'bmn_result_val.json', 'bmn_result_val.json']

        self.save_path = './dataset/mydataset/BMN_results'
        self.save_all_json = ['bmn_result_train.json',
                              'bmn_result_val.json', 'bmn_result_test.json']

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
            self.make_test_dataset()

        else:
            print('you may already have BMN result files.')

    def make_test_dataset(self):
        for i in range(len(self.annotation_all_files)):
            annotation_df = pd.read_json(os.path.join(
                self.annotation_path, self.annotation_all_files[i])).T
            bmn_result_df = pd.read_json(os.path.join(
                self.bmn_results_path, self.bmn_results_all_json[i]))
            bmn_result_df = bmn_result_df.rename(index=lambda a: "v_" + a)

            bmn_result_df = bmn_result_df[bmn_result_df.index.isin(
                annotation_df.index)]
            bmn_result_df.to_json(os.path.join(
                self.save_path, self.save_all_json[i]))
            pd.read_json(os.path.join(
                self.save_path, self.save_all_json[i]))


if __name__ == '__main__':
    start = time.time()
    MyFile()
    MyTestFile()
    time = time.time() - start
    print('{} s'.format(time))
