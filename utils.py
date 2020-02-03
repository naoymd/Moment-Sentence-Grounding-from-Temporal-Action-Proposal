import os
import torch
from collections import OrderedDict


def sec2str(sec):
    if sec < 60:
        return "elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)


def model_state_dict(save_dir, model_checkpoint):
    model_params = os.path.join(save_dir, model_checkpoint)
    state_dict = torch.load(model_params)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        # name = k[:]
        new_state_dict[name] = v
    return state_dict


def l2norm(out):
    norm = torch.norm(out, dim=1, keepdim=True)
    return out / norm


def collate_fn(data):
    out = {'video_id': [], 'context_video': [], 'video': [],
           'video_timestamp': [], 'sentence': [], 'sentence_timestamp': []}
    for obj in data:
        out['video_id'].append(obj['video_id'])
        out['context_video'].append(obj['context_video'])
        out['video'].append(obj['video'])
        out['video_timestamp'].append(obj['video_timestamp'])
        out['sentence'].append(obj['sentence'])
        out['sentence_timestamp'].append(obj['sentence_timestamp'])
    out['context_video'] = torch.stack(out['context_video'])
    out['video'] = torch.stack(out['video'])
    return out
