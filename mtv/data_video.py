import io
import os
import sys
from functools import partial
import math
import torchvision.transforms as TT
from sgm.webds import MetaDistributedWebDataset
import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
from torchvision.utils import save_image
import numpy as np
import torch
import json
import torch.nn.functional as F
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import decord
import pandas as pd
from decord import VideoReader
from torch.utils.data import Dataset
import imageio
import json
from PIL import Image
import cv2
import ast
from pathlib import Path
from icecream import ic

def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr



class MTV_Dataset(Dataset):
    def __init__(self, data_meta_path, video_size, max_num_frames, 
                 frame_interval=1,
                 skip_frms_num=3,
                 audio_margin=2,
                 is_test=False):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(MTV_Dataset, self).__init__()

        self.video_size = video_size
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.audio_margin = audio_margin
        self.frame_interval = frame_interval
        self.is_test = is_test

    
        self.vid_meta = pd.read_csv(data_meta_path)
        self.length = len(self.vid_meta)

        self.mask_ratios = {
            "random": 0.01,
            "intepolate": 0.002,
            "quarter_random": 0.002,
            "quarter_head": 0.002,
            "quarter_tail": 0.002,
            "quarter_head_tail": 0.002,
            "image_head": 0.22,
            "image_tail": 0.005,
            "image_head_tail": 0.005,
        }
        

    def get_latent_frame_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[0] // 4 + 1
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "intepolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1

        return mask

    def __getitem__(self, index):

        retries = 100
        for attempt in range(retries):
            try:
                # 随机选择一个索引
                if (attempt > 0):
                    if (self.is_test):
                        index = index
                    else:
                        index = random.randint(0, self.length - 1)
                item = self.getitem(index)
                return item
                # break  # 跳出循环
            except Exception as e:
                print(f"Error processing video at index {index}: {e}")
                if attempt == retries - 1:
                    print("Exceeded maximum retries. Failing...")
                else:
                    print("Retrying with a new random index...")
    
    def getitem(self, index):
        decord.bridge.set_bridge("torch")

        row_data = self.vid_meta.iloc[index]
        video_path = row_data['path']

        caption = row_data['caption']
            
        audio_emb_path = Path(video_path)
        audio_emb_path = audio_emb_path.parent / f"{audio_emb_path.stem}_dialog.pt"
        audio_emb_vocal = torch.load(str(audio_emb_path), weights_only=True)

        audio_emb_path = Path(video_path)
        audio_emb_path = audio_emb_path.parent / f"{audio_emb_path.stem}_effect.pt"
        audio_emb_accm = torch.load(str(audio_emb_path), weights_only=True)

        audio_emb_path = Path(video_path)
        audio_emb_path = audio_emb_path.parent / f"{audio_emb_path.stem}_music.pt"
        audio_emb_music = torch.load(str(audio_emb_path), weights_only=True)

        margin_indices = (
            torch.arange(2 * self.audio_margin + 1) - self.audio_margin
        )  # Generates [-2, -1, 0, 1, 2]

        vr = VideoReader(uri=video_path, height=-1, width=-1)
        ori_vlen = len(vr)
        video_fps = vr.get_avg_fps()
        
        sample_len = self.max_num_frames * self.frame_interval

        assert ori_vlen > sample_len+self.audio_margin, video_path
        
        rand_choice = random.random()
        
        start = random.randint(
            self.skip_frms_num,
            ori_vlen - sample_len - self.audio_margin - 1
        )
    

        end = min(start + self.max_num_frames * self.frame_interval, ori_vlen)
        ori_indices = np.arange(start, end, self.frame_interval).astype(int)
        temp_frms = vr.get_batch(np.arange(start, end))
        assert temp_frms is not None
        tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
        
        ori_indices = torch.from_numpy(ori_indices)
        new_indices = torch.tensor((ori_indices - start).tolist())
        tensor_frms = tensor_frms[new_indices]
        
        center_indices = ori_indices.unsqueeze(1) + margin_indices.unsqueeze(0)
        audio_tensor_vocal = audio_emb_vocal[center_indices]
        audio_tensor_accm = audio_emb_accm[center_indices]
        audio_emb_music = audio_emb_music[center_indices]

        if random.random() < 0.05:
            audio_tensor_vocal = torch.zeros_like(audio_tensor_vocal)
            audio_tensor_accm = torch.zeros_like(audio_tensor_accm)
            audio_emb_music = torch.zeros_like(audio_emb_music)
        
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")

        latent_frame_mask = self.get_latent_frame_mask(tensor_frms)
        
        assert tensor_frms.shape[0]==self.max_num_frames
        assert tensor_frms.shape[0]==audio_tensor_vocal.shape[0]
        assert tensor_frms.shape[0]==audio_tensor_accm.shape[0]
        assert tensor_frms.shape[0]==audio_emb_music.shape[0]
        
        tensor_frms = (tensor_frms - 127.5) / 127.5

        assert tensor_frms is not None, "tensor_frms is None"
        assert audio_tensor_vocal is not None, "audio_tensor_vocal is None"
        assert audio_tensor_accm is not None, "audio_tensor_accm is None"
        assert audio_emb_music is not None, "audio_emb_music is None"
        audio_emb_vocal_1 = torch.randn(1, 5, 3)
        # assert 
        item = {
            "mp4": tensor_frms,
            "txt": caption,
            "fps": video_fps,
            "num_frames": self.max_num_frames,
            "latent_frame_mask": latent_frame_mask,
            "audio_emb_vocal": audio_tensor_vocal,
            "audio_emb_vocal_1": audio_emb_vocal_1,
            "audio_emb_accm": audio_tensor_accm,
            "audio_emb_music": audio_emb_music,
        }

        return item

    def __len__(self):
        return len(self.vid_meta)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_meta_path=path, **kwargs)
    