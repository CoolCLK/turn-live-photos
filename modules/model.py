#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
模型的逻辑。

依赖库: 
- pillow==11.2.1
- torch==2.7.0+cu128
- accelerate==1.7.0
- diffusers==0.33.1
作者: CoolCLK
"""

from typing import Callable, Optional
import PIL.Image
import numpy as np
import torch
import PIL
from accelerate import Accelerator
from diffusers import StableVideoDiffusionPipeline
from modules.utils import export_to_fit_gif

class Instance:
    """处理模型的实例"""
    
    __accelerator = None
    __pipe = None

    def __init__(self, model, torch_dtype, variant):
        """
        初始化实例。

        :param model: 模型名或路径
        :param torch_dtype: 加载的精度
        :param variant: 选用的模型变种
        :type model: str
        :type torch_dtype: torch.dtype
        :type variant: str
        """
        self.__accelerator = Accelerator(
            mixed_precision = 'fp16'
        )
        self.__pipe = StableVideoDiffusionPipeline.from_pretrained(
            model,
            torch_dtype = torch_dtype,
            variant = variant
        )
        self.__pipe = self.__accelerator.prepare(self.__pipe)
        self.__pipe.enable_model_cpu_offload()

    def compile_unet(self):
        """加载 UNet 模型。"""
        self.__pipe = torch.compile(self.__pipe)

    def generate(
            self, 
            image: PIL.Image,
            output_gif_path: str,
            num_frames: int,
            num_inference_steps: Optional[int] = None,
            max_guidance_scale: Optional[float] = None,
            fps: Optional[int] = None,
            motion_bucket_id: Optional[int] = None,
            noise_aug_strength: Optional[float] = None,
            decode_chunk_size: Optional[int] = None,
            callback: Optional[Callable[[str], None]] = None
        ):
        """
        生成动图。

        :param image: 作为传入参数的图像
        :param num_inference_steps: 推理步数
        :param decode_chunk_size: 解析区块数目
        :param num_frames: 生成总帧数
        :param output_gif_path: 输出地址
        :param fps: 每秒帧数
        :type image: PIL.Image
        :type num_inference_steps: int
        :type decode_chunk_size: int
        :type num_frames: int
        :type output_gif_path: str
        :type fps: int
        """
        height = 576
        width = 1024
        num_inference_steps = num_inference_steps if num_inference_steps is not None else 25
        max_guidance_scale = max_guidance_scale if max_guidance_scale is not None else 3.0
        fps = fps if fps is not None else 7
        motion_bucket_id = motion_bucket_id if motion_bucket_id is not None else 127
        noise_aug_strength = noise_aug_strength if noise_aug_strength is not None else 0.02
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        with self.__accelerator.autocast():
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()  # [HWC] → [CHW]
            image_tensor = image_tensor.to().half()
            
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),  # [1, C, H, W]
                size=(width, height),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            frames = self.__pipe(
                image = image,
                height = height,
                width = width,
                num_frames = num_frames,
                num_inference_steps = num_inference_steps,
                max_guidance_scale = max_guidance_scale,
                fps = fps,
                motion_bucket_id = motion_bucket_id,
                noise_aug_strength = noise_aug_strength,
                decode_chunk_size = decode_chunk_size,
            ).frames[0]
            export_to_fit_gif(
                image = frames,
                output_gif_path = output_gif_path,
                fps = fps,
                ratio = width / height,
                any_rotations = True
            )
            if callback is not None:
                callback(output_gif_path)

def load_model(model, torch_dtype, variant):
    return Instance(model, torch_dtype, variant)