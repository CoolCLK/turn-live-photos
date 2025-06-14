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

from typing import Optional
import PIL.Image
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
            image: PIL.Image.Image,
            output_gif_path: str,
            num_frames: int,
            num_inference_steps: Optional[int] = None,
            max_guidance_scale: Optional[float] = None,
            fps: Optional[int] = None,
            motion_bucket_id: Optional[int] = None,
            noise_aug_strength: Optional[float] = None,
            decode_chunk_size: Optional[int] = None,
        ):
        """
        生成动图。

        :param image: 作为传入参数的图像
        :param output_gif_path: 输出地址
        :param num_frames: 生成总帧数
        :param num_inference_steps: 推理步数
        :param max_guidance_scale: 引导指数
        :param fps: 每秒帧数
        :param motion_bucket_id: 运动级别
        :param decode_chunk_size: 解析区块数目
        :type image: PIL.Image
        :type output_gif_path: str
        :type num_frames: int
        :type num_inference_steps: int
        :type max_guidance_scale: int
        :type fps: int
        :type motion_bucket_id: int
        :type decode_chunk_size: int
        """
        height = 1024
        width = 576
        image = image.convert('RGB').resize((width, height))
        num_inference_steps = num_inference_steps if num_inference_steps is not None else 25
        max_guidance_scale = max_guidance_scale if max_guidance_scale is not None else 3.0
        fps = fps if fps is not None else 7
        motion_bucket_id = motion_bucket_id if motion_bucket_id is not None else 127
        noise_aug_strength = noise_aug_strength if noise_aug_strength is not None else 0.02
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        
        with self.__accelerator.autocast():
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

def load_model(model, torch_dtype, variant):
    return Instance(model, torch_dtype, variant)