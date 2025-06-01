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

    def generate(self, image: PIL.Image, num_inference_steps, decode_chunk_size, num_frames: int, output_gif_path: str, fps: int, callback):
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
        with self.__accelerator.autocast():
            frames = self.__pipe(
                image,
                num_inference_steps = num_inference_steps,
                decode_chunk_size = decode_chunk_size,
                num_frames = num_frames,
            ).frames[0]
            export_to_fit_gif(
                image = frames,
                output_gif_path = output_gif_path,
                fps = fps,
                ratio = 1280 / 720,
                any_rotations = True
            )
            callback()

def load_model(model, torch_dtype, variant):
    return Instance(model, torch_dtype, variant)