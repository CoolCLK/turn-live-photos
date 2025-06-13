#!/usr/bin/python
# -*- coding: UTF-8 -*-

from configparser import ConfigParser
conf = ConfigParser()
conf.read('configuration.ini', encoding = 'utf-8')

app_host = conf.get('WebOptions', 'Host', fallback = '0.0.0.0')
app_port = conf.getint('WebOptions', 'Port', fallback = 5000)
app_max_file_size = conf.getint('WebOptions', 'MaxContentLength', fallback = 0)
output_folder = conf.get('StoreOptions', 'OutputsHome', fallback = 'outputs')
output_fps = conf.getint('ModelOptions', 'OutputFPS', fallback = 24)
output_frames = conf.getint('ModelOptions', 'OutputFrames', fallback = int(output_fps * 3))
model_folder = conf.get('StoreOptions', 'ModelsHome', fallback = 'models')
model_name = conf.get('ModelOptions', 'ModelName', fallback = 'stabilityai/stable-video-diffusion-img2vid-xt')
model_inference_steps = conf.getint('ModelOptions', 'InferenceSteps', fallback = 25)
model_decode_chunk_size = conf.getint('ModelOptions', 'DecodeChunkSize', fallback = None)
model_unet = conf.getboolean('ModelOptions', 'UNet', fallback = True)