#!/usr/bin/python
# -*- coding: UTF-8 -*-

app_host = '0.0.0.0'
app_port = 5000
app_max_file_size = 10 * 1024 * 1024
output_folder = './outputs/'
output_fps = 24
output_frames = int(output_fps * 3)
model_folder = './models/'
model_name = 'stabilityai/stable-video-diffusion-img2vid-xt'
model_inference_steps = 25
model_decode_chunk_size = None
model_unet = True