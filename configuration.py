#!/usr/bin/python
# -*- coding: UTF-8 -*-

app_host='0.0.0.0'
app_port=5000
app_max_file_size=10 * 1024 * 1024 # 单位 Byte
output_folder='./outputs/'
output_frame_duration=100 # 单位 ms
output_frames=int((1000 / output_frame_duration) * 3)
model_folder='./models/' # 请在末尾加上 / 
model_name='stabilityai/stable-video-diffusion-img2vid-xt'