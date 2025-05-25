#!/usr/bin/python
# -*- coding: UTF-8 -*-

from modules.logging import get_logger
logger = get_logger(__name__)

from flask import Flask, render_template, request, send_file
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import torch
import os
import tempfile
import configuration as conf
import sys
# import torch_directml

def allowed_file(filename):
    """确认文件是支持的图像文件"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'avif', 'bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff', 'webp'}

app = Flask(__name__)
pipe = None
output_temp=sys.argv.__contains__('--temp-output')

@app.route('/')
def index():
    """渲染上传页面"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_gif():
    """生成动图"""
    global pipe
    if 'file' not in request.files:
        return '{"message": "没有文件被上传"}', 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return '{"message": "仅支持JPG/PNG格式"}', 415
    
    temp_dir = tempfile.mkdtemp() if output_temp else conf.output_folder
    img_path = os.path.join(temp_dir, file.name)
    file.save(img_path)
    
    input_image = Image.open(img_path).convert("RGB")
    
    torch.cuda.empty_cache()
    frames = pipe(
        image=input_image,
        num_inference_steps=20,
        num_frames=conf.output_frames,
    ).frames[0]
    
    gif_path = os.path.join(conf.output_folder, "%s.gif" % file.name)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=conf.output_frame_duration,
        loop=0,
        optimize=True
    )
    logger.info('编译 UNet 模型成功')
    
    return send_file(gif_path, mimetype='image/gif')

def __main__():
    """主程序"""
    global app, pipe
    app.config['MAX_CONTENT_LENGTH'] = conf.app_max_file_size
    logger.info("设定了上传大小限制: %sMB", app.config['MAX_CONTENT_LENGTH'] / 1048576)
    if (not output_temp) and (not os.path.isdir(conf.output_folder)):
        os.makedirs(conf.output_folder)
    model_file="%s%s" % (conf.model_folder, conf.model_name)
    model_use_local=os.path.isdir(model_file) or os.path.isfile(model_file)
    use_model_name=model_file if model_use_local else conf.model_folder
    command_sets='cpu'
    # if torch_directml.is_available():
    #     command_sets='dml'
    if torch.cuda.is_available():
        command_sets='cuda'
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        use_model_name,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(command_sets)
    logger.info("成功以 %s 指令集加载了模型 %s" % (command_sets, use_model_name))
    pipe.enable_model_cpu_offload()
    if os.name.lower() == 'linux':
        pipe.unet = torch.compile(pipe.unet)
        logger.info('编译 UNet 模型成功')
    logger.info("在 %s:%s 启动了 Web 服务器 (http://%s:%s)" % (conf.app_host, conf.app_port, conf.app_host, conf.app_port))
    app.run(host=conf.app_host, port=conf.app_port, threaded=True)

if __name__ == '__main__':
    __main__()