#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
主要的 Web 框架和逻辑。

依赖库: 
- torch==2.7.0+cu128
- diffusers==0.33.1
- pillow==11.2.1
- Flask==3.1.1
- Werkzeug==3.1.3
- Jinja2==3.1.6
- 以及所有 modules 下的依赖库...
作者: CoolCLK
"""

from modules.logging import get_logger
logger = get_logger(__name__)

from modules import envvars
envvars.tensorflow.set_min_log_level(1)

from modules.argparsing import parse_args
args = parse_args()
if args.max_split_size_mb > 0:
    envvars.pytroch.set_expandable_segments(True)
    envvars.pytroch.set_max_split_size_mb(args.max_split_size_mb)
if args.ngrok:
    from flask_ngrok2 import run_with_ngrok
    if args.ngrok_authtoken is None or args.ngrok_authtoken == "":
        logger.warning('启用了 ngrok 但是 Auth Token 为空')

import os
from flask import Flask, render_template, request, send_file
import diffusers
from modules.model import load_model
from PIL import Image
import torch
import tempfile
import configuration as conf

diffusers.utils.logging.set_verbosity_error()
if not args.progress_bar:
    diffusers.utils.logging.disable_progress_bar()

app = Flask(__name__)
model = None

@app.route('/')
def route_root():
    """监听主页面请求"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def route_generate():
    """监听生成请求"""
    def allowed_file(filename) -> bool:
        """确认文件是支持的图像文件"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'avif', 'bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff', 'webp'}

    def read_image_file(file) -> Image:
        """读取文件"""
        img_path = os.path.join(tempfile.mkdtemp(), file.name)
        file.save(img_path)
        return Image.open(img_path)

    global model
    if 'file' not in request.files:
        return '{"message": "没有文件被上传"}', 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return '{"message": "不支持的格式"}', 415
    
    try:
        gif_path = os.path.join(tempfile.mkdtemp() if args.output_temp else conf.output_folder, "%s.gif" % file.name)
        model.generate(
            image = read_image_file(file = file).convert('RGB'),
            output_gif_path = gif_path,
            num_inference_steps = conf.model_inference_steps,
            decode_chunk_size = conf.model_decode_chunk_size,
            num_frames = conf.output_frames,
            fps = conf.output_fps,
        )
        return send_file(gif_path, mimetype = 'image/gif')
    except Exception as e:
        logger.warning("生成时出现了一些问题\n%s", e)
    
    return '{"message":"AI 出现了一些问题..."}', 500

def __main__():
    """主程序"""
    global app, args, logger, model
    if (not args.output_temp) and (not os.path.isdir(conf.output_folder)):
        os.makedirs(conf.output_folder)
    elif args.output_temp:
        logger.info("注意：你已禁用了文件输出。")
    app.config['MAX_CONTENT_LENGTH'] = conf.app_max_file_size
    model_path = "%s%s" % (conf.model_folder, conf.model_name)
    model_use_local = os.path.isdir(model_path) or os.path.isfile(model_path)
    use_model_name = model_path if model_use_local else conf.model_name
    model = load_model(model = use_model_name, torch_dtype = torch.float16, variant = 'fp16')
    logger.info("成功加载了模型 %s" % (use_model_name))
    if (not conf.model_unet) and (os.name.lower() == 'linux'):
        model.compile_unet()
        logger.info('编译 UNet 模型成功')
    if args.ngrok:
        logger.info("使用 ngrok 启动了 Web 服务器")
        run_with_ngrok(app = app, auth_token = args.ngrok_authtoken)
        app.run()
    else:
        logger.info("在 %s:%s 启动了 Web 服务器 (http://%s:%s)" % (conf.app_host, conf.app_port, conf.app_host, conf.app_port))
        app.run(host = conf.app_host, port = conf.app_port, threaded=True)

__main__()