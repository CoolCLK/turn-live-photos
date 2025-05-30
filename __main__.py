#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from modules.logging import get_logger
from flask import Flask, render_template, request, send_file
import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_gif
from PIL import Image
import torch
import os
import tempfile
import configuration as conf
import argparse
from accelerate import Accelerator

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--output-temp',
    action='store_true',
    type=bool,
    default=False,
    help='启用临时文件输出模式'
)
parser.add_argument(
    '--progress-bar',
    action='store_true',
    type=bool,
    default=False,
    help='启用临时文件输出模式'
)
parser.add_argument(
    '--max-split-size-mb',
    type=int,
    default=0,
    help='设置 PyTorch 的 CUDA 最大分区大小'
)
args = parser.parse_args()
if args.max_split_size_mb > 0:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:%s" % args.max_split_size_mb

logger = get_logger(__name__)
diffusers.utils.logging.set_verbosity_error()
if not parser.progress_bar:
    diffusers.utils.logging.disable_progress_bar()

app = Flask(__name__)
pipe = None
accelerator = None

@app.route('/')
def index():
    """渲染上传页面"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_gif():
    """生成动图"""
    def allowed_file(filename):
        """确认文件是支持的图像文件"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'avif', 'bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff', 'webp'}

    global pipe, accelerator
    if 'file' not in request.files:
        return '{"message": "没有文件被上传"}', 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return '{"message": "不支持的格式"}', 415
    
    try:
        temp_dir = tempfile.mkdtemp()
        img_path = os.path.join(temp_dir, file.name)
        file.save(img_path)
        input_image = Image.open(img_path).convert('RGB')
        frames = None
        with accelerator.autocast():
            frames = pipe(
                image=input_image,
                num_inference_steps=conf.model_inference_steps,
                decode_chunk_size=conf.model_decode_chunk_size,
                num_frames=conf.output_frames,
            ).frames[0]
        if frames == None:
            return '{"message":"结果为空"}', 500
        gif_path = os.path.join(temp_dir if args.output_temp else conf.output_folder, "%s.gif" % file.name)
        export_to_gif(
            images = frames,
            output_gif_path = gif_path,
            fps = conf.output_fps
        )
        return send_file(gif_path, mimetype='image/gif')
    except Exception as e:  
        logger.error("生成时出现了一些问题\n%s", e)
        return '{"message":"AI 出现了一些问题..."}', 500

def __main__():
    """主程序"""
    global app, logger, pipe, accelerator
    if (not args.output_temp) and (not os.path.isdir(conf.output_folder)):
        os.makedirs(conf.output_folder)
    elif args.output_temp:
        logger.info("注意：你已禁用了文件输出。")
    app.config['MAX_CONTENT_LENGTH'] = conf.app_max_file_size
    model_path="%s%s" % (conf.model_folder, conf.model_name)
    model_use_local=os.path.isdir(model_path) or os.path.isfile(model_path)
    use_model_name=model_path if model_use_local else conf.model_name
    accelerator = Accelerator(
        mixed_precision='fp16'
    )
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        use_model_name,
        torch_dtype = torch.float16,
        variant = 'fp16',
        use_safetensors = True,
    )
    pipe = accelerator.prepare(pipe)
    logger.info("成功加载了模型 %s" % (use_model_name))
    pipe.enable_model_cpu_offload()
    if (not conf.model_unet) and (os.name.lower() == 'linux'):
        pipe.unet = torch.compile(pipe.unet)
        logger.info('编译 UNet 模型成功')
    logger.info("在 %s:%s 启动了 Web 服务器 (http://%s:%s)" % (conf.app_host, conf.app_port, conf.app_host, conf.app_port))
    app.run(host=conf.app_host, port=conf.app_port, threaded=True)

if __name__ == '__main__':
    __main__()