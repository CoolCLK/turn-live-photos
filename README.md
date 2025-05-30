# turn-live-photos
将静态照片通过 AIGC 生成为实况照片

### 配置要求

Python：推荐使用 `3.10.6`，默认使用 `pytorch==2.7.0+cu128`。

显卡：至少显存大于 `8G`，且 [CUDA](https://developer.nvidia.com/cuda-toolkit) 版本为 `12.8`。

#### 使用 CPU 或其它版本的 CUDA 

注意，如果你想使用 CPU 或 [CUDA](https://developer.nvidia.com/cuda-toolkit) 其它版本的话，请打开 [PyTorch](https://pytorch.org/get-started/locally/) 选择适合你的版本，复制代码。比如 CPU 的代码是这样的：

`pip3 install torch torchvision torchaudio`

找到 [START.bat](START.bat) 的安装代码，它应该长这样：

`"%PYTHON_HOME%/pip3.exe" install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128>nul 1>nul`

为了应用 CPU 版本的 [PyTorch](https://pytorch.org/get-started/locally/)，应该将脚本中代码改为这样：

`"%PYTHON_HOME%/pip3.exe" install torch torchvision torchaudio>nul 1>nul`

当然，不同版本的 [CUDA](https://developer.nvidia.com/cuda-toolkit) 可能丢失部分特性，_更重要的是，我们**强烈**不推荐使用 CPU 进行运算_。

#### 使用 AMD 显卡

~~`torch-directml==0.2.5.dev240914` 已经被提前放入了 [requirments.txt](requirments.txt)，因而可以直接使用，_但是我们并不推荐使用 AMD 显卡进行运算_。~~

`torch-directml==0.2.5.dev240914` 与 `pytorch==2.7.0+cu128` 不兼容。

#### 指令集优先级

优先 `cuda`，其次 `dml`，最后 `cpu`，因而可能会出现双卡**只**跑了一张卡的情况。

### 下载/安装

#### 下载最新的稳定版

访问 [turn-live-photos](https://github.com/CoolCLK/turn-live-photos) 中的 [Releases](https://github.com/CoolCLK/turn-live-photos/releases)。

找到最新的 [Release](https://api.github.com/repos/CoolCLK/turn-live-photos/releases/latest) 即可下载。

#### 下载最新的实验版

_警告：实验板往往是不稳定、不确定能够正常运行的版本！_

访问 [turn-live-photos](https://github.com/CoolCLK/turn-live-photos)，找到 __Codes__ 并点击 __Download ZIP__ 或[直接下载](https://github.com/CoolCLK/turn-live-photos/archive/refs/heads/main.zip)。

当然，你也可以使用 [Git](https://git-scm.com/) 下载。

打开终端，输入 `https://github.com/CoolCLK/turn-live-photos.git` 后等待即可。

### 使用

假定你已经下载好了 [turn-live-photos](https://github.com/CoolCLK/turn-live-photos)，见表。

|系统|操作|
|---|---|
|Windows|在此文件夹中打开终端，输入 `.\START.bat`，或直接双击打开也可。 |
|MacOS/Linux|打开终端，`cd` 到项目文件夹，输入 `.\START` 即可。|

倘若你想要提前下载模型，不仅要安装 [Git](https://git-scm.com/)，还要安装 [Git LFS](https://git-lfs.com/)。

> 注：由于 [huggingface.co](https://huggingface.co/) 被 GFW 屏蔽，因而脚本中允许你使用 [hf-mirror.com](https://hf-mirror.com/) 下载模型，但这样只能够以本地模式运行模型，其存放在 `models` 文件夹下，意味着你也可以制作离线包。

之后，跟随脚本的指引打开部署的网页。

拖放或上传任意照片后等待即可。

倘若你不想让结果输出到 `outputs` 的话，可以添加参数 `--output-temp`。

### 配置

我们使用 [configuration.py](configuration.py) 进行配置，不用担心，它们易于配置！

见表：

|属性|值类型|说明|
|---|---|---|
|`app_host`|`string`|主机名|
|`app_port`|`unsigned short`|人话就是0~65535，运行端口|
|`app_max_file_size`|正整数，以字节为单位|最大允许上传文件的大小|
|`output_folder`|`string`|输出文件的位置|
|`output_fps`|`int`，以帧/秒作为单位|_无需多言_|
|`output_frames`|`int`|输出帧数|
|`model_folder`|`string`|储存模型的位置，**需要在末尾加上 `/`**|
|`model_inference_steps`|`int`|越高质量也会越高，但是要求的显存会更高|
|`model_decode_chunk_size`|`int`|越高的数值有利于减少显存，***小概率*会造成画面撕裂**|
|`model_name`|`string`|模型名称，此项目使用 [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)|
|`model_unet`|`bool`|如果可以使用 UNet 模型的话，那就使用，同时会占用一部分显存，*仅限 Linux 平台*|

倘若你想要配置启动参数的话，可以编辑 [run_args.txt](run_args.txt) 来修改。

如果像获取更多参数帮助，可以使用命令 `python __main__.py --help` 来查阅。

### 模型

我们使用 [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) 模型来生成内容，顺带一提，我们使用的精度是`fp16`。

你需要在遵守 [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) 模型协议的情况下才可以使用 AI 生成工具。

### 协议

你需要在遵守本项目[协议](LICENSE)的前提下对此项目进行二次修改（仅限于代码）。

### 其它平台

#### [Google Colab](https://colab.research.google.com/)

> 需要一个准备 [Google 账号](https://myaccount.google.com/)。

首先打开 [Google Colab](https://colab.research.google.com/)，之后[新建笔记本](https://colab.research.google.com/#create=true)。

接下来，找到**修改**>**笔记本设置**>**硬件加速器**，任意选择一个即可。

之后，使用新建代码单元格或使用快捷键**Ctrl+M B**，输入：

```
!apt-get install python3.10
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab Notebooks
!git clone https://github.com/CoolCLK/turn-live-photos.git
%cd /content/drive/MyDrive/Colab Notebooks/turn-live-photos
!pip3 install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
!pip install -r requirements.txt
!python __main__.py
```

运行单元格或者使用快捷键**Ctrl+Enter**，稍等即可。

运行完成后，我们会发现我们无法正常访问网址。那么此时我们需要内网穿透。

我们这里以 [ngrok](https://ngrok.com/) 做例子，提前注册好账号后，打开 [Your Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) 并复制身份验证码，此时对原先代码稍作修改：

```
!pip install flask-ngrok2
```

之后，在 [Google Drive](https://drive.google.com/) 找到 **__main__.py**，一般是在**Colab Notebooks/turn-live-photos**下，添加`from flask_ngrok2 import run_with_ngrok`，修改`app.run(host=conf.app_host, port=conf.app_port, threaded=True)`为`app.run()`，并在之前加上`run_with_ngrok(app=app, auth_token='<your-authtoken>')`，运行后应当可以看到了。

> 极力推荐 [Google Colab](https://colab.research.google.com/)，免费额度可以分到*至少 8G 显存*的 GPU。

> 比如我这里用的是**T4 GPU**，并且显存只有*15.0 GB*，这看起来很多，但对于视频生成远远不够，因而我们可以在 `!python __main__.py` 后面添加参数，`--max-split-size-mb=6144` 是比较合适的，*但这种方法会使得生成速度变慢*。~~你要氪金也可以。~~

#### [Hugging Face Spaces](https://huggingface.co/spaces)

> 需要准备一个 Hugging Face 账号

创建仓库，可以直接导入到 [Hugging Face Spaces](https://huggingface.co/spaces)，仅需在 [README.md] 前加上：

```
---
title: turn-live-photos
emoji: 😍
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 5000
---
```

然后等待即可。

> Hugging Face 免费额度只提供 CPU，不推荐使用。