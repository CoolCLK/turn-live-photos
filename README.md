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

`torch-directml==0.2.5.dev240914` 已经被提前放入了 [requirments.txt](requirments.txt)，因而可以直接使用，_但是我们并不推荐使用 AMD 显卡进行运算_。

#### 指令集优先级

优先 `cuda`，其次 `dml`，最后 `cpu`，因而可能会出现双卡__只__跑了一张卡的情况。

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

> 注：由于 [huggingface.co](https://huggingface.co/) 被 GFW 屏蔽，因而脚本中允许你使用 [hf-mirror.com](https://hf-mirror.com/) 下载模型，但这样只能够以本地模式运行模型，其存放在 `models` 文件夹下，意味着你也可以制作离线包。

之后，跟随脚本的指引打开部署的网页。

拖放或上传任意照片后等待即可。

### 配置

我们使用 [configuration.py](configuration.py) 进行配置，不用担心，它们易于配置！

见表：

|属性|值类型|说明|
|---|---|---|
|`app_host`|`string`|主机名|
|`app_port`|`unsigned short`|人话就是0~65535，运行端口|
|`app_max_file_size`|正整数，以字节为单位|最大允许上传文件的大小|
|`output_folder`|`string`|输出文件的位置|
|`output_frame_duration`|正整数，以毫秒作为单位|一秒内的帧数|
|`output_frames`|正整数|输出帧数|
|`model_folder`|`string`|储存模型的位置|
|`model_name`|`string`|模型名称，此项目使用 [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)|

### 协议

你需要在遵守本项目、[stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) 协议的情况下才可以使用 AI 生成工具。

你需要在遵守本项目[协议](LICENSE)的前提下对此项目进行二次修改。