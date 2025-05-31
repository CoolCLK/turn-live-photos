@echo off

:global-settings
chcp 65001
cls
cd /D "%~dp0"
setlocal enabledelayedexpansion
set "VENV_HOME=.venv"
set "PYTHON_HOME=%VENV_HOME%/Scripts/"
set "PYTHON_RECOMMENDED_VERSION=3.10.6"
set "MODELS_HOME=./models/"
set "LOGGING_PREFIX=[90m[[33mCoolCLK[90m/[36mturn-live-photos[90m] [0m"

:check-python
cmd /c "exit /b 0"
where python>nul
if not %errorlevel% equ 0 (
    echo %LOGGING_PREFIX%未检测到 Python 环境
    goto end
) else (
    for /F "delims=" %%i in ('python --version') do ( set "PYTHON_VERSION=%%i" )
    if not "!PYTHON_VERSION!"=="Python %PYTHON_RECOMMENDED_VERSION%" (
        echo %LOGGING_PREFIX%警告：你正在使用一个不被支持的 Python 版本 !PYTHON_VERSION!
        echo %LOGGING_PREFIX%我们推荐使用 Python %PYTHON_RECOMMENDED_VERSION%
        echo %LOGGING_PREFIX%详见 https://www.python.org/downloads/release/python-3106/ 
    )
)

:activate-venv
if not exist %VENV_HOME% (
    echo %LOGGING_PREFIX%正在创建虚拟环境中
    python -m venv %VENV_HOME%>nul 1>nul
    echo %LOGGING_PREFIX%在 %VENV_HOME% 目录创建了虚拟环境
)
echo %LOGGING_PREFIX%激活虚拟环境中...
call %VENV_HOME%\Scripts\activate.bat

:ask-requirements
set /p "DO_CHECK_REQUIRMENTS=%LOGGING_PREFIX%是否检查依赖 (Y/n) "
if /i "%DO_CHECK_REQUIRMENTS%"=="Y" (
    echo %LOGGING_PREFIX%从 requirements.txt 处理依赖中
    set /p=[31m<nul
    python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple>nul 1>nul
    "%PYTHON_HOME%/pip3.exe" install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128>nul 1>nul
    "%PYTHON_HOME%/pip.exe" install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple>nul 1>nul
    set /p=[0m<nul
)

:pretrained-model-downloading
set /p "DO_DOWNLOAD_MODEL=%LOGGING_PREFIX%是否提前下载/更新模型 (Y/n) "
if /i "%DO_DOWNLOAD_MODEL%"=="Y" (
    git --version>nul
    set "MODEL_URL=https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt"
    if not %errorlevel% equ 9009 (
        echo %LOGGING_PREFIX%我们默认你已安装 Git LFS（https://git-lfs.com/）

        set /p "DO_USE_MODEL_MIRROR_URL=%LOGGING_PREFIX%是否使用 hf-mirror.com 镜像 (Y/n) "
        if /i "!DO_USE_MODEL_MIRROR_URL!"=="Y" (set "MODEL_URL=https://hf-mirror.com/stabilityai/stable-video-diffusion-img2vid-xt")
        set "GIT_ACTION=clone"
        if exist "%MODELS_HOME%stabilityai/stable-video-diffusion-img2vid-xt" (set "GIT_ACTION=pull") else (mkdir "%MODELS_HOME%stabilityai">nul)
        cd /D "%MODELS_HOME%stabilityai"
        echo %LOGGING_PREFIX%开始拉取 stabilityai/stable-video-diffusion-img2vid-xt ...
        set /p "=[31m"<nul
        git lfs install>nul 1>nul
        git !GIT_ACTION! !MODEL_URL!>nul 1>nul
        set /p "=[0m"<nul
        cd /D "%~dp0"
    ) else (
        echo %LOGGING_PREFIX%检测到未安装 Git（https://git-scm.com/），正在跳过...
    )
)

:run-script
set /p "ARGUMENTS="<run_args.txt
if "%ARGUMENTS%"=="" (echo %LOGGING_PREFIX%启动 Python 脚本) else (echo %LOGGING_PREFIX%以 %ARGUMENTS% 的参数启动 Python 脚本)
"%PYTHON_HOME%/python.exe" app.py %ARGUMENTS%
echo %LOGGING_PREFIX%Python 脚本已停止

:deactivate-venv
echo %LOGGING_PREFIX%停用虚拟环境中
call %VENV_HOME%\Scripts\deactivate.bat

:end
echo %LOGGING_PREFIX%关闭脚本中