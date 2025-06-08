@echo off

:global-settings
chcp 65001>nul
cd /D "%~dp0"
setlocal enabledelayedexpansion

set "PYTHON_HOME="
set "PYTHON_RECOMMENDED_VERSION=3.10.6
set "MODELS_HOME=./models/"
set "LOGGING_PREFIX=[90m[[33mCoolCLK[90m/[36mturn-live-photos[90m] [0m"

:read-configuration
for /f "delims=" %%l in (configuration.ini) do (
    set "_=%%l"
    if "!_:~0,1!"=="[" (
        set "CATEGORY=!_:~1,-1!"
    ) else if not "!_:~0,1!"=="#" (
        for /f "tokens=1,2 delims==" %%i in ("%%l") do (
            set "!CATEGORY!.%%i=%%j"
        )
    )
    set "_="
)

:check-python
if /i not "!LaunchOptions.SkipPythonChecking!"=="True" (
    cmd /c "exit /b 0"
    where python>nul
    if not %errorlevel% equ 0 (
        echo %LOGGING_PREFIX%未检测到 Python 环境
        goto end
    ) else (
        for /f "delims=" %%i in ('python --version') do ( set "PYTHON_VERSION=%%i" )
        if not "!PYTHON_VERSION!"=="Python %PYTHON_RECOMMENDED_VERSION%" (
            echo %LOGGING_PREFIX%警告：你正在使用一个不被支持的 Python 版本 !PYTHON_VERSION!
            echo %LOGGING_PREFIX%我们推荐使用 Python %PYTHON_RECOMMENDED_VERSION%
            echo %LOGGING_PREFIX%详见 https://www.python.org/downloads/release/python-3106/ 
        )
    )
)

:activate-venv
if /i "!LaunchOptions.UseVenv!"=="True" (
    if not exist !StoreOptions.VenvHome! (
        echo %LOGGING_PREFIX%正在创建虚拟环境中
        "%PYTHON_HOME%python.exe" -m venv !StoreOptions.VenvHome!>nul 1>nul
        echo %LOGGING_PREFIX%在 !StoreOptions.VenvHome! 目录创建了虚拟环境
        set "PYTHON_HOME=!StoreOptions.VenvHome!/Scripts/"
    )
    echo %LOGGING_PREFIX%激活虚拟环境中...
    call !StoreOptions.VenvHome!\Scripts\activate.bat
)

:ask-requirements
set /p "DO_CHECK_REQUIRMENTS=%LOGGING_PREFIX%是否检查依赖 (Y/n) "
if /i "%DO_CHECK_REQUIRMENTS%"=="Y" (
    set "InstalltionOptions.PipMirrorArguments="
    if not "!InstalltionOptions.PyTorchIndexUrl!"=="" (
        set "InstalltionOptions.PipMirrorArguments=!InstalltionOptions.PipMirrorArguments! --extra-index-url !InstalltionOptions.PyTorchIndexUrl!"
    )
    if not "!InstalltionOptions.PipMirrorUrl!"=="" (
        set "InstalltionOptions.PipMirrorArguments=!InstalltionOptions.PipMirrorArguments! -i !InstalltionOptions.PipMirrorUrl!"
    )
    echo %LOGGING_PREFIX%从 requirements.txt 处理依赖中
    set /p=[31m<nul
    python -m pip install --upgrade pip!InstalltionOptions.PipMirrorArguments!>nul 1>nul
    "!PYTHON_HOME!pip.exe" install -r requirements.txt !InstalltionOptions.PipMirrorArguments!>nul 1>nul
    set /p=[0m<nul
)

:pretrained-model-downloading
set /p "DO_DOWNLOAD_MODEL=%LOGGING_PREFIX%是否提前下载/更新模型 (Y/n) "
if /i "%DO_DOWNLOAD_MODEL%"=="Y" (
    git --version>nul
    if not %errorlevel% equ 9009 (
        echo %LOGGING_PREFIX%我们默认你已安装 Git LFS（https://git-lfs.com/）

        set "GIT_ACTION=clone"
        if exist "!StoreOptions.ModelsHome!/!ModelOptions.ModelName!" (set "GIT_ACTION=pull") else (mkdir "!StoreOptions.ModelsHome!/!ModelOptions.ModelName!">nul)
        cd /D "!StoreOptions.ModelsHome!/!ModelOptions.ModelName!/../"
        echo %LOGGING_PREFIX%开始拉取 !ModelOptions.ModelName! ...
        set /p "=[31m"<nul
        git lfs install>nul 1>nul
        git !GIT_ACTION! !InstalltionOptions.ModelRepositoryUrl!>nul 1>nul
        set /p "=[0m"<nul
        cd /D "%~dp0"
    ) else (
        echo %LOGGING_PREFIX%检测到未安装 Git（https://git-scm.com/），正在跳过...
    )
)

:run-script
if "!LaunchOptions.RunArguments!"=="" (echo %LOGGING_PREFIX%启动 Python 脚本) else (echo %LOGGING_PREFIX%以 !LaunchOptions.RunArguments! 的参数启动 Python 脚本)
"!PYTHON_HOME!python.exe" app.py !LaunchOptions.RunArguments!
echo %LOGGING_PREFIX%Python 脚本已停止

:deactivate-venv
if /i "!LaunchOptions.UseVenv!"=="True" (
    echo %LOGGING_PREFIX%停用虚拟环境中
    call !StoreOptions.VenvHome!\Scripts\deactivate.bat
)

:end
echo %LOGGING_PREFIX%关闭脚本中