import argparse
from typing import Sequence

def parse_args(args: Sequence[str] | None = None):
    """
    解析运行时参数。

    :param args: 传入参数
    :type args: Sequence[str] | None
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--output-temp',
        action='store_true',
        default=False,
        help='启用临时文件输出模式'
    )
    parser.add_argument(
        '--progress-bar',
        action='store_true',
        default=False,
        help='允许输出进度条'
    )
    parser.add_argument(
        '--max-split-size-mb',
        type=int,
        default=0,
        help='设置 PyTorch 的 CUDA 最大分区大小'
    )
    parser.add_argument(
        '--ngrok',
        action='store_true',
        default=False,
        help='使用 ngrok 进行内网穿透'
    )
    parser.add_argument(
        '--ngrok-authtoken',
        type=str,
        default=None,
        help='使用 ngrok 的 Auth Token '
    )
    return parser.parse_args()