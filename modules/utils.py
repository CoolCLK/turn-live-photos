from typing import List, Optional, Union
from PIL import Image
import tempfile

def export_to_fit_gif(
    image: List[Image.Image],
    output_gif_path: Optional[str] = None,
    fps: int = 10,
    ratio: Optional[int] = None,
    any_rotations: bool = False,
) -> str:
    """
    整合图像输出为 Gif 并支持尺寸调整。

    :param image: 输入图像序列
    :param output_gif_path: 输出路径，默认临时文件
    :param fps: 帧率
    :param ratio: 目标比例
    :return: 实际保存路径
    """
    original_width, original_height = image[0].size
    if ratio is not None:
        original_ratio = original_width / original_height
        if any_rotations and (ratio < (1 / original_ratio)) if (original_ratio > 1) else (ratio > (1 / original_ratio)): # 化简后结果
            ratio = 1 / ratio
        new_size = (int(original_width*ratio), int(original_height*ratio))
    else:
        new_size = (original_width, original_height)

    resized_frames = []
    for frame in image:
        frame = frame.resize(new_size)
        resized_frames.append(frame)

    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name

    resized_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=resized_frames[1:],
        optimize=False,
        duration=1000 // fps,
        loop=0,
        disposal=2
    )

    return output_gif_path