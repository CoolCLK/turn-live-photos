import PIL.Image
import torch
import PIL
from accelerate import Accelerator
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_gif

class Instance:
    __accelerator = None
    __pipe = None

    def __init__(self, model, torch_dtype, variant):
        self.__accelerator = Accelerator(
            mixed_precision='fp16'
        )
        self.__pipe = StableVideoDiffusionPipeline.from_pretrained(
            model,
            torch_dtype = torch_dtype,
            variant = variant
        )
        self.__pipe = self.__accelerator.prepare(self.__pipe)
        self.__pipe.enable_model_cpu_offload()

    def compile_unet(self):
        self.__pipe = torch.compile(self.__pipe)

    def generate(self, image: PIL.Image, num_inference_steps, decode_chunk_size, num_frames: int, output_gif_path: str, fps: int):
        with self.__accelerator.autocast():
            frames = self.__pipe(
                image,
                num_inference_steps = num_inference_steps,
                decode_chunk_size = decode_chunk_size,
                num_frames = num_frames,
            ).frames[0]
            export_to_gif(
                image = frames,
                output_gif_path = output_gif_path,
                fps = fps,
            )

def load_model():
    return Instance()