import torch
import numpy as np
import gc
from .FastBlend.api import smooth_video

class SmoothVideo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orginalframe": ("IMAGE",),
                "keyframe": ("IMAGE",),
                "accuracy": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 3,
                    "step": 1,
                    "display": "number"
                }),
                "window_size": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "batch_size": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 100,
                    "step": 8,
                    "display": "number"
                }),
                "tracking_window_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "minimum_patch_size": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 2,
                    "display": "number"
                }),
                "num_iter": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "guide_weight": ("FLOAT", {
                    "default": 10.0,
                    "min": 1,
                    "max": 100,
                    "step": 0.5,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "AInseven"

    def execute(self, orginalframe, keyframe, accuracy, window_size, batch_size, tracking_window_size,
                minimum_patch_size, num_iter, guide_weight):
        try:
            if accuracy == 1:
                MODE = 'Fast'
            elif accuracy == 2:
                MODE = 'Balanced'
            else:
                MODE = "Accurate"

            print('begin blend keyframe:')

            print("orginalframe Type:", type(orginalframe))
            print("orginalframe shape:", orginalframe.shape)
            print("orginalframe Maximum value of the first item:", torch.max(orginalframe[0]))
            print("orginalframe Minimum value of the first item:", torch.min(orginalframe[0]))
            print(orginalframe.dtype)

            print("keyframe Type:", type(keyframe))
            print("keyframe shape:", keyframe.shape)
            print("keyframe Maximum value of the first item:", torch.max(keyframe[0]))
            print("keyframe Minimum value of the first item:", torch.min(keyframe[0]))
            print(keyframe.dtype)

            orginalframe_np = (orginalframe.cpu().detach().numpy() * 255).astype(np.uint8)
            keyframe_np = (keyframe.cpu().detach().numpy() * 255).astype(np.uint8)

            frames = smooth_video(
                video_guide=None,
                video_guide_folder=orginalframe_np,
                video_style=None,
                video_style_folder=keyframe_np,
                mode=MODE,
                window_size=window_size,
                batch_size=batch_size,
                tracking_window_size=tracking_window_size,
                output_path=None,
                fps=None,
                minimum_patch_size=minimum_patch_size,
                num_iter=num_iter,
                guide_weight=guide_weight,
                initialize="identity"
            )
            print('frames max min:', frames[0].max(), frames[0].min(), frames[0].shape, type(frames[0]), len(frames))

            print('numpy_images = np.stack(frames)')
            numpy_images = np.stack(frames)
            print("numpy_images.shape", numpy_images.shape)

            numpy_images = numpy_images.clip(0, 255)
            normalized_images = numpy_images / 255.0

            print('torch_images = torch.from_numpy(normalized_images)')
            torch_images = torch.from_numpy(normalized_images)
            print("torch_images.shape", torch_images.shape)
            print(torch_images.dtype)

            return (torch_images.type(torch.float32),)
        finally:
            # Move tensors to CPU before deleting
            orginalframe = orginalframe.cpu()
            keyframe = keyframe.cpu()
            torch.cuda.empty_cache()  # Free up GPU memory
            del orginalframe, keyframe, orginalframe_np, keyframe_np, frames, numpy_images, normalized_images, torch_images
            gc.collect()  # Collect garbage
            torch.cuda.empty_cache()  # Free up GPU memory
            print('VRAM cleared')
