"""API-format graph builders for ComfyUI.

API format is a flat dict: {node_id: {"class_type": str, "inputs": {...}}}.
Links are ["source_node_id", output_index] pairs.

Input key names come from INPUT_TYPES() on ComfyUI 0.10.0. They are not
guessable -- change them only against a fresh dump.
"""
from __future__ import annotations

Graph = dict


def _base(ckpt: str, prompt: str, negative: str, seed: int, steps: int,
          cfg: float, sampler: str, scheduler: str, denoise: float,
          prefix: str) -> Graph:
    """Shared spine: checkpoint -> two text encodes -> sampler -> decode -> save.

    Node "4" (the latent source) is deliberately left out; each builder
    supplies either EmptyLatentImage or VAEEncode.
    """
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": sampler,
                         "scheduler": scheduler, "positive": ["2", 0],
                         "negative": ["3", 0], "latent_image": ["4", 0],
                         "denoise": denoise}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }


def _empty_latent(width: int, height: int, batch: int) -> Graph:
    return {"class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch}}


def sdxl_txt2img(ckpt: str, prompt: str, negative: str = "", seed: int = 0,
                 steps: int = 25, cfg: float = 7.0, width: int = 1024,
                 height: int = 1024, batch: int = 1,
                 sampler: str = "dpmpp_2m", scheduler: str = "karras") -> Graph:
    g = _base(ckpt, prompt, negative, seed, steps, cfg, sampler, scheduler,
              1.0, "colab/sdxl")
    g["4"] = _empty_latent(width, height, batch)
    return g


def flux_txt2img(ckpt: str, prompt: str, negative: str = "", seed: int = 0,
                 steps: int = 20, cfg: float = 1.0, width: int = 1024,
                 height: int = 1024, batch: int = 1,
                 guidance: float = 3.5) -> Graph:
    """Flux ignores CFG entirely -- it must be 1.0, with real guidance
    supplied by FluxGuidance. Any other cfg produces scorched output, so the
    parameter is overridden rather than trusted."""
    g = _base(ckpt, prompt, negative, seed, steps, 1.0, "euler", "simple",
              1.0, "colab/flux")
    g["4"] = _empty_latent(width, height, batch)
    g["8"] = {"class_type": "FluxGuidance",
              "inputs": {"conditioning": ["2", 0], "guidance": guidance}}
    g["5"]["inputs"]["positive"] = ["8", 0]
    return g


def img2img(ckpt: str, prompt: str, image: str, negative: str = "",
            seed: int = 0, steps: int = 25, cfg: float = 7.0,
            denoise: float = 0.6, sampler: str = "dpmpp_2m",
            scheduler: str = "karras") -> Graph:
    """`image` is a filename already present in the server's input/ dir --
    upload it first via ComfyClient.upload_image()."""
    g = _base(ckpt, prompt, negative, seed, steps, cfg, sampler, scheduler,
              denoise, "colab/img2img")
    g["10"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["4"] = {"class_type": "VAEEncode",
              "inputs": {"pixels": ["10", 0], "vae": ["1", 2]}}
    return g


def upscale(ckpt: str, prompt: str, negative: str = "", seed: int = 0,
            steps: int = 25, cfg: float = 7.0, width: int = 1024,
            height: int = 1024, batch: int = 1,
            model_name: str = "4x-UltraSharp.pth",
            sampler: str = "dpmpp_2m", scheduler: str = "karras") -> Graph:
    """txt2img then a pure image-space upscale. No image input, so this is
    the one optional feature needing no upload path."""
    g = sdxl_txt2img(ckpt, prompt, negative, seed, steps, cfg, width, height,
                     batch, sampler, scheduler)
    g["11"] = {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": model_name}}
    g["12"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["11", 0], "image": ["6", 0]}}
    g["7"]["inputs"]["images"] = ["12", 0]
    g["7"]["inputs"]["filename_prefix"] = "colab/upscale"
    return g


def controlnet_canny(ckpt: str, prompt: str, image: str, negative: str = "",
                     seed: int = 0, steps: int = 25, cfg: float = 7.0,
                     width: int = 1024, height: int = 1024, batch: int = 1,
                     strength: float = 0.8, low_threshold: float = 0.4,
                     high_threshold: float = 0.8,
                     control_net: str = "controlnet-canny-sdxl.safetensors",
                     sampler: str = "dpmpp_2m",
                     scheduler: str = "karras") -> Graph:
    """SDXL only. Canny is a core node -- no comfyui_controlnet_aux needed.

    ControlNetApplyAdvanced emits BOTH conditionings, so the sampler's
    positive and negative must be rewired to outputs 0 and 1 of the same
    node. Rewiring only positive is a silent correctness bug.
    """
    g = sdxl_txt2img(ckpt, prompt, negative, seed, steps, cfg, width, height,
                     batch, sampler, scheduler)
    g["10"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["13"] = {"class_type": "Canny",
               "inputs": {"image": ["10", 0], "low_threshold": low_threshold,
                          "high_threshold": high_threshold}}
    g["14"] = {"class_type": "ControlNetLoader",
               "inputs": {"control_net_name": control_net}}
    g["15"] = {"class_type": "ControlNetApplyAdvanced",
               "inputs": {"positive": ["2", 0], "negative": ["3", 0],
                          "control_net": ["14", 0], "image": ["13", 0],
                          "strength": strength, "start_percent": 0.0,
                          "end_percent": 1.0}}
    g["5"]["inputs"]["positive"] = ["15", 0]
    g["5"]["inputs"]["negative"] = ["15", 1]
    g["7"]["inputs"]["filename_prefix"] = "colab/controlnet"
    return g
