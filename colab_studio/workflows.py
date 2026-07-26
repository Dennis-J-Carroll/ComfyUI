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


def is_flux(profile: str) -> bool:
    """True for every Flux profile name in registry.PROFILES."""
    return profile.startswith("flux")


def _spine(ckpt: str, prompt: str, negative: str, seed: int, steps: int,
           cfg: float, sampler: str, scheduler: str, denoise: float,
           prefix: str, profile: str, guidance: float) -> Graph:
    """`_base` plus the profile-dependent sampling contract.

    THE single place the Flux-vs-SDXL decision lives. Flux ignores classifier
    free guidance entirely: cfg must be 1.0 with the real guidance supplied by
    a FluxGuidance node wired into KSampler.positive. Any other cfg produces
    scorched output, so cfg/sampler/scheduler are overridden rather than
    trusted -- every builder routes through here so no graph can be handed a
    Flux checkpoint with SDXL-shaped sampling.
    """
    if is_flux(profile):
        cfg, sampler, scheduler = 1.0, "euler", "simple"
    g = _base(ckpt, prompt, negative, seed, steps, cfg, sampler, scheduler,
              denoise, prefix)
    if is_flux(profile):
        g["8"] = {"class_type": "FluxGuidance",
                  "inputs": {"conditioning": ["2", 0], "guidance": guidance}}
        g["5"]["inputs"]["positive"] = ["8", 0]
    return g


def _txt2img(ckpt: str, prompt: str, negative: str, seed: int, steps: int,
             cfg: float, width: int, height: int, batch: int, sampler: str,
             scheduler: str, prefix: str, profile: str,
             guidance: float) -> Graph:
    g = _spine(ckpt, prompt, negative, seed, steps, cfg, sampler, scheduler,
               1.0, prefix, profile, guidance)
    g["4"] = _empty_latent(width, height, batch)
    return g


def sdxl_txt2img(ckpt: str, prompt: str, negative: str = "", seed: int = 0,
                 steps: int = 25, cfg: float = 7.0, width: int = 1024,
                 height: int = 1024, batch: int = 1,
                 sampler: str = "dpmpp_2m", scheduler: str = "karras") -> Graph:
    """SDXL txt2img: 7 nodes, dpmpp_2m/karras. Use flux_txt2img for Flux."""
    return _txt2img(ckpt, prompt, negative, seed, steps, cfg, width, height,
                    batch, sampler, scheduler, "colab/sdxl", "sdxl", 3.5)


def flux_txt2img(ckpt: str, prompt: str, negative: str = "", seed: int = 0,
                 steps: int = 20, cfg: float = 1.0, width: int = 1024,
                 height: int = 1024, batch: int = 1,
                 guidance: float = 3.5) -> Graph:
    """Flux ignores CFG entirely -- it must be 1.0, with real guidance
    supplied by FluxGuidance. Any other cfg produces scorched output, so the
    parameter is overridden rather than trusted (see _spine)."""
    return _txt2img(ckpt, prompt, negative, seed, steps, cfg, width, height,
                    batch, "euler", "simple", "colab/flux", "flux", guidance)


def img2img(ckpt: str, prompt: str, image: str, negative: str = "",
            seed: int = 0, steps: int = 25, cfg: float = 7.0,
            denoise: float = 0.6, sampler: str = "dpmpp_2m",
            scheduler: str = "karras", profile: str = "sdxl",
            guidance: float = 3.5) -> Graph:
    """`image` is a filename already present in the server's input/ dir --
    upload it first via ComfyClient.upload_image().

    Pass `profile` so a Flux checkpoint gets Flux sampling; the default is
    SDXL-shaped.
    """
    g = _spine(ckpt, prompt, negative, seed, steps, cfg, sampler, scheduler,
               denoise, "colab/img2img", profile, guidance)
    g["10"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["4"] = {"class_type": "VAEEncode",
              "inputs": {"pixels": ["10", 0], "vae": ["1", 2]}}
    return g


def upscale(ckpt: str, prompt: str, negative: str = "", seed: int = 0,
            steps: int = 25, cfg: float = 7.0, width: int = 1024,
            height: int = 1024, batch: int = 1,
            model_name: str = "4x-UltraSharp.pth",
            sampler: str = "dpmpp_2m", scheduler: str = "karras",
            profile: str = "sdxl", guidance: float = 3.5) -> Graph:
    """txt2img then a pure image-space upscale. No image input, so this is
    the one optional feature needing no upload path.

    Pass `profile` so a Flux checkpoint gets Flux sampling; the default is
    SDXL-shaped.
    """
    g = _txt2img(ckpt, prompt, negative, seed, steps, cfg, width, height,
                 batch, sampler, scheduler, "colab/upscale", profile, guidance)
    g["11"] = {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": model_name}}
    g["12"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["11", 0], "image": ["6", 0]}}
    g["7"]["inputs"]["images"] = ["12", 0]
    return g


def controlnet_canny(ckpt: str, prompt: str, image: str, negative: str = "",
                     seed: int = 0, steps: int = 25, cfg: float = 7.0,
                     width: int = 1024, height: int = 1024, batch: int = 1,
                     strength: float = 0.8, low_threshold: float = 0.4,
                     high_threshold: float = 0.8,
                     control_net: str = "controlnet-canny-sdxl.safetensors",
                     sampler: str = "dpmpp_2m",
                     scheduler: str = "karras",
                     profile: str = "sdxl") -> Graph:
    """SDXL only. Canny is a core node -- no comfyui_controlnet_aux needed.

    ControlNetApplyAdvanced emits BOTH conditionings, so the sampler's
    positive and negative must be rewired to outputs 0 and 1 of the same
    node. Rewiring only positive is a silent correctness bug.

    Raises ValueError for a Flux profile: the canny weights really are SDXL
    (registry.CONTROLNET_CANNY), and Flux ControlNet is out of scope at
    22.17 GB. There is no Flux-correct graph to fall back to, so this refuses
    rather than silently producing a wrong one.
    """
    if is_flux(profile):
        raise ValueError(
            f"controlnet_canny is SDXL-only: {control_net!r} holds SDXL "
            f"weights and cannot condition the {profile!r} checkpoint "
            f"{ckpt!r}. Flux ControlNet (flux1-canny-dev, 22.17 GB) is out of "
            "scope -- use profile='sdxl', or img2img() for a Flux edit."
        )
    g = _txt2img(ckpt, prompt, negative, seed, steps, cfg, width, height,
                 batch, sampler, scheduler, "colab/controlnet", profile, 3.5)
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
    return g
