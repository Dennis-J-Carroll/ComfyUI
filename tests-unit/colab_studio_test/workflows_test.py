"""Validates generated graphs against ComfyUI's own prompt validator.

This is structural validation: class names, input keys, enum membership,
link types. It does not execute nodes.
"""
import asyncio
import contextlib
import io
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def comfy(tmp_path_factory):
    """Boot ComfyUI's node registry in CPU mode with dummy models present."""
    sys.argv = ["main.py", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()   # MUST precede `import nodes`

    root = tmp_path_factory.mktemp("models")
    for sub, name in [
        ("checkpoints", "sd_xl_base_1.0.safetensors"),
        ("checkpoints", "flux1-dev-fp8.safetensors"),
        ("vae", "sdxl_vae.safetensors"),
        ("controlnet", "controlnet-canny-sdxl.safetensors"),
        ("upscale_models", "4x-UltraSharp.pth"),
    ]:
        d = root / sub
        d.mkdir(exist_ok=True)
        (d / name).touch()

    probe = os.path.join(REPO, "input", "_wf_probe.png")
    os.makedirs(os.path.join(REPO, "input"), exist_ok=True)
    open(probe, "a").close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import folder_paths
        for sub in ("checkpoints", "vae", "controlnet", "upscale_models"):
            folder_paths.add_model_folder_path(sub, str(root / sub))
        import nodes
        import execution
        r = nodes.init_extra_nodes(init_api_nodes=False)
        if asyncio.iscoroutine(r):
            asyncio.run(r)
    yield execution
    os.remove(probe)


def assert_valid(execution, name, graph):
    res = asyncio.run(execution.validate_prompt(name, graph, None))
    assert res[0], f"{name} invalid: {res[1]} / {res[3]}"


def test_sdxl_txt2img_validates(comfy):
    from colab_studio.workflows import sdxl_txt2img
    assert_valid(comfy, "sdxl", sdxl_txt2img("sd_xl_base_1.0.safetensors", "a cat"))


def test_flux_txt2img_validates(comfy):
    from colab_studio.workflows import flux_txt2img
    assert_valid(comfy, "flux", flux_txt2img("flux1-dev-fp8.safetensors", "a cat"))


def test_flux_forces_cfg_one(comfy):
    # Flux ignores CFG; real guidance rides on FluxGuidance. cfg != 1.0 scorches output.
    from colab_studio.workflows import flux_txt2img
    g = flux_txt2img("flux1-dev-fp8.safetensors", "a cat", cfg=7.0)
    ks = [n for n in g.values() if n["class_type"] == "KSampler"][0]
    assert ks["inputs"]["cfg"] == 1.0
    fg = [n for n in g.values() if n["class_type"] == "FluxGuidance"]
    assert len(fg) == 1


def test_img2img_validates_and_uses_vaeencode(comfy):
    from colab_studio.workflows import img2img
    g = img2img("sd_xl_base_1.0.safetensors", "a cat", image="_wf_probe.png", denoise=0.6)
    assert_valid(comfy, "img2img", g)
    assert any(n["class_type"] == "VAEEncode" for n in g.values())
    assert not any(n["class_type"] == "EmptyLatentImage" for n in g.values())


def test_upscale_validates_and_saves_upscaled_image(comfy):
    from colab_studio.workflows import upscale
    g = upscale("sd_xl_base_1.0.safetensors", "a cat")
    assert_valid(comfy, "upscale", g)
    save = [n for n in g.values() if n["class_type"] == "SaveImage"][0]
    src = save["inputs"]["images"][0]
    assert g[src]["class_type"] == "ImageUpscaleWithModel"


def test_controlnet_validates_and_rewires_both_conditionings(comfy):
    from colab_studio.workflows import controlnet_canny
    g = controlnet_canny("sd_xl_base_1.0.safetensors", "a cat", image="_wf_probe.png")
    assert_valid(comfy, "controlnet", g)
    ks = [n for n in g.values() if n["class_type"] == "KSampler"][0]
    pos, neg = ks["inputs"]["positive"], ks["inputs"]["negative"]
    assert g[pos[0]]["class_type"] == "ControlNetApplyAdvanced"
    assert g[neg[0]]["class_type"] == "ControlNetApplyAdvanced"
    assert pos[1] == 0 and neg[1] == 1


def test_seed_and_size_are_threaded_through(comfy):
    from colab_studio.workflows import sdxl_txt2img
    g = sdxl_txt2img("sd_xl_base_1.0.safetensors", "a cat",
                     seed=1234, width=768, height=512, steps=12)
    ks = [n for n in g.values() if n["class_type"] == "KSampler"][0]
    lat = [n for n in g.values() if n["class_type"] == "EmptyLatentImage"][0]
    assert ks["inputs"]["seed"] == 1234
    assert ks["inputs"]["steps"] == 12
    assert (lat["inputs"]["width"], lat["inputs"]["height"]) == (768, 512)


def test_negative_control_bad_sampler_is_rejected(comfy):
    # Proves the validator is engaged rather than vacuously passing.
    from colab_studio.workflows import sdxl_txt2img
    g = sdxl_txt2img("sd_xl_base_1.0.safetensors", "a cat")
    ks_id = [k for k, n in g.items() if n["class_type"] == "KSampler"][0]
    g[ks_id]["inputs"]["sampler_name"] = "not_a_real_sampler"
    res = asyncio.run(comfy.validate_prompt("bad", g, None))
    assert not res[0]
