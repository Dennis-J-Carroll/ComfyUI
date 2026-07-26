import pytest
import torch
import av
from fractions import Fraction
from comfy_api.latest._input_impl.video_types import VideoFromFile, VideoFromComponents
from comfy_api.latest._util.video_types import VideoComponents, normalize_crop_rect


RED = torch.tensor([1.0, 0.0, 0.0])
GREEN = torch.tensor([0.0, 1.0, 0.0])
BLUE = torch.tensor([0.0, 0.0, 1.0])
WHITE = torch.tensor([1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def quadrant_components():
    """64x64, 2 frames; each 32x32 quadrant a distinct solid color to locate crops"""
    images = torch.zeros(2, 64, 64, 3)
    images[:, :32, :32] = RED
    images[:, :32, 32:] = GREEN
    images[:, 32:, :32] = BLUE
    images[:, 32:, 32:] = WHITE
    return VideoComponents(images=images, frame_rate=Fraction(30))


@pytest.fixture(scope="module")
def src(quadrant_components, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "src.mp4")
    VideoFromComponents(quadrant_components).save_to(path)
    return path


def probe_dimensions(path):
    with av.open(path) as container:
        stream = container.streams.video[0]
        return stream.width, stream.height


def decoded_center_color(path):
    """Mean RGB of the center 8x8 of the first decoded frame"""
    with av.open(path) as container:
        frame = next(container.decode(container.streams.video[0]))
        rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0
        h, w = rgb.shape[0], rgb.shape[1]
        return rgb[h // 2 - 4:h // 2 + 4, w // 2 - 4:w // 2 + 4].mean(dim=(0, 1))


def assert_color(actual, expected, tolerance=0.15):
    assert torch.allclose(actual, expected, atol=tolerance), f"{actual} != {expected}"


class TestNormalizeCropRect:
    def test_clamps_to_frame_and_even_aligns(self):
        assert normalize_crop_rect(10, 10, 100, 100, 64, 64) == (10, 10, 54, 54)
        assert normalize_crop_rect(0, 0, 33, 31, 64, 64) == (0, 0, 32, 30)

    def test_empty_and_full_frame_are_none(self):
        assert normalize_crop_rect(0, 0, 0, 0, 64, 64) is None
        assert normalize_crop_rect(0, 0, -2, 10, 64, 64) is None
        assert normalize_crop_rect(0, 0, 64, 64, 64, 64) is None

    def test_negative_origin_clamps_to_zero(self):
        assert normalize_crop_rect(-10, -10, 32, 32, 64, 64) == (0, 0, 32, 32)


class TestVideoFromComponentsCrop:
    def test_slices_images(self, quadrant_components):
        cropped = VideoFromComponents(quadrant_components).as_cropped(32, 0, 32, 32)
        components = cropped.get_components()
        assert components.images.shape == (2, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), GREEN, tolerance=0.01)

    def test_noop_returns_self(self, quadrant_components):
        video = VideoFromComponents(quadrant_components)
        assert video.as_cropped(0, 0, 0, 0) is video
        assert video.as_cropped(0, 0, 64, 64) is video

    def test_slices_alpha(self, quadrant_components):
        alpha = torch.zeros(2, 64, 64)
        alpha[:, :32, :32] = 1.0
        with_alpha = VideoComponents(
            images=quadrant_components.images,
            frame_rate=quadrant_components.frame_rate,
            alpha=alpha,
        )
        components = VideoFromComponents(with_alpha).as_cropped(0, 0, 32, 32).get_components()
        assert components.alpha.shape == (2, 32, 32)
        assert components.alpha.min() == 1.0


class TestVideoFromFileCrop:
    def test_dimensions_and_components(self, src):
        cropped = VideoFromFile(src).as_cropped(0, 32, 32, 32)
        assert cropped.get_dimensions() == (32, 32)
        components = cropped.get_components()
        assert components.images.shape == (2, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), BLUE)

    def test_save_streams_cropped_output(self, src, tmp_path):
        path = str(tmp_path / "cropped.mp4")
        VideoFromFile(src).as_cropped(32, 32, 32, 32).save_to(path)
        assert probe_dimensions(path) == (32, 32)
        assert_color(decoded_center_color(path), WHITE)

    def test_noop_returns_self(self, src):
        video = VideoFromFile(src)
        assert video.as_cropped(0, 0, 0, 0) is video

    def test_composes_with_trim(self, src, tmp_path):
        video = VideoFromFile(src).as_trimmed(0, 1 / 30, strict_duration=False).as_cropped(32, 0, 32, 32)
        components = video.get_components()
        assert components.images.shape == (1, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), GREEN)

        path = str(tmp_path / "trim_crop.mp4")
        video.save_to(path)
        assert probe_dimensions(path) == (32, 32)
        assert_color(decoded_center_color(path), GREEN)

    def test_crop_survives_trim(self, src):
        video = VideoFromFile(src).as_cropped(32, 0, 32, 32).as_trimmed(0, 1 / 30, strict_duration=False)
        components = video.get_components()
        assert components.images.shape == (1, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), GREEN)

    def test_crop_of_crop_composes(self, src):
        cropped = VideoFromFile(src).as_cropped(32, 0, 32, 64).as_cropped(0, 32, 32, 32)
        components = cropped.get_components()
        assert components.images.shape == (2, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), WHITE)
