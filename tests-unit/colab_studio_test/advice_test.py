from colab_studio.advice import recommend, Advice
from colab_studio.registry import PROFILES


def test_t4_is_mid_tier():
    # A Colab T4 reports ~14.7 GB, not a clean 16.
    a = recommend(14.7)
    assert a.tier == "mid"
    assert a.profile == "sdxl"


def test_t4_never_gets_highvram():
    # --highvram on a T4 is actively wrong; this is the regression guard.
    assert "--highvram" not in recommend(14.7).launch_flags


def test_l4_is_high_tier_and_gets_flux():
    a = recommend(22.5)
    assert a.tier == "high"
    assert a.profile.startswith("flux")
    assert "--highvram" in a.launch_flags


def test_a100_is_high_tier():
    assert recommend(40.0).tier == "high"


def test_low_vram_drops_resolution_and_avoids_flux():
    a = recommend(8.0)
    assert a.tier == "low"
    assert a.max_side <= 768
    assert not a.profile.startswith("flux")


def test_recommended_profile_always_exists_in_registry():
    for vram in (6.0, 8.0, 14.7, 16.0, 22.5, 40.0, 80.0):
        assert recommend(vram).profile in PROFILES


def test_low_disk_forces_small_profile_even_on_big_gpu():
    # Flux is 16 GB; refusing it when disk is tight prevents a mid-download death.
    a = recommend(40.0, disk_free_gb=12.0)
    assert a.profile == "sdxl"
    assert any("disk" in n.lower() for n in a.notes)


def test_advice_is_frozen():
    a = recommend(22.5)
    assert isinstance(a, Advice)
    try:
        a.tier = "low"          # type: ignore[misc]
        raise AssertionError("Advice should be immutable")
    except AttributeError:
        pass
