import numpy as np
import pytest

from colorcast.analysis.appearance import (
    CIELABSpace,
    ICtCpSpace,
    _pq_eotf_inv,
)
from colorcast.utils.color_utils import srgb_to_linear


def test_pq_eotf_inv_uses_absolute_luminance_reference() -> None:
    values = np.array([0.0, 10000.0], dtype=np.float64)

    result = _pq_eotf_inv(values)

    assert result[0] < 1e-6
    np.testing.assert_allclose(result[1], 1.0, rtol=1e-6, atol=1e-6)


def test_srgb_to_linear_does_not_mutate_source_array() -> None:
    rgb = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    original = rgb.copy()

    result = srgb_to_linear(rgb)

    np.testing.assert_array_equal(rgb, original)
    assert result is not rgb
    assert result.dtype == np.float64


def test_cielab_identical_images_give_zero_delta() -> None:
    rgb = np.random.rand(64, 64, 3).astype(np.float32)
    space = CIELABSpace()
    delta = space.delta_E(rgb, rgb)
    assert delta.space_name == "CIELAB (ciede2000)"
    assert delta.values.shape == (64, 64)
    assert np.all(delta.values >= 0)
    assert np.max(delta.values) < 1e-6


def test_ictcp_identical_images_give_zero_delta() -> None:
    rgb = np.random.rand(64, 64, 3).astype(np.float32)
    space = ICtCpSpace()
    delta = space.delta_E(rgb, rgb)
    assert delta.space_name == "ICtCp"
    assert delta.values.shape == (64, 64)
    assert np.all(delta.values >= 0)
    assert np.max(delta.values) < 1e-5


def test_ictcp_delta_is_monotonic_with_distortion() -> None:
    rgb = np.random.rand(32, 32, 3).astype(np.float32)
    space = ICtCpSpace()
    delta_small = space.delta_E(rgb, rgb * 0.95)
    delta_large = space.delta_E(rgb, rgb * 0.80)
    assert np.mean(delta_small.values) < np.mean(delta_large.values)


def test_cielab_from_rgb_returns_lab_shape() -> None:
    rgb = np.random.rand(16, 32, 3).astype(np.float32)
    space = CIELABSpace()
    lab = space.from_rgb(rgb)
    assert lab.shape == (16, 32, 3)
    assert lab.dtype == np.float32


def test_ictcp_from_rgb_returns_ictcp_shape() -> None:
    rgb = np.random.rand(16, 32, 3).astype(np.float32)
    space = ICtCpSpace()
    ictcp = space.from_rgb(rgb)
    assert ictcp.shape == (16, 32, 3)
    assert ictcp.dtype == np.float32


def test_ictcp_space_accepts_explicit_peak_luminance() -> None:
    rgb = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    space = ICtCpSpace(peak_luminance=200.0)
    ictcp = space.from_rgb(rgb)

    assert ictcp.shape == (1, 1, 3)
    assert ictcp.dtype == np.float32
    assert not np.isnan(ictcp).any()


def test_ictcp_space_rejects_unknown_transfer_function() -> None:
    with pytest.raises(ValueError, match="unsupported transfer function"):
        ICtCpSpace(transfer_function="unknown")


def test_appearance_spaces_have_different_names() -> None:
    assert CIELABSpace().name == "CIELAB (ciede2000)"
    assert CIELABSpace(metric="cie76").name == "CIELAB (cie76)"
    assert ICtCpSpace().name == "ICtCp"
    assert CIELABSpace().name != ICtCpSpace().name
    assert CIELABSpace().name != CIELABSpace(metric="cie76").name


def test_cielab_ciede2000_identical_images_give_zero_delta() -> None:
    rgb = np.random.rand(64, 64, 3).astype(np.float32)
    space = CIELABSpace(metric="ciede2000")
    delta = space.delta_E(rgb, rgb)
    assert delta.space_name == "CIELAB (ciede2000)"
    assert delta.values.shape == (64, 64)
    assert np.all(delta.values >= 0)
    assert np.max(delta.values) < 1e-6


def test_cielab_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unsupported metric"):
        CIELABSpace(metric="unknown")
