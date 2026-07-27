"""Tests for MethodComparison metrics and ranking."""

import numpy as np
import pytest
from colorcast.analysis.comparison import MethodComparison, _METRIC_DIRECTION


class TestComputePsnr:
    """Tests for PSNR computation."""

    def test_identical_images(self):
        img = np.random.rand(100, 100, 3).astype(np.float32)
        psnr = MethodComparison.compute_psnr(img, img)
        assert psnr == float("inf")

    def test_different_images(self):
        img1 = np.zeros((100, 100, 3), dtype=np.float32)
        img2 = np.ones((100, 100, 3), dtype=np.float32)
        psnr = MethodComparison.compute_psnr(img1, img2)
        assert np.isfinite(psnr)
        assert psnr < float("inf")

    def test_similar_images(self):
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = img1 + np.random.normal(0, 0.01, img1.shape).astype(np.float32)
        img2 = np.clip(img2, 0, 1)
        psnr = MethodComparison.compute_psnr(img1, img2)
        assert psnr > 20.0

    def test_return_is_finite(self):
        img1 = np.random.rand(50, 50, 3).astype(np.float32)
        img2 = np.random.rand(50, 50, 3).astype(np.float32)
        psnr = MethodComparison.compute_psnr(img1, img2)
        assert np.isfinite(psnr)

    def test_constant_images(self):
        img1 = np.full((50, 50, 3), 0.3, dtype=np.float32)
        img2 = np.full((50, 50, 3), 0.3, dtype=np.float32)
        psnr = MethodComparison.compute_psnr(img1, img2)
        assert psnr == float("inf")


class TestComputeSsim:
    """Tests for SSIM computation."""

    def test_identical_images(self):
        img = np.random.rand(100, 100, 3).astype(np.float32)
        ssim_val = MethodComparison.compute_ssim(img, img)
        np.testing.assert_allclose(ssim_val, 1.0, atol=1e-6)

    def test_different_images(self):
        img1 = np.zeros((100, 100, 3), dtype=np.float32)
        img2 = np.ones((100, 100, 3), dtype=np.float32)
        ssim_val = MethodComparison.compute_ssim(img1, img2)
        assert ssim_val < 0.1

    def test_return_range(self):
        img1 = np.random.rand(50, 50, 3).astype(np.float32)
        img2 = np.random.rand(50, 50, 3).astype(np.float32)
        ssim_val = MethodComparison.compute_ssim(img1, img2)
        assert -1.0 <= ssim_val <= 1.0

    def test_return_is_scalar(self):
        img1 = np.random.rand(50, 50, 3).astype(np.float32)
        img2 = np.random.rand(50, 50, 3).astype(np.float32)
        ssim_val = MethodComparison.compute_ssim(img1, img2)
        assert np.isscalar(ssim_val)


class TestComputeColorDistance:
    """Tests for color distance computation."""

    def test_identical_images(self):
        img = np.random.rand(100, 100, 3).astype(np.float32)
        distance = MethodComparison.compute_color_distance(img, img)
        assert distance == pytest.approx(0.0, abs=1e-7)

    def test_different_images(self):
        img1 = np.zeros((100, 100, 3), dtype=np.float32)
        img2 = np.ones((100, 100, 3), dtype=np.float32)
        distance = MethodComparison.compute_color_distance(img1, img2)
        assert distance > 0.0

    def test_return_type(self):
        img1 = np.random.rand(50, 50, 3).astype(np.float32)
        img2 = np.random.rand(50, 50, 3).astype(np.float32)
        distance = MethodComparison.compute_color_distance(img1, img2)
        assert isinstance(distance, (float, np.floating))

    def test_non_negative(self):
        img1 = np.random.rand(50, 50, 3).astype(np.float32)
        img2 = np.random.rand(50, 50, 3).astype(np.float32)
        distance = MethodComparison.compute_color_distance(img1, img2)
        assert distance >= 0.0


class TestComputeHistogramDistance:
    """Tests for histogram distance (EMD) computation."""

    def test_identical_images(self):
        img = (np.random.rand(100, 100, 3)).astype(np.float32)
        distance = MethodComparison.compute_histogram_distance(img, img)
        assert distance == pytest.approx(0.0, abs=1e-7)

    def test_different_images(self):
        img1 = np.zeros((100, 100, 3), dtype=np.float32)
        img2 = np.ones((100, 100, 3), dtype=np.float32)
        distance = MethodComparison.compute_histogram_distance(img1, img2)
        assert distance > 0.0

    def test_return_type(self):
        img1 = np.random.rand(50, 50, 3).astype(np.float32)
        img2 = np.random.rand(50, 50, 3).astype(np.float32)
        distance = MethodComparison.compute_histogram_distance(img1, img2)
        assert isinstance(distance, (float, np.floating))

    def test_empty_channel_does_not_produce_nan(self):
        """Zero-sum histograms must not produce NaN."""
        img1 = np.empty((0, 0, 3), dtype=np.float32)
        img2 = np.empty((0, 0, 3), dtype=np.float32)
        distance = MethodComparison.compute_histogram_distance(img1, img2)
        assert not np.isnan(distance)
        assert distance == pytest.approx(0.0, abs=1e-7)

    def test_custom_bins(self):
        img1 = np.random.rand(50, 50, 3).astype(np.float32)
        img2 = np.random.rand(50, 50, 3).astype(np.float32)
        distance = MethodComparison.compute_histogram_distance(img1, img2, bins=128)
        assert isinstance(distance, (float, np.floating))
        assert distance >= 0.0


class TestCompareMethods:
    """Tests for the compare_methods orchestration."""

    @pytest.fixture
    def source(self):
        return np.random.rand(100, 100, 3).astype(np.float32)

    @pytest.fixture
    def reference(self):
        return np.random.rand(100, 100, 3).astype(np.float32)

    @pytest.fixture
    def identity_method(self):
        return lambda src, ref: src

    @pytest.fixture
    def failing_method(self):
        def _fail(src, ref):
            raise RuntimeError("test failure")
        return _fail

    def test_baseline_included(self, source, reference, identity_method):
        cm = MethodComparison()
        methods = {"identity": identity_method}
        results = cm.compare_methods(source, reference, methods, include_baseline=True)
        assert "baseline" in results
        assert "identity" in results

    def test_baseline_excluded(self, source, reference, identity_method):
        cm = MethodComparison()
        methods = {"identity": identity_method}
        results = cm.compare_methods(source, reference, methods, include_baseline=False)
        assert "baseline" not in results
        assert "identity" in results

    def test_baseline_metrics(self, source, reference, identity_method):
        cm = MethodComparison()
        methods = {"identity": identity_method}
        results = cm.compare_methods(source, reference, methods, include_baseline=True)
        baseline = results["baseline"]
        assert "psnr" in baseline
        assert "ssim" in baseline
        assert baseline["color_distance"] == pytest.approx(0.0, abs=1e-7)
        assert "histogram_distance" in baseline

    def test_all_required_metrics_present(self, source, reference, identity_method):
        cm = MethodComparison()
        methods = {"identity": identity_method}
        results = cm.compare_methods(source, reference, methods)
        for metric in ("psnr", "ssim", "color_distance", "histogram_distance"):
            assert metric in results["identity"]

    def test_failing_method_returns_nan(self, source, reference, failing_method):
        cm = MethodComparison()
        methods = {"failer": failing_method}
        results = cm.compare_methods(source, reference, methods)
        entry = results["failer"]
        assert np.isnan(entry["psnr"])
        assert np.isnan(entry["ssim"])
        assert np.isnan(entry["color_distance"])
        assert np.isnan(entry["histogram_distance"])
        assert "_error" in entry

    def test_multiple_methods(self, source, reference, identity_method):
        cm = MethodComparison()
        methods = {
            "method_a": identity_method,
            "method_b": identity_method,
        }
        results = cm.compare_methods(source, reference, methods, include_baseline=True)
        assert len(results) == 3  # 2 methods + baseline


class TestRankMethods:
    """Tests for the rank_methods function."""

    def test_ascending_ranks_low_to_high(self):
        cm = MethodComparison()
        results = {
            "A": {"psnr": 10.0},
            "B": {"psnr": 20.0},
            "C": {"psnr": 5.0},
        }
        ranking = cm.rank_methods(results, "psnr", ascending=True)
        assert ranking[0][0] == "C"
        assert ranking[1][0] == "A"
        assert ranking[2][0] == "B"

    def test_descending_ranks_high_to_low(self):
        cm = MethodComparison()
        results = {
            "A": {"psnr": 10.0},
            "B": {"psnr": 20.0},
            "C": {"psnr": 5.0},
        }
        ranking = cm.rank_methods(results, "psnr", ascending=False)
        assert ranking[0][0] == "B"
        assert ranking[1][0] == "A"
        assert ranking[2][0] == "C"

    def test_auto_infer_direction_psnr(self):
        cm = MethodComparison()
        results = {"A": {"psnr": 10.0}, "B": {"psnr": 20.0}}
        ranking = cm.rank_methods(results, "psnr", ascending=None)
        assert ranking[0][0] == "B"

    def test_auto_infer_direction_color_distance(self):
        cm = MethodComparison()
        results = {"A": {"color_distance": 0.3}, "B": {"color_distance": 0.1}}
        ranking = cm.rank_methods(results, "color_distance", ascending=None)
        assert ranking[0][0] == "B"

    def test_excludes_baseline(self):
        cm = MethodComparison()
        results = {
            "baseline": {"psnr": 999.0},
            "A": {"psnr": 10.0},
            "B": {"psnr": 20.0},
        }
        ranking = cm.rank_methods(results, "psnr", ascending=False)
        assert len(ranking) == 2
        assert "baseline" not in [r[0] for r in ranking]

    def test_excludes_nan_entries(self):
        cm = MethodComparison()
        results = {
            "A": {"psnr": 10.0},
            "B": {"psnr": float("nan")},
            "C": {"psnr": 5.0},
        }
        ranking = cm.rank_methods(results, "psnr", ascending=False)
        names = [r[0] for r in ranking]
        assert "B" not in names
        assert len(ranking) == 2

    def test_explicit_ascending_overrides_direction(self):
        cm = MethodComparison()
        results = {"A": {"psnr": 10.0}, "B": {"psnr": 20.0}}
        ranking = cm.rank_methods(results, "psnr", ascending=True)
        assert ranking[0][0] == "A"


class TestGenerateComparisonReport:
    """Tests for report generation."""

    def test_report_is_string(self):
        cm = MethodComparison()
        results = {"method_a": {"psnr": 20.0, "ssim": 0.9, "color_distance": 0.1, "histogram_distance": 0.05}}
        report = cm.generate_comparison_report(results)
        assert isinstance(report, str)

    def test_report_contains_method_name(self):
        cm = MethodComparison()
        results = {"test_method": {"psnr": 20.0, "ssim": 0.9, "color_distance": 0.1, "histogram_distance": 0.05}}
        report = cm.generate_comparison_report(results)
        assert "test_method" in report

    def test_report_contains_rankings_section(self):
        cm = MethodComparison()
        results = {"A": {"psnr": 20.0, "ssim": 0.9, "color_distance": 0.1, "histogram_distance": 0.05}}
        report = cm.generate_comparison_report(results)
        assert "Rankings" in report


class TestFindBestMethod:
    """Tests for find_best_method."""

    def test_returns_best_by_ssim(self):
        cm = MethodComparison()
        results = {
            "A": {"ssim": 0.8, "psnr": 30.0},
            "B": {"ssim": 0.9, "psnr": 25.0},
        }
        name, value = cm.find_best_method(results, "ssim")
        assert name == "B"
        assert value == pytest.approx(0.9)

    def test_returns_best_by_color_distance(self):
        cm = MethodComparison()
        results = {
            "A": {"color_distance": 0.3, "psnr": 30.0},
            "B": {"color_distance": 0.1, "psnr": 25.0},
        }
        name, _ = cm.find_best_method(results, "color_distance")
        assert name == "B"

    def test_raises_on_empty_ranking(self):
        cm = MethodComparison()
        results = {"A": {"ssim": float("nan")}}
        with pytest.raises(ValueError, match="No valid results"):
            cm.find_best_method(results, "ssim")

    def test_excludes_baseline(self):
        cm = MethodComparison()
        results = {
            "baseline": {"ssim": 0.99},
            "A": {"ssim": 0.5},
        }
        name, _ = cm.find_best_method(results, "ssim")
        assert name == "A"


class TestMetricDirection:
    """Tests for metric direction mapping."""

    def test_psnr_higher_is_better(self):
        assert _METRIC_DIRECTION["psnr"] is True

    def test_ssim_higher_is_better(self):
        assert _METRIC_DIRECTION["ssim"] is True

    def test_color_distance_lower_is_better(self):
        assert _METRIC_DIRECTION["color_distance"] is False

    def test_histogram_distance_lower_is_better(self):
        assert _METRIC_DIRECTION["histogram_distance"] is False
