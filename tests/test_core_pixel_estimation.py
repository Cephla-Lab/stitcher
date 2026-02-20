"""Tests for pixel size estimation."""

import numpy as np
import pytest
from unittest.mock import MagicMock


class TestEstimatePixelSize:
    """Tests for TileFusion.estimate_pixel_size()."""

    def _create_mock_tilefusion(self, tile_positions, pixel_size, pairwise_metrics):
        """Create a mock TileFusion with required state."""
        from tilefusion.core import TileFusion

        mock = MagicMock(spec=TileFusion)
        mock._tile_positions = tile_positions
        mock._pixel_size = pixel_size
        mock.pairwise_metrics = pairwise_metrics

        # Bind the real method
        mock.estimate_pixel_size = lambda: TileFusion.estimate_pixel_size(mock)
        return mock

    def test_perfect_calibration(self):
        """When measured shifts match expected, deviation should be ~0%."""
        tile_positions = [(0, 0), (0, 90), (90, 0), (90, 90)]
        pixel_size = (1.0, 1.0)
        pairwise_metrics = {
            (0, 1): (0, 90, 0.95),
            (0, 2): (90, 0, 0.95),
            (1, 3): (90, 0, 0.95),
            (2, 3): (0, 90, 0.95),
        }

        tf = self._create_mock_tilefusion(tile_positions, pixel_size, pairwise_metrics)
        estimated, deviation = tf.estimate_pixel_size()

        assert abs(estimated - 1.0) < 0.01
        assert abs(deviation) < 1.0

    def test_pixel_size_underestimated(self):
        """When metadata pixel size is too small, estimated should be larger."""
        tile_positions = [(0, 0), (0, 90)]
        pixel_size = (1.0, 1.0)
        pairwise_metrics = {(0, 1): (0, 82, 0.95)}

        tf = self._create_mock_tilefusion(tile_positions, pixel_size, pairwise_metrics)
        estimated, deviation = tf.estimate_pixel_size()

        assert 1.05 < estimated < 1.15
        assert deviation > 5.0

    def test_no_metrics_raises(self):
        """Should raise if no pairwise metrics."""
        tf = self._create_mock_tilefusion([], (1.0, 1.0), {})

        with pytest.raises(ValueError, match="No pairwise metrics"):
            tf.estimate_pixel_size()
