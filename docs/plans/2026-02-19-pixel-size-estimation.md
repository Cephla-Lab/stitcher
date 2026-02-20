# Pixel Size Estimation - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Estimate true pixel size from registration results by comparing expected vs measured tile shifts.

**Architecture:** Add `estimate_pixel_size()` method to TileFusion that analyzes pairwise_metrics after registration. For each pair, compute ratio of expected/measured shift, take median, multiply by metadata pixel size. Add GUI checkbox to optionally use estimated value.

**Tech Stack:** Python, NumPy, PyQt5

---

### Task 1: Add estimate_pixel_size() method to TileFusion

**Files:**
- Modify: `src/tilefusion/core.py`
- Test: `tests/test_core_pixel_estimation.py`

**Step 1: Write the test**

Create `tests/test_core_pixel_estimation.py`:

```python
"""Tests for pixel size estimation."""

import numpy as np
import pytest


class TestEstimatePixelSize:
    """Tests for TileFusion.estimate_pixel_size()."""

    def test_perfect_calibration(self):
        """When measured shifts match expected, ratio should be 1.0."""
        from tilefusion.core import TileFusion

        # Create minimal mock - we'll set up the state directly
        # For this test, we need pairwise_metrics and tile_positions

        # Simulate 2x2 grid with 10% overlap, pixel_size=1.0
        # Tile size 100x100, so tiles at (0,0), (0,90), (90,0), (90,90)
        tile_positions = [(0, 0), (0, 90), (90, 0), (90, 90)]
        pixel_size = (1.0, 1.0)

        # Expected shift for horizontal neighbor: 90 pixels (stage) / 1.0 (px_size) = 90
        # If measured shift is also 90, ratio = 1.0
        # pairwise_metrics format: {(i, j): (dy, dx, score)}
        pairwise_metrics = {
            (0, 1): (0, 90, 0.95),   # horizontal pair
            (0, 2): (90, 0, 0.95),   # vertical pair
            (1, 3): (90, 0, 0.95),   # vertical pair
            (2, 3): (0, 90, 0.95),   # horizontal pair
        }

        estimated, deviation = _estimate_pixel_size_from_metrics(
            pairwise_metrics, tile_positions, pixel_size
        )

        assert abs(estimated - 1.0) < 0.01
        assert abs(deviation) < 1.0  # Less than 1% deviation

    def test_pixel_size_too_small_in_metadata(self):
        """When metadata pixel size is too small, estimated should be larger."""
        # If metadata says 1.0 but true is 1.1:
        # Expected shift = 90 / 1.0 = 90 pixels
        # Measured shift = 90 / 1.1 = 81.8 pixels (tiles closer in pixels)
        # Ratio = 90 / 81.8 = 1.1
        # Estimated = 1.0 * 1.1 = 1.1

        tile_positions = [(0, 0), (0, 90)]  # 90 µm apart
        pixel_size = (1.0, 1.0)  # metadata says 1.0

        # Measured shift is 82 pixels (as if true pixel size were ~1.1)
        pairwise_metrics = {
            (0, 1): (0, 82, 0.95),
        }

        estimated, deviation = _estimate_pixel_size_from_metrics(
            pairwise_metrics, tile_positions, pixel_size
        )

        # Expected: 90/82 * 1.0 ≈ 1.098
        assert 1.05 < estimated < 1.15
        assert deviation > 5.0  # More than 5% deviation


def _estimate_pixel_size_from_metrics(pairwise_metrics, tile_positions, pixel_size):
    """Helper to test the core algorithm without full TileFusion."""
    from tilefusion.core import TileFusion
    # This will be implemented as a static/class method or we test via TileFusion
    # For now, placeholder
    raise NotImplementedError("Implement estimate_pixel_size first")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core_pixel_estimation.py -v`
Expected: FAIL with "NotImplementedError"

**Step 3: Implement estimate_pixel_size() in core.py**

Add after `refine_tile_positions_with_cross_correlation` method (around line 815):

```python
def estimate_pixel_size(self) -> Tuple[float, float]:
    """
    Estimate pixel size from registration results.

    Compares expected shifts (from stage positions / metadata pixel size)
    with measured shifts (from cross-correlation) to estimate true pixel size.

    Returns
    -------
    estimated_pixel_size : float
        Estimated pixel size in same units as metadata (typically µm).
    deviation_percent : float
        Percentage deviation from metadata: (estimated/metadata - 1) * 100

    Raises
    ------
    ValueError
        If no valid pairwise metrics available.
    """
    if not self.pairwise_metrics:
        raise ValueError("No pairwise metrics available. Run registration first.")

    ratios = []

    for (i, j), (dy_measured, dx_measured, score) in self.pairwise_metrics.items():
        # Get stage positions
        pos_i = np.array(self._tile_positions[i])
        pos_j = np.array(self._tile_positions[j])

        # Expected shift in pixels = stage_distance / pixel_size
        stage_diff = pos_j - pos_i  # (dy, dx) in physical units
        expected_dy = stage_diff[0] / self._pixel_size[0]
        expected_dx = stage_diff[1] / self._pixel_size[1]

        # Compute ratio for non-zero shifts
        if abs(dx_measured) > 5:  # Horizontal shift
            ratio = expected_dx / dx_measured
            ratios.append(ratio)
        if abs(dy_measured) > 5:  # Vertical shift
            ratio = expected_dy / dy_measured
            ratios.append(ratio)

    if not ratios:
        raise ValueError("No valid shift measurements for pixel size estimation.")

    # Use median to filter outliers
    median_ratio = float(np.median(ratios))

    # Estimated pixel size (assume isotropic)
    estimated = self._pixel_size[0] * median_ratio
    deviation_percent = (median_ratio - 1.0) * 100.0

    return estimated, deviation_percent
```

**Step 4: Update the test to use the real method**

Update `tests/test_core_pixel_estimation.py`:

```python
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
        # Measured 82 pixels instead of expected 90
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
```

**Step 5: Run tests**

Run: `pytest tests/test_core_pixel_estimation.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/tilefusion/core.py tests/test_core_pixel_estimation.py
git commit -m "feat: Add estimate_pixel_size() method to TileFusion"
```

---

### Task 2: Add GUI checkbox and wire up pixel size estimation

**Files:**
- Modify: `gui/app.py`

**Step 1: Add state variable and checkbox**

In `StitcherGUI.__init__` after the dataset dimension state (around line 735), add:
```python
self.estimated_pixel_size = None
```

In `setup_ui`, in the registration sub-options section (after channel combo, before `reg_zt_layout.addStretch()`):
```python
self.use_estimated_px_checkbox = QCheckBox("Use estimated pixel size")
self.use_estimated_px_checkbox.setToolTip(
    "Estimate pixel size from registration and use it for stitching"
)
self.use_estimated_px_checkbox.setChecked(False)
settings_layout.addWidget(self.use_estimated_px_checkbox)
```

**Step 2: Call estimate_pixel_size after registration in FusionWorker**

In `FusionWorker.run()`, after registration completes (after `tf.save_pairwise_metrics`), add:
```python
# Estimate pixel size
try:
    estimated_px, deviation = tf.estimate_pixel_size()
    self.progress.emit(
        f"Estimated pixel size: {estimated_px:.4f} µm ({deviation:+.1f}% from metadata)"
    )
    if self.use_estimated_pixel_size and abs(deviation) > 1.0:
        tf._pixel_size = (estimated_px, estimated_px)
        self.progress.emit(f"Using estimated pixel size for stitching")
except ValueError as e:
    self.progress.emit(f"Could not estimate pixel size: {e}")
```

**Step 3: Add parameter to FusionWorker**

Add `use_estimated_pixel_size=False` to `FusionWorker.__init__` and store it.

**Step 4: Pass checkbox value in run_stitching**

In `run_stitching()`, pass `use_estimated_pixel_size=self.use_estimated_px_checkbox.isChecked()` to FusionWorker.

**Step 5: Commit**

```bash
git add gui/app.py
git commit -m "feat: Add GUI option to use estimated pixel size"
```

---

### Task 3: Add pixel size estimation to PreviewWorker

**Files:**
- Modify: `gui/app.py`

**Step 1: Add estimation to PreviewWorker.run()**

After registration in `PreviewWorker.run()` (after `tf.refine_tile_positions_with_cross_correlation`):
```python
# Estimate pixel size
try:
    estimated_px, deviation = tf.estimate_pixel_size()
    self.progress.emit(
        f"Estimated pixel size: {estimated_px:.4f} µm ({deviation:+.1f}% from metadata)"
    )
except ValueError:
    pass  # Not enough pairs for estimation
```

**Step 2: Commit**

```bash
git add gui/app.py
git commit -m "feat: Show estimated pixel size in preview"
```

---

### Task 4: Format and test

**Step 1: Run black**
```bash
python3 -m black --line-length=100 src/tilefusion/core.py gui/app.py tests/test_core_pixel_estimation.py
```

**Step 2: Run all tests**
```bash
pytest tests/ -v
```

**Step 3: Commit if needed**
```bash
git add -A
git commit -m "style: Format with black"
```

---

### Task 5: Manual testing

1. Run `python3 gui/app.py`
2. Load a multi-tile dataset
3. Enable registration
4. Run preview - check log shows estimated pixel size
5. Check "Use estimated pixel size"
6. Run full stitching - verify it uses estimated value
