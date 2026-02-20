# Pixel Size Estimation from Registration

**Date:** 2026-02-19
**Status:** Approved

## Summary

Estimate the true pixel size by comparing expected vs measured tile shifts during registration. This validates/corrects the pixel size from metadata.

## Algorithm

1. After registration completes, for each pair with valid metrics:
   - Expected shift (pixels) = `stage_distance / metadata_pixel_size`
   - Measured shift (pixels) = cross-correlation result
   - Ratio = `expected / measured`
2. Take median of all ratios (filters outliers)
3. Estimated pixel size = `metadata_pixel_size * median_ratio`
4. Report deviation as percentage: `(median_ratio - 1) * 100%`

## Core Changes (TileFusion)

- Add `estimate_pixel_size()` method in `core.py`
- Returns `(estimated_px_size, deviation_percent)`
- Called after `refine_tile_positions_with_cross_correlation()`
- Always log the result

## GUI Changes

- Add checkbox: "Use estimated pixel size" in Settings (under registration options)
- Show estimated value in log after registration
- If checkbox enabled, apply corrected pixel size before optimization/fusion

## Output

- Log: `"Estimated pixel size: 0.768 µm (2.1% deviation from metadata)"`
- Store in output zarr metadata
