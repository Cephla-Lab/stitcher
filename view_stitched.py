#!/usr/bin/env python3
"""
View stitched microscopy data in napari with proper contrast settings.

Usage:
    python view_stitched.py [path_to_ome_zarr]

Example:
    python view_stitched.py "/media/squid/Extreme SSD/Monkey_fused/manual0.ome.zarr/t0.ome.zarr"
"""

import sys
import numpy as np
import zarr
import napari
from pathlib import Path


def calculate_contrast_limits(data, percentile_low=1, percentile_high=99.5):
    """Calculate good contrast limits based on percentiles."""
    # Sample data to avoid loading everything
    if data.size > 10_000_000:
        # Sample every Nth pixel
        step = int(np.sqrt(data.size / 10_000_000))
        sample = data[::step, ::step]
    else:
        sample = data

    # Remove zeros for percentile calculation
    nonzero = sample[sample > 0]
    if len(nonzero) == 0:
        return (0, 1)

    vmin = np.percentile(nonzero, percentile_low)
    vmax = np.percentile(nonzero, percentile_high)

    return (float(vmin), float(vmax))


def main():
    if len(sys.argv) > 1:
        zarr_path = Path(sys.argv[1])
    else:
        # Default to first timepoint of manual0
        zarr_path = Path("/media/squid/Extreme SSD/Monkey_fused/manual0.ome.zarr/t0.ome.zarr")

    if not zarr_path.exists():
        print(f"Error: Path not found: {zarr_path}")
        print("\nUsage: python view_stitched.py [path_to_ome_zarr]")
        sys.exit(1)

    print(f"Loading: {zarr_path}")

    # Open the zarr array
    image_path = zarr_path / "scale0" / "image"
    if not image_path.exists():
        print(f"Error: Expected image path not found: {image_path}")
        sys.exit(1)

    store = zarr.DirectoryStore(str(image_path))
    z = zarr.open(store, mode="r")

    print(f"Shape: {z.shape}")
    print(f"Dtype: {z.dtype}")
    print(f"Chunks: {z.chunks}")

    # Load data - squeeze out singular dimensions
    # Shape is (T, C, Z, Y, X) -> squeeze to (C, Y, X) if T=1 and Z=1
    data = z[0, :, 0, :, :]  # All channels, first (only) time and z

    print(f"\nData loaded: {data.shape}")
    print("Calculating contrast limits for each channel...")

    # Create napari viewer
    viewer = napari.Viewer()

    # Channel names (update these based on your actual channels)
    channel_names = ["405 nm", "488 nm", "561 nm", "638 nm", "730 nm"]

    # Add each channel with auto-contrast
    for c in range(data.shape[0]):
        channel_data = data[c, :, :]

        # Calculate contrast limits
        contrast_limits = calculate_contrast_limits(channel_data)

        print(f"  Channel {c} ({channel_names[c] if c < len(channel_names) else f'Ch{c}'}):")
        print(f"    Data range: [{channel_data.min()}, {channel_data.max()}]")
        print(f"    Contrast limits: [{contrast_limits[0]:.0f}, {contrast_limits[1]:.0f}]")

        # Add to napari
        viewer.add_image(
            channel_data,
            name=channel_names[c] if c < len(channel_names) else f"Channel {c}",
            contrast_limits=contrast_limits,
            colormap="gray",
            blending="additive",
        )

    print(f"\nDisplaying in napari...")
    print("Tip: Use the layer controls to adjust contrast, toggle channels, or change colormaps")

    napari.run()


if __name__ == "__main__":
    main()
