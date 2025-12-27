"""
Individual TIFFs format reader.

Reads folder format with individual TIFF files per tile/channel and coordinates.csv.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tifffile


def _detect_filename_pattern(image_folder: Path, tiff_files: list):
    """
    Detect the filename pattern used in the folder.

    Returns pattern type and relevant info:
    - 'manual': manual_{fov}_{z}_{channel}.tiff
    - 'region': {region}_{fov}_{z}_{channel}.tiff (e.g., C7_0_0_..., current_0_0_...)
    """
    for f in tiff_files:
        stem = f.stem
        if stem.startswith("manual_"):
            return "manual", None
        # Check for region pattern: {region}_{fov}_{z}_{channel}
        # region can be well ID (C4, D5) or arbitrary name (current)
        parts = stem.split("_")
        if len(parts) >= 4:
            # Second part should be numeric (fov index)
            if parts[1].isdigit():
                return "region", None
    return "manual", None


def load_individual_tiffs_metadata(folder_path: Path) -> Dict[str, Any]:
    """
    Load metadata from individual TIFFs folder format.

    Supports two naming conventions:
    - manual_{fov}_{z}_{channel}.tiff with fov column in coordinates.csv
    - {well}_{fov}_{z}_{channel}.tiff with region column in coordinates.csv

    Parameters
    ----------
    folder_path : Path
        Path to the data folder.

    Returns
    -------
    metadata : dict
        Dictionary containing:
        - n_tiles: int
        - shape: (Y, X)
        - channels: int
        - channel_names: list of str
        - pixel_size: (py, px)
        - tile_positions: list of (y, x) tuples
        - tile_identifiers: list of (well, fov) or (fov,) tuples
        - image_folder: Path
    """
    # Find the subfolder containing images (usually "0" for single z-level)
    subfolders = [d for d in folder_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    # Filter to only directories that contain tiff files
    image_folder = None
    for sub in subfolders:
        if list(sub.glob("*.tiff")) or list(sub.glob("*.tif")):
            image_folder = sub
            break
    if image_folder is None:
        image_folder = folder_path

    # Load coordinates
    coords_path = image_folder / "coordinates.csv"
    if not coords_path.exists():
        coords_path = folder_path / "coordinates.csv"
    if not coords_path.exists():
        raise FileNotFoundError(f"coordinates.csv not found in {folder_path}")

    coords = pd.read_csv(coords_path)
    n_tiles = len(coords)

    # Get channel names from TIFF filenames
    tiff_files = list(image_folder.glob("*.tiff"))
    if not tiff_files:
        tiff_files = list(image_folder.glob("*.tif"))

    # Detect filename pattern
    pattern, _ = _detect_filename_pattern(image_folder, tiff_files)

    channel_names = set()
    for f in tiff_files:
        parts = f.stem.split("_")
        if len(parts) >= 4:
            channel_name = "_".join(parts[3:])
            channel_names.add(channel_name)

    channel_names = sorted(channel_names)
    channels = len(channel_names)

    # Determine tile identifiers based on coords columns and pattern
    if "region" in coords.columns and "fov" in coords.columns:
        # Region+fov format: {region}_{fov}_{z}_{channel}.tiff
        # Use the actual fov values from the CSV
        tile_identifiers = []
        for _, row in coords.iterrows():
            region = row["region"]
            fov = row["fov"]
            tile_identifiers.append((region, fov))
        first_region, first_fov = tile_identifiers[0]
        first_channel = channel_names[0]
        first_img_path = image_folder / f"{first_region}_{first_fov}_0_{first_channel}.tiff"
    elif "region" in coords.columns:
        # Region-only format: {region}_{fov}_{z}_{channel}.tiff (fov is sequential within region)
        region_fov_counts = {}
        tile_identifiers = []
        for _, row in coords.iterrows():
            region = row["region"]
            fov_in_region = region_fov_counts.get(region, 0)
            tile_identifiers.append((region, fov_in_region))
            region_fov_counts[region] = fov_in_region + 1
        first_region, first_fov = tile_identifiers[0]
        first_channel = channel_names[0]
        first_img_path = image_folder / f"{first_region}_{first_fov}_0_{first_channel}.tiff"
    elif "fov" in coords.columns:
        # Manual format: manual_{fov}_{z}_{channel}.tiff
        tile_identifiers = [(fov,) for fov in coords["fov"].tolist()]
        first_fov = coords["fov"].iloc[0]
        first_channel = channel_names[0]
        first_img_path = image_folder / f"manual_{first_fov}_0_{first_channel}.tiff"
    else:
        # Fallback: assume sequential FOVs
        tile_identifiers = [(i,) for i in range(n_tiles)]
        first_channel = channel_names[0]
        first_img_path = image_folder / f"manual_0_0_{first_channel}.tiff"

    if not first_img_path.exists():
        first_img_path = first_img_path.with_suffix(".tif")

    first_img = tifffile.imread(first_img_path)
    Y, X = first_img.shape[-2:]

    # Load pixel size from acquisition parameters
    params_path = folder_path / "acquisition parameters.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        magnification = params.get("objective", {}).get("magnification", 10.0)
        sensor_pixel_um = params.get("sensor_pixel_size_um", 7.52)
        pixel_size_um = sensor_pixel_um / magnification
    else:
        pixel_size_um = 0.752  # Default for 10x

    pixel_size = (pixel_size_um, pixel_size_um)

    # Convert mm coordinates to µm and store as (y, x)
    tile_positions = []
    for _, row in coords.iterrows():
        x_um = row["x (mm)"] * 1000
        y_um = row["y (mm)"] * 1000
        tile_positions.append((y_um, x_um))

    return {
        "n_tiles": n_tiles,
        "n_series": n_tiles,
        "shape": (Y, X),
        "channels": channels,
        "channel_names": channel_names,
        "time_dim": 1,
        "position_dim": n_tiles,
        "pixel_size": pixel_size,
        "tile_positions": tile_positions,
        "tile_identifiers": tile_identifiers,
        "image_folder": image_folder,
        "pattern": pattern,
    }


def _get_tile_filename(image_folder: Path, tile_id: tuple, channel_name: str) -> Path:
    """Get the TIFF filename for a tile based on its identifier."""
    if len(tile_id) == 2:
        # Region-based format: (region, fov)
        region, fov = tile_id
        img_path = image_folder / f"{region}_{fov}_0_{channel_name}.tiff"
    else:
        # Manual format: (fov,)
        fov = tile_id[0]
        img_path = image_folder / f"manual_{fov}_0_{channel_name}.tiff"

    if not img_path.exists():
        img_path = img_path.with_suffix(".tif")
    return img_path


def read_individual_tiffs_tile(
    image_folder: Path,
    channel_names: List[str],
    tile_identifiers: List[tuple],
    tile_idx: int,
) -> np.ndarray:
    """
    Read all channels of a tile from individual TIFFs folder format.

    Parameters
    ----------
    image_folder : Path
        Path to folder containing TIFF files.
    channel_names : list of str
        Channel names.
    tile_identifiers : list of tuple
        Tile identifiers: (well, fov) or (fov,) tuples.
    tile_idx : int
        Index of the tile.

    Returns
    -------
    arr : ndarray of shape (C, Y, X)
        Tile data as float32.
    """
    tile_id = tile_identifiers[tile_idx]

    channels = []
    for channel_name in channel_names:
        img_path = _get_tile_filename(image_folder, tile_id, channel_name)
        arr = tifffile.imread(img_path)
        channels.append(arr)

    stacked = np.stack(channels, axis=0)
    return stacked.astype(np.float32)


def read_individual_tiffs_region(
    image_folder: Path,
    channel_names: List[str],
    tile_identifiers: List[tuple],
    tile_idx: int,
    y_slice: slice,
    x_slice: slice,
    channel_idx: int = 0,
) -> np.ndarray:
    """
    Read a region of a single channel from individual TIFFs format.

    Parameters
    ----------
    image_folder : Path
        Path to folder containing TIFF files.
    channel_names : list of str
        Channel names.
    tile_identifiers : list of tuple
        Tile identifiers: (well, fov) or (fov,) tuples.
    tile_idx : int
        Index of the tile.
    y_slice, x_slice : slice
        Region to read.
    channel_idx : int
        Channel index.

    Returns
    -------
    arr : ndarray of shape (1, h, w)
        Tile region as float32.
    """
    tile_id = tile_identifiers[tile_idx]
    channel_name = channel_names[channel_idx]

    img_path = _get_tile_filename(image_folder, tile_id, channel_name)

    arr = tifffile.imread(img_path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return arr[:, y_slice, x_slice].astype(np.float32)
