"""
Tile caching for efficient registration.

LRU cache to avoid re-reading the same tile regions during registration.
"""

from collections import OrderedDict
from typing import Any

import numpy as np


class TileCache:
    """
    LRU cache for tile regions.

    Caches tile regions during registration to avoid redundant reads
    when the same tile appears in multiple adjacent pairs.

    Parameters
    ----------
    maxsize : int
        Maximum number of regions to cache.
    """

    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, reader: Any, tile_idx: int, y_slice: slice, x_slice: slice) -> np.ndarray:
        """
        Get a tile region, reading from disk if not cached.

        Parameters
        ----------
        reader : OMETiffReader or similar
            Reader with read_region method.
        tile_idx : int
            Tile index.
        y_slice, x_slice : slice
            Region bounds.

        Returns
        -------
        data : ndarray
            Tile region data.
        """
        key = (tile_idx, y_slice.start, y_slice.stop, x_slice.start, x_slice.stop)

        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self.misses += 1
        data = reader.read_region(tile_idx, y_slice, x_slice)
        self._cache[key] = data

        # Evict oldest if over limit
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

        return data

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
