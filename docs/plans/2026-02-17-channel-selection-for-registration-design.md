# Channel Selection for Registration

**Date:** 2026-02-17
**Status:** Approved

## Summary

Add a UI control to select which channel is used for tile registration. Currently, registration always uses channel 0. Users need to select specific channels (e.g., DAPI for nuclear staining) that provide better registration results.

## Current State

- `TileFusion` has a `channel_to_use` parameter (defaults to `0`)
- Used in `_read_tile_region` for reading overlap regions during registration
- GUI has z-level and timepoint selection for registration (commit ce47f22)
- No UI control to select channel

## Design

### UI Changes

Add a `QComboBox` dropdown to the registration sub-options widget (`reg_zt_widget`) that displays channel names and allows selection.

**Location:** Nested under "Enable registration refinement" checkbox, alongside z-level and timepoint controls.

**Visibility:** Only shown when:
1. Registration is enabled, AND
2. Dataset has multiple channels (n_channels > 1)

### State Management

New instance variables in `StitcherGUI`:
- `self.dataset_n_channels: int` - Number of channels in loaded dataset
- `self.dataset_channel_names: List[str]` - Channel names from metadata

### Data Flow

1. `on_file_dropped()` loads metadata, extracts channel count and names
2. `_update_reg_zt_controls()` updates channel combo visibility and contents
3. `run_stitching()` / `run_preview()` pass `channel_to_use=reg_channel_combo.currentIndex()` to workers
4. Workers pass to `TileFusion` constructor

### Files to Modify

- `gui/app.py`: Add channel combo UI and wire up data flow

## Alternatives Considered

**SpinBox with channel index:** Simpler but less user-friendly. Users think in terms of channel names, not indices. Rejected in favor of dropdown with names.
