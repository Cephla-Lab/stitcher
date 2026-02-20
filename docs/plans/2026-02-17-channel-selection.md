# Channel Selection for Registration - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a channel dropdown to the registration settings so users can select which channel to use for tile registration.

**Architecture:** Add QComboBox to existing `reg_zt_widget`, populate with channel names from metadata on file load, pass selection to TileFusion via `channel_to_use` parameter.

**Tech Stack:** PyQt5 (QComboBox, QLabel), TileFusion API

---

### Task 1: Add Channel Selection UI

**Files:**
- Modify: `gui/app.py:717-735` (add state variables in `__init__`)
- Modify: `gui/app.py:909-931` (add channel combo to `reg_zt_widget`)

**Step 1: Add state variables to `__init__`**

After line 735 (`self.dataset_n_t = 1`), add:
```python
self.dataset_n_channels = 1
self.dataset_channel_names = []
```

**Step 2: Add channel combo to `reg_zt_widget`**

Insert before `reg_zt_layout.addStretch()` (line 930):
```python
self.reg_channel_label = QLabel("Channel:")
reg_zt_layout.addWidget(self.reg_channel_label)
self.reg_channel_combo = QComboBox()
self.reg_channel_combo.setToolTip("Channel to use for registration")
self.reg_channel_combo.setMinimumWidth(120)
reg_zt_layout.addWidget(self.reg_channel_combo)
```

---

### Task 2: Populate Channel Combo on File Load

**Files:**
- Modify: `gui/app.py:1005-1038` (`on_file_dropped` method)

**Step 1: Extract channel info from metadata**

In `on_file_dropped`, inside the `try` block after loading `tf_temp`, add:
```python
self.dataset_n_channels = tf_temp.channels
if "channel_names" in tf_temp._metadata:
    self.dataset_channel_names = tf_temp._metadata["channel_names"]
else:
    self.dataset_channel_names = [f"Channel {i}" for i in range(self.dataset_n_channels)]
```

**Step 2: Reset channel state in except block**

In the `except Exception:` block, add:
```python
self.dataset_n_channels = 1
self.dataset_channel_names = []
```

---

### Task 3: Update `_update_reg_zt_controls` for Channel Visibility

**Files:**
- Modify: `gui/app.py:1061-1084` (`_update_reg_zt_controls` method)

**Step 1: Add channel visibility logic**

After the timepoint visibility logic, add:
```python
# Update channel combo
has_multi_channel = self.dataset_n_channels > 1
self.reg_channel_label.setVisible(has_multi_channel)
self.reg_channel_combo.setVisible(has_multi_channel)
if has_multi_channel:
    self.reg_channel_combo.clear()
    self.reg_channel_combo.addItems(self.dataset_channel_names)
    self.reg_channel_combo.setCurrentIndex(0)
```

**Step 2: Update widget visibility condition**

Change line ~1068-1069 from:
```python
show_zt = registration_enabled and (has_multi_z or has_multi_t)
```
to:
```python
has_multi_channel = self.dataset_n_channels > 1
show_zt = registration_enabled and (has_multi_z or has_multi_t or has_multi_channel)
```

---

### Task 4: Pass Channel Selection to Workers

**Files:**
- Modify: `gui/app.py:1295-1334` (`run_stitching` method)
- Modify: `gui/app.py:1374-1405` (`run_preview` method)
- Modify: `gui/app.py:119-146` (`PreviewWorker.__init__`)
- Modify: `gui/app.py:331-353` (`FusionWorker.__init__`)

**Step 1: Add `registration_channel` to PreviewWorker**

In `PreviewWorker.__init__`, add parameter `registration_channel=0` and store as `self.registration_channel`.

In `PreviewWorker.run`, pass to TileFusion:
```python
tf_full = TileFusion(
    ...
    channel_to_use=self.registration_channel,
)
```

**Step 2: Add `registration_channel` to FusionWorker**

Same pattern as PreviewWorker.

**Step 3: Pass selection in `run_stitching`**

Add before creating FusionWorker:
```python
registration_channel = self.reg_channel_combo.currentIndex() if self.dataset_n_channels > 1 else 0
```

Pass to FusionWorker constructor.

**Step 4: Pass selection in `run_preview`**

Same pattern as `run_stitching`.

---

### Task 5: Test Manually

1. Load a multi-channel dataset
2. Enable registration
3. Verify channel dropdown appears with correct names
4. Select different channel
5. Run preview/stitching
6. Verify registration uses selected channel (check logs)
