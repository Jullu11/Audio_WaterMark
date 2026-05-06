# src/watermark/__init__.py
"""Bridge to the official AUDIO WATERMARK code in external/."""

import sys
from pathlib import Path

# Point to official code
_EXT = Path(__file__).resolve().parents[2] / "external/audiowatermark.github.io/code"

def get_external_code_path() -> Path:
    return _EXT

def add_external_to_path():
    """Call this before importing official watermark modules."""
    p = str(_EXT)
    if p not in sys.path:
        sys.path.insert(0, p)

def checkpoints_exist() -> bool:
    """Check if required checkpoints are downloaded."""
    required = [
        _EXT / "checkpoint/benign",
        _EXT / "checkpoint/50_waveunet.pth",
        _EXT / "checkpoint/gst.pth",
    ]
    return all(p.exists() for p in required)

def print_checkpoint_status():
    files = {
        "Benign model":   _EXT / "checkpoint/benign",
        "Wave-U-Net":     _EXT / "checkpoint/50_waveunet.pth",
        "GST encoder":    _EXT / "checkpoint/gst.pth",
        "DeepAFX style":  _EXT / "checkpoint/deepafx_style.ckpt",
    }
    print("\n--- Checkpoint Status ---")
    for name, path in files.items():
        status = "✅" if path.exists() else "❌ MISSING"
        print(f"  {status}  {name}: {path}")
    print()
