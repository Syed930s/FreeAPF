#!/usr/bin/env python3


from __future__ import annotations

import hashlib
import io
import json
import lzma
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image

try:
    from PIL import ImageTk
except ImportError:
    ImageTk = None

try:
    import zstandard as zstd
except ImportError:
    zstd = None

try:
    import blosc2
except ImportError:
    blosc2 = None

try:
    import lz4.frame as lz4frame
except ImportError:
    lz4frame = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    RESAMPLING = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLING = Image.LANCZOS



APF_MAGIC = b"APF2"
APF_VERSION_MAJOR = 2
APF_VERSION_MINOR = 6919
APF_VERSION_STRING = "2.6919"

MAX_WIDTH = 15360
MAX_HEIGHT = 8640
MAX_CHANNELS = 65535

SAMPLE_FORMAT_FLOAT64 = 1

FILTER_NONE = 0
FILTER_SHUFFLE8 = 1

COMPRESSION_NONE = 0
COMPRESSION_ZSTD22 = 1
COMPRESSION_LZMA9E = 2
COMPRESSION_BLOSC2_ZSTD = 3
COMPRESSION_LZ4HC12 = 4

COMPRESSION_NAMES = {
    COMPRESSION_NONE: "Uncompressed",
    COMPRESSION_ZSTD22: "Zstandard-22 + SHUFFLE8",
    COMPRESSION_LZMA9E: "LZMA/XZ-9e + SHUFFLE8",
    COMPRESSION_BLOSC2_ZSTD: "Blosc2 + Zstd + internal shuffle",
    COMPRESSION_LZ4HC12: "LZ4HC-12 + SHUFFLE8",
}

FILTER_NAMES = {
    FILTER_NONE: "None",
    FILTER_SHUFFLE8: "SHUFFLE8 byte-plane transform",
}

FLAG_LINEAR_NUMERIC = 1 << 0
FLAG_ALLOW_NEGATIVE = 1 << 1
FLAG_ALLOW_ABOVE_WHITE = 1 << 2
FLAG_ALLOW_NONFINITE = 1 << 3
FLAG_STRAIGHT_ALPHA = 1 << 4

COLOR_DISPLAY_ROLES = {"red", "green", "blue", "luminance"}

FIXED_HEADER_FORMAT = "<4sHH" + "I" * 9 + "dddd" + "IQII" + "32s"
FIXED_HEADER_SIZE = struct.calcsize(FIXED_HEADER_FORMAT)

if FIXED_HEADER_SIZE != 128:
    raise RuntimeError(f"APF fixed header must be 128 bytes, got {FIXED_HEADER_SIZE}.")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compression_name(method: int) -> str:
    return COMPRESSION_NAMES.get(method, f"Unknown compression {method}")


def filter_name(method: int) -> str:
    return FILTER_NAMES.get(method, f"Unknown filter {method}")


def flags_to_text(flags: int) -> str:
    parts = []
    if flags & FLAG_LINEAR_NUMERIC:
        parts.append("linear-float64")
    if flags & FLAG_ALLOW_NEGATIVE:
        parts.append("negative-allowed")
    if flags & FLAG_ALLOW_ABOVE_WHITE:
        parts.append("HDR-above-white")
    if flags & FLAG_ALLOW_NONFINITE:
        parts.append("inf/nan-allowed")
    if flags & FLAG_STRAIGHT_ALPHA:
        parts.append("straight-alpha")
    return ", ".join(parts) if parts else "none"


def json_safe_float(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return number


def finite_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return json_safe_float(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return json_safe_float(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    return obj


def format_float(value: Any) -> str:
    if value is None:
        return "None"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number == 0 or 1e-6 <= abs(number) < 1e9:
        return f"{number:.6g}"
    return f"{number:.6e}"


def finite_min_max(arr: np.ndarray) -> tuple[float, float]:
    if arr.size == 0:
        return 0.0, 1.0

    mask = np.isfinite(arr)
    if mask.all():
        return float(arr.min()), float(arr.max())

    finite_values = arr[mask]
    if finite_values.size == 0:
        return 0.0, 1.0

    return float(finite_values.min()), float(finite_values.max())


def make_channel_map(
    channel_count: int,
    roles: Optional[list[str]] = None,
    names: Optional[list[str]] = None,
    units: Optional[list[str]] = None,
    ranges: Optional[list[Any]] = None,
    wavelengths_nm: Optional[list[Any]] = None,
    color_indices: Optional[list[int]] = None,
    alpha_index: Optional[int] = None,
) -> dict[str, Any]:
    """Create and validate an APF channel map."""

    if not isinstance(channel_count, int) or channel_count < 1 or channel_count > MAX_CHANNELS:
        raise ValueError(f"APF supports 1 to {MAX_CHANNELS} channels.")

    if roles is None:
        if channel_count == 1:
            roles = ["luminance"]
        elif channel_count == 2:
            roles = ["luminance", "alpha"]
        elif channel_count == 3:
            roles = ["red", "green", "blue"]
        elif channel_count == 4:
            roles = ["red", "green", "blue", "alpha"]
        else:
            roles = [f"data_{i}" for i in range(channel_count)]
    else:
        roles = [str(role).strip().lower() for role in roles]
        if len(roles) != channel_count:
            raise ValueError("Channel role count does not match channel count.")

    if names is None:
        names = []
        for i, role in enumerate(roles):
            if role == "red":
                names.append("R")
            elif role == "green":
                names.append("G")
            elif role == "blue":
                names.append("B")
            elif role == "alpha":
                names.append("A")
            elif role == "depth":
                names.append("Depth")
            elif role.startswith("normal_"):
                names.append(role.replace("normal_", "N_").upper())
            elif role.startswith("spectral"):
                names.append(f"S{i}")
            else:
                names.append(f"C{i}")
    else:
        names = [str(name) for name in names]
        if len(names) != channel_count:
            raise ValueError("Channel name count does not match channel count.")

    if units is None:
        units = []
        for role in roles:
            if role in ("red", "green", "blue", "luminance"):
                units.append("linear_radiance")
            elif role == "alpha":
                units.append("coverage")
            elif role == "depth":
                units.append("meters")
            elif role.startswith("normal_"):
                units.append("vector_component")
            elif role.startswith("spectral"):
                units.append("spectral_radiance")
            elif role in ("infrared", "ultraviolet"):
                units.append("radiance")
            else:
                units.append("data")
    else:
        units = [str(unit) for unit in units]
        if len(units) != channel_count:
            raise ValueError("Channel unit count does not match channel count.")

    if ranges is None:
        ranges = []
        for role in roles:
            if role in ("red", "green", "blue", "luminance", "alpha"):
                ranges.append([0.0, 1.0])
            elif role.startswith("normal_"):
                ranges.append([-1.0, 1.0])
            else:
                ranges.append([None, None])
    else:
        if len(ranges) != channel_count:
            raise ValueError("Channel range count does not match channel count.")
        normalized_ranges = []
        for item in ranges:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                normalized_ranges.append([
                    finite_float_or_none(item[0]),
                    finite_float_or_none(item[1]),
                ])
            else:
                normalized_ranges.append([None, None])
        ranges = normalized_ranges

    if wavelengths_nm is None:
        wavelengths_nm = [None] * channel_count
    else:
        if len(wavelengths_nm) != channel_count:
            raise ValueError("Wavelength count does not match channel count.")
        wavelengths_nm = [finite_float_or_none(value) for value in wavelengths_nm]

    if color_indices is None:
        red = green = blue = None
        for i, role in enumerate(roles):
            if role == "red" and red is None:
                red = i
            elif role == "green" and green is None:
                green = i
            elif role == "blue" and blue is None:
                blue = i

        if red is not None and green is not None and blue is not None:
            color_indices = [red, green, blue]
        elif channel_count >= 3:
            color_indices = [0, 1, 2]
        else:
            color_indices = [0, 0, 0]
    else:
        if not isinstance(color_indices, (list, tuple)) or len(color_indices) < 3:
            raise ValueError("color_indices must contain at least three channel indices.")
        color_indices = [int(index) for index in color_indices[:3]]
        if any(index < 0 or index >= channel_count for index in color_indices):
            raise ValueError("color_indices contains an out-of-range channel index.")

    if alpha_index is None:
        for i, role in enumerate(roles):
            if role in ("alpha", "opacity", "coverage"):
                alpha_index = i
                break
    else:
        alpha_index = int(alpha_index)
        if alpha_index == -1:
            alpha_index = None
        elif alpha_index < 0 or alpha_index >= channel_count:
            raise ValueError("alpha_index is out of range.")

    return {
        "schema": f"apf-channel-map-{APF_VERSION_STRING}",
        "count": channel_count,
        "roles": roles,
        "names": names,
        "units": units,
        "ranges": ranges,
        "wavelength_nm": wavelengths_nm,
        "color_indices": color_indices,
        "alpha_index": alpha_index,
    }


def normalize_channel_map(raw: Optional[dict[str, Any]], channel_count: int) -> dict[str, Any]:
    if raw is None:
        return make_channel_map(channel_count)
    if not isinstance(raw, dict):
        raise ValueError("APF channel map must be a JSON object.")

    return make_channel_map(
        channel_count,
        roles=raw.get("roles"),
        names=raw.get("names"),
        units=raw.get("units"),
        ranges=raw.get("ranges"),
        wavelengths_nm=raw.get("wavelength_nm"),
        color_indices=raw.get("color_indices"),
        alpha_index=raw.get("alpha_index"),
    )


def make_metadata(source: str = "unknown", **extra: Any) -> dict[str, Any]:
    metadata = {
        "format": "Advanced Picture Format",
        "format_version": APF_VERSION_STRING,
        "software": "FreeAPF",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "sample_type": "IEEE-754 float64 per channel",
        "hdr_policy": {
            "reference_white": "1.0 means diffuse white unless metadata says otherwise",
            "negative_values": "allowed when FLAG_ALLOW_NEGATIVE is set",
            "above_white_values": "allowed when FLAG_ALLOW_ABOVE_WHITE is set",
            "non_finite_values": "allowed when FLAG_ALLOW_NONFINITE is set",
            "interpretation": "linear scene-referred values by default",
        },
    }
    metadata.update(extra)
    return sanitize_json(metadata)


def make_rgba_channel_map() -> dict[str, Any]:
    return make_channel_map(
        4,
        roles=["red", "green", "blue", "alpha"],
        names=["R", "G", "B", "A"],
        units=["linear_radiance", "linear_radiance", "linear_radiance", "coverage"],
        ranges=[[0.0, 1.0] for _ in range(4)],
        color_indices=[0, 1, 2],
        alpha_index=3,
    )


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

class StreamCompressor:
    def __init__(self, method: int):
        self.method = method
        self.parts: list[bytes] = []

        if method == COMPRESSION_ZSTD22:
            if zstd is None:
                raise RuntimeError("zstandard is not installed.")
            cctx = zstd.ZstdCompressor(
                level=22,
                threads=max(1, os.cpu_count() or 1),
                write_checksum=True,
            )
            self.impl = cctx.compressobj()
        elif method == COMPRESSION_LZMA9E:
            self.impl = lzma.LZMACompressor(
                format=lzma.FORMAT_XZ,
                check=lzma.CHECK_CRC64,
                preset=9 | lzma.PRESET_EXTREME,
            )
        elif method == COMPRESSION_LZ4HC12:
            if lz4frame is None:
                raise RuntimeError("lz4 is not installed.")
            self.impl = lz4frame.LZ4FrameCompressor(
                compression_level=12,
                content_checksum=True,
            )
        else:
            raise ValueError(f"Unsupported stream compression method: {method}")

    def begin(self) -> None:
        if self.method == COMPRESSION_LZ4HC12:
            self.parts.append(self.impl.begin())

    def update(self, chunk: bytes) -> None:
        if not chunk:
            return
        out = self.impl.compress(chunk)
        if out:
            self.parts.append(out)

    def finish(self) -> bytes:
        out = self.impl.flush()
        if out:
            self.parts.append(out)
        return b"".join(self.parts)


def available_compression_options() -> list[tuple[int, str]]:
    options = []
    if zstd is not None:
        options.append((COMPRESSION_ZSTD22, "Zstandard-22 + SHUFFLE8 (recommended)"))
    options.append((COMPRESSION_LZMA9E, "LZMA/XZ-9e + SHUFFLE8 (strongest)"))
    if blosc2 is not None:
        options.append((COMPRESSION_BLOSC2_ZSTD, "Blosc2 + Zstd (scientific)"))
    if lz4frame is not None:
        options.append((COMPRESSION_LZ4HC12, "LZ4HC-12 + SHUFFLE8 (fastest)"))
    return options


def preferred_compression() -> int:
    return COMPRESSION_ZSTD22 if zstd is not None else COMPRESSION_LZMA9E


def compress_pixel_payload(data: np.ndarray, method: int) -> tuple[bytes, int]:
    data = np.ascontiguousarray(data, dtype="<f8")

    if method == COMPRESSION_NONE:
        return data.tobytes(), FILTER_NONE

    byte_view = data.view(np.uint8).reshape(-1, 8)

    if method == COMPRESSION_BLOSC2_ZSTD:
        if blosc2 is None:
            raise RuntimeError("blosc2 is not installed.")
        raw = byte_view.tobytes()
        payload = blosc2.compress(
            raw,
            typesize=8,
            clevel=9,
            filter=blosc2.SHUFFLE,
            codec=blosc2.ZSTD,
        )
        return payload, FILTER_NONE

    compressor = StreamCompressor(method)
    compressor.begin()

    for plane in range(8):
        chunk = np.ascontiguousarray(byte_view[:, plane]).tobytes()
        compressor.update(chunk)

    return compressor.finish(), FILTER_SHUFFLE8


def decompress_pixel_payload(
    payload: bytes,
    method: int,
    filter_method: int,
    uncompressed_size: int,
) -> bytes:
    if method == COMPRESSION_NONE:
        raw = payload
    elif method == COMPRESSION_ZSTD22:
        if zstd is None:
            raise RuntimeError("This file uses Zstandard, but zstandard is not installed.")
        raw = zstd.ZstdDecompressor().decompress(payload, max_output_size=uncompressed_size)
    elif method == COMPRESSION_LZMA9E:
        raw = lzma.decompress(payload, format=lzma.FORMAT_XZ)
    elif method == COMPRESSION_BLOSC2_ZSTD:
        if blosc2 is None:
            raise RuntimeError("This file uses Blosc2, but blosc2 is not installed.")
        raw = blosc2.decompress(payload)
    elif method == COMPRESSION_LZ4HC12:
        if lz4frame is None:
            raise RuntimeError("This file uses LZ4HC, but lz4 is not installed.")
        raw = lz4frame.decompress(payload)
    else:
        raise ValueError(f"Unsupported APF compression method: {method}")

    if len(raw) != uncompressed_size:
        raise ValueError("Decompressed payload size does not match APF header.")

    if filter_method == FILTER_SHUFFLE8:
        byte_view = np.frombuffer(raw, dtype=np.uint8).reshape(8, -1)
        raw = np.ascontiguousarray(byte_view.T).tobytes()
    elif filter_method != FILTER_NONE:
        raise ValueError(f"Unsupported APF filter method: {filter_method}")

    return raw



@dataclass
class APFImage:
    data: np.ndarray
    channel_map: dict[str, Any]
    metadata: dict[str, Any]
    width: int
    height: int
    channels: int
    compression: int
    filter_method: int
    range_min: float
    range_max: float
    reference_white: float
    exposure_scale: float
    flags: int
    uncompressed_size: int
    checksum_sha256: bytes


# ---------------------------------------------------------------------------
# Scaling computation
# ---------------------------------------------------------------------------

def compute_scaling(data: np.ndarray, reference_white: float) -> tuple[float, float, int]:
    if not math.isfinite(reference_white) or reference_white <= 0:
        reference_white = 1.0

    try:
        range_min = float(np.nanmin(data))
    except ValueError:
        range_min = 0.0

    try:
        range_max = float(np.nanmax(data))
    except ValueError:
        range_max = 1.0

    if math.isnan(range_min) or math.isnan(range_max):
        range_min = 0.0
        range_max = 1.0

    flags = FLAG_LINEAR_NUMERIC

    if range_min < 0.0:
        flags |= FLAG_ALLOW_NEGATIVE

    if range_max > reference_white:
        flags |= FLAG_ALLOW_ABOVE_WHITE

    if not np.isfinite(data).all():
        flags |= FLAG_ALLOW_NONFINITE

    return range_min, range_max, flags


# ---------------------------------------------------------------------------
# APF writer
# ---------------------------------------------------------------------------

def write_apf(
    path: str,
    data: np.ndarray,
    *,
    channel_map: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    reference_white: float = 1.0,
    exposure_scale: float = 1.0,
    compression: Optional[int] = None,
) -> APFImage:
    if compression is None:
        compression = preferred_compression()

    if not math.isfinite(reference_white) or reference_white <= 0:
        reference_white = 1.0

    if not math.isfinite(exposure_scale) or exposure_scale <= 0:
        exposure_scale = 1.0

    data = np.asarray(data, dtype="<f8")

    if data.ndim == 2:
        data = data[:, :, None]

    if data.ndim != 3:
        raise ValueError("APF pixel data must have shape (height, width, channels).")

    height, width, channels = data.shape

    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError("APF image dimensions must be positive.")

    if width > MAX_WIDTH or height > MAX_HEIGHT:
        raise ValueError(f"APF 2.6919 maximum resolution is {MAX_WIDTH}x{MAX_HEIGHT}.")

    if channels > MAX_CHANNELS:
        raise ValueError(f"APF 2.6919 supports at most {MAX_CHANNELS} channels.")

    data = np.ascontiguousarray(data, dtype="<f8")

    if channel_map is None:
        channel_map = make_channel_map(channels)
    else:
        channel_map = normalize_channel_map(channel_map, channels)

    if metadata is None:
        metadata = make_metadata("array")
    else:
        metadata = dict(metadata)

    range_min, range_max, flags = compute_scaling(data, reference_white)

    if channel_map.get("alpha_index") is not None:
        flags |= FLAG_STRAIGHT_ALPHA

    payload, filter_method = compress_pixel_payload(data, compression)

    uncompressed_size = height * width * channels * 8
    if uncompressed_size >= 2 ** 64:
        raise ValueError("APF uncompressed payload is too large for the 64-bit size field.")

    metadata["width"] = width
    metadata["height"] = height
    metadata["channels"] = channels
    metadata["bits_per_channel"] = 64
    metadata["sample_format"] = "IEEE-754 float64"
    metadata["total_bits_per_pixel"] = 64 * channels
    metadata["compression_id"] = compression
    metadata["compression"] = compression_name(compression)
    metadata["filter"] = filter_name(filter_method)
    metadata["measured_range"] = [json_safe_float(range_min), json_safe_float(range_max)]
    metadata["reference_white"] = json_safe_float(reference_white)
    metadata["exposure_scale"] = json_safe_float(exposure_scale)
    metadata = sanitize_json(metadata)

    channel_map_bytes = json.dumps(channel_map, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    metadata_bytes = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    if len(channel_map_bytes) >= 2 ** 32 or len(metadata_bytes) >= 2 ** 32:
        raise ValueError("APF header metadata is too large.")

    checksum = hashlib.sha256()
    checksum.update(channel_map_bytes)
    checksum.update(metadata_bytes)
    checksum.update(payload)
    checksum_digest = checksum.digest()

    header_size = FIXED_HEADER_SIZE + len(channel_map_bytes) + len(metadata_bytes)

    header = struct.pack(
        FIXED_HEADER_FORMAT,
        APF_MAGIC,
        APF_VERSION_MAJOR,
        APF_VERSION_MINOR,
        header_size,
        compression,
        filter_method,
        SAMPLE_FORMAT_FLOAT64,
        64,
        channels,
        64 * channels,
        width,
        height,
        range_min,
        range_max,
        reference_white,
        exposure_scale,
        flags,
        uncompressed_size,
        len(channel_map_bytes),
        len(metadata_bytes),
        checksum_digest,
    )

    with open(path, "wb") as file:
        file.write(header)
        file.write(channel_map_bytes)
        file.write(metadata_bytes)
        file.write(payload)

    return APFImage(
        data=data,
        channel_map=channel_map,
        metadata=metadata,
        width=width,
        height=height,
        channels=channels,
        compression=compression,
        filter_method=filter_method,
        range_min=range_min,
        range_max=range_max,
        reference_white=reference_white,
        exposure_scale=exposure_scale,
        flags=flags,
        uncompressed_size=uncompressed_size,
        checksum_sha256=checksum_digest,
    )


def read_apf(path: str) -> APFImage:
    with open(path, "rb") as file:
        fixed_header = file.read(FIXED_HEADER_SIZE)
        if len(fixed_header) != FIXED_HEADER_SIZE:
            raise ValueError("File is too small to contain an APF 2.6919 header.")

        (
            magic,
            version_major,
            version_minor,
            header_size,
            compression,
            filter_method,
            sample_format,
            bits_per_channel,
            channels,
            total_bits_per_pixel,
            width,
            height,
            range_min,
            range_max,
            reference_white,
            exposure_scale,
            flags,
            uncompressed_size,
            channel_map_size,
            metadata_size,
            stored_checksum,
        ) = struct.unpack(FIXED_HEADER_FORMAT, fixed_header)

        if magic == b"APF1":
            raise ValueError("Legacy APF1 file detected. APF 2.6919 is not backward compatible.")

        if magic != APF_MAGIC:
            raise ValueError("Not an APF 2.6919 file.")

        if version_major != APF_VERSION_MAJOR or version_minor != APF_VERSION_MINOR:
            raise ValueError(
                f"Unsupported APF version {version_major}.{version_minor}. "
                f"This program supports APF {APF_VERSION_STRING}."
            )

        if sample_format != SAMPLE_FORMAT_FLOAT64:
            raise ValueError("APF 2.6919 currently supports IEEE-754 float64 samples only.")

        if bits_per_channel != 64:
            raise ValueError("APF 2.6919 uses 64 bits per channel.")

        if channels < 1 or channels > MAX_CHANNELS:
            raise ValueError(f"Channel count must be between 1 and {MAX_CHANNELS}.")

        if total_bits_per_pixel != bits_per_channel * channels:
            raise ValueError("Header total_bits_per_pixel does not match bits_per_channel * channels.")

        if width < 1 or height < 1:
            raise ValueError("Invalid image dimensions.")

        if width > MAX_WIDTH or height > MAX_HEIGHT:
            raise ValueError("Image exceeds APF 16K maximum dimensions.")

        expected_uncompressed_size = width * height * channels * 8
        if uncompressed_size != expected_uncompressed_size:
            raise ValueError("Header uncompressed size does not match dimensions and channel count.")

        if header_size != FIXED_HEADER_SIZE + channel_map_size + metadata_size:
            raise ValueError("APF header size fields are inconsistent.")

        channel_map_bytes = file.read(channel_map_size)
        metadata_bytes = file.read(metadata_size)

        if len(channel_map_bytes) != channel_map_size:
            raise ValueError("APF channel map is truncated.")
        if len(metadata_bytes) != metadata_size:
            raise ValueError("APF metadata is truncated.")

        payload = file.read()

    checksum = hashlib.sha256()
    checksum.update(channel_map_bytes)
    checksum.update(metadata_bytes)
    checksum.update(payload)

    if checksum.digest() != stored_checksum:
        raise ValueError("SHA-256 checksum mismatch: file is corrupt or was modified incorrectly.")

    raw = decompress_pixel_payload(payload, compression, filter_method, uncompressed_size)
    data = np.frombuffer(raw, dtype="<f8").reshape((height, width, channels))

    if channel_map_size:
        channel_map = normalize_channel_map(json.loads(channel_map_bytes.decode("utf-8")), channels)
    else:
        channel_map = make_channel_map(channels)

    if metadata_size:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    else:
        metadata = {}

    if not math.isfinite(reference_white) or reference_white <= 0:
        reference_white = 1.0

    if not math.isfinite(exposure_scale) or exposure_scale <= 0:
        exposure_scale = 1.0

    return APFImage(
        data=data,
        channel_map=channel_map,
        metadata=metadata,
        width=width,
        height=height,
        channels=channels,
        compression=compression,
        filter_method=filter_method,
        range_min=range_min,
        range_max=range_max,
        reference_white=reference_white,
        exposure_scale=exposure_scale,
        flags=flags,
        uncompressed_size=uncompressed_size,
        checksum_sha256=stored_checksum,
    )




def select_color_indices(channel_map: dict[str, Any], channel_count: int) -> list[int]:
    roles = [str(role).lower() for role in channel_map.get("roles", [])]

    red = green = blue = None
    for i, role in enumerate(roles):
        if role == "red" and red is None:
            red = i
        elif role == "green" and green is None:
            green = i
        elif role == "blue" and blue is None:
            blue = i

    if red is not None and green is not None and blue is not None:
        return [red, green, blue]

    color_indices = channel_map.get("color_indices")
    if isinstance(color_indices, list) and len(color_indices) >= 3:
        try:
            values = [int(color_indices[0]), int(color_indices[1]), int(color_indices[2])]
            if all(0 <= value < channel_count for value in values):
                return values
        except (TypeError, ValueError):
            pass

    if channel_count >= 3:
        return [0, 1, 2]
    return [0, 0, 0]


def select_alpha_index(channel_map: dict[str, Any], channel_count: int) -> Optional[int]:
    alpha_index = channel_map.get("alpha_index")

    if isinstance(alpha_index, bool):
        return None

    if isinstance(alpha_index, int) and 0 <= alpha_index < channel_count:
        return alpha_index

    roles = [str(role).lower() for role in channel_map.get("roles", [])]
    for i, role in enumerate(roles):
        if role in ("alpha", "opacity", "coverage"):
            return i

    names = [str(name).lower() for name in channel_map.get("names", [])]
    for i, name in enumerate(names):
        if name in ("a", "alpha", "opacity"):
            return i

    return None


def prepare_preview_channels(
    data: np.ndarray,
    indices: list[int],
    max_width: Optional[int],
    max_height: Optional[int],
) -> tuple[dict[int, np.ndarray], int, int]:
    height, width, channels = data.shape

    unique_indices: list[int] = []
    for index in indices:
        if index < 0 or index >= channels:
            raise ValueError("Preview channel index is out of range.")
        if index not in unique_indices:
            unique_indices.append(index)

    if not unique_indices:
        raise ValueError("No preview channels selected.")

    max_width = int(max_width or 0)
    max_height = int(max_height or 0)

    if max_width <= 16 or max_height <= 16 or (width <= max_width and height <= max_height):
        return {index: data[:, :, index] for index in unique_indices}, height, width

    step_w = max(1, math.ceil(width / max_width))
    step_h = max(1, math.ceil(height / max_height))

    out_h = height // step_h
    out_w = width // step_w

    if out_h <= 0 or out_w <= 0:
        return {index: data[:, :, index] for index in unique_indices}, height, width

    result: dict[int, np.ndarray] = {}
    max_input_rows_per_chunk = 64
    chunk_out_rows = max(1, min(out_h, max_input_rows_per_chunk // step_h))

    for index in unique_indices:
        out = np.empty((out_h, out_w), dtype=np.float64)

        for y0 in range(0, out_h, chunk_out_rows):
            y1 = min(out_h, y0 + chunk_out_rows)
            block = data[y0 * step_h:y1 * step_h, :out_w * step_w, index]
            block = block.reshape((y1 - y0), step_h, out_w, step_w)
            out[y0:y1] = block.mean(axis=(1, 3))

        result[index] = out

    return result, out_h, out_w


def aces_tonemap(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 65504.0).astype(np.float32)
    numerator = x * (2.51 * x + 0.03)
    denominator = x * (2.43 * x + 0.59) + 0.14
    return np.clip(numerator / denominator, 0.0, 1.0)


def linear_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0).astype(np.float32)
    srgb = np.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def normalize_data_channel(
    channel: np.ndarray,
    role: str,
    channel_range: Any,
) -> np.ndarray:
    lo = hi = None

    if isinstance(channel_range, (list, tuple)) and len(channel_range) == 2:
        lo = finite_float_or_none(channel_range[0])
        hi = finite_float_or_none(channel_range[1])

    if lo is None or hi is None or not hi > lo:
        auto_lo, auto_hi = finite_min_max(channel)
        if lo is None:
            lo = auto_lo
        if hi is None:
            hi = auto_hi

    if not math.isfinite(lo):
        lo = 0.0
    if not math.isfinite(hi):
        hi = 1.0
    if hi <= lo:
        hi = lo + 1.0

    f32max = float(np.finfo(np.float32).max)
    lo = max(min(float(lo), f32max), -f32max)
    hi = max(min(float(hi), f32max), -f32max)

    if hi <= lo:
        lo, hi = -1.0, 1.0

    ch = np.asarray(channel, dtype=np.float32)
    ch = np.nan_to_num(ch, nan=0.0, posinf=hi, neginf=lo)

    if role.startswith("normal_"):
        out = (ch + 1.0) * 0.5
    else:
        out = (ch - lo) / (hi - lo)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def render_preview(
    apf: APFImage,
    exposure_stops: float = 0.0,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> Image.Image:
    channel_map = apf.channel_map
    color_indices = select_color_indices(channel_map, apf.channels)
    alpha_index = select_alpha_index(channel_map, apf.channels)

    needed_indices = list(color_indices)
    if alpha_index is not None:
        needed_indices.append(alpha_index)

    channels, height, width = prepare_preview_channels(
        apf.data,
        needed_indices,
        max_width,
        max_height,
    )

    roles = [str(role).lower() for role in channel_map.get("roles", [])]
    ranges = channel_map.get("ranges", [])

    unique_color_indices = set(color_indices)
    is_color = all(
        roles[index] in COLOR_DISPLAY_ROLES
        for index in unique_color_indices
        if index < len(roles)
    )

    rgb = np.empty((height, width, 3), dtype=np.float32)

    if is_color:
        for out_index, src_index in enumerate(color_indices):
            rgb[..., out_index] = channels[src_index]

        color_min, color_max = finite_min_max(rgb)

        exposure = apf.exposure_scale * (2.0 ** float(exposure_stops))
        if math.isfinite(color_max) and color_max > 16.0:
            exposure *= 16.0 / color_max

        rgb = np.nan_to_num(rgb, nan=0.0, posinf=65504.0, neginf=0.0)
        rgb *= exposure
        np.clip(rgb, 0.0, 65504.0, out=rgb)

        needs_hdr = (
            not math.isfinite(color_max)
            or not math.isfinite(color_min)
            or color_max > apf.reference_white * 1.0001
            or color_min < 0.0
        )

        if needs_hdr:
            rgb = aces_tonemap(rgb)
        else:
            np.clip(rgb, 0.0, 1.0, out=rgb)

        rgb8 = linear_to_srgb_u8(rgb)
    else:
        for out_index, src_index in enumerate(color_indices):
            role = roles[src_index] if src_index < len(roles) else ""
            channel_range = ranges[src_index] if src_index < len(ranges) else None
            rgb[..., out_index] = normalize_data_channel(channels[src_index], role, channel_range)

        rgb8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if alpha_index is not None:
        alpha = np.asarray(channels[alpha_index], dtype=np.float32)
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1.0, neginf=0.0)
        alpha = np.clip(alpha, 0.0, 1.0)
    else:
        alpha = np.ones((height, width), dtype=np.float32)

    alpha8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)

    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb8
    rgba[..., 3] = alpha8

    return Image.fromarray(rgba, mode="RGBA")


def fit_image(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    if max_width < 16 or max_height < 16:
        return img

    width, height = img.size
    if width <= 0 or height <= 0:
        return img

    ratio = min(max_width / width, max_height / height)
    if ratio >= 1.0:
        return img

    new_width = max(1, int(round(width * ratio)))
    new_height = max(1, int(round(height * ratio)))
    return img.resize((new_width, new_height), RESAMPLING)


def composite_on_checkerboard(img: Image.Image, cell: int = 8) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    width, height = img.size
    if width <= 0 or height <= 0:
        return img

    y, x = np.ogrid[0:height, 0:width]
    board = (((x // cell) + (y // cell)) % 2) == 0

    background = np.empty((height, width, 3), dtype=np.uint8)
    background[board] = (225, 225, 225)
    background[~board] = (170, 170, 170)

    background_img = Image.fromarray(background, mode="RGB").convert("RGBA")
    return Image.alpha_composite(background_img, img)




def srgb_u8_to_linear_float64(rgb_u8: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb_u8, dtype=np.float64) / 255.0
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def pil_image_to_linear_float64_rgba(img: Image.Image) -> np.ndarray:
    img = img.convert("RGBA")
    arr = np.asarray(img, dtype=np.uint8)

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("Expected an RGBA image array.")

    linear_rgb = srgb_u8_to_linear_float64(arr[..., :3])

    out = np.empty(arr.shape[:2] + (4,), dtype="<f8")
    out[..., :3] = linear_rgb
    out[..., 3] = arr[..., 3].astype(np.float64) / 255.0
    return out



def _capture_with_temp_file(args_builder, backend_name: str) -> tuple[Image.Image, str]:
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    try:
        subprocess.run(
            args_builder(tmp_path),
            check=True,
            capture_output=True,
            timeout=20,
        )
        img = Image.open(tmp_path)
        img.load()
        return img.convert("RGBA"), backend_name
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def capture_screen_pil() -> tuple[Image.Image, str]:
    errors: list[str] = []

    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        if img.size[0] > 0 and img.size[1] > 0:
            return img.convert("RGBA"), "PIL.ImageGrab"
        errors.append("ImageGrab returned an empty image.")
    except Exception as exc:
        errors.append(f"ImageGrab: {exc}")

    if shutil.which("grim"):
        try:
            proc = subprocess.run(["grim", "-"], check=True, capture_output=True, timeout=20)
            img = Image.open(io.BytesIO(proc.stdout))
            img.load()
            return img.convert("RGBA"), "grim"
        except Exception as exc:
            errors.append(f"grim: {exc}")

    external_backends = []

    if shutil.which("spectacle"):
        external_backends.append((lambda path: ["spectacle", "-b", "-n", "-o", path], "spectacle"))

    if shutil.which("gnome-screenshot"):
        external_backends.append((lambda path: ["gnome-screenshot", "-f", path], "gnome-screenshot"))

    if shutil.which("scrot"):
        external_backends.append((lambda path: ["scrot", "-o", path], "scrot"))

    for args_builder, backend_name in external_backends:
        try:
            return _capture_with_temp_file(args_builder, backend_name)
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")

    hint = ""
    if os.environ.get("WAYLAND_DISPLAY"):
        hint = "\n\nFedora appears to be using Wayland. Install/use grim, spectacle, gnome-screenshot, or scrot."

    raise RuntimeError("No screenshot backend succeeded:\n" + "\n".join(errors) + hint)




class FreeAPFApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_apf: Optional[APFImage] = None
        self.current_path: Optional[str] = None
        self.preview_photo = None
        self._preview_job: Optional[str] = None

        self.root.title(f"FreeAPF {APF_VERSION_STRING} — float64 HDR Advanced Picture Format")
        self.root.geometry("1150x800")

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill=tk.X)

        ttk.Button(top, text="Open APF", command=self.open_apf).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(top, text="Screenshot → APF", command=self.take_screenshot).grid(row=0, column=1, padx=6)
        ttk.Button(top, text="Webcam → APF", command=self.take_webcam).grid(row=0, column=2, padx=6)

        ttk.Label(top, text="Compression:").grid(row=0, column=3, sticky="e", padx=(18, 4))

        self.compression_options = available_compression_options()
        self.compression_box = ttk.Combobox(
            top,
            state="readonly",
            width=44,
            values=[label for _, label in self.compression_options],
        )
        self.compression_box.current(0)
        self.compression_box.grid(row=0, column=4, sticky="w")

        ttk.Label(top, text="Exposure:").grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.exposure_var = tk.DoubleVar(value=0.0)
        self.exposure_scale = ttk.Scale(
            top,
            from_=-10.0,
            to=10.0,
            variable=self.exposure_var,
            orient=tk.HORIZONTAL,
            length=320,
            command=self._on_exposure_change,
        )
        self.exposure_scale.grid(row=1, column=1, columnspan=3, sticky="w", pady=(10, 0))

        self.exposure_label = ttk.Label(top, text="+0.0 stops", width=12)
        self.exposure_label.grid(row=1, column=4, sticky="w", pady=(10, 0))

        self.status_var = tk.StringVar()
        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            wraplength=1120,
            justify="left",
            foreground="blue",
        )
        status.pack(fill=tk.X, padx=10)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        backend = "Zstandard available" if zstd is not None else "Zstandard missing; using LZMA fallback"
        self.set_status(
            f"APF {APF_VERSION_STRING} ready — float64/channel, RGBA=256 bits/pixel, arbitrary channels, HDR, {backend}.",
            "blue",
        )

    def set_status(self, text: str, color: str = "black") -> None:
        self.status_var.set(text)
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Label) and child.cget("textvariable") == str(self.status_var):
                child.configure(foreground=color)
                break

    def selected_compression(self) -> int:
        index = self.compression_box.current()
        if index < 0 or index >= len(self.compression_options):
            return preferred_compression()
        return self.compression_options[index][0]

    def _on_exposure_change(self, value: Any = None) -> None:
        try:
            stops = float(value)
        except (TypeError, ValueError):
            stops = float(self.exposure_var.get())

        self.exposure_label.config(text=f"{stops:+0.1f} stops")

        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(90, self.refresh_preview)

    def _on_canvas_configure(self, _event=None) -> None:
        if self.current_apf is None:
            return
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(120, self.refresh_preview)

    def describe_apf(self, apf: APFImage, path: Optional[str] = None) -> str:
        names = apf.channel_map.get("names", [f"C{i}" for i in range(apf.channels)])
        names_text = ", ".join(str(name) for name in names[:8])
        if len(names) > 8:
            names_text += f", … ({apf.channels} total)"

        location = f"{path} | " if path else ""

        return (
            f"{location}APF {APF_VERSION_STRING} | {apf.width}×{apf.height}×{apf.channels} | "
            f"{apf.channels * 64} bits/pixel float64 | {compression_name(apf.compression)} | "
            f"range [{format_float(apf.range_min)}, {format_float(apf.range_max)}] | "
            f"white {format_float(apf.reference_white)} | flags: {flags_to_text(apf.flags)} | "
            f"channels: {names_text}"
        )

    def refresh_preview(self) -> None:
        self._preview_job = None

        if self.current_apf is None or ImageTk is None:
            return

        self.root.update_idletasks()

        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width < 32 or canvas_height < 32:
                canvas_width, canvas_height = 1100, 650

            max_width = max(16, canvas_width - 20)
            max_height = max(16, canvas_height - 20)

            img = render_preview(
                self.current_apf,
                float(self.exposure_var.get()),
                max_width,
                max_height,
            )

            img = fit_image(img, max_width, max_height)
            img = composite_on_checkerboard(img)

            self.preview_photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.preview_photo,
                anchor="center",
            )

            self.set_status(self.describe_apf(self.current_apf, self.current_path), "green")
        except Exception as exc:
            self.set_status(f"Preview failed: {exc}", "red")

    def open_apf(self) -> None:
        path = filedialog.askopenfilename(
            title="Open APF 2.6919 file",
            filetypes=[("APF 2.6919 files", "*.apf"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.set_status("Validating checksum, decompressing, and loading float64 APF...", "orange")
            self.root.update_idletasks()

            apf = read_apf(path)
            self.current_apf = apf
            self.current_path = path
            self.exposure_var.set(0.0)
            self.exposure_label.config(text="+0.0 stops")
            self.refresh_preview()
        except Exception as exc:
            self.current_apf = None
            self.current_path = None
            self.canvas.delete("all")
            messagebox.showerror("Open APF failed", str(exc))
            self.set_status("Open failed.", "red")

    def save_capture(
        self,
        data: np.ndarray,
        channel_map: dict[str, Any],
        metadata: dict[str, Any],
        source_label: str,
    ) -> None:
        initial = f"{source_label}_{time.strftime('%Y%m%d_%H%M%S')}.apf"

        path = filedialog.asksaveasfilename(
            title=f"Save {source_label} as APF 2.6919",
            defaultextension=".apf",
            initialfile=initial,
            filetypes=[("APF 2.6919 files", "*.apf")],
        )

        if not path:
            return

        if not path.lower().endswith(".apf"):
            path += ".apf"

        try:
            method = self.selected_compression()

            self.set_status(
                f"Compressing {data.shape[1]}×{data.shape[0]}×{data.shape[2]} float64 APF using {compression_name(method)}...",
                "orange",
            )
            self.root.update_idletasks()

            apf = write_apf(
                path,
                data,
                channel_map=channel_map,
                metadata=metadata,
                compression=method,
            )

            self.current_apf = apf
            self.current_path = path
            self.exposure_var.set(0.0)
            self.exposure_label.config(text="+0.0 stops")
            self.refresh_preview()

            file_size = os.path.getsize(path)

            messagebox.showinfo(
                "APF saved",
                f"Saved:\n{path}\n\n"
                f"Resolution: {apf.width}×{apf.height}\n"
                f"Channels: {apf.channels}\n"
                f"Bits/channel: 64 float\n"
                f"Total bits/pixel: {apf.channels * 64}\n"
                f"Compression: {compression_name(apf.compression)}\n"
                f"File size: {file_size:,} bytes\n"
                f"SHA-256: {apf.checksum_sha256.hex()[:32]}...",
            )
        except Exception as exc:
            messagebox.showerror("Save APF failed", str(exc))
            self.set_status("Save failed.", "red")

    def take_screenshot(self) -> None:
        self.set_status("Hiding FreeAPF and capturing screen in 1 second...", "orange")
        self.root.update_idletasks()
        self.root.iconify()

        try:
            time.sleep(1.0)
            img, backend = capture_screen_pil()
        except Exception as exc:
            self.root.deiconify()
            self.root.update_idletasks()
            messagebox.showerror("Screenshot failed", str(exc))
            self.set_status("Screenshot failed.", "red")
            return

        self.root.deiconify()
        self.root.update_idletasks()

        try:
            data = pil_image_to_linear_float64_rgba(img)

            metadata = make_metadata(
                "screenshot",
                color_space="linear-srgb",
                display_transfer="sRGB",
                alpha_mode="straight",
                capture_backend=backend,
                capture_width=int(data.shape[1]),
                capture_height=int(data.shape[0]),
                source_precision="8-bit sRGB converted losslessly to float64 scene-linear",
            )

            self.save_capture(data, make_rgba_channel_map(), metadata, "screenshot")
        except Exception as exc:
            messagebox.showerror("Screenshot conversion failed", str(exc))
            self.set_status("Screenshot conversion failed.", "red")

    def take_webcam(self) -> None:
        if cv2 is None:
            messagebox.showerror(
                "Webcam unavailable",
                "OpenCV is required for webcam capture.\n\n"
                "Install it with:\n"
                "pip install opencv-python",
            )
            return

        self.set_status("Starting webcam...", "orange")
        self.root.update_idletasks()

        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open camera index 0.")

            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            except Exception:
                pass

            for _ in range(5):
                cap.read()

            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Camera did not return a frame.")
        except Exception as exc:
            messagebox.showerror("Webcam failed", str(exc))
            self.set_status("Webcam failed.", "red")
            return
        finally:
            if cap is not None:
                cap.release()

        try:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            linear_rgb = srgb_u8_to_linear_float64(frame_rgb)

            height, width = linear_rgb.shape[:2]
            data = np.empty((height, width, 4), dtype="<f8")
            data[..., :3] = linear_rgb
            data[..., 3] = 1.0

            metadata = make_metadata(
                "webcam",
                color_space="linear-srgb",
                display_transfer="sRGB",
                alpha_mode="straight",
                camera_index=0,
                frame_width=int(width),
                frame_height=int(height),
                source_precision="8-bit BGR converted losslessly to float64 scene-linear",
            )

            self.save_capture(data, make_rgba_channel_map(), metadata, "webcam")
        except Exception as exc:
            messagebox.showerror("Webcam conversion failed", str(exc))
            self.set_status("Webcam conversion failed.", "red")

def run_self_test() -> None:
    print(f"FreeAPF {APF_VERSION_STRING} self-test")

    rng = np.random.default_rng(26919)

    height, width = 37, 53
    data = np.empty((height, width, 5), dtype="<f8")

    data[..., 0] = rng.random((height, width)) * 2.0
    data[..., 1] = rng.random((height, width)) - 0.25
    data[..., 2] = rng.random((height, width))
    data[..., 3] = rng.random((height, width))
    data[..., 4] = rng.random((height, width)) * 1000.0

    data[0, 0, 4] = np.inf
    data[1, 1, 2] = np.nan

    channel_map = make_channel_map(
        5,
        roles=["red", "green", "blue", "alpha", "depth"],
        names=["R", "G", "B", "A", "Z"],
        units=["linear_radiance", "linear_radiance", "linear_radiance", "coverage", "meters"],
        ranges=[[0.0, 1.0], [-1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, None]],
        color_indices=[0, 1, 2],
        alpha_index=3,
    )

    metadata = make_metadata(
        "self-test",
        color_space="linear-srgb",
        test_case="RGBA plus depth, negative values, HDR above white, inf and NaN",
    )

    methods = [(COMPRESSION_NONE, "Uncompressed baseline")] + available_compression_options()

    for method, label in methods:
        fd, path = tempfile.mkstemp(suffix=".apf")
        os.close(fd)

        try:
            written = write_apf(
                path,
                data,
                channel_map=channel_map,
                metadata=metadata,
                compression=method,
            )

            loaded = read_apf(path)

            assert loaded.width == width
            assert loaded.height == height
            assert loaded.channels == 5
            assert loaded.compression == method
            assert loaded.uncompressed_size == data.nbytes
            assert loaded.checksum_sha256 == written.checksum_sha256
            assert np.array_equal(loaded.data, data, equal_nan=True)
            assert loaded.flags & FLAG_ALLOW_NEGATIVE
            assert loaded.flags & FLAG_ALLOW_ABOVE_WHITE
            assert loaded.flags & FLAG_ALLOW_NONFINITE

            preview = render_preview(loaded, 0.0, 64, 64)
            assert preview.size == (width, height)

            file_size = os.path.getsize(path)

            print(
                f"  OK  {label:<45} "
                f"file={file_size:>10,} bytes  "
                f"range=[{format_float(loaded.range_min)}, {format_float(loaded.range_max)}]"
            )
        finally:
            if os.path.exists(path):
                os.unlink(path)

    spectral_data = rng.random((17, 21, 7), dtype="<f8") * 1_000_000.0

    spectral_map = make_channel_map(
        7,
        roles=["spectral"] * 7,
        names=[f"{400 + i * 50}nm" for i in range(7)],
        units=["spectral_radiance"] * 7,
        wavelengths_nm=[400.0 + i * 50.0 for i in range(7)],
        color_indices=[0, 2, 4],
        alpha_index=None,
    )

    fd, path = tempfile.mkstemp(suffix=".apf")
    os.close(fd)

    try:
        written = write_apf(
            path,
            spectral_data,
            channel_map=spectral_map,
            metadata=make_metadata("self-test-spectral", spectral_band_count=7),
        )

        loaded = read_apf(path)

        assert loaded.channels == 7
        assert loaded.width == 21
        assert loaded.height == 17
        assert np.array_equal(loaded.data, spectral_data)
        assert loaded.flags & FLAG_ALLOW_ABOVE_WHITE
        assert loaded.channel_map.get("wavelength_nm") == [400.0 + i * 50.0 for i in range(7)]

        preview = render_preview(loaded, 0.0, 64, 64)
        assert preview.size == (21, 17)

        print(f"  OK  7-channel spectral APF                  file={os.path.getsize(path):>10,} bytes")
    finally:
        if os.path.exists(path):
            os.unlink(path)

    print("APF 2.6919 self-test passed.")
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("--selftest", "--test", "-t"):
        run_self_test()
        return

    if ImageTk is None:
        raise SystemExit(
            "PIL.ImageTk is missing.\n\n"
            ":\n"
            "    \n\n"
            "in a virtual environment:\n"
            "    pip install --upgrade Pillow"
        )

    root = tk.Tk()
    FreeAPFApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
