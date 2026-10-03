#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risotone - tone calibration for risograph printing (an EDN-style workflow for riso)

Commands
  chart    Make a high-resolution calibration chart: every 8-bit tone value (0-255) as a
           square patch, replicated and shuffled across the sheet, inside a thick frame so
           the scan can be located automatically.
  profile  Read one or more scans of the printed chart, measure every patch, fit the
           printer's tone response and write:
             - correction LUTs: Photoshop curves (.amp exact, .acv approximate),
               1D + 3D .cube, CSV table
             - print-preview LUTs: 3D .cube "screen proof" files for a Photoshop
               Color Lookup layer (the same trick EDN uses)
             - a report (PNG + TXT) and a detection overlay for every scan
  preview  Render how an image will look printed: tones in the ink + paper color,
           no halftone texture. Works for several inks layered together too.
  apply    Apply the correction to an image so it is ready to send to the riso.

Typical use
  python3 risotone.py chart                                  # risotone_chart.tif (+ .layout.json)
  ... print the chart on the riso, scan it at 300-600 dpi, all auto-corrections off ...
  python3 risotone.py profile scan1.tif scan2.tif --name pink
  python3 risotone.py preview photo.tif --profile pink_risotone --compare
  python3 risotone.py apply   photo.tif --profile pink_risotone

Requirements: Python 3.8+, numpy, scipy, Pillow.
Optional:     matplotlib (report graphs), tifffile (full 16-bit precision for TIFF scans).
Run any command with -h for its options.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
import traceback

VERSION = "1.0.0"
PROG = "risotone"


def _check_dependencies():
    missing = []
    for module, package in (("numpy", "numpy"), ("scipy", "scipy"), ("PIL", "Pillow")):
        try:
            __import__(module)
        except Exception as exc:  # pragma: no cover - depends on the machine
            missing.append((package, "%s: %s" % (type(exc).__name__, exc)))
    if missing:
        pkgs = " ".join(p for p, _ in missing)
        sys.stderr.write(
            "%s: error: missing Python package(s): %s\n"
            "  Install with:\n    python3 -m pip install %s\n"
            "  (if pip complains about an externally managed environment, use a venv:\n"
            "     python3 -m venv ~/risotone-venv && ~/risotone-venv/bin/pip install %s matplotlib\n"
            "   then run ~/risotone-venv/bin/python risotone.py ...)\n"
            "  Import errors:\n    %s\n"
            % (PROG, ", ".join(p for p, _ in missing), pkgs, pkgs,
               "\n    ".join(e for _, e in missing)))
        sys.exit(2)


_check_dependencies()

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # scans and charts are big; this is not untrusted web input

# ----------------------------------------------------------------------------------------
# Chart geometry, in units of one patch cell.  The scan reader depends on these.
# ----------------------------------------------------------------------------------------
FRAME_GAP = 0.30        # paper gap between the patch grid and the frame
FRAME_THICK = 0.40      # thickness of the solid-ink frame
FRAME_OFF = FRAME_GAP + FRAME_THICK
GUTTER = 0.06           # paper line between neighbouring patches
SAMPLE_INSET = 0.28     # each side of a cell that is ignored when measuring (central 44% read)
GRID_N = 255 * 16 + 1    # dense curve resolution: exactly 1/16 of a code value
GRID = np.linspace(0.0, 255.0, GRID_N)
METRICS = ("sctv", "de", "l", "density")
PAPER_SIZES_IN = {
    "letter": (8.5, 11.0), "legal": (8.5, 14.0), "tabloid": (11.0, 17.0), "ledger": (17.0, 11.0),
    "a5": (148 / 25.4, 210 / 25.4), "a4": (210 / 25.4, 297 / 25.4), "a3": (297 / 25.4, 420 / 25.4),
    "b5": (182 / 25.4, 257 / 25.4), "b4": (257 / 25.4, 364 / 25.4),
}


# ----------------------------------------------------------------------------------------
# Errors and logging
# ----------------------------------------------------------------------------------------
class RisoError(Exception):
    """An expected problem: printed as a readable message with hints, no traceback."""

    def __init__(self, message, hints=(), details=None):
        super().__init__(message)
        self.message = message
        self.hints = list(hints)
        self.details = details


class _Log:
    level = 1

    def __init__(self):
        self.warnings = []

    def info(self, msg):
        if self.level >= 1:
            print("[%s] %s" % (PROG, msg), file=sys.stderr, flush=True)

    def detail(self, msg):
        if self.level >= 2:
            print("[%s]     %s" % (PROG, msg), file=sys.stderr, flush=True)

    def warn(self, msg):
        self.warnings.append(msg)
        print("[%s] WARNING: %s" % (PROG, msg), file=sys.stderr, flush=True)


LOG = _Log()


def _format_error(err):
    lines = ["%s: error: %s" % (PROG, err.message)]
    if err.details:
        lines += ["  " + ln for ln in str(err.details).rstrip().splitlines()]
    if err.hints:
        lines.append("  What to try:")
        lines += ["    - " + h for h in err.hints]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Color math (sRGB / D65)
# ----------------------------------------------------------------------------------------
_M_SRGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
_WHITE = _M_SRGB2XYZ.sum(axis=1)
_LUM = _M_SRGB2XYZ[1]


def srgb_decode(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, np.power((np.clip(v, 0.0, None) + 0.055) / 1.055, 2.4))


def srgb_encode(lin):
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)


def _flab(t):
    t = np.asarray(t, dtype=np.float64)
    return np.where(t > (6.0 / 29.0) ** 3, np.cbrt(t), t * (841.0 / 108.0) + 4.0 / 29.0)


def lin_to_xyz(rgb):
    return np.asarray(rgb, dtype=np.float64) @ _M_SRGB2XYZ.T


def lin_to_lab(rgb):
    f = _flab(lin_to_xyz(rgb) / _WHITE)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def sctv_coords(rgb):
    """ISO 20654 (spot colour tone value) coordinates: CIELAB-companded X, Y, Z."""
    return 116.0 * _flab(lin_to_xyz(rgb) / _WHITE) - 16.0


def luminance(rgb):
    return np.asarray(rgb, dtype=np.float64) @ _LUM


def lstar_of_Y(Y):
    return 116.0 * _flab(Y) - 16.0


def lin_to_hex(rgb):
    e = np.round(srgb_encode(rgb) * 255).astype(int)
    return "#%02x%02x%02x" % tuple(int(x) for x in e)


def fmt_lab(lab):
    return "L*%.1f a*%.1f b*%.1f" % tuple(lab)


def tone_values(rgb, paper, solid, metric):
    """Tone value of each patch: 0 = paper, 1 = solid ink (per scan normalisation)."""
    rgb = np.atleast_2d(np.asarray(rgb, dtype=np.float64))
    paper = np.asarray(paper, dtype=np.float64)
    solid = np.asarray(solid, dtype=np.float64)
    if metric == "sctv":
        v, vp, vs = sctv_coords(rgb), sctv_coords(paper), sctv_coords(solid)
        num = np.linalg.norm(v - vp, axis=-1)
        den = float(np.linalg.norm(vs - vp))
    elif metric == "de":
        lab, lp, ls = lin_to_lab(rgb), lin_to_lab(paper), lin_to_lab(solid)
        num = np.linalg.norm(lab - lp, axis=-1)
        den = float(np.linalg.norm(ls - lp))
    elif metric == "l":
        L = lin_to_lab(rgb)[:, 0]
        lp, ls = float(lin_to_lab(paper)[0]), float(lin_to_lab(solid)[0])
        num = lp - L
        den = lp - ls
    elif metric == "density":
        eps = 1e-5
        dp = -np.log10(np.maximum(paper, eps))
        ds = -np.log10(np.maximum(solid, eps))
        ch = int(np.argmax(np.abs(ds - dp)))
        num = -np.log10(np.maximum(rgb[:, ch], eps)) - dp[ch]
        den = float(ds[ch] - dp[ch])
    else:
        raise RisoError("unknown metric %r (choose from %s)" % (metric, ", ".join(METRICS)))
    if abs(den) < 1e-6:
        raise RisoError("the solid ink and the paper measure identically with metric '%s'" % metric,
                        hints=["check that the chart was actually printed and scanned in color",
                               "try --metric sctv (works for any ink color)"])
    return num / den


def parse_target(spec):
    s = str(spec).strip().lower()
    if s in ("srgb", "screen", "lstar", "l*"):
        return ("srgb", None)
    if s in ("linear", "lin"):
        return ("linear", None)
    m = re.match(r"^(?:gamma|g)[:=]?\s*([0-9]*\.?[0-9]+)$", s)
    if m:
        g = float(m.group(1))
        if not 0.3 <= g <= 4.0:
            raise RisoError("gamma %.3g is out of range (0.3 - 4.0)" % g)
        return ("gamma", g)
    raise RisoError("unknown --target %r" % spec,
                    hints=["use 'srgb' (print matches on-screen lightness), 'linear' "
                           "(equal tone steps per code value) or 'gamma:2.2' / 'gamma:1.8'"])


def target_tone(u, target):
    """Requested tone value (0 = paper, 1 = solid) for image code u (0..255)."""
    kind, g = parse_target(target) if isinstance(target, str) else target
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 255.0)
    if kind == "linear":
        t = 1.0 - u / 255.0
    else:
        Y = srgb_decode(u / 255.0) if kind == "srgb" else np.power(u / 255.0, g)
        t = 1.0 - lstar_of_Y(Y) / 100.0
    return np.clip(t, 0.0, 1.0)


# ----------------------------------------------------------------------------------------
# Small numeric helpers
# ----------------------------------------------------------------------------------------
def pava_increasing(y, w=None):
    """Weighted isotonic (non-decreasing) regression, pool-adjacent-violators."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    vals = np.empty(n)
    wts = np.empty(n)
    cnt = np.empty(n, dtype=np.int64)
    top = -1
    for i in range(n):
        top += 1
        vals[top], wts[top], cnt[top] = y[i], w[i], 1
        while top > 0 and vals[top - 1] > vals[top]:
            ws = wts[top - 1] + wts[top]
            vals[top - 1] = (vals[top - 1] * wts[top - 1] + vals[top] * wts[top]) / ws
            wts[top - 1] = ws
            cnt[top - 1] += cnt[top]
            top -= 1
    return np.repeat(vals[:top + 1], cnt[:top + 1])


def isotonic_r2(codes, tv):
    """How well tone follows a monotone (more ink = more tone) curve of printer code: 0..1."""
    codes = np.asarray(codes)
    tv = np.asarray(tv, dtype=np.float64)
    uc, inv = np.unique(codes, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    means = np.bincount(inv, tv) / cnt
    fit = -pava_increasing(-means, cnt)
    sst = float(((tv - tv.mean()) ** 2).sum())
    if sst <= 0:
        return 0.0
    return 1.0 - float(((tv - fit[inv]) ** 2).sum()) / sst


def nw_smooth(xk, fk, wk, xq, sigma, reflect=True):
    """Gaussian kernel (Nadaraya-Watson) smoothing with point-reflection at both ends.
    Keeps a monotone input monotone and has no first-order bias at the ends."""
    xk = np.asarray(xk, dtype=np.float64)
    fk = np.asarray(fk, dtype=np.float64)
    if fk.ndim == 1:
        fk = fk[:, None]
    wk = np.asarray(wk, dtype=np.float64)
    if reflect and len(xk) > 2:
        span = 5.0 * sigma
        lo, hi = xk[0], xk[-1]
        sl = (xk > lo) & (xk <= lo + span)
        sr = (xk < hi) & (xk >= hi - span)
        xk = np.concatenate([2 * lo - xk[sl], xk, 2 * hi - xk[sr]])
        fk = np.concatenate([2 * fk[0] - fk[sl], fk, 2 * fk[-1] - fk[sr]])
        wk = np.concatenate([wk[sl], wk, wk[sr]])
    xq = np.asarray(xq, dtype=np.float64)
    out = np.empty((len(xq), fk.shape[1]))
    for a in range(0, len(xq), 1024):
        k = np.exp(-0.5 * ((xq[a:a + 1024, None] - xk[None, :]) / sigma) ** 2) * wk[None, :]
        s = k.sum(axis=1)
        s[s <= 0] = 1e-300
        out[a:a + 1024] = (k @ fk) / s[:, None]
    return out


def homography(src, dst):
    """3x3 H mapping src (N,2) to dst (N,2), normalized DLT."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    def norm(p):
        c = p.mean(axis=0)
        s = math.sqrt(2.0) / max(np.mean(np.hypot(*(p - c).T)), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])

    ts, td = norm(src), norm(dst)
    sh = np.c_[src, np.ones(len(src))] @ ts.T
    dh = np.c_[dst, np.ones(len(dst))] @ td.T
    rows = []
    for (x, y, _), (u, v, _) in zip(sh, dh):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    return h / h[2, 2]


def apply_h(h, pts):
    pts = np.asarray(pts, dtype=np.float64)
    shp = pts.shape
    p = pts.reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ h.T
    return (q[:, :2] / q[:, 2:3]).reshape(shp)


def _sanitize(name):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return s or "risotone"


def _r(a, nd=6):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def _short_hash(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:8]


def _now():
    return _dt.datetime.now().replace(microsecond=0).isoformat()


# ----------------------------------------------------------------------------------------
# Image loading / saving
# ----------------------------------------------------------------------------------------
class Raster:
    """Pixel data in its native type. data: (H,W) or (H,W,3); maxval: 255, 65535 or 1.0."""

    def __init__(self, path, data, maxval, icc=None, dpi=None, desc=""):
        self.path = path
        self.data = data
        self.maxval = maxval
        self.icc = icc
        self.dpi = dpi
        self.desc = desc

    @property
    def channels(self):
        return 1 if self.data.ndim == 2 else self.data.shape[2]

    @property
    def size_txt(self):
        return "%dx%d px" % (self.data.shape[1], self.data.shape[0])


def _composite_white_numpy(arr, maxval):
    """(H,W,C+alpha) -> (H,W,C) composited over white."""
    a = arr[..., -1:].astype(np.float32) / maxval
    c = arr[..., :-1].astype(np.float32)
    out = c * a + maxval * (1.0 - a)
    return np.clip(np.round(out), 0, maxval).astype(arr.dtype) if maxval != 1.0 else out


def _load_tifffile(path):
    import tifffile  # optional
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        if len(tf.pages) > 1:
            LOG.warn("%s has %d pages/layers; using the first one" % (os.path.basename(path), len(tf.pages)))
        arr = page.asarray()
        photometric = str(getattr(page, "photometric", "")).upper()
        axes = getattr(page, "axes", "")
        icc = None
        dpi = None
        try:
            t = page.tags.get(34675)
            icc = bytes(t.value) if t is not None else None
        except Exception:
            icc = None
        try:
            xr = page.tags.get("XResolution")
            ru = page.tags.get("ResolutionUnit")
            if xr is not None:
                v = xr.value
                val = v[0] / v[1] if isinstance(v, tuple) else float(v)
                unit = int(getattr(ru, "value", 2)) if ru is not None else 2
                dpi = val * 2.54 if unit == 3 else val
                dpi = (dpi, dpi) if dpi > 1 else None
        except Exception:
            dpi = None
    if "SEPARATED" in photometric or "CMYK" in photometric:
        raise RisoError("%s is a CMYK TIFF" % path,
                        hints=["save the separation as Grayscale (or RGB) and try again"])
    if "PALETTE" in photometric:
        raise ValueError("palette TIFF; let Pillow handle it")
    if arr.ndim == 3 and axes.startswith("S") and arr.shape[0] in (2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim > 3:
        arr = arr.reshape((-1,) + arr.shape[-3:])[0]
    if arr.dtype == np.bool_:
        arr = np.where(arr, 255, 0).astype(np.uint8)
    if arr.dtype == np.uint8:
        maxval = 255
    elif arr.dtype == np.uint16:
        maxval = 65535
    elif arr.dtype.kind == "f":
        arr = arr.astype(np.float32)
        maxval = 1.0 if float(np.nanmax(arr)) <= 1.0001 else (255.0 if float(np.nanmax(arr)) <= 255.01 else 65535.0)
    else:
        raise ValueError("unsupported TIFF sample type %s" % arr.dtype)
    if "MINISWHITE" in photometric:
        arr = (maxval - arr).astype(arr.dtype)
    if arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            arr = arr[..., 0]
        elif c == 2:
            arr = _composite_white_numpy(arr, maxval)[..., 0]
        elif c == 4:
            arr = _composite_white_numpy(arr, maxval)
        elif c > 4:
            arr = arr[..., :3]
    desc = "TIFF %s, %s" % (arr.dtype, "gray" if arr.ndim == 2 else "RGB")
    return Raster(path, arr, maxval, icc, dpi, desc)


def _load_pil(path):
    im = Image.open(path)
    n_frames = getattr(im, "n_frames", 1)
    if n_frames > 1:
        LOG.warn("%s has %d frames/pages; using the first one" % (os.path.basename(path), n_frames))
    im.load()
    icc = im.info.get("icc_profile")
    dpi = im.info.get("dpi")
    mode = im.mode
    if mode == "1":
        im = im.convert("L")
    elif mode == "P":
        im = im.convert("RGBA")
    elif mode in ("CMYK",):
        raise RisoError("%s is a CMYK image" % path,
                        hints=["save the separation as Grayscale (one channel) or RGB, then try again"])
    elif mode in ("YCbCr", "HSV"):
        im = im.convert("RGB")
    elif mode == "LAB":
        raise RisoError("%s is a Lab-mode image" % path, hints=["save it as RGB or Grayscale"])
    mode = im.mode
    if mode in ("LA", "La", "PA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA")).convert("L")
    elif mode in ("RGBA", "RGBa", "RGBX"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        arr = np.asarray(im).astype(np.uint16)
        maxval = 65535
    elif mode == "I":
        a = np.asarray(im).astype(np.int64)
        mx = int(a.max()) if a.size else 0
        if mx <= 255:
            arr, maxval = a.astype(np.uint8), 255
        elif mx <= 65535:
            arr, maxval = a.astype(np.uint16), 65535
        else:
            raise RisoError("%s: 32-bit integer image with values up to %d is not supported" % (path, mx),
                            hints=["save it as an 8- or 16-bit TIFF/PNG"])
    elif mode == "F":
        arr = np.asarray(im).astype(np.float32)
        mx = float(np.nanmax(arr)) if arr.size else 1.0
        maxval = 1.0 if mx <= 1.0001 else (255.0 if mx <= 255.01 else 65535.0)
    elif mode == "L":
        arr = np.asarray(im, dtype=np.uint8)
        maxval = 255
    elif mode == "RGB":
        arr = np.asarray(im, dtype=np.uint8)
        maxval = 255
    else:
        arr = np.asarray(im.convert("RGB")).astype(np.uint8)
        maxval = 255
    desc = "%s %s" % (getattr(im, "format", None) or os.path.splitext(path)[1].upper().strip("."), mode)
    return Raster(path, arr, maxval, icc, dpi, desc)


def load_raster(path, role="image"):
    if not os.path.exists(path):
        raise RisoError("%s file not found: %s" % (role, path),
                        hints=["current folder is %s" % os.getcwd(),
                               "check the spelling, or drag the file into the terminal to paste its full path"])
    if os.path.isdir(path):
        raise RisoError("%s path is a folder, not an image file: %s" % (role, path))
    ext = os.path.splitext(path)[1].lower()
    problems = []
    if ext in (".tif", ".tiff"):
        try:
            return _load_tifffile(path)
        except RisoError:
            raise
        except ImportError:
            pass
        except Exception as exc:
            problems.append("tifffile: %s: %s" % (type(exc).__name__, exc))
    try:
        r = _load_pil(path)
        if ext in (".tif", ".tiff") and r.maxval == 255 and r.channels == 3:
            LOG.detail("(install tifffile to read 16-bit RGB TIFFs at full precision)")
        return r
    except RisoError:
        raise
    except Exception as exc:
        problems.append("Pillow: %s: %s" % (type(exc).__name__, exc))
    hints = ["supported: TIFF, PNG, JPEG, BMP, GIF, WebP (8 or 16 bit, gray or RGB)",
             "iPhone HEIC photos: export as JPEG or TIFF first"]
    if ext in (".tif", ".tiff"):
        hints.append("for unusual TIFF compression: python3 -m pip install tifffile imagecodecs")
    raise RisoError("could not read %s %s" % (role, path), hints=hints, details="\n".join(problems))


def _icc_description(icc):
    try:
        from PIL import ImageCms
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except Exception:
        return "unreadable profile"


def _icc_to_srgb_8bit(arr8, icc):
    from PIL import ImageCms
    src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    dst = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    im = Image.fromarray(np.ascontiguousarray(arr8, dtype=np.uint8))
    out = ImageCms.profileToProfile(im, src, dst, renderingIntent=1, outputMode="RGB")
    return np.asarray(out).astype(np.uint8)


def _decoder(encoding):
    kind, g = encoding
    if kind == "srgb":
        return lambda v: srgb_decode(v)
    if kind == "linear":
        return lambda v: np.asarray(v, dtype=np.float64)
    return lambda v: np.power(np.clip(np.asarray(v, dtype=np.float64), 0, None), g)


def parse_scan_encoding(spec):
    s = str(spec).strip().lower()
    if s in ("auto", "srgb"):
        return (s, None)
    if s in ("linear", "lin", "gamma1", "gamma:1", "gamma:1.0"):
        return ("linear", None)
    m = re.match(r"^(?:gamma|g)[:=]?\s*([0-9]*\.?[0-9]+)$", s)
    if m:
        return ("gamma", float(m.group(1)))
    raise RisoError("unknown --scan-encoding %r" % spec,
                    hints=["use auto (embedded profile, else sRGB), srgb, linear, or gamma:2.2"])


def scan_to_linear(r, encoding, maxdim):
    """Scan -> linear-light sRGB float32 (H,W,3), block-averaged down to <= maxdim px.
    Averaging in linear light is exactly how the eye merges halftone dots into a tone."""
    data = r.data
    enc = encoding
    if enc[0] == "auto":
        enc = ("srgb", None)
        if r.icc:
            desc = _icc_description(r.icc)
            if re.search(r"srgb", desc, re.I):
                LOG.detail("embedded profile '%s' -> treated as sRGB" % desc)
            else:
                try:
                    a8 = data if data.dtype == np.uint8 else np.clip(
                        np.round(data.astype(np.float32) / r.maxval * 255.0), 0, 255).astype(np.uint8)
                    if data.dtype != np.uint8:
                        LOG.warn("%s: 16-bit scan with embedded profile '%s' was converted to sRGB "
                                 "through 8 bits. If the scan is linear (gamma 1.0), rerun with "
                                 "--scan-encoding linear." % (os.path.basename(r.path), desc))
                    data = _icc_to_srgb_8bit(a8, r.icc)
                    LOG.info("  converted from embedded color profile '%s' to sRGB" % desc)
                except Exception as exc:
                    LOG.warn("could not apply embedded color profile '%s' (%s); assuming sRGB"
                             % (desc, exc))
    decode = _decoder(enc)
    H, W = data.shape[:2]
    k = max(1, int(math.ceil(max(H, W) / float(maxdim))))
    Hk, Wk = H // k, W // k
    if Hk < 50 or Wk < 50:
        raise RisoError("scan %s is too small (%s)" % (r.path, r.size_txt),
                        hints=["scan at 300 dpi or more"])
    lut = None
    if data.dtype == np.uint8:
        lut = decode(np.arange(256) / 255.0).astype(np.float32)
    elif data.dtype == np.uint16:
        lut = decode(np.arange(65536) / 65535.0).astype(np.float32)
    out = np.empty((Hk, Wk, 3), np.float32)
    step = max(1, 1024 // k) * k
    for y0 in range(0, Hk * k, step):
        y1 = min(Hk * k, y0 + step)
        blk = data[y0:y1, :Wk * k]
        f = lut[blk] if lut is not None else decode(
            np.clip(blk.astype(np.float32) / r.maxval, 0, 1)).astype(np.float32)
        if f.ndim == 2:
            f = f[..., None]
        f = f.reshape((y1 - y0) // k, k, Wk, k, f.shape[-1]).mean(axis=(1, 3))
        if f.shape[-1] == 1:
            f = np.repeat(f, 3, axis=-1)
        out[y0 // k:y1 // k] = f[..., :3]
    return out, k


def block_reduce(img, maxdim):
    H, W = img.shape[:2]
    k = max(1, int(math.ceil(max(H, W) / float(maxdim))))
    if k == 1:
        return img.copy(), 1
    Hk, Wk = H // k, W // k
    c = img[:Hk * k, :Wk * k]
    return c.reshape(Hk, k, Wk, k, -1).mean(axis=(1, 3)).astype(np.float32), k


def gray_codes(r):
    """Image -> gray code values u (float32 0..255) as the printer would receive them.
    Gray images are used as-is (no color management: the printer gets the raw values)."""
    d = r.data
    if r.channels == 1:
        if d.dtype == np.uint8:
            return d.astype(np.float32)
        return (d.astype(np.float32) * (255.0 / r.maxval)).astype(np.float32)
    LOG.warn("%s is RGB; converting to gray with sRGB luminance. For exact control, convert your "
             "separation to Grayscale yourself." % os.path.basename(r.path))
    out = np.empty(d.shape[:2], np.float32)
    lut = None
    if d.dtype == np.uint8:
        lut = srgb_decode(np.arange(256) / 255.0).astype(np.float32)
    for y0 in range(0, d.shape[0], 512):
        blk = d[y0:y0 + 512]
        lin = lut[blk] if lut is not None else srgb_decode(
            np.clip(blk.astype(np.float32) / r.maxval, 0, 1)).astype(np.float32)
        Y = lin @ _LUM.astype(np.float32)
        out[y0:y0 + 512] = (srgb_encode(Y) * 255.0).astype(np.float32)
    return out


def _srgb_icc_bytes():
    try:
        from PIL import ImageCms
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None


def save_image(arr, path, dpi=None, srgb_tag=False):
    """Save uint8 (H,W)/(H,W,3) or uint16 (H,W). TIFF gets deflate compression."""
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        raise RisoError("output folder does not exist: %s" % d)
    if arr.dtype == np.uint16:
        im = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint16))
    elif arr.ndim == 2:
        im = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))
    else:
        im = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))
    ext = os.path.splitext(path)[1].lower()
    kw = {}
    if dpi:
        kw["dpi"] = (float(dpi[0]), float(dpi[1]))
    if srgb_tag and ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        icc = _srgb_icc_bytes()
        if icc:
            kw["icc_profile"] = icc
    try:
        if ext in (".tif", ".tiff"):
            try:
                im.save(path, compression="tiff_deflate", **kw)
            except Exception:
                im.save(path, **kw)
        elif ext in (".jpg", ".jpeg"):
            im.save(path, quality=95, **kw)
        else:
            im.save(path, **kw)
    except Exception as exc:
        raise RisoError("could not write %s: %s: %s" % (path, type(exc).__name__, exc),
                        hints=["check the folder is writable and the extension is .tif, .png or .jpg"])
    return path


_FONT_CACHE = {}


def _font(px):
    px = max(6, int(round(px)))
    if px in _FONT_CACHE:
        return _FONT_CACHE[px]
    f = None
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc", "HelveticaNeue.ttc",
                 "/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            f = ImageFont.truetype(name, px)
            break
        except Exception:
            continue
    if f is None:
        try:
            f = ImageFont.load_default(size=px)
        except TypeError:
            f = None
    _FONT_CACHE[px] = f
    return f


def draw_text(img, xy, text, px, fill):
    """Text of roughly px pixels height; falls back to an upscaled bitmap font on old Pillow."""
    f = _font(px)
    if f is not None:
        ImageDraw.Draw(img).text(xy, text, font=f, fill=fill)
        return
    base = ImageFont.load_default()
    tmp = Image.new("L", (max(1, 7 * len(text) + 4), 14), 0)
    ImageDraw.Draw(tmp).text((0, 0), text, font=base, fill=255)
    s = max(1, int(round(px / 11.0)))
    tmp = tmp.resize((tmp.width * s, tmp.height * s), Image.NEAREST)
    color = Image.new(img.mode, tmp.size, fill)
    img.paste(color, (int(xy[0]), int(xy[1])), tmp)


# ----------------------------------------------------------------------------------------
# Chart layout
# ----------------------------------------------------------------------------------------
def _det_shuffle(n, seed):
    """Deterministic Fisher-Yates (same on every numpy/python version)."""
    idx = list(range(n))
    state = (int(seed) * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    for i in range(n - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        j = (state >> 33) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def make_layout(levels=256, replicates=2, order="shuffled", seed=1, min_anchors=8):
    if not 2 <= levels <= 256:
        raise RisoError("--levels must be between 2 and 256 (got %d)" % levels,
                        hints=["256 = every 8-bit value; to put more patches on the sheet raise "
                               "--replicates instead"])
    if not 1 <= replicates <= 16:
        raise RisoError("--replicates must be between 1 and 16 (got %d)" % replicates)
    if order not in ("shuffled", "ordered"):
        raise RisoError("--order must be 'shuffled' or 'ordered'")
    values = np.unique(np.round(np.linspace(0, 255, levels)).astype(int))
    levels_seq = [(int(v), "level") for _ in range(replicates) for v in values]
    n = int(math.ceil(math.sqrt(len(levels_seq) + max(0, min_anchors))))
    n_anchor = n * n - len(levels_seq)
    anchors = [(255, "paper") if i % 2 == 0 else (0, "solid") for i in range(n_anchor)]
    if order == "shuffled":
        entries = levels_seq + anchors
        perm = _det_shuffle(len(entries), seed)
        entries = [entries[i] for i in perm]
    else:  # ordered ramp(s), with the anchors spread evenly through the sheet
        pos = set(np.round(np.linspace(0, n * n - 1, n_anchor)).astype(int).tolist()) if n_anchor else set()
        while len(pos) < n_anchor:
            pos.add(max(set(range(n * n)) - pos))
        it_l, it_a = iter(levels_seq), iter(anchors)
        entries = [next(it_a) if i in pos else next(it_l) for i in range(n * n)]
    rows = [i // n for i in range(len(entries))]
    cols = [i % n for i in range(len(entries))]
    lay = {
        "format": "risotone-layout", "version": 1, "rows": n, "cols": n,
        "geometry": {"frame_gap": FRAME_GAP, "frame_thickness": FRAME_THICK, "gutter": GUTTER},
        "levels": int(len(values)), "replicates": int(replicates), "order": order, "seed": int(seed),
        "anchors": int(n_anchor),
        "patches": {"row": rows, "col": cols, "value": [e[0] for e in entries],
                    "code": [e[0] for e in entries], "kind": [e[1] for e in entries]},
        "precorrected_with": None,
    }
    lay["id"] = _short_hash(lay["rows"], lay["cols"], lay["patches"]["value"], lay["patches"]["code"])
    return lay


def precorrect_layout(lay, prof):
    """Run every level patch through a profile's correction (pass-2 chart)."""
    p = lay["patches"]
    t8 = prof.table8
    p["code"] = [int(t8[v]) if k == "level" else int(v) for v, k in zip(p["value"], p["kind"])]
    lay["precorrected_with"] = {"name": prof.name, "id": prof.id, "path": os.path.abspath(prof.path)}
    lay["id"] = _short_hash(lay["rows"], lay["cols"], p["value"], p["code"])
    return lay


def validate_layout(lay, src="layout"):
    try:
        if lay.get("format") == "risotone-profile":
            raise RisoError("%s is a profile file, not a chart layout file" % src,
                            hints=["pass the *.layout.json written next to the chart by 'chart'"])
        if lay.get("format") != "risotone-layout":
            raise RisoError("%s is not a risotone layout file" % src)
        p = lay["patches"]
        n = len(p["code"])
        assert n == len(p["row"]) == len(p["col"]) == len(p["value"]) == len(p["kind"])
        assert n == lay["rows"] * lay["cols"]
        g = lay.get("geometry", {})
        if abs(g.get("frame_gap", FRAME_GAP) - FRAME_GAP) > 1e-9 or \
                abs(g.get("frame_thickness", FRAME_THICK) - FRAME_THICK) > 1e-9:
            raise RisoError("%s was made by a different version of risotone (frame geometry differs)" % src)
    except RisoError:
        raise
    except Exception as exc:
        raise RisoError("%s is damaged or incomplete (%s: %s)" % (src, type(exc).__name__, exc),
                        hints=["re-create it with the same 'chart' options you printed with"])
    return lay


def paper_size_px(spec, dpi, landscape=False):
    s = str(spec).strip().lower()
    if s in PAPER_SIZES_IN:
        w, h = PAPER_SIZES_IN[s]
    else:
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*x\s*([0-9]*\.?[0-9]+)\s*(in|mm|cm)?$", s)
        if not m:
            raise RisoError("unknown paper size %r" % spec,
                            hints=["use one of: " + ", ".join(sorted(PAPER_SIZES_IN)),
                                   "or a custom size like 8.5x11in, 210x297mm, 11x17in"])
        w, h = float(m.group(1)), float(m.group(2))
        unit = m.group(3) or "in"
        f = {"in": 1.0, "mm": 1 / 25.4, "cm": 1 / 2.54}[unit]
        w, h = w * f, h * f
    if landscape:
        w, h = max(w, h), min(w, h)
    return int(round(w * dpi)), int(round(h * dpi)), (w, h)


def render_chart(lay, paper, dpi, margin_in, landscape, labels, ramp, prof=None):
    Wpx, Hpx, (win, hin) = paper_size_px(paper, dpi, landscape)
    m = int(round(margin_in * dpi))
    title_h = int(round(0.42 * dpi))
    ramp_h = int(round(0.66 * dpi)) if ramp else 0
    rows, cols = lay["rows"], lay["cols"]
    avail_w, avail_h = Wpx - 2 * m, Hpx - 2 * m - title_h - ramp_h
    cell = int(math.floor(min(avail_w / (cols + 2 * FRAME_OFF), avail_h / (rows + 2 * FRAME_OFF))))
    cell_mm = cell / dpi * 25.4
    if cell_mm < 4.0:
        raise RisoError("patches would be only %.1f mm wide on %s paper - too small to measure reliably"
                        % (cell_mm, paper),
                        hints=["use bigger paper (--paper tabloid / a3)", "lower --replicates",
                               "lower --margin"])
    if cell_mm < 6.0:
        LOG.warn("patches are %.1f mm; 6 mm+ measures more reliably on coarse riso screens" % cell_mm)
    fw = int(round(cell * (cols + 2 * FRAME_OFF)))
    fh = int(round(cell * (rows + 2 * FRAME_OFF)))
    fx0 = m + (avail_w - fw) // 2
    fy0 = m + title_h + (avail_h - fh) // 2
    off = int(round(FRAME_OFF * cell))
    ox, oy = fx0 + off, fy0 + off

    def X(u):
        return int(round(ox + u * cell))

    def Y(v):
        return int(round(oy + v * cell))

    img = np.full((Hpx, Wpx), 255, np.uint8)
    img[Y(-FRAME_OFF):Y(rows + FRAME_OFF), X(-FRAME_OFF):X(cols + FRAME_OFF)] = 0
    img[Y(-FRAME_GAP):Y(rows + FRAME_GAP), X(-FRAME_GAP):X(cols + FRAME_GAP)] = 255
    g = max(1, int(round(GUTTER * cell)))
    g1, g2 = g // 2, g - g // 2
    p = lay["patches"]
    for r, c, code in zip(p["row"], p["col"], p["code"]):
        img[Y(r) + g1:Y(r + 1) - g2, X(c) + g1:X(c + 1) - g2] = code
    if ramp:
        ry0 = Y(rows + FRAME_OFF) + int(round(0.16 * dpi))
        ry1 = ry0 + int(round(0.26 * dpi))
        xa, xb = X(-FRAME_OFF), X(cols + FRAME_OFF)
        vals = np.round(np.linspace(0, 255, xb - xa)).astype(int)
        if prof is not None:
            vals = prof.table8[vals]
        img[ry0:ry1, xa:xb] = vals[None, :].astype(np.uint8)
    pim = Image.fromarray(img)
    if labels:
        fpx = max(8, 0.11 * cell)
        for r, c, v, code in zip(p["row"], p["col"], p["value"], p["code"]):
            draw_text(pim, (X(c) + g1 + 0.05 * cell, Y(r) + g1 + 0.03 * cell), str(v), fpx,
                      0 if code >= 128 else 255)
    if ramp:
        tick_h = int(round(0.06 * dpi))
        d = ImageDraw.Draw(pim)
        for v in (0, 32, 64, 96, 128, 160, 192, 224, 255):
            x = xa + int(round(v / 255.0 * (xb - xa - 1)))
            d.rectangle([x - max(1, dpi // 300), ry1, x + max(1, dpi // 300), ry1 + tick_h], fill=0)
            draw_text(pim, (x - 0.05 * dpi if v else x, ry1 + tick_h + 0.01 * dpi), str(v), 0.085 * dpi, 0)
    t1 = "risotone chart   %d levels x %d (%s, seed %d)   %dx%d grid   id %s   %d dpi %s" % (
        lay["levels"], lay["replicates"], lay["order"], lay["seed"], cols, rows, lay["id"], dpi, paper)
    t2 = ("Print at 100% (no fit-to-page), no color management, one ink, same driver settings as real "
          "prints. Scan at 300-600 dpi, auto-corrections off. Bottom ramp = visual check only.")
    if prof is not None:
        t2 = "PASS 2: pre-corrected with profile '%s' (%s). Scan, then run profile with this chart's layout file." % (
            prof.name, prof.id)
    draw_text(pim, (m, m), t1, 0.12 * dpi, 0)
    draw_text(pim, (m, m + 0.19 * dpi), t2, 0.085 * dpi, 0)
    lay["render"] = {"dpi": dpi, "paper": str(paper), "landscape": bool(landscape), "page_px": [Wpx, Hpx],
                     "cell_px": cell, "cell_mm": round(cell_mm, 3), "origin_px": [ox, oy]}
    return np.asarray(pim), cell_mm, (win, hin)


# ----------------------------------------------------------------------------------------
# Finding the chart in a scan
# ----------------------------------------------------------------------------------------
def unit_frame_corners(lay):
    a = FRAME_OFF
    C, R = lay["cols"], lay["rows"]
    return np.array([[-a, -a], [C + a, -a], [C + a, R + a], [-a, R + a]], dtype=np.float64)


def _expected_frame_fill(lay):
    C, R = lay["cols"], lay["rows"]
    return 1.0 - ((C + 2 * FRAME_GAP) * (R + 2 * FRAME_GAP)) / ((C + 2 * FRAME_OFF) * (R + 2 * FRAME_OFF))


def _poly_area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _quad_from_points(pts):
    from scipy.spatial import ConvexHull
    if len(pts) < 20:
        return None
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    hv = pts[hull.vertices]
    if len(hv) > 1500:
        hv = hv[np.linspace(0, len(hv) - 1, 1500).astype(int)]
    d2 = ((hv[:, None, :] - hv[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    p, q = hv[i], hv[j]
    v = q - p
    nrm = np.array([-v[1], v[0]]) / (np.hypot(v[0], v[1]) + 1e-12)
    s = (hv - p) @ nrm
    k3, k4 = int(np.argmax(s)), int(np.argmin(s))
    if s[k3] <= 2 or s[k4] >= -2:
        return None
    quad = np.array([p, hv[k3], q, hv[k4]], dtype=np.float64)
    c = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))]
    quad = np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)
    return quad, float(hull.volume)


def _refine_quad(quad, pts, tol, iters=2):
    for _ in range(iters):
        lines = []
        for k in range(4):
            p, q = quad[k], quad[(k + 1) % 4]
            v = q - p
            L = float(np.hypot(v[0], v[1]))
            d = v / L
            n = np.array([-d[1], d[0]])
            rel = pts - p
            t = rel @ d / L
            dist = np.abs(rel @ n)
            sel = (t > 0.06) & (t < 0.94) & (dist < tol)
            if sel.sum() < 15:
                lines.append((p, d))
                continue
            P = pts[sel]
            c = P.mean(axis=0)
            _, _, vt = np.linalg.svd(P - c, full_matrices=False)
            lines.append((c, vt[0]))
        new = []
        for k in range(4):
            (c1, d1), (c2, d2) = lines[(k - 1) % 4], lines[k]
            A = np.array([d1, -d2]).T
            if abs(np.linalg.det(A)) < 1e-9:
                new.append(quad[k])
                continue
            s = np.linalg.solve(A, c2 - c1)
            new.append(c1 + s[0] * d1)
        quad = np.array(new)
    return quad


def find_frame(E, lay):
    """Locate the thick solid frame in a 'distance from paper' map. Returns (quad, info)."""
    H, W = E.shape
    b = max(2, int(0.02 * min(H, W)))
    core = E[b:H - b, b:W - b]
    eref = float(np.percentile(core, 99.5))
    info = {"eref": eref, "tried": []}
    if eref < 3.0:
        info["reason"] = "almost nothing on the page differs from the paper (max dE %.1f)" % eref
        return None, info
    exp_fill = _expected_frame_fill(lay)
    exp_aspect = (lay["cols"] + 2 * FRAME_OFF) / (lay["rows"] + 2 * FRAME_OFF)
    best = None
    st = np.ones((3, 3), bool)
    big = max(3, int(round(1.5 * max(H, W) / 290.0)))  # ~1.5 mm: bridges ink streaks through the frame
    passes = ((2, 3.0 * exp_fill, (0.5, 0.35, 0.65, 0.25, 0.8)),
              (big, 0.92, (0.5, 0.35, 0.25)))
    for close_it, fill_max, fracs in passes:
        for frac in fracs:
            T = frac * eref
            mask = ndimage.binary_closing(E > T, structure=st, iterations=close_it)
            lab, nlab = ndimage.label(mask, structure=st)
            if nlab == 0:
                continue
            counts = np.bincount(lab.ravel())
            for idx, sl in enumerate(ndimage.find_objects(lab), start=1):
                if sl is None:
                    continue
                h = sl[0].stop - sl[0].start
                w = sl[1].stop - sl[1].start
                if h * w < 0.06 * H * W:
                    continue
                comp = lab[sl] == idx
                ys, xs = np.nonzero(comp)
                pts = np.column_stack([xs + sl[1].start, ys + sl[0].start]).astype(np.float64)
                res = _quad_from_points(pts)
                if res is None:
                    continue
                quad, hull_area = res
                qa = _poly_area(quad)
                if qa <= 0:
                    continue
                rect = hull_area / qa
                fill = counts[idx] / qa
                sides = np.hypot(*(np.roll(quad, -1, axis=0) - quad).T)
                aspect = (sides[0] + sides[2]) / max(sides[1] + sides[3], 1e-9)
                aerr = min(abs(math.log(aspect / exp_aspect)), abs(math.log(aspect * exp_aspect)))
                ok = (0.93 < rect < 1.07) and (0.3 * exp_fill <= fill <= fill_max) and aerr < 0.15
                touches = sl[0].start <= 4 or sl[1].start <= 4 or sl[0].stop >= H - 4 or sl[1].stop >= W - 4
                info["tried"].append({"threshold": round(T, 2), "closing": close_it,
                                      "area_frac": round(qa / (H * W), 3),
                                      "rect": round(rect, 3), "fill": round(fill, 3),
                                      "fill_expected": round(exp_fill, 3), "aspect_err": round(aerr, 3),
                                      "touches_edge": bool(touches), "ok": ok})
                score = qa * (0.4 if touches else 1.0)
                if ok and (best is None or score > best[0] * 1.02):
                    best = (score, quad, sl, comp, T, sides)
            if best is not None and frac == fracs[0]:
                break
        if best is not None:
            if close_it != 2:
                LOG.detail("frame found only after bridging gaps in it (streaky or broken frame?)")
            break
    if best is None:
        info["reason"] = "no frame-shaped (thick hollow rectangle) ink region was found"
        return None, info
    qa, quad, sl, comp, T, sides = best
    inner = ndimage.binary_erosion(comp, structure=st)
    ys, xs = np.nonzero(comp & ~inner)
    bpts = np.column_stack([xs + sl[1].start, ys + sl[0].start]).astype(np.float64)
    thick = FRAME_THICK * float(np.mean(sides)) / (0.5 * (lay["rows"] + lay["cols"]) + 2 * FRAME_OFF)
    quad = _refine_quad(quad, bpts, tol=max(2.0, 0.35 * thick))
    info.update({"threshold": T, "thickness_px": thick})
    return quad, info


def _orientations(quad):
    out = []
    for r in range(4):
        out.append((r, False, np.array([quad[(i + r) % 4] for i in range(4)])))
        out.append((r, True, np.array([quad[(r - i) % 4] for i in range(4)])))
    return out


def _patch_unit_points(lay, inset, m):
    p = lay["patches"]
    cols = np.asarray(p["col"], dtype=np.float64)
    rows = np.asarray(p["row"], dtype=np.float64)
    o = inset + (np.arange(m) + 0.5) / m * (1.0 - 2.0 * inset)
    ox, oy = np.meshgrid(o, o)
    ux = cols[:, None] + ox.ravel()[None, :]
    uy = rows[:, None] + oy.ravel()[None, :]
    return np.stack([ux, uy], axis=-1)  # (N, m*m, 2)


def _sample(img, pts):
    """Bilinear sample (H,W,C) at pts (...,2) given as x,y. Returns (..., C) and in-bounds mask."""
    H, W = img.shape[:2]
    x = pts[..., 0].ravel()
    y = pts[..., 1].ravel()
    inb = (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)
    out = np.stack([ndimage.map_coordinates(img[..., c], [y, x], order=1, mode="nearest")
                    for c in range(img.shape[2])], axis=-1)
    return out.reshape(pts.shape[:-1] + (img.shape[2],)), inb.reshape(pts.shape[:-1])


def _paper_candidates(det_lin, det_lab):
    enc = srgb_encode(det_lin)
    q = np.clip((enc * 24).astype(np.int64), 0, 23)
    codes = q[..., 0] * 576 + q[..., 1] * 24 + q[..., 2]
    H, W = codes.shape
    b = max(1, int(0.03 * min(H, W)))
    inner = codes[b:H - b, b:W - b].ravel()
    lab_in = det_lab[b:H - b, b:W - b].reshape(-1, 3)
    lin_in = det_lin[b:H - b, b:W - b].reshape(-1, 3)
    counts = np.bincount(inner, minlength=24 ** 3)
    cands = []
    for binid in np.argsort(counts)[::-1][:8]:
        if counts[binid] < 0.01 * inner.size:
            break
        sel = inner == binid
        lab0 = lab_in[sel].mean(axis=0)
        near = np.linalg.norm(lab_in - lab0, axis=1) < 5.0
        lab1 = lab_in[near].mean(axis=0)
        lin1 = lin_in[near].mean(axis=0)
        if all(np.linalg.norm(lab1 - c[0]) > 6.0 for c in cands):
            cands.append((lab1, lin1, float(near.mean())))
    return cands


def _rgb8(lin):
    return np.round(srgb_encode(lin) * 255).astype(np.uint8)


def read_scan(path, lay, opts):
    """Measure every patch of one scanned chart. Returns a dict (see keys at the end)."""
    LOG.info("reading scan %s" % path)
    r = load_raster(path, "scan")
    LOG.detail("%s, %s%s" % (r.desc, r.size_txt, ", %.0f dpi" % r.dpi[0] if r.dpi else ""))
    lin, k = scan_to_linear(r, opts["scan_encoding"], opts["work_size"])
    det, k2 = block_reduce(lin, 1400)
    det_lab = lin_to_lab(det)
    det_rgb8 = _rgb8(det)
    base = os.path.splitext(os.path.basename(path))[0]
    cands = _paper_candidates(det, det_lab)
    if not cands:
        raise RisoError("could not find a dominant paper color in %s" % path,
                        hints=["scan the whole sheet including some blank margin"])
    attempts = []
    chosen = None
    codes = np.asarray(lay["patches"]["code"])
    U = unit_frame_corners(lay)
    upts_quick = _patch_unit_points(lay, 0.3, 5)
    det_lab32 = det_lab.astype(np.float32)
    for ci, (plab, plin, frac) in enumerate(cands[:4]):
        E = np.linalg.norm(det_lab - plab, axis=-1).astype(np.float32)
        quad, finfo = find_frame(E, lay)
        att = {"paper_lab": plab, "frac": frac, "frame": quad is not None, "finfo": finfo, "quad": quad}
        attempts.append(att)
        if quad is None:
            continue
        scores = []
        for rot, mir, q in _orientations(quad):
            Hm = homography(U, q)
            px = apply_h(Hm, upts_quick)
            vals, inb = _sample(det_lab32, px)
            plab_patch = vals.mean(axis=1)
            tv = np.linalg.norm(plab_patch - plab, axis=-1)
            r2 = isotonic_r2(codes, tv) if inb.mean() > 0.9 else -1.0
            scores.append((r2, rot, mir, q, Hm))
        scores.sort(key=lambda s: -s[0])
        att["r2"] = [round(s[0], 3) for s in scores]
        if scores[0][0] >= 0.85 or (scores[0][0] >= 0.5 and scores[0][0] - scores[1][0] >= 0.3):
            chosen = (plab, plin, quad, finfo, scores)
            break
    if chosen is None:
        fail_png = os.path.join(opts["outdir"], "%s_detect_%s_FAILED.png" % (opts["name"], _sanitize(base)))
        _failure_overlay(det_rgb8, attempts, fail_png)
        det_lines = []
        frame_found = any(a["frame"] for a in attempts)
        blank = all(a["finfo"].get("eref", 99) < 3.0 for a in attempts if not a["frame"]) and not frame_found
        cut = False
        for i, a in enumerate(attempts):
            line = "paper guess %d (%s, %.0f%% of the scan): " % (i + 1, fmt_lab(a["paper_lab"]), 100 * a["frac"])
            if not a["frame"]:
                line += "frame not found - " + a["finfo"].get("reason", "?")
                tried = [t for t in a["finfo"].get("tried", []) if t["area_frac"] < 0.95]
                if tried:
                    t = max(tried, key=lambda d: d["area_frac"])
                    line += " (closest shape: %.0f%% of the scan, ink fill %.3f vs %.3f expected, side ratio off by %.0f%%%s)" % (
                        100 * t["area_frac"], t["fill"], t["fill_expected"], 100 * t["aspect_err"],
                        ", touches the scan edge" if t["touches_edge"] else "")
                    cut = cut or (t["touches_edge"] and i == 0)
            else:
                line += "frame found, but the patches don't follow any tone curve (best fit %.2f, need 0.85)" % a["r2"][0]
            if i == 0 or a["frame"]:
                det_lines.append(line)
        if len(attempts) > len(det_lines):
            det_lines.append("(%d other paper-color guesses also failed)" % (len(attempts) - len(det_lines)))
        hints = ["look at %s" % fail_png]
        if blank:
            hints.insert(0, "the scan looks blank (no ink found) - is it the right file / the printed side?")
        elif frame_found:
            hints += ["the scan doesn't match the layout file: use the .layout.json of the chart you actually "
                      "printed (--layout); a different --seed/--levels/--replicates or a pass-2 chart gives this",
                      "if the scan is a camera photo, make sure it is sharp and evenly lit"]
        else:
            if cut:
                hints.insert(0, "part of the chart seems cut off: the whole chart including the thick black "
                                "outer frame must be inside the scan")
            hints += ["the thick frame around the patches must be fully visible and unbroken",
                      "scan with auto exposure / auto color / sharpening / descreen OFF",
                      "for a very light ink, scan in 48-bit color and check the scan isn't clipped to white"]
        raise RisoError("could not read the chart in %s" % path, hints=hints, details="\n".join(det_lines))
    plab, plin, quad, finfo, scores = chosen
    r2, rot, mir, qdet, Hdet = scores[0]
    second = scores[1][0]
    if r2 - second < 0.05:
        LOG.warn("%s: chart orientation is ambiguous (fit %.3f vs %.3f). Check the detection overlay."
                 % (base, r2, second))
    # ---- full-resolution measurement
    qlin = qdet * k2 + (k2 - 1) / 2.0
    Hlin = homography(U, qlin)
    sides = np.hypot(*(np.roll(qlin, -1, axis=0) - qlin).T)
    cell_px = float(np.mean(sides)) / (0.5 * (lay["rows"] + lay["cols"]) + 2 * FRAME_OFF)
    w = max(1, int(round(0.12 * cell_px)))
    filt = ndimage.uniform_filter(lin, size=(w, w, 1), mode="nearest")
    m = int(np.clip(round((1 - 2 * SAMPLE_INSET) * cell_px), 6, 40))
    upts = _patch_unit_points(lay, SAMPLE_INSET, m)
    vals, inb = _sample(filt, apply_h(Hlin, upts))
    del filt
    outside = 1.0 - inb.mean(axis=1)
    if (outside > 0.1).any():
        raise RisoError("part of the chart lies outside the scan %s (%d patches cut off)"
                        % (path, int((outside > 0.1).sum())),
                        hints=["rescan with the whole sheet on the glass"])
    Yv = vals @ _LUM
    med = np.median(Yv, axis=1)
    mad = np.median(np.abs(Yv - med[:, None]), axis=1)
    keep = np.abs(Yv - med[:, None]) <= (4.0 * 1.4826 * mad + 0.002)[:, None]
    rgb = (vals * keep[..., None]).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)[:, None]
    spread = 1.4826 * mad / np.maximum(med, 1e-4)
    clip_frac = (vals.max(axis=2) >= 0.985).mean(axis=1)
    black_frac = (vals.max(axis=2) <= 0.0015).mean(axis=1)
    p = lay["patches"]
    kinds = np.asarray(p["kind"])
    if opts.get("flatfield"):
        rgb = _flatfield(rgb, lay, codes, base)
    is_paper = codes == 255
    is_solid = codes == 0
    if is_paper.sum() == 0 or is_solid.sum() == 0:
        raise RisoError("layout has no paper (255) or solid (0) patches - cannot normalise",
                        hints=["re-create the chart with this version of risotone"])
    paper = np.median(rgb[is_paper], axis=0)
    solid = np.median(rgb[is_solid], axis=0)
    paper_lab, solid_lab = lin_to_lab(paper), lin_to_lab(solid)
    de_ps = float(np.linalg.norm(paper_lab - solid_lab))
    rot_deg = math.degrees(math.atan2(qlin[1, 1] - qlin[0, 1], qlin[1, 0] - qlin[0, 0]))
    quarter = int(round(rot_deg / 90.0)) * 90
    tilt = rot_deg - quarter
    rot_txt = {0: "upright", 90: "turned 90 deg clockwise", -90: "turned 90 deg counter-clockwise",
               180: "upside down", -180: "upside down"}.get(quarter, "rotated %.0f deg" % quarter)
    if abs(tilt) >= 0.05:
        rot_txt += ", tilted %.1f deg" % tilt
    LOG.info("  chart found: %s%s, %.1f px per patch, orientation fit %.3f (next best %.3f)"
             % (rot_txt, ", mirrored" if mir else "", cell_px * k, r2, second))
    LOG.info("  paper %s | solid ink %s | dE %.1f" % (fmt_lab(paper_lab), fmt_lab(solid_lab), de_ps))
    if de_ps < 2.0:
        raise RisoError("solid ink and paper are almost identical in %s (dE %.1f)" % (path, de_ps),
                        hints=["is this the printed chart?", "check the scan isn't blank or overexposed"])
    if de_ps < 8.0:
        LOG.warn("%s: very low ink contrast (dE %.1f) - measurements will be noisy; scan in 48-bit "
                 "if you can and use several scans" % (base, de_ps))
    pde = np.linalg.norm(lin_to_lab(rgb[is_paper]) - paper_lab, axis=1)
    if (np.percentile(pde, 90) > 1.0 or pde.max() > 1.2) and not opts.get("flatfield"):
        LOG.warn("%s: the bare-paper patches vary by up to dE %.1f across the sheet (uneven lighting?). "
                 "Add --flatfield to correct it - recommended for camera captures." % (base, float(pde.max())))
    if np.median(clip_frac[is_paper]) > 0.5:
        LOG.warn("%s: the paper is clipped to pure white in the scan, so the lightest tones can't be "
                 "measured. Lower scanner brightness / turn auto exposure off." % base)
    if np.median(black_frac[is_solid]) > 0.5:
        LOG.warn("%s: the solid ink is clipped to pure black in the scan, so the darkest tones can't be "
                 "told apart. Turn off auto levels / auto exposure in the scanner software." % base)
    noisy = spread > max(0.08, 6 * float(np.median(spread)))
    if noisy.sum():
        LOG.detail("%d patches look non-uniform (dust, scratches or misalignment); robust averaging used"
                   % int(noisy.sum()))
    return {
        "name": base, "path": os.path.abspath(path), "layout_id": lay["id"],
        "precorrected": lay.get("precorrected_with") is not None,
        "rows": np.asarray(p["row"]), "cols": np.asarray(p["col"]), "values": np.asarray(p["value"]),
        "codes": codes, "kinds": kinds, "rgb": rgb, "spread": spread, "noisy": noisy,
        "paper": paper, "solid": solid,
        "orientation": {"rotation_deg": round(rot_deg, 2), "mirrored": bool(mir), "fit_r2": round(r2, 4),
                        "next_best_r2": round(second, 4)},
        "cell_px": cell_px * k, "dpi": r.dpi[0] if r.dpi else None,
        "_det_rgb8": det_rgb8, "_qdet": qdet, "_Hdet": Hdet,
    }


def _flatfield(rgb, lay, codes, base):
    p = lay["patches"]
    x = (np.asarray(p["col"]) + 0.5) / lay["cols"] - 0.5
    y = (np.asarray(p["row"]) + 0.5) / lay["rows"] - 0.5
    A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    sel = codes == 255
    if sel.sum() < 8:
        LOG.warn("%s: only %d paper patches - flat-field skipped" % (base, int(sel.sum())))
        return rgb
    gain = np.ones_like(rgb)
    for c in range(3):
        s = sel.copy()
        for _ in range(2):
            coef, *_ = np.linalg.lstsq(A[s], rgb[s, c], rcond=None)
            res = rgb[sel, c] - A[sel] @ coef
            mad = np.median(np.abs(res)) * 1.4826 + 1e-6
            s = sel.copy()
            s[sel] = np.abs(res) < 4 * mad
        fit = A @ coef
        gain[:, c] = fit / np.mean(fit[sel])
    LOG.info("  flat-field: corrected lighting variation of %.1f%%" % (100 * float(gain.max() - gain.min())))
    return rgb / gain


def _failure_overlay(det_rgb8, attempts, path):
    try:
        img = Image.fromarray(det_rgb8)
        d = ImageDraw.Draw(img)
        y = 6
        d = ImageDraw.Draw(img)
        for i, a in enumerate(attempts):
            txt = "paper guess %d: %s, frame %s" % (i + 1, fmt_lab(a["paper_lab"]),
                                                   "found (red outline)" if a["frame"] else "NOT found")
            draw_text(img, (8, y), txt, 16, (255, 0, 0))
            y += 22
            q = a.get("quad")
            if q is not None:
                d.line([tuple(map(float, pt)) for pt in q] + [tuple(map(float, q[0]))], fill=(255, 0, 0), width=3)
        img.save(path)
    except Exception as exc:
        LOG.detail("could not write failure overlay: %s" % exc)


def detection_overlay(scan, lay, path, rejected=None):
    img = Image.fromarray(scan["_det_rgb8"]).convert("RGB")
    d = ImageDraw.Draw(img)
    q = scan["_qdet"]
    pts = [tuple(map(float, pt)) for pt in q] + [tuple(map(float, q[0]))]
    d.line(pts, fill=(255, 0, 0), width=3)
    pp = lay["patches"]
    c0 = np.asarray(pp["col"], dtype=np.float64)[:, None]
    r0 = np.asarray(pp["row"], dtype=np.float64)[:, None]
    lo, hi = SAMPLE_INSET, 1.0 - SAMPLE_INSET
    corners = np.stack([np.concatenate([c0 + lo, c0 + hi, c0 + hi, c0 + lo], axis=1),
                        np.concatenate([r0 + lo, r0 + lo, r0 + hi, r0 + hi], axis=1)], axis=-1)
    px = apply_h(scan["_Hdet"], corners)
    bad = scan["noisy"].copy()
    if rejected is not None:
        bad |= rejected
    for i in range(len(px)):
        poly = [tuple(map(float, pt)) for pt in px[i]] + [tuple(map(float, px[i][0]))]
        d.line(poly, fill=(255, 40, 40) if bad[i] else (0, 200, 60), width=2 if bad[i] else 1)
    tl = q[0]
    d.ellipse([tl[0] - 9, tl[1] - 9, tl[0] + 9, tl[1] + 9], outline=(255, 0, 255), width=3)
    lx = tl[0] + 12 if tl[0] + 140 < img.width else tl[0] - 132
    ly = tl[1] + 4 if tl[1] + 26 < img.height else tl[1] - 24
    draw_text(img, (lx, ly), "chart top-left", 15, (255, 0, 255))
    o = scan["orientation"]
    head = "%s | rotated %.1f deg%s | orientation fit %.3f | green = measured area, red = outlier/non-uniform" % (
        scan["name"], o["rotation_deg"], " mirrored" if o["mirrored"] else "", o["fit_r2"])
    bar = Image.new("RGB", (img.width, 26), (0, 0, 0))
    draw_text(bar, (6, 4), head, 15, (255, 255, 255))
    out = Image.new("RGB", (img.width, img.height + 26), (0, 0, 0))
    out.paste(bar, (0, 0))
    out.paste(img, (0, 26))
    out.save(path)
    return path


# ----------------------------------------------------------------------------------------
# Fitting the response and building the correction
# ----------------------------------------------------------------------------------------
def _code_centers(codes, tv):
    uc, inv = np.unique(codes, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    center = np.empty(len(uc))
    for i in range(len(uc)):
        v = tv[inv == i]
        center[i] = np.median(v) if len(v) >= 3 else v.mean()
    return uc, center, cnt


def _fit_tone(codes, tv, sigma):
    uc, center, cnt = _code_centers(codes, tv)
    iso = -pava_increasing(-center, cnt)
    sm = nw_smooth(uc, iso, cnt, GRID, sigma)[:, 0]
    return sm, uc, center


def choose_sigma(uc, center, cnt):
    """Pick the smoothing width by leave-one-out cross-validation on the per-code tone values:
    as smooth as the measurement noise calls for, no smoother."""
    spacing = float(np.median(np.diff(uc))) if len(uc) > 1 else 1.0
    cands = np.array([0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]) * max(1.0, spacing)
    d = uc[:, None] - uc[None, :]
    scores = []
    for s in cands:
        K = np.exp(-0.5 * (d / s) ** 2) * cnt[None, :]
        np.fill_diagonal(K, 0.0)
        den = K.sum(axis=1)
        den[den <= 0] = 1e-300
        pred = (K @ center) / den
        scores.append(float(np.sum(cnt * (center - pred) ** 2) / np.sum(cnt)))
    scores = np.asarray(scores)
    i = int(np.argmin(scores))
    # prefer a smoother curve when it is statistically as good (within 2%)
    for j in range(len(cands) - 1, i, -1):
        if scores[j] <= scores[i] * 1.02:
            i = j
            break
    return float(cands[i]), scores


def build_model(scans, settings):
    metric = settings["metric"]
    auto = str(settings["smooth"]).lower() == "auto"
    sigma = 2.0 if auto else float(settings["smooth"])
    tvs, codes, rgb_abs, rgb_rel, sidx = [], [], [], [], []
    for i, s in enumerate(scans):
        tv = tone_values(s["rgb"], s["paper"], s["solid"], metric)
        tvs.append(tv)
        codes.append(np.asarray(s["codes"]))
        rgb_abs.append(s["rgb"])
        rgb_rel.append(s["rgb"] / np.maximum(s["paper"], 1e-6))
        sidx.append(np.full(len(tv), i))
    tv = np.concatenate(tvs)
    code = np.concatenate(codes).astype(np.float64)
    rgb_abs = np.concatenate(rgb_abs)
    rgb_rel = np.concatenate(rgb_rel)
    sidx = np.concatenate(sidx)
    N = len(tv)
    keep = np.ones(N, bool)
    for _ in range(3):
        sm, _, _ = _fit_tone(code[keep], tv[keep], sigma)
        res = tv - np.interp(code, GRID, sm)
        rk = res[keep]
        sig = 1.4826 * float(np.median(np.abs(rk - np.median(rk)))) + 1e-9
        thr = max(6.0 * sig, 0.05)
        new = np.abs(res) <= thr
        if new.sum() < 0.85 * N:
            thr = float(np.quantile(np.abs(res), 0.85))
            new = np.abs(res) <= thr
        if (new == keep).all():
            break
        keep = new
    if auto:
        uc0, c0, n0 = _code_centers(code[keep], tv[keep])
        sigma, _ = choose_sigma(uc0, c0, n0)
        LOG.detail("smoothing chosen by cross-validation: %.2f codes" % sigma)
    sm, ucodes, centers = _fit_tone(code[keep], tv[keep], sigma)
    res = tv - np.interp(code, GRID, sm)
    span = float(sm[0] - sm[-1])
    if span < 0.3:
        raise RisoError("the measured tones barely change from code 0 to 255 (range %.2f)" % span,
                        hints=["check the detection overlays", "was the chart printed with the ink actually on?",
                               "with very light inks try --metric sctv"])
    D = (sm - sm[-1]) / span
    D = np.minimum.accumulate(np.clip(D, 0.0, 1.0))
    D = D + 1e-7 * (255.0 - GRID) / 255.0
    D = (D - D[-1]) / (D[0] - D[-1])
    # ---- correction: image code u -> printer code with the target tone
    t_dense = target_tone(GRID, settings["target_parsed"])
    C = np.interp(t_dense, D[::-1], GRID[::-1])
    C[0], C[-1] = 0.0, 255.0
    if settings["ends"] == "soft":
        W = float(settings["soft_width"])
        hi0 = 255.0 - W
        c_hi = float(np.interp(hi0, GRID, C))
        c_lo = float(np.interp(W, GRID, C))
        sel = GRID > hi0
        C[sel] = c_hi + (255.0 - c_hi) * (GRID[sel] - hi0) / W
        sel = GRID < W
        C[sel] = c_lo * GRID[sel] / W
    C = np.maximum.accumulate(np.clip(C, 0.0, 255.0))
    C_int = C[::16]
    table8 = np.clip(np.round(C_int), 0, 255).astype(int)
    D_int = D[::16]
    t_int = target_tone(np.arange(256), settings["target_parsed"])
    pred = D_int[table8]
    err = pred - t_int
    # ---- colors along the printer code axis (for previews)
    uc, inv = np.unique(code[keep], return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    abs_mean = np.stack([np.bincount(inv, rgb_abs[keep][:, c]) / cnt for c in range(3)], axis=1)
    rel_mean = np.stack([np.bincount(inv, rgb_rel[keep][:, c]) / cnt for c in range(3)], axis=1)
    abs_tab = np.clip(nw_smooth(uc, abs_mean, cnt, GRID, sigma), 0.0, 1.5)
    rel_tab = np.clip(nw_smooth(uc, rel_mean, cnt, GRID, sigma), 0.0, 1.5)
    # ---- stats
    paper = np.mean([s["paper"] for s in scans], axis=0)
    solid = np.mean([s["solid"] for s in scans], axis=0)
    above = np.nonzero(D >= 0.01)[0]
    x_hi = float(GRID[above[-1]]) if len(above) else 0.0
    below = np.nonzero(D <= 0.99)[0]
    x_lo = float(GRID[below[0]]) if len(below) else 255.0
    groups = {}
    for i in np.nonzero(keep)[0]:
        groups.setdefault((int(sidx[i]), int(code[i])), []).append(tv[i])
    devs, dof = 0.0, 0
    for v in groups.values():
        if len(v) >= 2:
            a = np.asarray(v)
            devs += float(((a - a.mean()) ** 2).sum())
            dof += len(a) - 1
    rep_sd = math.sqrt(devs / dof) if dof else None
    rejected_list = []
    for i in np.nonzero(~keep)[0]:
        s = scans[sidx[i]]
        j = i - int(np.nonzero(sidx == sidx[i])[0][0])
        rejected_list.append({"scan": s["name"], "row": int(s["rows"][j]) + 1, "col": int(s["cols"][j]) + 1,
                              "code": int(code[i]), "tone": float(tv[i]), "fit": float(tv[i] - res[i])})
    accuracy = []
    for s_i, s in enumerate(scans):
        if not s.get("precorrected"):
            continue
        selm = (sidx == s_i) & keep
        kinds = np.asarray(s["kinds"])
        vals = np.asarray(s["values"])
        lv = kinds == "level"
        loc = np.nonzero(sidx == s_i)[0]
        e, ev = [], []
        for v in np.unique(vals[lv]):
            mm = lv & (vals == v) & selm[loc]
            if mm.any():
                e.append(float(np.mean(tv[loc][mm])) - float(target_tone(v, settings["target_parsed"])))
                ev.append(float(v))
        if len(e) >= 4:
            e, ev = np.asarray(e), np.asarray(ev)
            e_s = nw_smooth(ev, e, np.ones_like(e), ev, 4.0 * max(1.0, float(np.median(np.diff(ev)))))[:, 0]
            accuracy.append({"scan": s["name"], "mean_abs": float(np.mean(np.abs(e))),
                             "max_abs": float(np.max(np.abs(e))), "rms": float(np.sqrt(np.mean(e ** 2))),
                             "systematic_max": float(np.max(np.abs(e_s)))})
    stats = {
        "paper_lab": _r(lin_to_lab(paper), 2), "solid_lab": _r(lin_to_lab(solid), 2),
        "paper_hex": lin_to_hex(paper), "solid_hex": lin_to_hex(solid),
        "dE_paper_solid": round(float(np.linalg.norm(lin_to_lab(paper) - lin_to_lab(solid))), 2),
        "density_solid_rel": round(float(math.log10(max(luminance(paper), 1e-6) / max(luminance(solid), 1e-6))), 3),
        "density_solid_abs": round(float(-math.log10(max(luminance(solid), 1e-6))), 3),
        "density_paper_abs": round(float(-math.log10(max(luminance(paper), 1e-6))), 3),
        "highlight_dropout_from": round(x_hi, 2), "shadow_plug_below": round(x_lo, 2),
        "fit_rms": round(float(np.sqrt(np.mean(res[keep] ** 2))), 5),
        "smoothing_codes": round(float(sigma), 3),
        "replicate_sd": round(rep_sd, 5) if rep_sd is not None else None,
        "samples": int(N), "rejected": int((~keep).sum()), "rejected_list": rejected_list[:40],
        "distinct_printer_codes": int(len(np.unique(table8))),
        "quantization_max_err": round(float(np.max(np.abs(err))), 5),
        "quantization_rms_err": round(float(np.sqrt(np.mean(err ** 2))), 5),
        "pass2_accuracy": accuracy,
    }
    return {
        "D": D, "C": C, "table8": table8, "t_int": t_int, "pred_int": pred,
        "abs_tab": abs_tab, "rel_tab": rel_tab, "paper": paper, "solid": solid,
        "samples": {"code": code, "tv": tv, "keep": keep, "sidx": sidx, "res": res},
        "unique_codes": ucodes, "centers": centers, "stats": stats, "span": span,
    }


# ----------------------------------------------------------------------------------------
# Profile object (loaded from JSON)
# ----------------------------------------------------------------------------------------
class Profile:
    def __init__(self, d, path):
        self.d = d
        self.path = path
        self.name = d.get("name", "profile")
        self.id = d.get("id", "?")
        self.C = np.asarray(d["correction"]["dense"], dtype=np.float64)
        self.table8 = np.asarray(d["correction"]["table8"], dtype=int)
        self.D = np.asarray(d["response"]["dense"], dtype=np.float64)
        self.abs_tab = np.asarray(d["color"]["abs_lin"], dtype=np.float64)
        self.rel_tab = np.asarray(d["color"]["rel_lin"], dtype=np.float64)
        self.paper = np.asarray(d["paper_lin"], dtype=np.float64)
        self.settings = d.get("settings", {})
        if len(self.C) != GRID_N or self.abs_tab.shape != (GRID_N, 3) or len(self.table8) != 256:
            raise ValueError("table sizes don't match this version")

    def color_at(self, x, white=False):
        tab = self.rel_tab if white else self.abs_tab
        x = np.asarray(x, dtype=np.float64)
        return np.stack([np.interp(x, GRID, tab[:, c]) for c in range(3)], axis=-1)

    def correct(self, u):
        return np.interp(np.asarray(u, dtype=np.float64), GRID, self.C)

    def preview_table8(self, corrected=True, white=False):
        """(GRID_N, 3) uint8 sRGB lookup indexed by round(u * 16)."""
        x = self.C if corrected else GRID
        return _rgb8(self.color_at(x, white))


def load_profile(path):
    if path is None:
        raise RisoError("no profile given", hints=["pass --profile <folder or .risotone.json from 'profile'>"])
    p = path
    if os.path.isdir(p):
        found = sorted(glob.glob(os.path.join(p, "*.risotone.json")))
        if not found:
            raise RisoError("no *.risotone.json profile inside folder %s" % p,
                            hints=["pass the folder written by 'profile' (named <name>_risotone)"])
        if len(found) > 1:
            raise RisoError("several profiles in %s; pick one" % p, details="\n".join(found))
        p = found[0]
    if not os.path.exists(p):
        alt = p + "_risotone" if os.path.isdir(p + "_risotone") else None
        if alt:
            return load_profile(alt)
        raise RisoError("profile not found: %s" % path, hints=["current folder is %s" % os.getcwd()])
    try:
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as exc:
        raise RisoError("could not read profile %s (%s: %s)" % (p, type(exc).__name__, exc))
    if d.get("format") == "risotone-layout":
        raise RisoError("%s is a chart layout file, not a profile" % p,
                        hints=["profiles are written by the 'profile' command as <name>.risotone.json"])
    if d.get("format") != "risotone-profile":
        raise RisoError("%s is not a risotone profile" % p)
    try:
        return Profile(d, p)
    except Exception as exc:
        raise RisoError("profile %s is damaged or from an incompatible version (%s)" % (p, exc),
                        hints=["re-run 'profile' on the scans"])


# ----------------------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------------------
def write_cube_1d(path, values01, title, comments=()):
    n = len(values01)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        for c in comments:
            fh.write("# %s\n" % c)
        fh.write('TITLE "%s"\nLUT_1D_SIZE %d\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n' % (title, n))
        for v in values01:
            fh.write("%.6f %.6f %.6f\n" % (v, v, v))


def write_cube_3d(path, fn, size, title, comments=()):
    g = np.arange(size) / (size - 1.0)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([r.ravel(), gg.ravel(), b.ravel()], axis=-1)  # red changes fastest
    out = np.clip(fn(rgb), 0.0, 1.0)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        for c in comments:
            fh.write("# %s\n" % c)
        fh.write('TITLE "%s"\nLUT_3D_SIZE %d\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n' % (title, size))
        np.savetxt(fh, out, fmt="%.6f")


def write_amp(path, table8):
    """Photoshop Arbitrary Map: one 256-byte table, applied to the composite or the single gray channel."""
    with open(path, "wb") as fh:
        fh.write(bytes(int(v) for v in np.clip(table8, 0, 255)))


def acv_points(c_int, d_dense, max_points=14):
    """Pick <= max_points so Photoshop's spline curve prints (almost) the same tones as the exact
    correction. Error is judged in printed tone, so code differences inside dropout/plugged
    ranges (which print identically) cost nothing."""
    from scipy.interpolate import CubicSpline
    target = np.asarray(c_int, dtype=np.float64)
    xs = np.arange(256)
    pts = [0, 255]

    def tone(c):
        return np.interp(np.clip(np.round(c), 0, 255), GRID, d_dense)

    t_target = tone(target)

    def approx(points):
        p = sorted(points)
        y = target[p]
        if len(p) == 2:
            a = np.interp(xs, p, y)
        else:
            a = CubicSpline(p, y, bc_type="natural")(xs)
        return np.clip(a, 0, 255)

    def err(points):
        return np.abs(tone(approx(points)) - t_target)

    while len(pts) < max_points:
        e = err(pts)
        for p in pts:
            e[max(0, p - 2):p + 3] = -1
        cand = int(np.argmax(e))
        if e[cand] < 0.002:
            break
        pts.append(cand)
    for _ in range(3):  # local polish: nudge each interior point if it lowers the max error
        for i in range(len(pts)):
            if pts[i] in (0, 255):
                continue
            best = (float(np.max(err(pts))), pts[i])
            for d in (-4, -3, -2, -1, 1, 2, 3, 4):
                cand = pts[i] + d
                if cand <= 0 or cand >= 255 or any(abs(cand - q) < 3 for j, q in enumerate(pts) if j != i):
                    continue
                trial = list(pts)
                trial[i] = cand
                e = float(np.max(err(trial)))
                if e < best[0]:
                    best = (e, cand)
            pts[i] = best[1]
    p = sorted(pts)
    return [(int(x), int(round(target[x]))) for x in p], float(np.max(err(p)))


def write_acv(path, points):
    """Photoshop Curves preset, version 4: master curve + 4 null curves."""
    data = struct.pack(">HH", 4, 5)
    data += struct.pack(">H", len(points))
    for x, y in points:
        data += struct.pack(">HH", int(np.clip(y, 0, 255)), int(np.clip(x, 0, 255)))
    for _ in range(4):
        data += struct.pack(">HHHHH", 2, 0, 0, 255, 255)
    with open(path, "wb") as fh:
        fh.write(data)


def _proof_fn(prof_like, corrected, white):
    """RGB (encoded 0..1) -> preview color (encoded 0..1). Input gray = luminance of the pixel."""
    C, abs_tab, rel_tab = prof_like

    def fn(rgb):
        y = luminance(srgb_decode(rgb))
        u = srgb_encode(y) * 255.0
        x = np.interp(u, GRID, C) if corrected else u
        tab = rel_tab if white else abs_tab
        lin = np.stack([np.interp(x, GRID, tab[:, c]) for c in range(3)], axis=-1)
        return srgb_encode(lin)
    return fn


def write_profile_outputs(model, scans, lay_info, settings, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, name)
    files = []
    C, D, table8 = model["C"], model["D"], model["table8"]
    pid = _short_hash(_r(model["samples"]["tv"], 4), settings["metric"], settings["target"],
                      settings["ends"], settings["smooth"])
    prof = {
        "format": "risotone-profile", "version": 1, "id": pid, "name": name, "created": _now(),
        "tool": "%s %s" % (PROG, VERSION),
        "settings": {k: v for k, v in settings.items() if k != "target_parsed"},
        "layouts": lay_info,
        "paper_lin": _r(model["paper"]), "solid_lin": _r(model["solid"]),
        "grid": {"start": 0, "stop": 255, "n": GRID_N},
        "response": {"dense": _r(D), "note": "tone value (0 paper .. 1 solid) printed by printer code GRID"},
        "correction": {"dense": _r(C, 4), "table8": [int(v) for v in table8],
                       "note": "printer code to send for image code GRID"},
        "color": {"abs_lin": _r(model["abs_tab"]), "rel_lin": _r(model["rel_tab"]),
                  "note": "linear sRGB print color produced by printer code GRID (abs = as scanned, rel = paper->white)"},
        "stats": model["stats"],
        "scans": [],
    }
    for s in scans:
        prof["scans"].append({
            "name": s["name"], "path": s.get("path"), "layout_id": s.get("layout_id"),
            "precorrected": bool(s.get("precorrected")), "orientation": s.get("orientation"),
            "merged_from": s.get("merged_from"),
            "paper_lin": _r(s["paper"]), "solid_lin": _r(s["solid"]),
            "samples": {"row": [int(v) for v in s["rows"]], "col": [int(v) for v in s["cols"]],
                        "value": [int(v) for v in s["values"]], "code": [int(v) for v in s["codes"]],
                        "kind": [str(v) for v in s["kinds"]], "rgb_lin": _r(s["rgb"])},
        })
    jpath = stem + ".risotone.json"
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(prof, fh, separators=(",", ":"))
    files.append((jpath, "the profile (used by preview / apply / chart --precorrect)"))
    # correction files
    p = stem + "_correction.amp"
    write_amp(p, table8)
    files.append((p, "Photoshop Curves, exact 256 values (Curves > Load Preset; works in Grayscale and RGB)"))
    pts, acv_err = acv_points(C[::16], D)
    p = stem + "_correction.acv"
    write_acv(p, pts)
    files.append((p, "Photoshop Curves, %d points, approximate (max %.1f%% tone off the exact curve)"
                  % (len(pts), 100 * acv_err)))
    if acv_err > 0.03:
        LOG.warn("the 14-point Photoshop .acv curve is up to %.1f%% tone off the exact correction; "
                 "use the .amp file or the 'apply' command for exact results" % (100 * acv_err))
    title = "risotone %s correction" % name
    cm = ["risotone %s - correction for profile '%s' (%s)" % (VERSION, name, pid),
          "image gray value in -> value to send to the riso out"]
    p = stem + "_correction_1D.cube"
    n1 = 4096
    u1 = np.linspace(0, 255, n1)
    write_cube_1d(p, np.interp(u1, GRID, C) / 255.0, title, cm)
    files.append((p, "correction as 1D LUT (apps that take 1D .cube)"))
    p = stem + "_correction_3D.cube"
    write_cube_3d(p, lambda rgb: np.interp(rgb * 255.0, GRID, C) / 255.0, 65, title, cm)
    files.append((p, "correction as 3D LUT (Photoshop Color Lookup on an RGB document)"))
    # proof (preview) cubes
    size = int(settings.get("cube_size", 33))
    tabs = (C, model["abs_tab"], model["rel_tab"])
    for suffix, corrected, white, what in (
            ("proof", True, False, "PRINT PREVIEW: how your image prints with the correction (paper color shown)"),
            ("proof_white", True, True, "same, paper shown as white (stack several inks with Multiply)"),
            ("proof_raw", False, False, "preview of a file that is ALREADY corrected (or of an uncorrected print)")):
        p = stem + "_%s.cube" % suffix
        write_cube_3d(p, _proof_fn(tabs, corrected, white), size, "risotone %s %s" % (name, suffix),
                      ["risotone %s - screen proof for profile '%s' (%s)" % (VERSION, name, pid), what])
        files.append((p, what))
    # CSV tables
    p = stem + "_curve.csv"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("image_code,target_tone_pct,printer_code,printer_code_8bit,predicted_tone_pct,"
                 "uncorrected_tone_pct,preview_srgb,uncorrected_preview_srgb\n")
        prev = model["abs_tab"]
        for u in range(256):
            x = C[u * 16]
            col = np.array([np.interp(x, GRID, prev[:, c]) for c in range(3)])
            fh.write("%d,%.2f,%.3f,%d,%.2f,%.2f,%s,%s\n" % (
                u, 100 * model["t_int"][u], x, table8[u], 100 * model["pred_int"][u], 100 * D[u * 16],
                lin_to_hex(col), lin_to_hex(prev[u * 16])))
    files.append((p, "the curve as a table (image code -> printer code, tones, preview colors)"))
    p = stem + "_samples.csv"
    sm = model["samples"]
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("scan,row,col,value,printer_code,kind,L,a,b,tone_pct,fit_pct,used\n")
        off = 0
        for s in scans:
            lab = lin_to_lab(s["rgb"])
            for j in range(len(s["codes"])):
                i = off + j
                fh.write("%s,%d,%d,%d,%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n" % (
                    s["name"], s["rows"][j] + 1, s["cols"][j] + 1, s["values"][j], s["codes"][j], s["kinds"][j],
                    lab[j, 0], lab[j, 1], lab[j, 2], 100 * sm["tv"][i], 100 * (sm["tv"][i] - sm["res"][i]),
                    int(sm["keep"][i])))
            off += len(s["codes"])
    files.append((p, "every measured patch"))
    return files, pid


def report_text(model, scans, settings, name, pid, files):
    st = model["stats"]
    L = []
    L.append("risotone %s - profile '%s' (%s)  %s" % (VERSION, name, pid, _now()))
    L.append("")
    L.append("settings   metric %s | target %s | ends %s%s | smoothing %s (%.2f codes)" % (
        settings["metric"], settings["target"], settings["ends"],
        " (%g codes)" % settings["soft_width"] if settings["ends"] == "soft" else "", settings["smooth"],
        st["smoothing_codes"]))
    L.append("scans      %d (%s)" % (len(scans), ", ".join(s["name"] for s in scans)))
    L.append("")
    L.append("PRINT")
    L.append("  paper           %s  %s  (density %.2f)" % (st["paper_hex"], fmt_lab(st["paper_lab"]),
                                                           st["density_paper_abs"]))
    L.append("  solid ink       %s  %s  (density %.2f, %.2f above paper)" % (
        st["solid_hex"], fmt_lab(st["solid_lab"]), st["density_solid_abs"], st["density_solid_rel"]))
    L.append("  ink contrast    dE %.1f" % st["dE_paper_solid"])
    L.append("  highlights      codes above %.0f print within 1%% of bare paper (dropout)"
             % st["highlight_dropout_from"])
    L.append("  shadows         codes below %.0f print within 1%% of solid (plugging)" % st["shadow_plug_below"])
    L.append("  usable range    printer codes %.0f - %.0f" % (st["shadow_plug_below"], st["highlight_dropout_from"]))
    if st["replicate_sd"] is not None:
        L.append("  uniformity      identical patches differ by +/-%.1f%% tone (1 sd) across the sheet"
                 % (100 * st["replicate_sd"]))
    L.append("  fit             rms %.2f%% tone; %d of %d patches rejected as outliers"
             % (100 * st["fit_rms"], st["rejected"], st["samples"]))
    L.append("")
    L.append("CORRECTION")
    L.append("  uses %d distinct printer codes for the 256 image codes" % st["distinct_printer_codes"])
    L.append("  predicted error after correction (8-bit rounding%s): max %.2f%% tone, rms %.2f%%" % (
        " + soft ends" if settings["ends"] == "soft" else "",
        100 * st["quantization_max_err"], 100 * st["quantization_rms_err"]))
    for a in st["pass2_accuracy"]:
        L.append("  pass-2 check (%s): corrected print vs target: systematic error max %.1f%% (smoothed);"
                 " single values mean %.1f%%, max %.1f%% (includes sheet unevenness)" % (
                     a["scan"], 100 * a["systematic_max"], 100 * a["mean_abs"], 100 * a["max_abs"]))
    if st["rejected_list"]:
        L.append("")
        L.append("OUTLIER PATCHES (ignored)")
        for r in st["rejected_list"][:15]:
            L.append("  %s row %d col %d code %d: measured %.1f%% vs curve %.1f%%" % (
                r["scan"], r["row"], r["col"], r["code"], 100 * r["tone"], 100 * r["fit"]))
    if LOG.warnings:
        L.append("")
        L.append("WARNINGS")
        L += ["  - " + w for w in LOG.warnings]
    L.append("")
    L.append("FILES")
    for f, what in files:
        L.append("  %-34s %s" % (os.path.basename(f), what))
    L.append("")
    L.append("Key numbers: image code -> printer code")
    t8 = model["table8"]
    for row in range(0, 256, 16):
        L.append("  " + "  ".join("%3d>%3d" % (u, t8[u]) for u in range(row, min(256, row + 16), 2)))
    return "\n".join(L) + "\n"


def report_png(model, scans, settings, name, pid, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        LOG.warn("matplotlib is not installed, so the report graph was skipped "
                 "(python3 -m pip install matplotlib). The text report has the numbers.")
        return None
    st = model["stats"]
    D, C = model["D"], model["C"]
    sm = model["samples"]
    ink = srgb_encode(model["solid"])
    if luminance(model["solid"]) > 0.55:
        ink = np.array([0.3, 0.3, 0.3])
    fig = plt.figure(figsize=(13, 10), dpi=110)
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 1.05, 2.3], hspace=0.42, wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    k = sm["keep"]
    ax.axvspan(st["highlight_dropout_from"], 255, color="#f2c0c0", alpha=0.5, lw=0, label="prints as paper")
    ax.axvspan(0, st["shadow_plug_below"], color="#c0c8f2", alpha=0.5, lw=0, label="prints as solid")
    ax.scatter(sm["code"][k], 100 * sm["tv"][k], s=5, color="#555555", alpha=0.55, lw=0, label="patches")
    if (~k).any():
        ax.scatter(sm["code"][~k], 100 * sm["tv"][~k], s=18, marker="x", color="red", label="outliers")
    ax.plot(GRID, 100 * D, color=ink, lw=2.2, label="printer response (fit)")
    ax.plot(GRID, 100 * target_tone(GRID, settings["target_parsed"]), "--", color="#2a7", lw=1.4,
            label="target (%s)" % settings["target"])
    ax.set_xlim(0, 255)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("printer code (0 = solid ink, 255 = no ink)")
    ax.set_ylabel("tone value %% (%s)" % settings["metric"])
    ax.set_title("What the riso prints for each code")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot([0, 255], [0, 255], ":", color="#999999", lw=1)
    ax.plot(GRID, C, color="#d03060", lw=2.2, label="correction")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel("image gray value (what you design)")
    ax.set_ylabel("value sent to the riso")
    ax.set_title("Correction curve (%s ends, %d distinct codes)" % (settings["ends"], st["distinct_printer_codes"]))
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1, :])
    u = np.linspace(0, 255, 1024)
    target_row = np.repeat(srgb_encode(srgb_decode(u / 255.0))[:, None], 3, axis=1)
    raw_row = srgb_encode(np.stack([np.interp(u, GRID, model["abs_tab"][:, c]) for c in range(3)], -1))
    cx = np.interp(u, GRID, C)
    cor_row = srgb_encode(np.stack([np.interp(cx, GRID, model["abs_tab"][:, c]) for c in range(3)], -1))
    strip = np.stack([target_row, raw_row, cor_row], axis=0)
    ax.imshow(np.repeat(strip, 40, axis=0), aspect="auto", extent=[0, 255, 3, 0], interpolation="nearest")
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["on screen", "printed, no correction", "printed, corrected"], fontsize=9)
    ax.set_xticks(range(0, 256, 32))
    ax.set_title("Tone preview (tones only, no halftone)")
    ax = fig.add_subplot(gs[2, 0])
    uu = np.arange(256)
    ax.plot(uu, 100 * model["t_int"], "--", color="#2a7", lw=1.4, label="target")
    ax.plot(uu, 100 * model["pred_int"], color="#d03060", lw=1.6, label="predicted print after correction")
    ax.plot(uu, 100 * D[::16], color="#888888", lw=1.2, label="print without correction")
    ax2 = ax.twinx()
    ax2.bar(uu, 100 * (model["pred_int"] - model["t_int"]), color="#d03060", alpha=0.35, width=1.0)
    ax2.set_ylim(-6, 6)
    ax2.set_ylabel("error % (bars)")
    ax.set_xlim(0, 255)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("image gray value")
    ax.set_ylabel("tone value %")
    ax.set_title("After correction (incl. 8-bit rounding)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    lines = [
        "profile   %s  (%s)" % (name, pid),
        "scans     %d" % len(scans),
        "metric    %s   target %s" % (settings["metric"], settings["target"]),
        "",
        "paper     %s  %s" % (st["paper_hex"], fmt_lab(st["paper_lab"])),
        "solid     %s  %s" % (st["solid_hex"], fmt_lab(st["solid_lab"])),
        "contrast  dE %.1f   density %.2f" % (st["dE_paper_solid"], st["density_solid_rel"]),
        "dropout   codes > %.0f print as paper" % st["highlight_dropout_from"],
        "plugging  codes < %.0f print as solid" % st["shadow_plug_below"],
        "fit rms   %.2f%%   outliers %d/%d" % (100 * st["fit_rms"], st["rejected"], st["samples"]),
    ]
    if st["replicate_sd"] is not None:
        lines.append("sheet uniformity  +/-%.1f%% tone" % (100 * st["replicate_sd"]))
    lines.append("predicted error after correction  max %.1f%%" % (100 * st["quantization_max_err"]))
    for a in st["pass2_accuracy"]:
        lines.append("pass-2 %s: within %.1f%% (systematic)" % (a["scan"][:14], 100 * a["systematic_max"]))
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5,
            transform=ax.transAxes)
    fig.suptitle("risotone - %s" % name, fontsize=14, x=0.07, ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------
def cmd_chart(a):
    prof = load_profile(a.precorrect) if a.precorrect else None
    lay = make_layout(a.levels, a.replicates, a.order, a.seed, a.anchors)
    if prof is not None:
        lay = precorrect_layout(lay, prof)
    if a.dpi < 150 or a.dpi > 2400:
        raise RisoError("--dpi %d is outside 150-2400" % a.dpi,
                        hints=["riso masters are made at 600 dpi (some models 300x600); 600 is the useful maximum"])
    out = a.out
    ext = os.path.splitext(out)[1].lower()
    if ext not in (".tif", ".tiff", ".png"):
        raise RisoError("chart must be saved as .tif or .png (got %r)" % out,
                        hints=["JPEG would blur the exact tone values"])
    img, cell_mm, (win, hin) = render_chart(lay, a.paper, a.dpi, a.margin, a.landscape,
                                            not a.no_labels, not a.no_ramp, prof)
    lay["created"] = _now()
    lay["tool"] = "%s %s" % (PROG, VERSION)
    lay["chart_file"] = os.path.basename(out)
    save_image(img, out, dpi=(a.dpi, a.dpi))
    lpath = os.path.splitext(out)[0] + ".layout.json"
    with open(lpath, "w", encoding="utf-8") as fh:
        json.dump(lay, fh, indent=1)
    LOG.info("wrote %s  (%.2f x %.2f in, %d dpi, %dx%d patches of %.1f mm, %d tone levels x %d, %d paper/solid anchors)"
             % (out, win, hin, a.dpi, lay["cols"], lay["rows"], cell_mm, lay["levels"], lay["replicates"], lay["anchors"]))
    LOG.info("wrote %s  (keep it next to the chart; 'profile' needs it to know which patch is which)" % lpath)
    if prof is not None:
        LOG.info("pass-2 chart: level patches were run through '%s'. Print + scan it, then:" % prof.name)
        LOG.info("  python3 risotone.py profile <scan> --layout %s" % lpath)
    else:
        LOG.info("next: print it on the riso at 100%, one ink, then scan and run: python3 risotone.py profile <scan(s)>")


def _find_layout(a):
    if a.layout:
        lp = a.layout
        if not os.path.exists(lp):
            raise RisoError("layout file not found: %s" % lp, hints=["current folder is %s" % os.getcwd()])
        try:
            with open(lp, "r", encoding="utf-8") as fh:
                lay = json.load(fh)
        except Exception as exc:
            raise RisoError("could not read layout %s (%s)" % (lp, exc))
        return validate_layout(lay, lp), lp
    dirs = []
    for s in a.scans:
        d = os.path.dirname(os.path.abspath(s))
        if d not in dirs:
            dirs.append(d)
    if os.getcwd() not in dirs:
        dirs.append(os.getcwd())
    found = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.layout.json"))):
            if os.path.abspath(f) not in found:
                found.append(os.path.abspath(f))
    if len(found) == 1:
        LOG.info("using chart layout %s" % found[0])
        with open(found[0], "r", encoding="utf-8") as fh:
            return validate_layout(json.load(fh), found[0]), found[0]
    if len(found) > 1:
        desc = []
        for f in found:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    l = json.load(fh)
                pw = l.get("precorrected_with")
                desc.append("%s   (id %s, %s levels x %s, %s, seed %s%s, made %s)" % (
                    f, l.get("id"), l.get("levels"), l.get("replicates"), l.get("order"), l.get("seed"),
                    ", PASS 2 via '%s'" % pw.get("name") if pw else "", l.get("created", "?")))
            except Exception:
                desc.append(f + "   (unreadable)")
        raise RisoError("found several chart layout files; say which chart you printed with --layout",
                        details="\n".join(desc),
                        hints=["the id is printed in the top line of each chart",
                               "e.g.  python3 risotone.py profile scan.tif --layout %s" % os.path.relpath(found[0])])
    lay = make_layout(a.levels, a.replicates, a.order, a.seed, a.anchors)
    if a.precorrected_with:
        lay = precorrect_layout(lay, load_profile(a.precorrected_with))
    LOG.warn("no *.layout.json found; assuming a chart made with: --levels %d --replicates %d --order %s "
             "--seed %d --anchors %d%s (the top line printed on the chart shows the real settings)"
             % (a.levels, a.replicates, a.order, a.seed, a.anchors,
                " --precorrect %s" % a.precorrected_with if a.precorrected_with else ""))
    return lay, None


def _scans_from_profile(prof):
    out = []
    for s in prof.d.get("scans", []):
        sm = s["samples"]
        out.append({
            "name": s["name"] + "@" + prof.name, "path": s.get("path"), "layout_id": s.get("layout_id"),
            "precorrected": bool(s.get("precorrected")), "orientation": s.get("orientation"),
            "merged_from": prof.name,
            "rows": np.asarray(sm["row"]), "cols": np.asarray(sm["col"]), "values": np.asarray(sm["value"]),
            "codes": np.asarray(sm["code"]), "kinds": np.asarray(sm["kind"]),
            "rgb": np.asarray(sm["rgb_lin"], dtype=np.float64),
            "paper": np.asarray(s["paper_lin"]), "solid": np.asarray(s["solid_lin"]),
        })
    return out


def cmd_profile(a):
    if a.metric not in METRICS:
        raise RisoError("unknown --metric %r" % a.metric, hints=["choose from: " + ", ".join(METRICS)])
    target_parsed = parse_target(a.target)
    if str(a.smooth).lower() != "auto":
        try:
            sm_v = float(a.smooth)
        except ValueError:
            raise RisoError("--smooth must be 'auto' or a number of code values (e.g. 2)")
        if sm_v < 0.3 or sm_v > 20:
            raise RisoError("--smooth must be between 0.3 and 20 codes")
        a.smooth = sm_v
    if a.ends == "soft" and not 1 <= a.soft_width <= 64:
        raise RisoError("--soft-width must be between 1 and 64 codes")
    if a.cube_size < 2 or a.cube_size > 129:
        raise RisoError("--cube-size must be between 2 and 129")
    enc = parse_scan_encoding(a.scan_encoding)
    lay, lpath = _find_layout(a)
    name = _sanitize(a.name or os.path.splitext(os.path.basename(a.scans[0]))[0])
    outdir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.scans[0])), name + "_risotone")
    os.makedirs(outdir, exist_ok=True)
    opts = {"scan_encoding": enc, "work_size": a.work_size, "flatfield": a.flatfield,
            "outdir": outdir, "name": name}
    scans = []
    for sp in a.scans:
        scans.append(read_scan(sp, lay, opts))
    merged = []
    merge_paths = list(a.merge or [])
    pw = lay.get("precorrected_with")
    if pw and not a.no_merge and not merge_paths:
        cand = pw.get("path")
        if cand and os.path.exists(cand):
            merge_paths.append(cand)
            LOG.info("pass-2 chart: also using the measurements from '%s' (use --no-merge to skip)" % pw.get("name"))
        else:
            LOG.info("pass-2 chart: earlier profile %s not found, using only these scans" % cand)
    for mp in merge_paths:
        mprof = load_profile(mp)
        ms = _scans_from_profile(mprof)
        LOG.info("merging %d scan(s) from profile '%s'" % (len(ms), mprof.name))
        merged += ms
    settings = {"metric": a.metric, "target": a.target, "target_parsed": target_parsed, "ends": a.ends,
                "soft_width": a.soft_width, "smooth": a.smooth, "flatfield": bool(a.flatfield),
                "scan_encoding": a.scan_encoding, "cube_size": a.cube_size}
    all_scans = scans + merged
    LOG.info("fitting the tone response (%d scans, %d patches, metric %s)"
             % (len(all_scans), sum(len(s["codes"]) for s in all_scans), a.metric))
    if a.metric == "l":
        de = np.linalg.norm(lin_to_lab(scans[0]["paper"]) - lin_to_lab(scans[0]["solid"]))
        dl = abs(lin_to_lab(scans[0]["paper"])[0] - lin_to_lab(scans[0]["solid"])[0])
        if dl < 0.4 * de:
            LOG.warn("this ink changes color much more than lightness (dL %.1f vs dE %.1f); "
                     "--metric sctv will be more accurate" % (dl, de))
    model = build_model(all_scans, settings)
    lay_info = {"id": lay["id"], "path": lpath, "rows": lay["rows"], "cols": lay["cols"],
                "levels": lay.get("levels"), "replicates": lay.get("replicates"),
                "precorrected_with": lay.get("precorrected_with")}
    files, pid = write_profile_outputs(model, all_scans, lay_info, settings, outdir, name)
    rej_by_scan = {}
    off = 0
    for s in all_scans:
        n = len(s["codes"])
        rej_by_scan[id(s)] = ~model["samples"]["keep"][off:off + n]
        off += n
    for s in scans:
        p = os.path.join(outdir, "%s_detect_%s.png" % (name, _sanitize(s["name"])))
        detection_overlay(s, lay, p, rej_by_scan[id(s)])
        files.append((p, "where the patches were measured on scan %s" % s["name"]))
    rp = os.path.join(outdir, name + "_report.png")
    if report_png(model, all_scans, settings, name, pid, rp):
        files.append((rp, "graphs: response, correction, tone preview, accuracy"))
    tp = os.path.join(outdir, name + "_report.txt")
    files.append((tp, "this summary"))
    txt = report_text(model, all_scans, settings, name, pid, files)
    with open(tp, "w", encoding="utf-8") as fh:
        fh.write(txt)
    st = model["stats"]
    LOG.info("done: profile '%s' (%s) -> %s" % (name, pid, outdir))
    LOG.info("  dropout above code %.0f, plugging below %.0f, %d distinct printer codes, fit rms %.1f%%"
             % (st["highlight_dropout_from"], st["shadow_plug_below"], st["distinct_printer_codes"],
                100 * st["fit_rms"]))
    for acc in st["pass2_accuracy"]:
        LOG.info("  pass-2 check %s: corrected print is within %.1f%% of target (systematic); single "
                 "patches mean %.1f%% off" % (acc["scan"], 100 * acc["systematic_max"], 100 * acc["mean_abs"]))
    if a.quiet:
        return
    print(txt.split("FILES")[0].rstrip())
    print("\nFILES in %s" % outdir)
    for f, what in files:
        print("  %-34s %s" % (os.path.basename(f), what))


def _gray_input(path):
    r = load_raster(path, "image")
    return r, gray_codes(r)


def _compare_image(panels, captions, maxside=1300):
    ims = []
    for arr in panels:
        im = Image.fromarray(arr)
        s = maxside / float(max(im.size))
        if s < 1:
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        ims.append(im)
    w, h = ims[0].size
    horizontal = w / float(h) <= 1.6
    cap = 34
    pad = 12
    if horizontal:
        W = len(ims) * w + (len(ims) + 1) * pad
        H = h + cap + 2 * pad
    else:
        W = w + 2 * pad
        H = len(ims) * (h + cap) + (len(ims) + 1) * pad
    out = Image.new("RGB", (W, H), (200, 200, 200))
    for i, (im, c) in enumerate(zip(ims, captions)):
        x = pad + i * (w + pad) if horizontal else pad
        y = pad if horizontal else pad + i * (h + cap + pad)
        draw_text(out, (x, y + 4), c, 20, (20, 20, 20))
        out.paste(im.convert("RGB"), (x, y + cap))
    return np.asarray(out)


def cmd_preview(a):
    layers = []
    if a.layer:
        if a.image or a.profile:
            raise RisoError("use either IMAGE --profile P, or one or more --layer IMAGE PROFILE, not both")
        for img_path, prof_path in a.layer:
            layers.append((img_path, load_profile(prof_path)))
    else:
        if not a.image:
            raise RisoError("no image given", hints=["python3 risotone.py preview photo.tif --profile pink_risotone"])
        layers.append((a.image, load_profile(a.profile)))
    corrected = not a.raw
    inputs = []
    for img_path, prof in layers:
        r, u = _gray_input(img_path)
        inputs.append((img_path, prof, r, u))
        LOG.info("%s: %s, %s, profile '%s'" % (os.path.basename(img_path), r.desc, r.size_txt, prof.name))
    shp = inputs[0][3].shape
    for img_path, _, _, u in inputs[1:]:
        if u.shape != shp:
            raise RisoError("layer images must be the same size: %s is %dx%d but %s is %dx%d" % (
                os.path.basename(inputs[0][0]), shp[1], shp[0], os.path.basename(img_path), u.shape[1], u.shape[0]),
                hints=["export every separation from the same document/canvas"])
    first_path, first_prof, first_r, first_u = inputs[0]
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(first_path)),
                                os.path.splitext(os.path.basename(first_path))[0] + "_riso_preview.png")

    def render(corr):
        if len(inputs) == 1:
            tab = first_prof.preview_table8(corrected=corr, white=a.white)
            return tab[np.clip(np.rint(first_u * 16), 0, GRID_N - 1).astype(np.int32)]
        paper = np.ones(3) if a.white else first_prof.paper
        res = np.empty(shp + (3,), np.uint8)
        for y0 in range(0, shp[0], 256):
            acc = np.ones((min(256, shp[0] - y0), shp[1], 3), np.float64) * paper
            for _, prof, _, u in inputs:
                x = u[y0:y0 + 256].astype(np.float64)
                if corr:
                    x = prof.correct(x)
                acc *= prof.color_at(x, white=True)
            res[y0:y0 + 256] = _rgb8(acc)
        return res

    img = render(corrected)
    save_image(img, out, dpi=first_r.dpi, srgb_tag=True)
    LOG.info("wrote %s (%s)" % (out, "print preview" if corrected else "preview of an already-corrected file"))
    if a.compare:
        cpath = os.path.splitext(out)[0] + "_compare.png"
        if corrected:
            panels = [render(False), img]
            caps = ["riso, NO correction", "riso, WITH correction"]
            if len(inputs) == 1:
                panels.insert(0, np.repeat(np.clip(np.rint(first_u), 0, 255).astype(np.uint8)[..., None], 3, axis=2))
                caps.insert(0, "your file (on screen)")
        else:
            panels = [np.repeat(np.clip(np.rint(first_u), 0, 255).astype(np.uint8)[..., None], 3, axis=2), img]
            caps = ["file values (already corrected)", "riso print"]
        save_image(_compare_image(panels, caps), cpath, srgb_tag=True)
        LOG.info("wrote %s" % cpath)


def cmd_apply(a):
    prof = load_profile(a.profile)
    r = load_raster(a.image, "image")
    u = gray_codes(r)
    if r.channels == 1 and r.data.dtype == np.uint8:
        out_codes = prof.table8[r.data].astype(np.float64)
    else:
        out_codes = prof.correct(u)
    if a.bits == 8:
        arr = np.clip(np.rint(out_codes), 0, 255).astype(np.uint8)
    else:
        arr = np.clip(np.rint(out_codes / 255.0 * 65535.0), 0, 65535).astype(np.uint16)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.image)),
                                os.path.splitext(os.path.basename(a.image))[0] + "_riso.tif")
    if os.path.abspath(out) == os.path.abspath(a.image):
        raise RisoError("refusing to overwrite the input image; choose another -o")
    if os.path.splitext(out)[1].lower() in (".jpg", ".jpeg"):
        LOG.warn("JPEG output changes tone values slightly; .tif or .png is better for printing")
        if a.bits == 16:
            raise RisoError("16-bit output needs .tif or .png")
    save_image(arr, out, dpi=r.dpi)
    a8 = np.clip(np.rint(out_codes), 0, 255)
    LOG.info("wrote %s (%d-bit gray, %s) with profile '%s'" % (out, a.bits, r.size_txt, prof.name))
    LOG.info("  %.1f%% of pixels are bare paper (255), %.1f%% solid ink (0). Print it with the same "
             "settings as the chart, no color management." % (100 * float((a8 >= 255).mean()), 100 * float((a8 <= 0).mean())))


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def build_parser():
    fmt = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(prog="risotone.py", formatter_class=fmt,
                                description=__doc__.split("Requirements:")[0].strip())
    p.add_argument("--version", action="version", version="%s %s" % (PROG, VERSION))
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="more detail")
    common.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")
    common.add_argument("--traceback", action="store_true", help="show the Python traceback on errors")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    c = sub.add_parser("chart", parents=[common], formatter_class=fmt, help="make the calibration chart",
                       description="Make the calibration chart (+ a .layout.json the scan reader needs).\n\n"
                                   "Default: all 256 tone values, each printed twice at shuffled positions, plus\n"
                                   "paper/solid anchor patches (23x23 = 529 patches), 600 dpi Letter.")
    c.add_argument("-o", "--out", default="risotone_chart.tif", help="output .tif or .png (default %(default)s)")
    c.add_argument("--paper", default="letter",
                   help="letter, legal, tabloid, ledger, a5, a4, a3, b5, b4, or e.g. 8.5x11in / 210x297mm (default %(default)s)")
    c.add_argument("--landscape", action="store_true", help="landscape page")
    c.add_argument("--dpi", type=int, default=600, help="pixels per inch of the chart file (default %(default)s)")
    c.add_argument("--levels", type=int, default=256, help="tone levels, 2-256, evenly spaced (default %(default)s = every value)")
    c.add_argument("--replicates", type=int, default=2, help="copies of every level (default %(default)s)")
    c.add_argument("--order", choices=("shuffled", "ordered"), default="shuffled",
                   help="patch placement; shuffled averages out uneven inking (default %(default)s)")
    c.add_argument("--seed", type=int, default=1, help="shuffle seed (default %(default)s)")
    c.add_argument("--anchors", type=int, default=8, help="minimum extra paper/solid patches (default %(default)s)")
    c.add_argument("--margin", type=float, default=0.4, help="page margin in inches (default %(default)s)")
    c.add_argument("--no-labels", action="store_true", help="don't print the value in each patch")
    c.add_argument("--no-ramp", action="store_true", help="don't print the continuous ramp")
    c.add_argument("--precorrect", metavar="PROFILE", help="pass-2 chart: run the patches through this profile's correction")

    pr = sub.add_parser("profile", parents=[common], formatter_class=fmt, help="read scans -> LUTs + preview + report",
                        description="Read scan(s) of the printed chart and build the correction and preview LUTs.\n"
                                    "More scans (several sheets from the run) = more accurate.")
    pr.add_argument("scans", nargs="+", help="scan(s) of the printed chart")
    pr.add_argument("--layout", help="the chart's .layout.json (default: auto-find next to the scans / current folder)")
    pr.add_argument("--name", help="profile name (default: first scan's file name)")
    pr.add_argument("-o", "--out", help="output folder (default: <name>_risotone next to the first scan)")
    pr.add_argument("--metric", default="sctv", choices=METRICS,
                    help="tone measure: sctv = ISO 20654 spot color tone value, any ink color (default); "
                         "de = CIE dE76 from paper; l = L* lightness; density = channel density")
    pr.add_argument("--target", default="srgb",
                    help="what 'correct' means: srgb = print matches on-screen lightness (default); "
                         "linear = equal tone steps per value; gamma:2.2 etc.")
    pr.add_argument("--ends", choices=("exact", "soft"), default="exact",
                    help="exact = every tone as measured (lightest tones jump to the smallest printable dot); "
                         "soft = ramp the ends gently, losing a few extreme tones (EDN 'gamma line')")
    pr.add_argument("--soft-width", type=float, default=10.0, help="codes of ramp for --ends soft (default %(default)s)")
    pr.add_argument("--smooth", default="auto",
                    help="curve smoothing width in code values, or auto = chosen from the data's noise (default %(default)s)")
    pr.add_argument("--scan-encoding", default="auto",
                    help="auto (embedded profile, else sRGB), srgb, linear (raw 16-bit linear scans), gamma:2.2")
    pr.add_argument("--flatfield", action="store_true",
                    help="correct uneven lighting using the paper patches (recommended for camera captures)")
    pr.add_argument("--merge", action="append", metavar="PROFILE", help="also use the measurements of an earlier profile")
    pr.add_argument("--no-merge", action="store_true", help="for pass-2 charts: don't merge the pass-1 measurements")
    pr.add_argument("--precorrected-with", metavar="PROFILE",
                    help="pass-2 chart whose layout file is lost: rebuild its patch values from this profile")
    pr.add_argument("--cube-size", type=int, default=33, help="size of the preview 3D LUTs (default %(default)s)")
    pr.add_argument("--work-size", type=int, default=3600, help=argparse.SUPPRESS)
    g = pr.add_argument_group("chart settings, only used when no .layout.json is found")
    g.add_argument("--levels", type=int, default=256)
    g.add_argument("--replicates", type=int, default=2)
    g.add_argument("--order", choices=("shuffled", "ordered"), default="shuffled")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--anchors", type=int, default=8)

    pv = sub.add_parser("preview", parents=[common], formatter_class=fmt, help="render how an image will print",
                        description="Render how an image will look printed (tones in the ink and paper color, no\n"
                                    "halftone). One ink:   preview IMAGE --profile P\n"
                                    "Several inks:        preview --layer pink.tif pink_risotone --layer blue.tif blue_risotone")
    pv.add_argument("image", nargs="?", help="grayscale separation (RGB is converted by luminance)")
    pv.add_argument("--profile", help="profile folder or .risotone.json")
    pv.add_argument("--layer", nargs=2, action="append", metavar=("IMAGE", "PROFILE"),
                    help="one ink layer (repeat for multi-ink prints; inks multiply like translucent riso ink)")
    pv.add_argument("-o", "--out", help="output PNG (default <image>_riso_preview.png)")
    pv.add_argument("--raw", action="store_true", help="the image is ALREADY corrected (output of 'apply'), preview as is")
    pv.add_argument("--white", action="store_true", help="show the paper as white instead of its scanned color")
    pv.add_argument("--compare", action="store_true", help="also write a side-by-side: on screen / no correction / corrected")

    ap = sub.add_parser("apply", parents=[common], formatter_class=fmt, help="apply the correction to an image",
                        description="Apply a profile's correction to a grayscale separation -> file to send to the riso.")
    ap.add_argument("image", help="grayscale separation (RGB is converted by luminance)")
    ap.add_argument("--profile", required=True, help="profile folder or .risotone.json")
    ap.add_argument("-o", "--out", help="output file (default <image>_riso.tif)")
    ap.add_argument("--bits", type=int, choices=(8, 16), default=8, help="output bit depth (default %(default)s)")
    return p


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    if not a.cmd:
        parser.print_help()
        return 1
    LOG.level = 0 if a.quiet else (2 if a.verbose else 1)
    try:
        {"chart": cmd_chart, "profile": cmd_profile, "preview": cmd_preview, "apply": cmd_apply}[a.cmd](a)
        return 0
    except RisoError as e:
        print(_format_error(e), file=sys.stderr)
        if a.traceback:
            traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        print("%s: interrupted" % PROG, file=sys.stderr)
        return 130
    except MemoryError:
        print("%s: error: ran out of memory. Scan at 300-600 dpi instead of higher, or close other apps."
              % PROG, file=sys.stderr)
        return 1
    except Exception as e:
        print("%s: unexpected error: %s: %s" % (PROG, type(e).__name__, e), file=sys.stderr)
        print("  This is a bug or an unhandled case. Full traceback:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
