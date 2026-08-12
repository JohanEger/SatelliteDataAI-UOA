"""
mapfig.py - turn a geemap / ipyleaflet screendump into a publication-standard figure.

Adds a scale bar, north arrow, optional graticule, attribution, panel label and
keyline as *vector* objects in a margin below the map, so nothing is flattened
into the raster and no text sits on top of busy imagery.

Conforms to the Nature research figure specifications:
  - RGB, minimum 300 dpi at the stated reproduction width (checked, warns if not)
  - Arial / Helvetica (Liberation Sans is metric-compatible), 5-7 pt text
  - panel labels 8 pt bold upright lowercase
  - pdf.fonttype = 42, so text stays editable rather than outlined
  - black text only, no drop shadows, no decorative elements
  - vector .pdf export alongside a high-dpi .png

Usage
-----
    import mapfig

    # 1. capture the map (see mapfig.CAPTURE_RECIPE)
    # 2. get the ground resolution straight from the live map object:
    mpp = mapfig.m_per_px_from_bounds(Map_q1, capture_width_px=1314)

    # 3. compose
    mapfig.publication_figure(
        "auckland_raw.png",
        m_per_px = mpp,
        out_stem = "fig1_auckland",
        bar_km   = 10,
        panel    = "a",
    )
"""

from __future__ import annotations

import math
import warnings

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType 42 -> editable text
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
]

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OSM_ATTRIBUTION = (
    "Map data \u00a9 OpenStreetMap contributors, SRTM  |  "
    "Map style \u00a9 OpenTopoMap (CC-BY-SA)"
)

# --------------------------------------------------------------- tile render

TILE_SERVERS = {
    "OpenTopoMap": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}

# OpenTopoMap is a small volunteer-run service. Its usage policy requires a
# descriptive User-Agent and forbids bulk downloading. This module caches every
# tile to disk so a re-run costs zero requests, and a single figure at
# hidpi=True is on the order of a few dozen tiles - well within fair use.
USER_AGENT = "UoA-SatelliteDataAI-lab-figure/1.0 (student coursework; contact: your.email@aucklanduni.ac.nz)"

TILE_SIZE = 256


def _lonlat_to_global_px(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Web Mercator (EPSG:3857) lon/lat -> global pixel coordinates at zoom z."""
    world = TILE_SIZE * 2 ** z
    x = (lon + 180.0) / 360.0 * world
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * world
    return x, y


def _global_px_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    """Inverse of _lonlat_to_global_px."""
    world = TILE_SIZE * 2 ** z
    lon = x / world * 360.0 - 180.0
    n = math.pi * (1 - 2 * y / world)
    lat = math.degrees(math.atan(math.sinh(n)))
    return lon, lat


def _fetch_tile(url: str, cache_dir, session=None):
    """Download one tile, caching it on disk. Returns a PIL Image."""
    import hashlib
    from pathlib import Path

    from PIL import Image

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = cache / (hashlib.sha1(url.encode()).hexdigest() + ".png")
    if key.exists():
        return Image.open(key).convert("RGB")

    import requests

    sess = session or requests
    r = sess.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    key.write_bytes(r.content)
    return Image.open(key).convert("RGB")


def render_basemap(
    center: tuple[float, float],
    zoom: int,
    size_px: tuple[int, int] = (1200, 900),
    basemap: str = "OpenTopoMap",
    hidpi: bool = True,
    out_png: str = "basemap_raw.png",
    cache_dir: str = ".tilecache",
):
    """Build the map raster from XYZ tiles - no screenshot, no browser.

    center  : (lat, lon), same convention as geemap.Map(center=...)
    zoom    : the zoom level you would have passed to geemap.Map(zoom=...)
    size_px : output size in CSS pixels at that zoom.
    hidpi   : fetch one zoom level deeper and return a 2x raster. This is the
              honest way to reach 300 dpi - it is real extra tile detail, not
              interpolation, which the Nature guidance rightly says gains
              nothing.

    Returns (out_png, m_per_px, bounds) where bounds is ((s, w), (n, e)),
    ready to hand straight to publication_figure().
    """
    from PIL import Image

    lat, lon = center
    z = zoom + (1 if hidpi else 0)
    mult = 2 if hidpi else 1
    W, H = size_px[0] * mult, size_px[1] * mult

    cx, cy = _lonlat_to_global_px(lon, lat, z)
    left, top = cx - W / 2.0, cy - H / 2.0

    tx0, ty0 = int(math.floor(left / TILE_SIZE)), int(math.floor(top / TILE_SIZE))
    tx1 = int(math.floor((left + W - 1) / TILE_SIZE))
    ty1 = int(math.floor((top + H - 1) / TILE_SIZE))

    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    if n_tiles > 200:
        raise ValueError(
            f"{n_tiles} tiles requested. That is heavy for a volunteer tile "
            f"server - reduce zoom, reduce size_px, or set hidpi=False."
        )

    template = TILE_SERVERS[basemap]
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_SIZE,
                              (ty1 - ty0 + 1) * TILE_SIZE))
    n_max = 2 ** z
    for i, tx in enumerate(range(tx0, tx1 + 1)):
        for j, ty in enumerate(range(ty0, ty1 + 1)):
            if not (0 <= ty < n_max):
                continue                       # off the top/bottom of the world
            url = template.format(s="a", z=z, x=tx % n_max, y=ty)
            mosaic.paste(_fetch_tile(url, cache_dir),
                         (i * TILE_SIZE, j * TILE_SIZE))

    ox, oy = left - tx0 * TILE_SIZE, top - ty0 * TILE_SIZE
    img = mosaic.crop((int(round(ox)), int(round(oy)),
                       int(round(ox)) + W, int(round(oy)) + H))
    img.save(out_png)

    m_per_px = 156_543.03392 * math.cos(math.radians(lat)) / (2 ** z)
    w_deg, n_deg = _global_px_to_lonlat(left, top, z)
    e_deg, s_deg = _global_px_to_lonlat(left + W, top + H, z)

    return out_png, m_per_px, ((s_deg, w_deg), (n_deg, e_deg))


CAPTURE_RECIPE = """\
Capturing the raster
--------------------
1.  Show only the layers that belong in the figure. Superfluous layers are
    explicitly discouraged by the Nature guidance:
        for lyr in Map_q1.layers:
            print(lyr.name)          # then untick the rest in the layer control
2.  Give the map room and capture at 2x if you can (retina screenshot, or set
    the browser to 200% zoom then screenshot):
        Map_q1 = geemap.Map(center=(-36.8485, 174.7633), zoom=12, height="900px")
3.  Screenshot the map canvas ONLY - no browser tabs, no Colab toolbar, no
    layer-control widget. Do not crop after that, or the bounds-based scale
    calculation below stops being valid.
4.  Record the pixel width of what you captured; pass it as capture_width_px.
"""


# --------------------------------------------------------------------- scale


def m_per_px_from_bounds(m, capture_width_px: int) -> float:
    """Ground resolution (metres per screenshot pixel) from a live map object.

    Preferred method: reads the map's own visible extent, so the scale bar is
    derived from the map state rather than assumed. Requires that the capture
    covers the full map canvas and was not cropped afterwards.

    ipyleaflet reports bounds as ((south, west), (north, east)).
    """
    b = m.bounds
    if not b:
        raise RuntimeError(
            "Map.bounds is empty - the map must be rendered in the notebook "
            "output before it reports its extent. Scroll to the map, then "
            "re-run this cell. Otherwise use m_per_px_from_zoom()."
        )
    (south, west), (north, east) = b
    lat_mid = 0.5 * (south + north)
    width_m = (east - west) * 111_320.0 * math.cos(math.radians(lat_mid))
    return width_m / capture_width_px


def m_per_px_from_zoom(lat: float, zoom: int, device_pixel_ratio: float = 1.0) -> float:
    """Ground resolution from the Web Mercator tile pyramid.

    Fallback for when bounds are unavailable. device_pixel_ratio is 2.0 for a
    retina/HiDPI screenshot, since each CSS pixel became two image pixels.
    """
    return 156_543.03392 * math.cos(math.radians(lat)) / (2 ** zoom) / device_pixel_ratio


def _nice_bar_km(map_width_m: float, target_fraction: float = 0.25) -> float:
    """Pick a round scale-bar length covering roughly a quarter of the map."""
    raw_km = map_width_m * target_fraction / 1000.0
    exp = math.floor(math.log10(raw_km))
    for mult in (1, 2, 5, 10):
        cand = mult * 10 ** exp
        if cand >= raw_km:
            return cand
    return 10 ** (exp + 1)


# ------------------------------------------------------------------- figure


def publication_figure(
    src_png: str,
    m_per_px: float,
    out_stem: str = "figure",
    fig_width_mm: float = 183.0,
    bar_km: float | None = None,
    panel: str | None = "a",
    attribution: str = OSM_ATTRIBUTION,
    bounds: tuple | None = None,
    graticule: bool = False,
    dpi: int = 450,
):
    """Compose and export the annotated figure.

    Parameters
    ----------
    src_png       : cropped screenshot of the map canvas.
    m_per_px      : ground resolution of that screenshot, from the helpers above.
    fig_width_mm  : intended reproduction width. 89 = single column,
                    183 = double column. This is what the 5-7 pt text sizes and
                    the 300 dpi minimum are judged against, so set it to the
                    width you will actually place the image at in your document.
    bar_km        : scale-bar length. None picks a round number automatically.
    panel         : 8 pt bold lowercase panel letter, or None to omit.
    bounds        : ((south, west), (north, east)) - required if graticule=True.
    graticule     : draw ticked latitude/longitude axes with units.
    """
    img = mpimg.imread(src_png)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]                       # drop alpha -> clean RGB
    h_px, w_px = img.shape[:2]

    fig_w_in = fig_width_mm / 25.4
    eff_dpi = w_px / fig_w_in
    if eff_dpi < 300:
        warnings.warn(
            f"Effective resolution is {eff_dpi:.0f} dpi at {fig_width_mm:.0f} mm "
            f"wide, below the 300 dpi minimum. Either place the figure at "
            f"<= {w_px / 300 * 25.4:.0f} mm, or recapture the screenshot at 2x. "
            f"Upsampling the file will not help.",
            stacklevel=2,
        )

    if bar_km is None:
        bar_km = _nice_bar_km(w_px * m_per_px)

    pad_in = 0.62 if graticule else 0.0   # room for the rotated y-axis label
    map_h_in = (fig_w_in - pad_in) * h_px / w_px
    strip_in = 0.60
    fig_h_in = map_h_in + strip_in + (pad_in if graticule else 0.0)

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([
        pad_in / fig_w_in,
        (strip_in + (pad_in if graticule else 0.0)) / fig_h_in,
        (fig_w_in - pad_in) / fig_w_in,
        map_h_in / fig_h_in,
    ])
    ax.imshow(img, interpolation="none")
    for s in ax.spines.values():                  # thin keyline, no shadow
        s.set_linewidth(0.6)
        s.set_color("black")

    # ---- graticule: ticked axes with units in parentheses
    if graticule:
        if bounds is None:
            raise ValueError("graticule=True requires bounds=((s, w), (n, e))")
        (south, west), (north, east) = bounds
        lon_ticks = _nice_ticks(west, east)
        lat_ticks = _nice_ticks(south, north)
        ax.set_xticks([(t - west) / (east - west) * w_px for t in lon_ticks])
        ax.set_xticklabels([f"{t:.2f}" for t in lon_ticks], fontsize=6)
        ax.set_yticks([(north - t) / (north - south) * h_px for t in lat_ticks])
        ax.set_yticklabels([f"{t:.2f}" for t in lat_ticks], fontsize=6)
        ax.set_xlabel("Longitude (\u00b0E)", fontsize=7)
        ax.set_ylabel("Latitude (\u00b0)", fontsize=7)
        ax.tick_params(direction="out", length=2.5, width=0.6, colors="black",
                       labelcolor="black")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # ---- panel label: 8 pt bold, upright, lowercase
    if panel:
        ax.text(0.008, 0.992, panel, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.4))

    # ---- annotation strip, measured in inches
    sx = fig.add_axes([0, 0, 1, strip_in / fig_h_in])
    sx.set_xlim(0, fig_w_in)
    sx.set_ylim(0, strip_in)
    sx.set_axis_off()

    bar_in = (bar_km * 1000 / m_per_px) * ((fig_w_in - pad_in) / w_px)
    x0, y0, bh, nseg = pad_in + 0.10, 0.30, 0.055, 4
    for i in range(nseg):
        sx.add_patch(Rectangle(
            (x0 + i * bar_in / nseg, y0), bar_in / nseg, bh,
            facecolor=("black" if i % 2 else "white"),
            edgecolor="black", linewidth=0.5,
        ))
    for frac in (0.0, 0.5, 1.0):
        xt = x0 + frac * bar_in
        sx.plot([xt, xt], [y0, y0 - 0.035], color="black", linewidth=0.5)
        sx.text(xt, y0 - 0.055, f"{frac * bar_km:g}", ha="center", va="top",
                fontsize=6, color="black")
    sx.text(x0 + bar_in / 2, y0 + bh + 0.025, "Distance (km)",
            ha="center", va="bottom", fontsize=6, color="black")

    # ---- north arrow. Web Mercator is conformal with meridians vertical,
    #      so grid north equals true north and a vertical arrow is correct.
    nx = fig_w_in - 0.35
    sx.annotate("", xy=(nx, 0.46), xytext=(nx, 0.14),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                linewidth=0.9, mutation_scale=7))
    sx.text(nx, 0.50, "N", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color="black")

    # ---- attribution: a licence condition of the tiles, not decoration
    if attribution:
        sx.text(pad_in + 0.10, 0.055, attribution, ha="left", va="bottom",
                fontsize=5, color="black")

    fig.savefig(f"{out_stem}.pdf", facecolor="white")   # vector, preferred
    fig.savefig(f"{out_stem}.png", dpi=dpi, facecolor="white")
    plt.close(fig)

    scale_denom = m_per_px * eff_dpi / 0.0254
    report = {
        "raster_px": (w_px, h_px),
        "repro_width_mm": fig_width_mm,
        "effective_dpi": round(eff_dpi),
        "m_per_px": round(m_per_px, 3),
        "scale_bar_km": bar_km,
        "representative_fraction": f"1:{scale_denom:,.0f}",
        "outputs": [f"{out_stem}.pdf", f"{out_stem}.png"],
    }
    return report


def figure_from_tiles(
    center: tuple[float, float],
    zoom: int,
    out_stem: str = "figure",
    size_px: tuple[int, int] = (1200, 900),
    basemap: str = "OpenTopoMap",
    hidpi: bool = True,
    fig_width_mm: float = 111.0,
    bar_km: float | None = None,
    panel: str | None = "a",
    graticule: bool = False,
    cache_dir: str = ".tilecache",
    dpi: int = 450,
):
    """End-to-end: fetch tiles, compose the figure, export. No screenshot.

    Because the raster is built from the tile pyramid, the ground resolution
    and the geographic bounds are known exactly rather than inferred, so the
    scale bar and graticule are correct by construction.
    """
    raw, m_per_px, bounds = render_basemap(
        center=center, zoom=zoom, size_px=size_px, basemap=basemap,
        hidpi=hidpi, out_png=f"{out_stem}_raw.png", cache_dir=cache_dir,
    )
    report = publication_figure(
        raw, m_per_px=m_per_px, out_stem=out_stem,
        fig_width_mm=fig_width_mm, bar_km=bar_km, panel=panel,
        bounds=bounds, graticule=graticule, dpi=dpi,
    )
    report["raw_png"] = raw
    report["bounds_SWNE"] = bounds
    report["tile_zoom_fetched"] = zoom + (1 if hidpi else 0)
    return report


def _nice_ticks(lo: float, hi: float, n: int = 4):
    """Round tick positions inside [lo, hi]."""
    span = hi - lo
    exp = math.floor(math.log10(span / n))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * 10 ** exp
        if span / step <= n + 1:
            break
    start = math.ceil(lo / step) * step
    return list(np.arange(start, hi + step * 1e-9, step))


if __name__ == "__main__":
    rep = publication_figure(
        "/mnt/user-data/uploads/1786491580582_image.png",
        m_per_px=m_per_px_from_zoom(lat=-36.8485, zoom=12),
        out_stem="/home/claude/fig1_auckland",
        fig_width_mm=111,
        bar_km=10,
        panel="a",
    )
    for k, v in rep.items():
        print(f"{k:>26} : {v}")
