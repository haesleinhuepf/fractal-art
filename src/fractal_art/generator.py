from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib import image as mpimg
from scipy.ndimage import gaussian_filter
from skimage import exposure

THEMES = ("landscape", "tree", "planet", "nebula", "ocean")
STYLES = ("classic", "neon", "pastel", "mono")

STYLE_CMAPS = {
    "classic": "viridis",
    "neon": "plasma",
    "pastel": "magma",
    "mono": "gray",
}


def _fractal_field(size: int, seed: int, max_iter: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
    zoom = rng.uniform(0.7, 1.6)

    axis = np.linspace(-1.5, 1.5, size)
    x, y = np.meshgrid(axis, axis)
    c = (x / zoom + cx) + 1j * (y / zoom + cy)
    z = np.zeros_like(c)
    field = np.zeros(c.shape, dtype=float)
    active = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        z[active] = z[active] * z[active] + c[active]
        escaped = np.abs(z) > 2.0
        newly_escaped = escaped & active
        field[newly_escaped] = i
        active &= ~escaped
        if not np.any(active):
            break

    field[active] = max_iter
    field = gaussian_filter(field, sigma=1.2)
    field = exposure.rescale_intensity(field, out_range=(0.0, 1.0))
    return field


def _render_landscape(field: np.ndarray) -> np.ndarray:
    gradient = np.gradient(field)[0]
    terrain = np.clip(field * 1.4 - gradient * 0.6, 0.0, 1.0)
    sky = np.linspace(0.15, 1.0, field.shape[0])[:, None]
    return np.maximum(terrain, sky * 0.6)


def _render_tree(field: np.ndarray) -> np.ndarray:
    h, w = field.shape
    y, x = np.indices((h, w))
    x0 = w / 2
    trunk = np.exp(-((x - x0) ** 2) / (2 * (w * 0.02) ** 2)) * (y > h * 0.45)
    canopy = np.clip(field * (1.5 - y / h), 0.0, 1.0)
    return np.clip(gaussian_filter(trunk + canopy, 1.0), 0.0, 1.0)


def _render_planet(field: np.ndarray) -> np.ndarray:
    h, w = field.shape
    y, x = np.indices((h, w))
    r = min(h, w) * 0.34
    cx, cy = w / 2, h / 2
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    img = np.zeros_like(field)
    img[mask] = field[mask]
    glow = gaussian_filter(mask.astype(float), sigma=8) * 0.5
    return np.clip(img + glow, 0.0, 1.0)


def _render_nebula(field: np.ndarray) -> np.ndarray:
    return np.clip(gaussian_filter(field ** 0.7, sigma=2.0) * 1.1, 0.0, 1.0)


def _render_ocean(field: np.ndarray) -> np.ndarray:
    waves = np.sin(np.linspace(0, 8 * np.pi, field.shape[0]))[:, None] * 0.2
    return np.clip(field * 0.8 + waves + 0.2, 0.0, 1.0)


RENDERERS = {
    "landscape": _render_landscape,
    "tree": _render_tree,
    "planet": _render_planet,
    "nebula": _render_nebula,
    "ocean": _render_ocean,
}


def generate_art(
    output_path: str | Path,
    *,
    theme: str = "landscape",
    style: str = "classic",
    seed: int = 0,
    size: int = 512,
) -> Path:
    if theme not in THEMES:
        raise ValueError(f"Unknown theme '{theme}'. Available: {', '.join(THEMES)}")
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Available: {', '.join(STYLES)}")
    if size != 512:
        raise ValueError("Only 512x512 output is supported.")

    field = _fractal_field(size=size, seed=seed)
    grayscale = RENDERERS[theme](field)
    rgba = colormaps[STYLE_CMAPS[style]](grayscale)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(output, rgb)
    return output


def generate_gallery(output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    combos = [
        ("landscape", "classic", 1),
        ("landscape", "neon", 2),
        ("landscape", "pastel", 3),
        ("landscape", "mono", 4),
        ("tree", "classic", 5),
        ("tree", "neon", 6),
        ("tree", "pastel", 7),
        ("tree", "mono", 8),
        ("planet", "classic", 9),
        ("planet", "neon", 10),
        ("planet", "pastel", 11),
        ("planet", "mono", 12),
        ("nebula", "classic", 13),
        ("nebula", "neon", 14),
        ("nebula", "pastel", 15),
        ("nebula", "mono", 16),
        ("ocean", "classic", 17),
        ("ocean", "neon", 18),
        ("ocean", "pastel", 19),
        ("ocean", "mono", 20),
    ]

    paths: list[Path] = []
    for i, (theme, style, seed) in enumerate(combos, start=1):
        path = output / f"example_{i:02d}_{theme}_{style}.png"
        paths.append(generate_art(path, theme=theme, style=style, seed=seed))
    return paths
