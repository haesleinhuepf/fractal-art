from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import image as mpimg
from scipy.ndimage import gaussian_filter
from skimage import exposure, transform

THEMES = ("landscape", "tree", "planet", "nebula", "ocean")
STYLES = ("classic", "neon", "pastel", "mono")

STYLE_PALETTES = {
    "classic": ("#0b1d2a", "#2c5f57", "#6f8f5c", "#d0b277", "#f1e3c6"),
    "neon": ("#04070d", "#1d2854", "#3960b5", "#5ca7d6", "#c5f3ff"),
    "pastel": ("#1b2534", "#5e7c8a", "#93b0a9", "#c6ccb1", "#efe0c8"),
    "mono": ("#111111", "#5d5d5d", "#a7a7a7", "#f2f2f2"),
}


def _normalize(field: np.ndarray) -> np.ndarray:
    return exposure.rescale_intensity(field, out_range=(0.0, 1.0))


def _mandelbrot_field(size: int, seed: int, max_iter: int = 90) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
    zoom = rng.uniform(0.7, 1.8)

    axis = np.linspace(-1.6, 1.6, size)
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
    return _normalize(gaussian_filter(field, sigma=1.1))


def _julia_field(size: int, seed: int, max_iter: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.uniform(-0.75, 0.75) + 1j * rng.uniform(-0.75, 0.75)

    axis = np.linspace(-1.6, 1.6, size)
    x, y = np.meshgrid(axis, axis)
    z = x + 1j * y
    field = np.zeros(z.shape, dtype=float)
    active = np.ones(z.shape, dtype=bool)

    for i in range(max_iter):
        z[active] = z[active] * z[active] + c
        escaped = np.abs(z) > 2.0
        newly_escaped = escaped & active
        field[newly_escaped] = i
        active &= ~escaped
        if not np.any(active):
            break

    field[active] = max_iter
    return _normalize(gaussian_filter(field, sigma=1.0))


def _burning_ship_field(size: int, seed: int, max_iter: int = 85) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(-1.95, -1.55), rng.uniform(-0.15, 0.25)
    zoom = rng.uniform(1.2, 2.2)

    axis = np.linspace(-1.8, 1.8, size)
    x, y = np.meshgrid(axis, axis)
    c = (x / zoom + cx) + 1j * (y / zoom + cy)
    z = np.zeros_like(c)
    field = np.zeros(c.shape, dtype=float)
    active = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        zr = np.abs(z.real)
        zi = np.abs(z.imag)
        z[active] = (zr[active] + 1j * zi[active]) ** 2 + c[active]
        escaped = np.abs(z) > 2.0
        newly_escaped = escaped & active
        field[newly_escaped] = i
        active &= ~escaped
        if not np.any(active):
            break

    field[active] = max_iter
    return _normalize(gaussian_filter(field, sigma=1.0))


def _tricorn_field(size: int, seed: int, max_iter: int = 95) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(-0.9, 0.4), rng.uniform(-0.7, 0.7)
    zoom = rng.uniform(0.9, 1.9)

    axis = np.linspace(-1.7, 1.7, size)
    x, y = np.meshgrid(axis, axis)
    c = (x / zoom + cx) + 1j * (y / zoom + cy)
    z = np.zeros_like(c)
    field = np.zeros(c.shape, dtype=float)
    active = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        z[active] = np.conj(z[active]) * np.conj(z[active]) + c[active]
        escaped = np.abs(z) > 2.0
        newly_escaped = escaped & active
        field[newly_escaped] = i
        active &= ~escaped
        if not np.any(active):
            break

    field[active] = max_iter
    return _normalize(gaussian_filter(field, sigma=1.0))


def _newton_field(size: int, seed: int, max_iter: int = 45) -> np.ndarray:
    rng = np.random.default_rng(seed)
    zoom = rng.uniform(0.8, 1.6)
    axis = np.linspace(-1.5, 1.5, size)
    x, y = np.meshgrid(axis, axis)
    z = x / zoom + 1j * (y / zoom)

    roots = np.array([1 + 0j, -0.5 + np.sqrt(3) / 2j, -0.5 - np.sqrt(3) / 2j])
    field = np.zeros(z.shape, dtype=float)

    for i in range(max_iter):
        dz = 3.0 * z * z
        safe = np.abs(dz) > 1e-8
        z_next = z.copy()
        z_next[safe] = z[safe] - (z[safe] ** 3 - 1.0) / dz[safe]
        z = z_next
        nearest = np.min(np.abs(z[:, :, None] - roots[None, None, :]), axis=2)
        field += np.exp(-nearest * 4.5)
        if np.max(nearest) < 1e-3:
            break

    return _normalize(gaussian_filter(field, sigma=1.2))


def _fractal_layers(size: int, seed: int) -> dict[str, np.ndarray]:
    return {
        "mandelbrot": _mandelbrot_field(size, seed + 11),
        "julia": _julia_field(size, seed + 23),
        "burning_ship": _burning_ship_field(size, seed + 37),
        "tricorn": _tricorn_field(size, seed + 47),
        "newton": _newton_field(size, seed + 59),
    }


def _alpha_blend(canvas: np.ndarray, layer: np.ndarray, alpha: np.ndarray, top: int, left: int) -> None:
    h, w = layer.shape
    c_h, c_w = canvas.shape

    y1, x1 = max(0, top), max(0, left)
    y2, x2 = min(c_h, top + h), min(c_w, left + w)
    if y1 >= y2 or x1 >= x2:
        return

    ly1, lx1 = y1 - top, x1 - left
    ly2, lx2 = ly1 + (y2 - y1), lx1 + (x2 - x1)

    a = np.clip(alpha[ly1:ly2, lx1:lx2], 0.0, 1.0)
    canvas[y1:y2, x1:x2] = canvas[y1:y2, x1:x2] * (1.0 - a) + layer[ly1:ly2, lx1:lx2] * a


def _extract_tree_sprite(layers: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    texture = _normalize(0.55 * layers["julia"] + 0.45 * layers["burning_ship"])
    src_h, src_w = texture.shape

    crop_h = int(rng.integers(src_h // 7, src_h // 5))
    crop_w = int(rng.integers(src_w // 10, src_w // 7))
    y0 = int(rng.integers(src_h // 5, src_h - crop_h - 1))
    x0 = int(rng.integers(0, src_w - crop_w - 1))

    crop = texture[y0 : y0 + crop_h, x0 : x0 + crop_w]
    sprite = transform.resize(crop, (160, 96), anti_aliasing=True)

    h, w = sprite.shape
    yy, xx = np.indices((h, w))
    center = w / 2
    cone = (yy < h * 0.84) & (np.abs(xx - center) < (w * 0.52) * (1.0 - yy / (h * 0.9)))
    trunk = (yy >= h * 0.58) & (np.abs(xx - center) < w * 0.08)
    alpha = gaussian_filter((cone | trunk).astype(float), sigma=1.2)

    detail = _normalize(gaussian_filter(sprite, sigma=0.8))
    return np.clip(detail * alpha * 1.15, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)


def _add_planet(canvas: np.ndarray, texture: np.ndarray, center: tuple[int, int], radius: int) -> None:
    patch = transform.resize(texture, (2 * radius, 2 * radius), anti_aliasing=True)
    yy, xx = np.indices(patch.shape)
    rr = np.sqrt((yy - radius) ** 2 + (xx - radius) ** 2)
    mask = rr <= radius
    highlight = np.clip(1.1 - rr / max(radius, 1), 0.0, 1.0)
    planet = np.clip(patch * 0.8 + highlight * 0.28, 0.0, 1.0)
    alpha = gaussian_filter(mask.astype(float), sigma=max(radius / 22, 1))
    _alpha_blend(canvas, planet, alpha, center[0] - radius, center[1] - radius)


def _add_planet_ring(
    canvas: np.ndarray, texture: np.ndarray, center: tuple[int, int], radius: int, tilt: float = 0.45
) -> None:
    ring_size = int(radius * 2.7)
    patch = transform.resize(texture, (ring_size, ring_size), anti_aliasing=True)
    yy, xx = np.indices((ring_size, ring_size))
    cy = (ring_size - 1) / 2
    cx = (ring_size - 1) / 2
    x_t = (xx - cx) * np.cos(tilt) + (yy - cy) * np.sin(tilt)
    y_t = -(xx - cx) * np.sin(tilt) + (yy - cy) * np.cos(tilt)
    a = radius * 1.34
    b = radius * 0.48
    ellipse = np.sqrt((x_t / max(a, 1e-6)) ** 2 + (y_t / max(b, 1e-6)) ** 2)
    ring = (ellipse >= 0.78) & (ellipse <= 1.08)
    alpha = gaussian_filter(ring.astype(float), sigma=1.2) * 0.58
    layer = np.clip(patch * 0.95 + 0.12, 0.0, 1.0)
    _alpha_blend(canvas, layer, alpha, center[0] - ring_size // 2, center[1] - ring_size // 2)


def _add_horizon(canvas: np.ndarray, horizon: int, intensity: float = 0.16) -> None:
    band = np.exp(-((np.arange(canvas.shape[0]) - horizon) / 3.5) ** 2)[:, None]
    canvas += band * intensity


def _add_stars(canvas: np.ndarray, seed: int, density: float = 0.015, strength: float = 0.75) -> None:
    rng = np.random.default_rng(seed)
    stars = (rng.random(canvas.shape) < density).astype(float)
    twinkle = gaussian_filter(stars, sigma=0.55)
    canvas += twinkle * strength


def _colorize(grayscale: np.ndarray, style: str) -> np.ndarray:
    palette = STYLE_PALETTES[style]
    stops = np.linspace(0.0, 1.0, len(palette))
    rgb = np.zeros((*grayscale.shape, 3), dtype=float)
    g = np.clip(grayscale, 0.0, 1.0)
    for c in range(3):
        channel = np.array([int(p[1 + 2 * c : 3 + 2 * c], 16) for p in palette], dtype=float) / 255.0
        rgb[:, :, c] = np.interp(g, stops, channel)
    return rgb


def _render_landscape(layers: dict[str, np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = layers["mandelbrot"].shape[0]

    sky = _normalize(gaussian_filter(0.66 * layers["julia"] + 0.34 * layers["tricorn"], sigma=2.0))
    terrain = np.clip(layers["mandelbrot"] * 1.35 - np.gradient(layers["mandelbrot"])[0] * 0.8, 0.0, 1.0)
    fog = np.linspace(0.35, 0.0, size)[:, None]

    canvas = np.clip(sky * 0.55 + fog, 0.0, 1.0)
    horizon = int(size * 0.56)
    lake_top = int(size * 0.62)
    canvas[horizon:, :] = np.maximum(canvas[horizon:, :], terrain[horizon:, :] * 0.95)
    _add_horizon(canvas, horizon, intensity=0.14)

    sprite, alpha = _extract_tree_sprite(layers, seed + 101)
    for i in range(22):
        scale = 0.45 + i / 32
        tree = transform.resize(sprite, (int(sprite.shape[0] * scale), int(sprite.shape[1] * scale)), anti_aliasing=True)
        a = transform.resize(alpha, tree.shape, anti_aliasing=True) * (0.6 + 0.35 * scale)
        x = int((i / 22) * (size - tree.shape[1]) + rng.integers(-8, 8))
        y = horizon + int(rng.integers(8, 36)) - tree.shape[0]
        _alpha_blend(canvas, tree, a, y, x)

    reflection = np.flipud(canvas[: size - lake_top, :])
    lake_tex = _normalize(0.6 * layers["newton"][lake_top:, :] + 0.4 * reflection[: size - lake_top, :])
    canvas[lake_top:, :] = np.maximum(canvas[lake_top:, :], gaussian_filter(lake_tex, sigma=1.2) * 0.85)

    _add_planet(canvas, layers["burning_ship"], (int(size * 0.22), int(size * 0.78)), int(size * 0.08))
    return np.clip(canvas, 0.0, 1.0)


def _render_tree(layers: dict[str, np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = layers["mandelbrot"].shape[0]

    canvas = _normalize(0.25 * layers["julia"] + 0.35 * layers["burning_ship"] + 0.4 * layers["newton"])
    canvas = np.clip(canvas * 0.5 + np.linspace(0.2, 0.8, size)[:, None] * 0.25, 0.0, 1.0)
    horizon = int(size * 0.57)
    _add_horizon(canvas, horizon, intensity=0.1)

    sprite, alpha = _extract_tree_sprite(layers, seed + 301)
    for row in range(3):
        depth = 0.65 + row * 0.35
        base_y = int(size * (0.52 + row * 0.12))
        for col in range(9):
            jitter = rng.integers(-12, 12)
            scale = depth * (0.7 + rng.uniform(-0.08, 0.1))
            tree = transform.resize(sprite, (int(sprite.shape[0] * scale), int(sprite.shape[1] * scale)), anti_aliasing=True)
            a = transform.resize(alpha, tree.shape, anti_aliasing=True) * (0.55 + 0.25 * depth)
            x = int(col * (size / 9) + jitter)
            y = base_y - tree.shape[0]
            _alpha_blend(canvas, tree, a, y, x)

    return np.clip(canvas, 0.0, 1.0)


def _render_planet(layers: dict[str, np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = layers["mandelbrot"].shape[0]

    stars = np.clip((layers["julia"] > 0.952).astype(float) * 0.9, 0.0, 1.0)
    galaxy = _normalize(gaussian_filter(0.62 * layers["tricorn"] + 0.38 * layers["newton"], sigma=2.3))
    nebula = _normalize(gaussian_filter(0.35 * layers["julia"] + 0.35 * layers["burning_ship"] + 0.3 * layers["tricorn"], sigma=3.0))
    swirl = np.linspace(0.0, 1.0, size)[None, :]
    canvas = np.clip(nebula * 0.42 + galaxy * 0.4 * (1.1 - np.abs(swirl - 0.58)) + gaussian_filter(stars, sigma=0.7) * 0.6, 0.0, 1.0)
    _add_stars(canvas, seed + 777, density=0.012, strength=0.52)

    center = (int(size * 0.54), int(size * 0.42))
    radius = int(size * 0.22)
    _add_planet_ring(canvas, layers["newton"], center, radius)
    _add_planet(canvas, layers["burning_ship"], center, radius)
    _add_planet(canvas, layers["mandelbrot"], (int(size * 0.22), int(size * 0.82)), int(size * 0.07))

    sprite, alpha = _extract_tree_sprite(layers, seed + 401)
    for i in range(7):
        scale = 0.55 + i / 14
        tree = transform.resize(sprite, (int(sprite.shape[0] * scale), int(sprite.shape[1] * scale)), anti_aliasing=True)
        a = transform.resize(alpha, tree.shape, anti_aliasing=True) * 0.65
        x = int((i / 7) * (size - tree.shape[1])) + int(rng.integers(-8, 8))
        y = int(size * 0.9) - tree.shape[0]
        _alpha_blend(canvas, tree * 0.5, a, y, x)

    return np.clip(canvas, 0.0, 1.0)


def _render_nebula(layers: dict[str, np.ndarray], seed: int) -> np.ndarray:
    base = _normalize(
        0.26 * layers["mandelbrot"]
        + 0.24 * layers["julia"]
        + 0.2 * layers["burning_ship"]
        + 0.15 * layers["tricorn"]
        + 0.15 * layers["newton"]
    )
    filaments = np.clip(gaussian_filter(base ** 0.65, sigma=1.6) * 1.15, 0.0, 1.0)
    stars = gaussian_filter((layers["julia"] > 0.96).astype(float), sigma=0.8) * 0.9
    canvas = np.clip(filaments * 0.85 + stars, 0.0, 1.0)
    _add_stars(canvas, seed + 991, density=0.009, strength=0.32)
    return np.clip(canvas, 0.0, 1.0)


def _render_ocean(layers: dict[str, np.ndarray], seed: int) -> np.ndarray:
    size = layers["mandelbrot"].shape[0]
    horizon = int(size * 0.52)

    sky = _normalize(gaussian_filter(0.42 * layers["julia"] + 0.35 * layers["burning_ship"] + 0.23 * layers["tricorn"], sigma=1.8))
    water = _normalize(gaussian_filter(0.44 * layers["mandelbrot"] + 0.33 * layers["julia"] + 0.23 * layers["newton"], sigma=1.2))
    waves = np.sin(np.linspace(0, 22 * np.pi, size))[:, None] * 0.09

    canvas = np.clip(sky * 0.58, 0.0, 1.0)
    water_region = np.clip(water[horizon:, :] * 0.9 + waves[horizon:, :] + 0.08, 0.0, 1.0)
    canvas[horizon:, :] = np.maximum(canvas[horizon:, :], water_region)
    _add_horizon(canvas, horizon, intensity=0.18)

    planet_center = (int(size * 0.23), int(size * 0.72))
    _add_planet(canvas, layers["burning_ship"], planet_center, int(size * 0.09))

    reflection = np.flipud(canvas[: size - horizon, :])
    blend = np.clip(gaussian_filter(reflection, sigma=2.5) * 0.35, 0.0, 1.0)
    canvas[horizon:, :] = np.maximum(canvas[horizon:, :], blend[: size - horizon, :])
    _add_stars(canvas[:horizon, :], seed + 2026, density=0.011, strength=0.4)
    return np.clip(canvas, 0.0, 1.0)


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
    """Generate a single 512x512 fractal art PNG image.

    Args:
        output_path: File path where the PNG will be written.
        theme: Content theme for rendering.
        style: Colormap style.
        seed: Random seed for reproducible variation.
        size: Image size; only 512 is supported.

    Returns:
        The path to the written image.

    Raises:
        ValueError: If theme, style, or size is invalid.
    """
    if theme not in THEMES:
        raise ValueError(f"Unknown theme '{theme}'. Available: {', '.join(THEMES)}")
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Available: {', '.join(STYLES)}")
    if size != 512:
        raise ValueError("Only 512x512 output is supported.")

    layers = _fractal_layers(size=size, seed=seed)
    grayscale = RENDERERS[theme](layers, seed)
    rgb = (_colorize(grayscale, style) * 255).astype(np.uint8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(output, rgb)
    return output


def generate_gallery(output_dir: str | Path) -> list[Path]:
    """Generate 20 example images across available themes and styles.

    Args:
        output_dir: Directory where examples are written.

    Returns:
        A list of paths to generated PNG files.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    combos = [
        ("landscape", "classic", 101),
        ("landscape", "neon", 102),
        ("landscape", "pastel", 103),
        ("landscape", "mono", 104),
        ("tree", "classic", 201),
        ("tree", "neon", 202),
        ("tree", "pastel", 203),
        ("tree", "mono", 204),
        ("planet", "classic", 301),
        ("planet", "neon", 302),
        ("planet", "pastel", 303),
        ("planet", "mono", 304),
        ("nebula", "classic", 401),
        ("nebula", "neon", 402),
        ("nebula", "pastel", 403),
        ("nebula", "mono", 404),
        ("ocean", "classic", 501),
        ("ocean", "neon", 502),
        ("ocean", "pastel", 503),
        ("ocean", "mono", 504),
    ]

    paths: list[Path] = []
    for i, (theme, style, seed) in enumerate(combos, start=1):
        path = output / f"example_{i:02d}_{theme}_{style}.png"
        paths.append(generate_art(path, theme=theme, style=style, seed=seed))
    return paths
