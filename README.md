# fractal-art

A Python package for generating 512x512 fractal art PNG images using common scientific Python tools:

- `scipy`
- `matplotlib`
- `scikit-image`

The generator combines multiple fractal families (`Mandelbrot`, `Julia`, `Burning Ship`, `Tricorn`, and `Newton`) and composites cropped fractal elements (for example tree sprites) into layered scenes with natural color palettes, horizons, lakes, reflections, stars, galaxies, and ringed planets.

## Installation

```bash
pip install -e .
```

## Create your own fractal art

Generate one image:

```bash
fractal-art --theme landscape --style neon --seed 42 --output my_art.png
```

Options:

- `--theme`: `landscape`, `tree`, `planet`, `nebula`, `ocean`
- `--style`: `classic`, `neon`, `pastel`, `mono`
- `--seed`: integer random seed for reproducible variations
- `--size`: must be `512`
- `--output`: output PNG path

Generate a full set of 20 examples:

```bash
fractal-art --gallery --gallery-dir examples
```

## Python API

```python
from fractal_art import generate_art

generate_art(
    "my_planet.png",
    theme="planet",
    style="pastel",
    seed=123,
    size=512,
)
```

## 20 example artworks

![Detailed gallery screenshot](examples/showcase_detailed.png)

| # | Preview |
|---|---|
| 1 | ![example 1](examples/example_01_landscape_classic.png) |
| 2 | ![example 2](examples/example_02_landscape_neon.png) |
| 3 | ![example 3](examples/example_03_landscape_pastel.png) |
| 4 | ![example 4](examples/example_04_landscape_mono.png) |
| 5 | ![example 5](examples/example_05_tree_classic.png) |
| 6 | ![example 6](examples/example_06_tree_neon.png) |
| 7 | ![example 7](examples/example_07_tree_pastel.png) |
| 8 | ![example 8](examples/example_08_tree_mono.png) |
| 9 | ![example 9](examples/example_09_planet_classic.png) |
| 10 | ![example 10](examples/example_10_planet_neon.png) |
| 11 | ![example 11](examples/example_11_planet_pastel.png) |
| 12 | ![example 12](examples/example_12_planet_mono.png) |
| 13 | ![example 13](examples/example_13_nebula_classic.png) |
| 14 | ![example 14](examples/example_14_nebula_neon.png) |
| 15 | ![example 15](examples/example_15_nebula_pastel.png) |
| 16 | ![example 16](examples/example_16_nebula_mono.png) |
| 17 | ![example 17](examples/example_17_ocean_classic.png) |
| 18 | ![example 18](examples/example_18_ocean_neon.png) |
| 19 | ![example 19](examples/example_19_ocean_pastel.png) |
| 20 | ![example 20](examples/example_20_ocean_mono.png) |
