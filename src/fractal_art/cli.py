from __future__ import annotations

import argparse

from .generator import STYLES, THEMES, generate_art, generate_gallery


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 512x512 fractal art PNG files.")
    parser.add_argument("--theme", choices=THEMES, default="landscape")
    parser.add_argument("--style", choices=STYLES, default="classic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--output", default="fractal.png")
    parser.add_argument("--gallery", action="store_true", help="Generate 20 themed examples")
    parser.add_argument("--gallery-dir", default="examples")

    args = parser.parse_args()

    if args.gallery:
        paths = generate_gallery(args.gallery_dir)
        print(f"Generated {len(paths)} examples in {args.gallery_dir}")
        return

    path = generate_art(args.output, theme=args.theme, style=args.style, seed=args.seed, size=args.size)
    print(f"Generated {path}")


if __name__ == "__main__":
    main()
