import tempfile
import unittest
from pathlib import Path

from PIL import Image

from fractal_art import generate_art, generate_gallery


class TestGenerator(unittest.TestCase):
    def test_generate_512_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sample.png"
            generate_art(output, theme="planet", style="classic", seed=7, size=512)
            self.assertTrue(output.exists())
            with Image.open(output) as image:
                self.assertEqual(image.size, (512, 512))

    def test_rejects_non_512_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "invalid.png"
            with self.assertRaisesRegex(ValueError, "Only 512x512 output is supported\\."):
                generate_art(output, size=256)

    def test_generate_gallery_outputs_20_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_gallery(tmp)
            self.assertEqual(len(paths), 20)
            with Image.open(paths[0]) as image:
                self.assertEqual(image.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
