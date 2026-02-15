import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image

    from data import PairedSRDataset
    _DEPS_AVAILABLE = True
except ModuleNotFoundError:
    _DEPS_AVAILABLE = False


def _save_image(path: Path):
    img = Image.new("RGB", (16, 16), (128, 128, 128))
    img.save(path)


@unittest.skipUnless(_DEPS_AVAILABLE, "torch + pillow dependencies are required")
class DataPairingTest(unittest.TestCase):
    def test_pairing_uses_full_basename_without_digit_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lr_dir = tmp_path / "lr"
            hr_dir = tmp_path / "hr"
            lr_dir.mkdir(parents=True, exist_ok=True)
            hr_dir.mkdir(parents=True, exist_ok=True)

            _save_image(lr_dir / "scene_01_tile_01_lr.png")
            _save_image(lr_dir / "scene_01_tile_02_lr.png")
            _save_image(hr_dir / "scene_01_tile_01_hr.png")
            _save_image(hr_dir / "scene_01_tile_02_hr.png")

            ds = PairedSRDataset(
                str(lr_dir),
                str(hr_dir),
                scale=4,
                hr_crop=16,
                train=False,
            )
            self.assertEqual(len(ds), 2)

    def test_duplicate_keys_raise_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lr_dir = tmp_path / "lr"
            hr_dir = tmp_path / "hr"
            lr_dir.mkdir(parents=True, exist_ok=True)
            hr_dir.mkdir(parents=True, exist_ok=True)

            # Both LR filenames collapse to key "sample" after stripping _lr suffix.
            _save_image(lr_dir / "sample_lr.png")
            _save_image(lr_dir / "sample.png")
            _save_image(hr_dir / "sample_hr.png")

            with self.assertRaisesRegex(ValueError, "Duplicate LR pair keys"):
                PairedSRDataset(
                    str(lr_dir),
                    str(hr_dir),
                    scale=4,
                    hr_crop=16,
                    train=False,
                )


if __name__ == "__main__":
    unittest.main()
