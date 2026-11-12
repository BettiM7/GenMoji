"""Exercise actual SVG rendering, split isolation, and ImageFolder ingestion."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from genmoji.data import family_key, prepare_dataset, prepare_svg


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "data").mkdir(parents=True)
        (self.source / "color/svg").mkdir(parents=True)
        (self.source / "LICENSE.txt").write_text("Test fixture, not OpenMoji artwork")
        self.rows = [{"hexcode": f"{code:X}", "annotation": f"test shape {code}",
                      "openmoji_author": "Fixture author"} for code in range(0x1F600, 0x1F640)]
        self.rows.append({"hexcode": "1F600-1F3FB", "annotation": "variant",
                          "skintone_base_hexcode": "1F600"})
        self.write_metadata()
        for row in self.rows:
            (self.source / "color/svg" / f"{row['hexcode']}.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72">'
                '<rect x="18" y="18" width="36" height="36" fill="red"/></svg>')

    def write_metadata(self):
        (self.source / "data/openmoji.json").write_text(json.dumps(self.rows))

    def test_conversion_and_both_trainers_imagefolder_contract(self):
        output = self.root / "prepared"
        manifest = prepare_dataset(self.source, output, resolution=32)
        self.assertEqual(sum(manifest["counts"].values()), len(self.rows))
        self.assertTrue(all(manifest["counts"].values()))
        records = json.loads((output / "provenance/records.json").read_text())
        by_code = {row["hexcode"]: row for row in records}
        self.assertEqual(by_code["1F600"]["split"], by_code["1F600-1F3FB"]["split"])
        with Image.open(output / "imagefolder" / records[0]["file_name"]) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (32, 32))
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(image.getpixel((16, 16)), (255, 0, 0))
        # This executes the exact local-imagefolder loader used by both trainers,
        # and independently checks provenance against every decoded example.
        check_script = Path(__file__).resolve().parents[1] / "scripts/check_dataset.py"
        result = subprocess.run([sys.executable, str(check_script), str(output)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Verified 65 examples", result.stdout)
        from datasets import load_dataset
        splits = load_dataset("imagefolder", data_dir=str(output / "imagefolder"))
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertEqual(len(splits["train"]), manifest["counts"]["train"])

    def test_repeatable_splits_independent_of_metadata_order(self):
        prepare_dataset(self.source, self.root / "first", resolution=8)
        self.rows.reverse()
        self.write_metadata()
        prepare_dataset(self.source, self.root / "second", resolution=8)
        for split in ("train", "validation", "test"):
            first = self.root / "first/imagefolder" / split / "metadata.jsonl"
            second = self.root / "second/imagefolder" / split / "metadata.jsonl"
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_reject_overwrite_and_missing_assets(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_dataset(self.source, output)
        (self.source / "color/svg/1F600.svg").unlink()
        with self.assertRaisesRegex(ValueError, "Missing color SVG"):
            prepare_dataset(self.source, self.root / "missing")
        self.assertFalse((self.root / "missing").exists())

    def test_failed_render_does_not_publish_partial_dataset(self):
        (self.source / "color/svg/1F600.svg").write_text("broken SVG")
        output = self.root / "broken"
        with self.assertRaises(Exception):
            prepare_dataset(self.source, output, resolution=8)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".openmoji-*")))

    def test_invalid_fractions_and_family_normalization(self):
        with self.assertRaisesRegex(ValueError, "fractions"):
            prepare_dataset(self.source, self.root / "invalid", validation_fraction=0.8, test_fraction=0.3)
        self.assertEqual(family_key({"hexcode": "270D-1F3FD-FE0F"}), "270D")

    def test_dangling_marker_renders_and_records_adjustment(self):
        path = self.source / "color/svg/1F600.svg"
        original = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72">'
            '<path d="M 10 36 L 62 36" stroke="red" stroke-width="4" '
            'marker-end="url(#Dot)"/></svg>'
        )
        path.write_text(original)
        output = self.root / "markers"
        manifest = prepare_dataset(self.source, output, resolution=72)
        self.assertEqual(manifest["adjusted_svg_count"], 1)
        records = json.loads((output / "provenance/records.json").read_text())
        row = next(row for row in records if row["hexcode"] == "1F600")
        self.assertEqual(row["svg_adjustments"], [{"attribute": "marker-end", "value": "url(#Dot)"}])
        with Image.open(output / "imagefolder" / row["file_name"]) as image:
            self.assertEqual(image.getpixel((36, 36)), (255, 0, 0))
        self.assertEqual(path.read_text(), original)

    def test_defined_markers_are_preserved(self):
        svg = (
            b'<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72">'
            b'<defs><marker id="Dot"><circle r="1"/></marker></defs>'
            b'<path d="M 10 36 L 62 36" marker-end="url(#Dot)"/></svg>'
        )
        self.assertEqual(prepare_svg(svg), (svg, []))


if __name__ == "__main__":
    unittest.main()
