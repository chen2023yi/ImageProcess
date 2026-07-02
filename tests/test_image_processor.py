from __future__ import annotations

import os
import unittest

from PIL import Image, ImageDraw

os.environ.setdefault("IMAGEPRO_DISABLE_REMBG", "1")

from image_processor import (
    OUTPUT_SOLID,
    crop_image,
    detect_subject_mask,
    extract_subject,
    extract_subject_from_mask,
    paint_subject_mask,
    _refine_local_model_mask,
)


class CropImageTests(unittest.TestCase):
    def test_crop_image_returns_selected_box_as_rgba(self) -> None:
        image = Image.new("RGB", (4, 3), "white")
        image.putpixel((1, 1), (255, 0, 0))
        image.putpixel((2, 1), (0, 255, 0))

        cropped = crop_image(image, (1, 1, 3, 2))

        self.assertEqual(cropped.mode, "RGBA")
        self.assertEqual(cropped.size, (2, 1))
        self.assertEqual(cropped.getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(cropped.getpixel((1, 0)), (0, 255, 0, 255))

    def test_crop_image_rejects_out_of_bounds_box(self) -> None:
        image = Image.new("RGBA", (4, 3), "white")

        with self.assertRaises(ValueError):
            crop_image(image, (0, 0, 5, 3))

    def test_crop_image_rejects_empty_box(self) -> None:
        image = Image.new("RGBA", (4, 3), "white")

        with self.assertRaises(ValueError):
            crop_image(image, (2, 1, 2, 3))


class ExtractSubjectTests(unittest.TestCase):
    def test_extract_subject_makes_background_transparent(self) -> None:
        image = Image.new("RGBA", (6, 6), (250, 250, 250, 255))
        image.putpixel((3, 3), (255, 0, 0, 255))

        result = extract_subject(image, [(1, 1), (5, 1), (5, 5), (1, 5)])

        self.assertEqual(result.image.mode, "RGBA")
        self.assertEqual(result.image.getpixel((0, 0))[3], 0)
        self.assertEqual(result.image.getpixel((3, 3)), (255, 0, 0, 255))
        self.assertGreater(result.removed_pixels, 0)

    def test_extract_subject_replaces_background_with_solid_color(self) -> None:
        image = Image.new("RGBA", (6, 6), (255, 0, 0, 255))

        result = extract_subject(
            image,
            [(1, 1), (5, 1), (5, 5), (1, 5)],
            output_mode=OUTPUT_SOLID,
            replacement_color=(10, 20, 30),
        )

        self.assertEqual(result.image.getpixel((0, 0)), (10, 20, 30, 255))
        self.assertEqual(result.image.getpixel((3, 3)), (255, 0, 0, 255))

    def test_extract_subject_rejects_empty_selection(self) -> None:
        image = Image.new("RGBA", (6, 6), "white")

        with self.assertRaises(ValueError):
            extract_subject(image, [(1, 1), (2, 2)])

    def test_extract_subject_from_mask_uses_mask_alpha(self) -> None:
        image = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
        mask = Image.new("L", (4, 4), 0)
        mask.putpixel((2, 2), 255)

        result = extract_subject_from_mask(image, mask)

        self.assertEqual(result.image.getpixel((0, 0))[3], 0)
        self.assertEqual(result.image.getpixel((2, 2)), (255, 0, 0, 255))

    def test_detect_subject_mask_finds_center_object(self) -> None:
        image = Image.new("RGBA", (30, 30), (0, 0, 255, 255))
        for y in range(10, 21):
            for x in range(10, 21):
                image.putpixel((x, y), (255, 0, 0, 255))

        mask = detect_subject_mask(image)

        self.assertEqual(mask.mode, "L")
        self.assertGreater(mask.getpixel((15, 15)), 128)
        self.assertLess(mask.getpixel((0, 0)), 128)

    def test_detect_subject_mask_fills_internal_background_colored_holes(self) -> None:
        image = Image.new("RGBA", (60, 60), (0, 0, 255, 255))
        for y in range(14, 47):
            for x in range(14, 47):
                image.putpixel((x, y), (255, 0, 0, 255))
        for y in range(27, 34):
            for x in range(27, 34):
                image.putpixel((x, y), (0, 0, 255, 255))

        mask = detect_subject_mask(image)

        self.assertGreater(mask.getpixel((30, 30)), 128)
        self.assertLess(mask.getpixel((2, 2)), 128)

    def test_detect_subject_mask_bridges_nearby_body_gap(self) -> None:
        image = Image.new("RGBA", (80, 80), (0, 0, 255, 255))
        for y in range(18, 62):
            for x in range(16, 35):
                image.putpixel((x, y), (230, 150, 105, 255))
        for y in range(18, 62):
            for x in range(43, 62):
                image.putpixel((x, y), (230, 150, 105, 255))

        mask = detect_subject_mask(image)

        self.assertGreater(mask.getpixel((39, 40)), 128)
        self.assertLess(mask.getpixel((4, 4)), 128)

    def test_detect_subject_mask_has_softened_edges(self) -> None:
        image = Image.new("RGBA", (60, 60), (0, 0, 255, 255))
        for y in range(14, 47):
            for x in range(14, 47):
                image.putpixel((x, y), (255, 0, 0, 255))

        mask = detect_subject_mask(image)
        histogram = mask.histogram()

        self.assertTrue(
            any(count for value, count in enumerate(histogram) if 0 < value < 255)
        )

    def test_detect_subject_mask_rejects_border_connected_background_leak(self) -> None:
        image = Image.new("RGBA", (90, 80), (235, 225, 205, 255))
        for y in range(18, 72):
            for x in range(28, 68):
                image.putpixel((x, y), (230, 150, 105, 255))
        for y in range(10, 70):
            for x in range(0, 18):
                image.putpixel((x, y), (235, 225, 205, 255))

        mask = detect_subject_mask(image)

        self.assertGreater(mask.getpixel((45, 45)), 128)
        self.assertLess(mask.getpixel((8, 40)), 128)

    def test_model_refinement_keeps_large_background_gaps_open(self) -> None:
        image = Image.new("RGBA", (100, 100), (235, 225, 205, 255))
        image_draw = ImageDraw.Draw(image)
        image_draw.rectangle((10, 10, 90, 90), fill=(225, 145, 105, 255))
        image_draw.rectangle((42, 20, 58, 80), fill=(235, 225, 205, 255))

        raw_mask = Image.new("L", (100, 100), 0)
        mask_draw = ImageDraw.Draw(raw_mask)
        mask_draw.rectangle((10, 10, 90, 90), fill=225)
        mask_draw.rectangle((42, 20, 58, 80), fill=45)

        refined = _refine_local_model_mask(raw_mask, image)

        self.assertGreater(refined.getpixel((25, 50)), 128)
        self.assertGreater(refined.getpixel((75, 50)), 128)
        self.assertLess(refined.getpixel((50, 50)), 128)

    def test_model_refinement_trims_weak_background_colored_fringe(self) -> None:
        image = Image.new("RGBA", (100, 100), (240, 236, 226, 255))
        image_draw = ImageDraw.Draw(image)
        image_draw.rectangle((22, 18, 82, 82), fill=(224, 142, 102, 255))
        image_draw.rectangle((16, 24, 22, 76), fill=(240, 236, 226, 255))

        raw_mask = Image.new("L", (100, 100), 0)
        mask_draw = ImageDraw.Draw(raw_mask)
        mask_draw.rectangle((22, 18, 82, 82), fill=225)
        mask_draw.rectangle((16, 24, 22, 76), fill=118)

        refined = _refine_local_model_mask(raw_mask, image)

        self.assertGreater(refined.getpixel((50, 50)), 128)
        self.assertLess(refined.getpixel((17, 50)), 64)

    def test_paint_subject_mask_adds_and_removes_regions(self) -> None:
        mask = Image.new("L", (10, 10), 0)

        mask = paint_subject_mask(mask, [(5, 5)], radius=2, value=255)
        self.assertEqual(mask.getpixel((5, 5)), 255)

        mask = paint_subject_mask(mask, [(5, 5)], radius=1, value=0)
        self.assertEqual(mask.getpixel((5, 5)), 0)


if __name__ == "__main__":
    unittest.main()
