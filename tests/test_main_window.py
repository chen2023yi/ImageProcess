from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("IMAGEPRO_DISABLE_REMBG", "1")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from image_processor import ProcessResult  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402


def get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class MainWindowCropExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = get_app()
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def wait_until(self, predicate, timeout: float = 5.0) -> bool:  # noqa: ANN001
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        self.app.processEvents()
        return bool(predicate())

    def mouse_event(
        self,
        event_type: QEvent.Type,
        position: QPoint,
        button: Qt.MouseButton,
        buttons: Qt.MouseButton,
    ) -> QMouseEvent:
        local_position = QPointF(position)
        return QMouseEvent(
            event_type,
            local_position,
            local_position,
            button,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )

    def test_cropped_image_can_be_exported_without_background_processing(self) -> None:
        image = Image.new("RGBA", (8, 6), "white")
        self.window.source_path = Path("sample.jpg")
        self.window.loaded_image = image.copy()
        self.window.original_image = image.copy()
        self.window.crop_box = (2, 1, 7, 5)

        self.window.crop_current_image()

        export_image = self.window._export_image()
        self.assertIsNotNone(export_image)
        self.assertEqual(export_image.size, (5, 4))
        self.assertIsNone(self.window.processed_result)
        self.assertTrue(self.window.export_button.isEnabled())
        self.assertEqual(self.window._default_export_name(), "sample_cropped.png")

    def test_resetting_cropped_image_disables_direct_export(self) -> None:
        image = Image.new("RGBA", (8, 6), "white")
        self.window.loaded_image = image.copy()
        self.window.original_image = image.copy()
        self.window.crop_box = (2, 1, 7, 5)

        self.window.crop_current_image()
        self.window.reset_cropped_image()

        self.assertIsNone(self.window._export_image())
        self.assertFalse(self.window.export_button.isEnabled())

    def test_subject_selection_can_be_exported_without_color_detection(self) -> None:
        image = Image.new("RGBA", (8, 8), "white")
        image.putpixel((4, 4), (255, 0, 0, 255))
        mask = Image.new("L", (8, 8), 0)
        for y in range(2, 7):
            for x in range(2, 7):
                mask.putpixel((x, y), 255)
        self.window.source_path = Path("portrait.jpg")
        self.window.loaded_image = image.copy()
        self.window.original_image = image.copy()
        self.window.subject_mask = mask

        self.window.apply_subject_selection()

        export_image = self.window._export_image()
        self.assertIsNotNone(export_image)
        self.assertEqual(export_image.getpixel((0, 0))[3], 0)
        self.assertEqual(export_image.getpixel((4, 4)), (255, 0, 0, 255))
        self.assertEqual(self.window._default_export_name(), "portrait_subject.png")
        self.assertTrue(self.window.export_button.isEnabled())

    def test_background_parameter_change_keeps_subject_result(self) -> None:
        image = Image.new("RGBA", (8, 8), "white")
        image.putpixel((4, 4), (255, 0, 0, 255))
        mask = Image.new("L", (8, 8), 0)
        for y in range(2, 7):
            for x in range(2, 7):
                mask.putpixel((x, y), 255)
        self.window.loaded_image = image.copy()
        self.window.original_image = image.copy()
        self.window.subject_mask = mask

        self.window.apply_subject_selection()
        self.window.tolerance_slider.setValue(91)

        self.assertEqual(self.window.processed_result_source, "subject")
        self.assertIsNotNone(self.window._export_image())
        self.assertTrue(self.window.export_button.isEnabled())
        self.assertTrue(self.window.process_button.isEnabled())
        self.assertIn("fills the background", self.window.mode_help_label.text())

    def test_picking_background_color_keeps_subject_background_apply_available(self) -> None:
        image = Image.new("RGBA", (8, 8), "white")
        image.putpixel((4, 4), (255, 0, 0, 255))
        mask = Image.new("L", (8, 8), 0)
        for y in range(2, 7):
            for x in range(2, 7):
                mask.putpixel((x, y), 255)
        self.window.loaded_image = image.copy()
        self.window.original_image = image.copy()
        self.window.subject_mask = mask

        self.window.apply_subject_selection()
        self.window._apply_background_color_choice((0, 255, 0))

        self.assertEqual(self.window.processed_result_source, "subject")
        self.assertIsNotNone(self.window._export_image())
        self.assertTrue(self.window.process_button.isEnabled())
        self.assertIn("Subject background", self.window.color_label.text())

    def test_apply_background_uses_active_subject_mask(self) -> None:
        image = Image.new("RGBA", (8, 8), "white")
        image.putpixel((4, 4), (255, 0, 0, 255))
        mask = Image.new("L", (8, 8), 0)
        for y in range(2, 7):
            for x in range(2, 7):
                mask.putpixel((x, y), 255)
        self.window.loaded_image = image.copy()
        self.window.original_image = image.copy()
        self.window.subject_mask = mask
        errors: list[tuple[str, str]] = []
        self.window._show_error = lambda title, message: errors.append((title, message))

        self.window.apply_subject_selection()
        self.window._apply_background_color_choice((10, 20, 30))
        self.window.process_current_image()
        export_image = self.window._export_image()

        self.assertEqual(self.window.processed_result_source, "subject")
        self.assertEqual(errors, [])
        self.assertIsNotNone(export_image)
        self.assertEqual(export_image.getpixel((0, 0)), (10, 20, 30, 255))
        self.assertEqual(export_image.getpixel((4, 4)), (255, 0, 0, 255))
        self.assertIn("Subject mask background", self.window.result_meta_label.text())
        self.assertIn("Solid #0A141E", self.window.result_meta_label.text())
        self.assertTrue(self.window.process_button.isEnabled())

        self.window.clear_subject_mask()
        self.assertTrue(self.window.process_button.isEnabled())
        self.assertNotIn("fills the background", self.window.mode_help_label.text())

    def test_clearing_subject_selection_discards_subject_result(self) -> None:
        image = Image.new("RGBA", (8, 8), "white")
        mask = Image.new("L", (8, 8), 255)
        self.window.original_image = image.copy()
        self.window.subject_mask = mask

        self.window.apply_subject_selection()
        self.window.clear_subject_mask()

        self.assertIsNone(self.window.processed_result)
        self.assertIsNone(self.window._export_image())
        self.assertFalse(self.window.export_button.isEnabled())

    def test_id_photo_crop_resizes_processed_result_to_one_inch(self) -> None:
        image = Image.new("RGBA", (80, 120), (0, 0, 255, 255))
        self.window.source_path = Path("portrait.jpg")
        self.window.processed_result = ProcessResult(image=image, removed_pixels=0)
        self.window.processed_result_source = "background"
        self.window._set_processed_preview(image)

        self.window.toggle_photo_crop_selection()
        self.assertIsNotNone(self.window.photo_crop_box)
        left, top, right, bottom = self.window.photo_crop_box
        self.assertAlmostEqual(
            (right - left) / (bottom - top),
            295 / 413,
            delta=0.02,
        )

        self.window.apply_photo_crop()
        export_image = self.window._export_image()

        self.assertIsNotNone(export_image)
        self.assertEqual(export_image.size, (295, 413))
        self.assertEqual(self.window.processed_result_source, "photo")
        self.assertEqual(self.window._default_export_name(), "portrait_photo.png")
        self.assertIn("Photo output: 295 x 413px", self.window.result_meta_label.text())

    def test_apply_background_after_id_photo_crop_preserves_photo_size(self) -> None:
        image = Image.new("RGBA", (20, 20), "white")
        for y in range(7, 14):
            for x in range(7, 14):
                image.putpixel((x, y), (255, 0, 0, 255))
        mask = Image.new("L", (20, 20), 0)
        for y in range(7, 14):
            for x in range(7, 14):
                mask.putpixel((x, y), 255)

        self.window.source_path = Path("portrait.jpg")
        self.window.loaded_image = image.copy()
        self.window.original_image = image.copy()
        self.window.subject_mask = mask

        self.window.apply_subject_selection()
        self.window.toggle_photo_crop_selection()
        self.window.processed_canvas.set_crop_box((3, 0, 17, 20), notify=True)
        self.window.apply_photo_crop()
        photo_image = self.window._export_image()
        self.assertIsNotNone(photo_image)
        self.assertEqual(photo_image.size, (295, 413))
        self.assertEqual(self.window.processed_result_source, "photo")
        self.assertIsNotNone(self.window.photo_subject_mask)

        self.window._apply_background_color_choice((10, 20, 30))
        self.window.process_current_image()
        export_image = self.window._export_image()

        self.assertIsNotNone(export_image)
        self.assertEqual(export_image.size, (295, 413))
        self.assertEqual(self.window.processed_result_source, "photo")
        self.assertEqual(self.window._default_export_name(), "portrait_photo.png")
        self.assertEqual(export_image.getpixel((0, 0)), (10, 20, 30, 255))
        self.assertEqual(export_image.getpixel((147, 206)), (255, 0, 0, 255))
        self.assertIn("ID photo subject background", self.window.result_meta_label.text())

    def test_id_photo_crop_starts_inside_zoomed_visible_area(self) -> None:
        image = Image.new("RGBA", (400, 400), (0, 0, 255, 255))
        self.window.processed_result = ProcessResult(image=image, removed_pixels=0)
        self.window.processed_result_source = "background"
        self.window.processed_canvas.resize(460, 520)
        self.window._set_processed_preview(image)
        self.window.processed_canvas._zoom_at(QPoint(230, 260), 4)
        self.window.processed_canvas._pan_zoom_by(QPoint(100, 0))
        visible_box = self.window.processed_canvas.visible_image_box()

        self.window.toggle_photo_crop_selection()
        photo_box = self.window.photo_crop_box

        self.assertIsNotNone(visible_box)
        self.assertIsNotNone(photo_box)
        visible_left, visible_top, visible_right, visible_bottom = visible_box
        photo_left, photo_top, photo_right, photo_bottom = photo_box
        self.assertGreaterEqual(photo_left, visible_left)
        self.assertGreaterEqual(photo_top, visible_top)
        self.assertLessEqual(photo_right, visible_right)
        self.assertLessEqual(photo_bottom, visible_bottom)

    def test_id_photo_crop_frame_stays_fixed_when_preview_zoom_changes(self) -> None:
        image = Image.new("RGBA", (400, 400), (0, 0, 255, 255))
        self.window.processed_result = ProcessResult(image=image, removed_pixels=0)
        self.window.processed_result_source = "background"
        self.window.processed_canvas.resize(460, 520)
        self.window._set_processed_preview(image)
        self.window.toggle_photo_crop_selection()
        initial_box = self.window.photo_crop_box
        initial_frame = self.window.processed_canvas._current_crop_widget_rect()

        self.window.processed_canvas._zoom_at(QPoint(230, 260), 3)
        self.window.processed_canvas._pan_zoom_by(QPoint(40, 30))
        zoomed_box = self.window.photo_crop_box
        zoomed_frame = self.window.processed_canvas._current_crop_widget_rect()
        self.window.reset_processed_zoom()
        reset_box = self.window.photo_crop_box
        reset_frame = self.window.processed_canvas._current_crop_widget_rect()

        self.assertIsNotNone(initial_box)
        self.assertEqual(zoomed_frame, initial_frame)
        self.assertNotEqual(zoomed_box, initial_box)
        self.assertEqual(reset_frame, initial_frame)
        self.assertEqual(reset_box, initial_box)

    def test_processed_preview_uses_right_drag_for_zoom_pan_with_crop_box(
        self,
    ) -> None:
        image = Image.new("RGBA", (400, 400), (0, 0, 255, 255))
        self.window.processed_result = ProcessResult(image=image, removed_pixels=0)
        self.window.processed_result_source = "background"
        self.window.processed_canvas.resize(460, 520)
        self.window._set_processed_preview(image)
        self.window.toggle_photo_crop_selection()
        self.window.processed_canvas._zoom_at(QPoint(230, 260), 3)

        canvas = self.window.processed_canvas
        initial_pan = QPoint(canvas._zoom_pan)
        left_start = canvas._current_crop_widget_rect().center()
        left_end = left_start + QPoint(20, 12)

        canvas.mousePressEvent(
            self.mouse_event(
                QEvent.Type.MouseButtonPress,
                left_start,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
            )
        )
        canvas.mouseMoveEvent(
            self.mouse_event(
                QEvent.Type.MouseMove,
                left_end,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
            )
        )
        canvas.mouseReleaseEvent(
            self.mouse_event(
                QEvent.Type.MouseButtonRelease,
                left_end,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
            )
        )

        self.assertEqual(canvas._zoom_pan, initial_pan)

        right_start = canvas._current_crop_widget_rect().center()
        right_end = right_start + QPoint(36, 24)
        canvas.mousePressEvent(
            self.mouse_event(
                QEvent.Type.MouseButtonPress,
                right_start,
                Qt.MouseButton.RightButton,
                Qt.MouseButton.RightButton,
            )
        )
        canvas.mouseMoveEvent(
            self.mouse_event(
                QEvent.Type.MouseMove,
                right_end,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.RightButton,
            )
        )
        canvas.mouseReleaseEvent(
            self.mouse_event(
                QEvent.Type.MouseButtonRelease,
                right_end,
                Qt.MouseButton.RightButton,
                Qt.MouseButton.NoButton,
            )
        )

        self.assertNotEqual(canvas._zoom_pan, initial_pan)
        self.assertFalse(canvas._zoom_pan_active)

    def test_custom_id_photo_crop_uses_custom_dimensions(self) -> None:
        image = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        self.window.processed_result = ProcessResult(image=image, removed_pixels=0)
        self.window.processed_result_source = "background"
        self.window._set_processed_preview(image)
        custom_index = self.window.photo_size_combo.findData("custom")
        self.window.photo_size_combo.setCurrentIndex(custom_index)
        self.window.photo_width_input.setValue(600)
        self.window.photo_height_input.setValue(800)

        self.window.toggle_photo_crop_selection()
        self.window.processed_canvas.set_crop_box((10, 0, 70, 80), notify=True)
        self.window.apply_photo_crop()
        export_image = self.window._export_image()

        self.assertIsNotNone(export_image)
        self.assertEqual(export_image.size, (600, 800))
        self.assertEqual(self.window.processed_result_source, "photo")
        self.assertIn("Photo output: 600 x 800px", self.window.result_meta_label.text())

    def test_processed_preview_zoom_pan_and_reset_do_not_resize_export_image(self) -> None:
        image = Image.new("RGBA", (400, 400), (0, 0, 255, 255))
        self.window.processed_result = ProcessResult(image=image, removed_pixels=0)
        self.window.processed_result_source = "background"
        self.window.processed_canvas.resize(460, 520)
        self.window._set_processed_preview(image)

        self.window.processed_canvas._zoom_at(QPoint(230, 260), 3)
        self.window.processed_canvas._pan_zoom_by(QPoint(40, 30))
        export_image = self.window._export_image()

        self.assertGreater(self.window.processed_canvas._zoom_factor, 1.0)
        self.assertNotEqual(self.window.processed_canvas._zoom_pan, QPoint(0, 0))
        self.assertTrue(self.window.reset_zoom_button.isEnabled())
        self.assertIsNotNone(export_image)
        self.assertEqual(export_image.size, (400, 400))

        self.window.reset_processed_zoom()
        self.assertEqual(self.window.processed_canvas._zoom_factor, 1.0)
        self.assertEqual(self.window.processed_canvas._zoom_pan, QPoint(0, 0))
        self.assertFalse(self.window.reset_zoom_button.isEnabled())

    def test_detect_subject_creates_mask_and_result(self) -> None:
        image = Image.new("RGBA", (20, 20), (0, 0, 255, 255))
        for y in range(6, 15):
            for x in range(6, 15):
                image.putpixel((x, y), (255, 0, 0, 255))
        self.window.source_path = Path("simple.png")
        self.window.loaded_image = image.copy()
        self.window.original_image = image.copy()

        self.window.detect_subject()

        self.assertTrue(
            self.wait_until(lambda: self.window.processed_result is not None)
        )
        self.assertIsNotNone(self.window.subject_mask)
        self.assertIsNotNone(self.window.processed_result)
        self.assertTrue(self.window.subject_apply_button.isEnabled())
        self.assertTrue(self.window.subject_refine_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
