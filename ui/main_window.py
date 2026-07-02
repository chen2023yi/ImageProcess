from __future__ import annotations

from collections.abc import Callable
import time
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from image_processor import (
    DEFAULT_LIGHTNESS_THRESHOLD,
    DEFAULT_NEUTRAL_CHROMA_THRESHOLD,
    DEFAULT_REPLACEMENT_COLOR,
    DEFAULT_TOLERANCE,
    MODE_COLOR,
    MODE_LIGHT_NEUTRAL,
    OUTPUT_SOLID,
    OUTPUT_TRANSPARENT,
    PURE_GREEN,
    ProcessResult,
    crop_and_resize_image,
    crop_and_resize_mask,
    crop_image,
    detect_subject_mask,
    extract_subject_from_mask,
    is_supported_image,
    load_image,
    paint_subject_mask,
    remove_background,
    save_png,
)


PHOTO_SIZE_1_INCH = (295, 413)
PHOTO_SIZE_2_INCH = (413, 579)
PHOTO_SIZE_CUSTOM = "custom"


class ImageTaskThread(QThread):
    task_finished = Signal(object, float)
    task_failed = Signal(str)

    def __init__(self, task: Callable[[], object], parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._task = task

    def run(self) -> None:
        start = time.perf_counter()
        try:
            result = self._task()
        except Exception as exc:  # noqa: BLE001
            self.task_failed.emit(str(exc))
            return
        self.task_finished.emit(result, time.perf_counter() - start)


class ColorSwatch(QLabel):
    clicked = Signal()

    def __init__(self, tooltip: str) -> None:
        super().__init__()
        self._click_enabled = True
        self.setObjectName("ColorSwatch")
        self.setFixedSize(36, 36)
        self.setToolTip(tooltip)
        self.set_click_enabled(True)

    def set_click_enabled(self, enabled: bool) -> None:
        self._click_enabled = enabled
        self.setEnabled(enabled)
        if enabled:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if self._click_enabled and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class ScrollFriendlyComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # noqa: ANN001
        event.ignore()


class ScrollFriendlySlider(QSlider):
    def wheelEvent(self, event) -> None:  # noqa: ANN001
        event.ignore()


class ImageCanvas(QWidget):
    image_pixel_selected = Signal(int, int)
    crop_selection_changed = Signal(int, int, int, int)
    crop_selection_cleared = Signal()
    subject_selection_changed = Signal(object)
    subject_selection_cleared = Signal()
    zoom_changed = Signal(float)

    def __init__(self, empty_text: str, checkerboard: bool = False) -> None:
        super().__init__()
        self.empty_text = empty_text
        self.background_style = "checkerboard" if checkerboard else "white"
        self._pixmap: QPixmap | None = None
        self._image_rect = QRect()
        self._pick_enabled = False
        self._crop_enabled = False
        self._crop_anchor: QPoint | None = None
        self._crop_resize_handle: str | None = None
        self._crop_move_offset: QPoint | None = None
        self._active_crop_rect = QRect()
        self._crop_box: tuple[int, int, int, int] | None = None
        self._crop_fixed_to_view = False
        self._crop_view_rect = QRect()
        self._crop_handle_size = 12
        self._crop_aspect_ratio: float | None = None
        self._zoom_enabled = False
        self._zoom_factor = 1.0
        self._zoom_pan = QPoint(0, 0)
        self._zoom_pan_active = False
        self._zoom_pan_last_position: QPoint | None = None
        self._min_zoom = 1.0
        self._max_zoom = 6.0
        self._subject_enabled = False
        self._subject_mask: Image.Image | None = None
        self._subject_brush_value = 255
        self._subject_brush_radius = 24
        self._subject_brush_active = False
        self._subject_brush_last_point: tuple[int, int] | None = None
        self._subject_brush_preview_point: QPoint | None = None
        self.setMinimumSize(320, 360)
        self.setMouseTracking(True)

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self._image_rect = QRect()
        self._crop_anchor = None
        self._crop_resize_handle = None
        self._crop_move_offset = None
        self._active_crop_rect = QRect()
        self._crop_box = None
        self._crop_view_rect = QRect()
        self._zoom_factor = 1.0
        self._zoom_pan = QPoint(0, 0)
        self._zoom_pan_active = False
        self._zoom_pan_last_position = None
        self._subject_enabled = False
        self._subject_mask = None
        self._subject_brush_active = False
        self._subject_brush_last_point = None
        self._subject_brush_preview_point = None
        self.update()
        self.zoom_changed.emit(self._zoom_factor)

    def set_pick_enabled(self, enabled: bool) -> None:
        self._pick_enabled = enabled
        if enabled:
            self._crop_enabled = False
            self._subject_enabled = False
            self._zoom_pan_active = False
            self._zoom_pan_last_position = None
            self._crop_anchor = None
            self._crop_resize_handle = None
            self._crop_move_offset = None
            self._active_crop_rect = QRect()
        self._refresh_cursor()

    def set_crop_enabled(self, enabled: bool) -> None:
        self._crop_enabled = enabled
        if enabled:
            self._pick_enabled = False
            self._subject_enabled = False
            self._zoom_pan_active = False
            self._zoom_pan_last_position = None
        else:
            self._crop_anchor = None
            self._crop_resize_handle = None
            self._crop_move_offset = None
            self._active_crop_rect = QRect()
        self._refresh_cursor()
        self.update()

    def is_crop_enabled(self) -> bool:
        return self._crop_enabled

    def set_subject_enabled(self, enabled: bool) -> None:
        self._subject_enabled = enabled
        if enabled:
            self._pick_enabled = False
            self._crop_enabled = False
            self._zoom_pan_active = False
            self._zoom_pan_last_position = None
            self._crop_anchor = None
            self._crop_resize_handle = None
            self._crop_move_offset = None
            self._active_crop_rect = QRect()
        else:
            self._subject_brush_active = False
            self._subject_brush_last_point = None
            self._subject_brush_preview_point = None
        self._refresh_cursor()
        self.update()

    def is_subject_enabled(self) -> bool:
        return self._subject_enabled

    def clear_crop_selection(self, notify: bool = False) -> None:
        had_selection = self._crop_box is not None or not self._active_crop_rect.isNull()
        self._crop_anchor = None
        self._crop_resize_handle = None
        self._crop_move_offset = None
        self._active_crop_rect = QRect()
        self._crop_box = None
        self._crop_view_rect = QRect()
        self.update()
        if notify and had_selection:
            self.crop_selection_cleared.emit()

    def set_crop_fixed_to_view(self, enabled: bool) -> None:
        self._crop_fixed_to_view = enabled
        if not enabled:
            self._crop_view_rect = QRect()

    def set_crop_box(
        self,
        box: tuple[int, int, int, int] | None,
        notify: bool = False,
    ) -> None:
        self._crop_anchor = None
        self._crop_resize_handle = None
        self._crop_move_offset = None
        self._active_crop_rect = QRect()
        self._crop_box = box
        if self._crop_fixed_to_view:
            self._crop_view_rect = (
                self._image_box_to_widget_rect(box) if box is not None else QRect()
            )
            mapped_box = (
                self._widget_rect_to_image_box(self._crop_view_rect)
                if box is not None
                else None
            )
            if mapped_box is not None:
                self._crop_box = mapped_box
        self.update()
        if notify:
            if box is None:
                self.crop_selection_cleared.emit()
            else:
                self.crop_selection_changed.emit(*self._crop_box)

    def current_crop_image_box(self) -> tuple[int, int, int, int] | None:
        if self._crop_fixed_to_view:
            if self._crop_view_rect.isNull():
                return None
            return self._widget_rect_to_image_box(self._crop_view_rect)
        return self._crop_box

    def set_crop_aspect_ratio(self, aspect_ratio: float | None) -> None:
        if aspect_ratio is None or aspect_ratio <= 0:
            self._crop_aspect_ratio = None
        else:
            self._crop_aspect_ratio = float(aspect_ratio)

    def visible_image_box(self) -> tuple[int, int, int, int] | None:
        if not self._pixmap or self._pixmap.isNull():
            return None

        image_rect = self._current_image_rect()

        viewport = self.rect().adjusted(24, 24, -24, -24)
        if viewport.width() <= 0 or viewport.height() <= 0:
            viewport = self.rect()

        visible_rect = image_rect.intersected(viewport)
        if visible_rect.width() <= 0 or visible_rect.height() <= 0:
            return None

        return self._widget_rect_to_image_box_for_rect(visible_rect, image_rect)

    def clear_subject_selection(self, notify: bool = False) -> None:
        had_selection = self._subject_mask is not None
        self._subject_brush_active = False
        self._subject_brush_last_point = None
        self._subject_brush_preview_point = None
        self._subject_mask = None
        self.update()
        if notify and had_selection:
            self.subject_selection_cleared.emit()

    def set_subject_mask(self, mask: Image.Image | None) -> None:
        self._subject_mask = mask.convert("L").copy() if mask is not None else None
        self._subject_brush_active = False
        self._subject_brush_last_point = None
        self.update()

    def set_subject_brush_value(self, value: int) -> None:
        self._subject_brush_value = 255 if int(value) >= 128 else 0

    def set_subject_brush_radius(self, radius: int) -> None:
        self._subject_brush_radius = max(1, int(radius))
        self.update()

    def set_background_style(self, style: str) -> None:
        self.background_style = style
        self.update()

    def set_zoom_enabled(self, enabled: bool) -> None:
        self._zoom_enabled = enabled
        if not enabled:
            self.reset_zoom()

    def reset_zoom(self) -> None:
        self._zoom_factor = 1.0
        self._zoom_pan = QPoint(0, 0)
        self._zoom_pan_active = False
        self._zoom_pan_last_position = None
        self._sync_fixed_crop_box()
        self.update()
        self.zoom_changed.emit(self._zoom_factor)

    def sizeHint(self) -> QSize:
        return QSize(460, 520)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self.background_style == "checkerboard":
            self._paint_checkerboard(painter)
        else:
            painter.fillRect(self.rect(), self._background_color_for_style())

        painter.setPen(QPen(Qt.GlobalColor.transparent))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)

        if not self._pixmap or self._pixmap.isNull():
            painter.setPen("#667085")
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return

        self._zoom_pan = self._clamped_zoom_pan_offset(
            self._zoom_factor,
            self._zoom_pan,
        )
        self._image_rect = self._image_rect_for_zoom(
            self._zoom_factor,
            self._zoom_pan,
        )
        scaled = self._pixmap.scaled(
            self._image_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(self._image_rect.topLeft(), scaled)
        self._paint_crop_selection(painter)
        self._paint_subject_selection(painter)

    def wheelEvent(self, event) -> None:  # noqa: ANN001
        if (
            not self._zoom_enabled
            or not self._pixmap
            or self._pixmap.isNull()
            or event.angleDelta().y() == 0
        ):
            super().wheelEvent(event)
            return

        self._zoom_at(event.position().toPoint(), event.angleDelta().y() / 120)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if self._subject_enabled:
            if (
                event.button() != Qt.MouseButton.LeftButton
                or not self._pixmap
                or self._pixmap.isNull()
                or self._image_rect.isNull()
                or self._subject_mask is None
            ):
                super().mousePressEvent(event)
                return

            position = event.position().toPoint()
            if not self._image_rect.contains(position):
                super().mousePressEvent(event)
                return

            self._subject_brush_active = True
            self._subject_brush_preview_point = position
            self._paint_subject_brush(position)
            event.accept()
            return

        if (
            self._can_pan_zoom()
            and event.button() == Qt.MouseButton.RightButton
            and self._pan_bounds_rect().contains(event.position().toPoint())
        ):
            self._zoom_pan_active = True
            self._zoom_pan_last_position = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self._crop_enabled:
            bounds_rect = self._crop_bounds_rect()
            if (
                event.button() != Qt.MouseButton.LeftButton
                or not self._pixmap
                or self._pixmap.isNull()
                or bounds_rect.isNull()
            ):
                super().mousePressEvent(event)
                return

            position = event.position().toPoint()
            handle = self._crop_handle_at(position)
            if handle is not None:
                crop_rect = self._current_crop_widget_rect()
                self._crop_resize_handle = handle
                self._crop_anchor = self._opposite_crop_corner(crop_rect, handle)
                self._crop_move_offset = None
                self._active_crop_rect = self._constrained_crop_rect(
                    self._crop_anchor,
                    self._clamp_to_crop_bounds(position),
                )
                self._active_crop_rect = self._fit_rect_in_crop_bounds(
                    self._active_crop_rect
                )
                self.update()
                event.accept()
                return

            crop_rect = self._current_crop_widget_rect().intersected(bounds_rect)
            if not crop_rect.isNull() and crop_rect.contains(position):
                self._crop_resize_handle = None
                self._crop_anchor = None
                self._crop_move_offset = position - crop_rect.topLeft()
                self._active_crop_rect = crop_rect
                self.update()
                event.accept()
                return

            if self._crop_fixed_to_view:
                event.accept()
                return

            if not bounds_rect.contains(position):
                self.clear_crop_selection(notify=True)
                event.accept()
                return

            self._crop_resize_handle = None
            self._crop_move_offset = None
            self._crop_anchor = self._clamp_to_crop_bounds(position)
            self._active_crop_rect = QRect(self._crop_anchor, self._crop_anchor)
            self._crop_box = None
            self._crop_view_rect = QRect()
            self.update()
            event.accept()
            return

        if (
            not self._pick_enabled
            or not self._pixmap
            or self._pixmap.isNull()
            or self._image_rect.isNull()
        ):
            super().mousePressEvent(event)
            return

        position = event.position().toPoint()
        if not self._image_rect.contains(position):
            super().mousePressEvent(event)
            return

        image_x = int(
            (position.x() - self._image_rect.x())
            * self._pixmap.width()
            / self._image_rect.width()
        )
        image_y = int(
            (position.y() - self._image_rect.y())
            * self._pixmap.height()
            / self._image_rect.height()
        )
        image_x = max(0, min(self._pixmap.width() - 1, image_x))
        image_y = max(0, min(self._pixmap.height() - 1, image_y))
        self.image_pixel_selected.emit(image_x, image_y)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._subject_enabled:
            position = event.position().toPoint()
            self._subject_brush_preview_point = position
            if self._subject_brush_active and self._image_rect.contains(position):
                self._paint_subject_brush(position)
            else:
                self.update()
            event.accept()
            return

        if self._crop_enabled and self._crop_anchor is not None:
            current = self._clamp_to_crop_bounds(event.position().toPoint())
            self._active_crop_rect = self._constrained_crop_rect(
                self._crop_anchor,
                current,
            )
            self._active_crop_rect = self._fit_rect_in_crop_bounds(
                self._active_crop_rect
            )
            self.update()
            event.accept()
            return
        if self._crop_enabled and self._crop_move_offset is not None:
            crop_rect = self._current_crop_widget_rect()
            top_left = event.position().toPoint() - self._crop_move_offset
            crop_rect.moveTopLeft(top_left)
            self._active_crop_rect = self._fit_rect_in_crop_bounds(crop_rect)
            self.update()
            event.accept()
            return
        if self._zoom_pan_active and self._zoom_pan_last_position is not None:
            position = event.position().toPoint()
            delta = position - self._zoom_pan_last_position
            self._pan_zoom_by(delta)
            self._zoom_pan_last_position = position
            event.accept()
            return
        if self._crop_enabled:
            self._refresh_cursor(event.position().toPoint())
            event.accept()
            return
        if self._can_pan_zoom():
            self._refresh_cursor(event.position().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._subject_enabled and self._subject_brush_active:
            self._subject_brush_active = False
            self._subject_brush_last_point = None
            if self._subject_mask is not None:
                self.subject_selection_changed.emit(self._subject_mask.copy())
            self.update()
            event.accept()
            return

        if self._crop_enabled and self._crop_anchor is not None:
            current = self._clamp_to_crop_bounds(event.position().toPoint())
            selected_rect = self._constrained_crop_rect(self._crop_anchor, current)
            selected_rect = self._fit_rect_in_crop_bounds(selected_rect)
            was_resizing = self._crop_resize_handle is not None
            self._crop_anchor = None
            self._crop_resize_handle = None
            self._crop_move_offset = None

            crop_box = self._widget_rect_to_image_box(selected_rect)
            if crop_box is None:
                if was_resizing and self._crop_box is not None:
                    self._active_crop_rect = QRect()
                    self.update()
                else:
                    self.clear_crop_selection(notify=True)
            else:
                self._crop_box = crop_box
                if self._crop_fixed_to_view:
                    self._crop_view_rect = selected_rect
                self._active_crop_rect = QRect()
                self.crop_selection_changed.emit(*crop_box)
                self.update()

            self._refresh_cursor(event.position().toPoint())
            event.accept()
            return
        if self._crop_enabled and self._crop_move_offset is not None:
            selected_rect = self._fit_rect_in_crop_bounds(self._active_crop_rect)
            self._crop_move_offset = None
            self._active_crop_rect = QRect()

            crop_box = self._widget_rect_to_image_box(selected_rect)
            if crop_box is None:
                self.clear_crop_selection(notify=True)
            else:
                self._crop_box = crop_box
                if self._crop_fixed_to_view:
                    self._crop_view_rect = selected_rect
                self.crop_selection_changed.emit(*crop_box)
                self.update()

            self._refresh_cursor(event.position().toPoint())
            event.accept()
            return
        if self._zoom_pan_active:
            self._zoom_pan_active = False
            self._zoom_pan_last_position = None
            self._refresh_cursor(event.position().toPoint())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _paint_checkerboard(self, painter: QPainter) -> None:
        cell_size = 16
        light = "#F8FAFC"
        dark = "#DDE3EA"

        for y in range(0, self.height(), cell_size):
            for x in range(0, self.width(), cell_size):
                row = y // cell_size
                column = x // cell_size
                painter.fillRect(
                    QRect(x, y, cell_size, cell_size),
                    light if (row + column) % 2 == 0 else dark,
                )

    def _paint_crop_selection(self, painter: QPainter) -> None:
        crop_rect = self._current_crop_widget_rect()
        if crop_rect.isNull() or self._image_rect.isNull():
            return

        bounds_rect = self._crop_bounds_rect()
        crop_rect = crop_rect.intersected(bounds_rect)
        if crop_rect.width() <= 1 or crop_rect.height() <= 1:
            return

        shade = QColor(15, 23, 42, 110)

        self._fill_if_positive(
            painter,
            QRect(
                bounds_rect.left(),
                bounds_rect.top(),
                bounds_rect.width(),
                crop_rect.top() - bounds_rect.top(),
            ),
            shade,
        )
        self._fill_if_positive(
            painter,
            QRect(
                bounds_rect.left(),
                crop_rect.bottom() + 1,
                bounds_rect.width(),
                bounds_rect.bottom() - crop_rect.bottom(),
            ),
            shade,
        )
        self._fill_if_positive(
            painter,
            QRect(
                bounds_rect.left(),
                crop_rect.top(),
                crop_rect.left() - bounds_rect.left(),
                crop_rect.height(),
            ),
            shade,
        )
        self._fill_if_positive(
            painter,
            QRect(
                crop_rect.right() + 1,
                crop_rect.top(),
                bounds_rect.right() - crop_rect.right(),
                crop_rect.height(),
            ),
            shade,
        )

        painter.setPen(QPen(QColor("#2563EB"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(crop_rect.adjusted(0, 0, -1, -1))
        self._paint_crop_handles(painter, crop_rect)

    def _paint_crop_handles(self, painter: QPainter, crop_rect: QRect) -> None:
        painter.setPen(QPen(QColor("#1D4ED8"), 2))
        painter.setBrush(QColor("#FFFFFF"))
        for handle_rect in self._crop_handle_rects(crop_rect).values():
            painter.drawRect(handle_rect)

    def _paint_subject_selection(self, painter: QPainter) -> None:
        if self._subject_mask is None or self._image_rect.isNull():
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        overlay = self._subject_mask.resize(
            (self._image_rect.width(), self._image_rect.height()),
            Image.Resampling.NEAREST,
        )
        alpha = overlay.point(lambda value: 72 if value >= 128 else 0)
        rgba_overlay = Image.new("RGBA", overlay.size, (16, 185, 129, 0))
        rgba_overlay.putalpha(alpha)
        painter.drawPixmap(self._image_rect.topLeft(), self._pil_to_pixmap(rgba_overlay))

        if self._subject_enabled and self._subject_brush_preview_point is not None:
            widget_radius = max(
                4,
                round(
                    self._subject_brush_radius
                    * self._image_rect.width()
                    / max(1, self._pixmap.width())
                ),
            )
            color = QColor("#059669") if self._subject_brush_value >= 128 else QColor("#DC2626")
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                self._subject_brush_preview_point,
                widget_radius,
                widget_radius,
            )

    def _current_crop_widget_rect(self) -> QRect:
        if not self._active_crop_rect.isNull():
            return self._active_crop_rect
        if self._crop_fixed_to_view and not self._crop_view_rect.isNull():
            return self._crop_view_rect
        if self._crop_box is not None:
            return self._image_box_to_widget_rect(self._crop_box)
        return QRect()

    def _crop_handle_at(self, point: QPoint) -> str | None:
        crop_rect = self._current_crop_widget_rect()
        bounds_rect = self._crop_bounds_rect()
        if crop_rect.isNull() or bounds_rect.isNull():
            return None

        crop_rect = crop_rect.intersected(bounds_rect)
        for handle, handle_rect in self._crop_handle_rects(crop_rect).items():
            if handle_rect.contains(point):
                return handle
        return None

    def _crop_handle_rects(self, crop_rect: QRect) -> dict[str, QRect]:
        half = self._crop_handle_size // 2
        corners = {
            "top_left": crop_rect.topLeft(),
            "top_right": crop_rect.topRight(),
            "bottom_left": crop_rect.bottomLeft(),
            "bottom_right": crop_rect.bottomRight(),
        }
        return {
            handle: QRect(
                corner.x() - half,
                corner.y() - half,
                self._crop_handle_size,
                self._crop_handle_size,
            )
            for handle, corner in corners.items()
        }

    def _zoom_at(self, position: QPoint, wheel_steps: float) -> None:
        if not self._zoom_enabled or not self._pixmap or self._pixmap.isNull():
            return

        old_zoom = self._zoom_factor
        new_zoom = old_zoom * (1.15**wheel_steps)
        new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
        if abs(new_zoom - old_zoom) < 0.001:
            return

        old_rect = self._image_rect
        if old_rect.isNull():
            old_rect = self._image_rect_for_zoom(old_zoom, self._zoom_pan)

        if old_rect.width() <= 0 or old_rect.height() <= 0:
            ratio_x = 0.5
            ratio_y = 0.5
        else:
            ratio_x = (position.x() - old_rect.left()) / old_rect.width()
            ratio_y = (position.y() - old_rect.top()) / old_rect.height()
            ratio_x = max(0.0, min(1.0, ratio_x))
            ratio_y = max(0.0, min(1.0, ratio_y))

        if new_zoom <= self._min_zoom + 0.001:
            self._zoom_factor = self._min_zoom
            self._zoom_pan = QPoint(0, 0)
            self._zoom_pan_active = False
            self._zoom_pan_last_position = None
            self.update()
            self.zoom_changed.emit(self._zoom_factor)
            return

        base_size = self._base_scaled_size()
        zoomed_width = max(1, round(base_size.width() * new_zoom))
        zoomed_height = max(1, round(base_size.height() * new_zoom))
        centered_left = (self.width() - zoomed_width) // 2
        centered_top = (self.height() - zoomed_height) // 2
        desired_left = position.x() - round(ratio_x * zoomed_width)
        desired_top = position.y() - round(ratio_y * zoomed_height)

        self._zoom_factor = new_zoom
        self._zoom_pan = self._clamped_zoom_pan_offset(
            new_zoom,
            QPoint(desired_left - centered_left, desired_top - centered_top),
        )
        self._sync_fixed_crop_box()
        self.update()
        self.zoom_changed.emit(self._zoom_factor)

    def _pan_zoom_by(self, delta: QPoint) -> None:
        if not self._can_pan_zoom():
            return

        self._zoom_pan = self._clamped_zoom_pan_offset(
            self._zoom_factor,
            QPoint(
                self._zoom_pan.x() + delta.x(),
                self._zoom_pan.y() + delta.y(),
            ),
        )
        self._sync_fixed_crop_box()
        self.update()

    def _can_pan_zoom(self) -> bool:
        return (
            self._zoom_enabled
            and self._zoom_factor > self._min_zoom + 0.001
            and self._pixmap is not None
            and not self._pixmap.isNull()
            and not self._pick_enabled
            and (not self._crop_enabled or self._crop_fixed_to_view)
            and not self._subject_enabled
        )

    def _pan_bounds_rect(self) -> QRect:
        if self._crop_enabled and self._crop_fixed_to_view:
            return self._crop_bounds_rect()
        return self._current_image_rect()

    def _base_scaled_size(self) -> QSize:
        if not self._pixmap or self._pixmap.isNull():
            return QSize(0, 0)
        available = self.rect().adjusted(24, 24, -24, -24).size()
        available = QSize(max(1, available.width()), max(1, available.height()))
        return self._pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ).size()

    def _image_rect_for_zoom(self, zoom_factor: float, pan: QPoint) -> QRect:
        base_size = self._base_scaled_size()
        width = max(1, round(base_size.width() * zoom_factor))
        height = max(1, round(base_size.height() * zoom_factor))
        left = (self.width() - width) // 2 + pan.x()
        top = (self.height() - height) // 2 + pan.y()
        return QRect(left, top, width, height)

    def _clamped_zoom_pan_offset(self, zoom_factor: float, pan: QPoint) -> QPoint:
        if not self._zoom_enabled or zoom_factor <= self._min_zoom + 0.001:
            return QPoint(0, 0)

        base_size = self._base_scaled_size()
        zoomed_width = max(1, round(base_size.width() * zoom_factor))
        zoomed_height = max(1, round(base_size.height() * zoom_factor))
        centered_left = (self.width() - zoomed_width) // 2
        centered_top = (self.height() - zoomed_height) // 2
        viewport = self.rect().adjusted(24, 24, -24, -24)
        if viewport.width() <= 0 or viewport.height() <= 0:
            viewport = self.rect()

        pan_x = self._clamp_zoom_axis(
            pan.x(),
            centered_left,
            zoomed_width,
            viewport.left(),
            viewport.right() + 1,
        )
        pan_y = self._clamp_zoom_axis(
            pan.y(),
            centered_top,
            zoomed_height,
            viewport.top(),
            viewport.bottom() + 1,
        )
        return QPoint(pan_x, pan_y)

    @staticmethod
    def _clamp_zoom_axis(
        pan_value: int,
        centered_start: int,
        zoomed_length: int,
        viewport_start: int,
        viewport_end: int,
    ) -> int:
        viewport_length = max(1, viewport_end - viewport_start)
        if zoomed_length <= viewport_length:
            return 0

        min_pan = viewport_end - (centered_start + zoomed_length)
        max_pan = viewport_start - centered_start
        return max(min_pan, min(max_pan, pan_value))

    def _constrained_crop_rect(self, anchor: QPoint, current: QPoint) -> QRect:
        if self._crop_aspect_ratio is None:
            return QRect(anchor, current).normalized()

        delta_x = current.x() - anchor.x()
        delta_y = current.y() - anchor.y()
        if delta_x == 0 or delta_y == 0:
            return QRect(anchor, current).normalized()

        sign_x = 1 if delta_x >= 0 else -1
        sign_y = 1 if delta_y >= 0 else -1
        width = max(1, abs(delta_x))
        height = max(1, abs(delta_y))

        if width / height > self._crop_aspect_ratio:
            width = max(1, round(height * self._crop_aspect_ratio))
        else:
            height = max(1, round(width / self._crop_aspect_ratio))

        constrained = QPoint(
            anchor.x() + sign_x * width,
            anchor.y() + sign_y * height,
        )
        return QRect(anchor, constrained).normalized()

    def _fit_rect_in_image(self, rect: QRect) -> QRect:
        if self._image_rect.isNull():
            return rect.normalized()

        fitted = rect.normalized()
        if fitted.width() > self._image_rect.width():
            fitted.setWidth(self._image_rect.width())
        if fitted.height() > self._image_rect.height():
            fitted.setHeight(self._image_rect.height())

        if fitted.left() < self._image_rect.left():
            fitted.moveLeft(self._image_rect.left())
        if fitted.top() < self._image_rect.top():
            fitted.moveTop(self._image_rect.top())
        if fitted.right() > self._image_rect.right():
            fitted.moveRight(self._image_rect.right())
        if fitted.bottom() > self._image_rect.bottom():
            fitted.moveBottom(self._image_rect.bottom())
        return fitted

    def _fit_rect_in_crop_bounds(self, rect: QRect) -> QRect:
        bounds_rect = self._crop_bounds_rect()
        if bounds_rect.isNull():
            return rect.normalized()

        fitted = rect.normalized()
        if fitted.width() > bounds_rect.width():
            fitted.setWidth(bounds_rect.width())
        if fitted.height() > bounds_rect.height():
            fitted.setHeight(bounds_rect.height())

        if fitted.left() < bounds_rect.left():
            fitted.moveLeft(bounds_rect.left())
        if fitted.top() < bounds_rect.top():
            fitted.moveTop(bounds_rect.top())
        if fitted.right() > bounds_rect.right():
            fitted.moveRight(bounds_rect.right())
        if fitted.bottom() > bounds_rect.bottom():
            fitted.moveBottom(bounds_rect.bottom())
        return fitted

    def _crop_bounds_rect(self) -> QRect:
        if not self._crop_fixed_to_view:
            return self._current_image_rect()

        viewport = self.rect().adjusted(24, 24, -24, -24)
        if viewport.width() <= 0 or viewport.height() <= 0:
            return self.rect()
        return viewport

    def _clamp_to_crop_bounds(self, point: QPoint) -> QPoint:
        bounds_rect = self._crop_bounds_rect()
        return QPoint(
            max(bounds_rect.left(), min(bounds_rect.right(), point.x())),
            max(bounds_rect.top(), min(bounds_rect.bottom(), point.y())),
        )

    def _sync_fixed_crop_box(self) -> None:
        if (
            not self._crop_fixed_to_view
            or not self._crop_enabled
            or self._crop_view_rect.isNull()
        ):
            return

        crop_box = self._widget_rect_to_image_box(self._crop_view_rect)
        if crop_box is None:
            return

        self._crop_box = crop_box
        self.crop_selection_changed.emit(*crop_box)

    def _paint_subject_brush(self, widget_point: QPoint) -> None:
        if self._subject_mask is None:
            return

        image_point = self._widget_point_to_image_point(
            self._clamp_to_image_rect(widget_point)
        )
        points = [image_point]
        if self._subject_brush_last_point is not None:
            points.insert(0, self._subject_brush_last_point)

        self._subject_mask = paint_subject_mask(
            self._subject_mask,
            points,
            self._subject_brush_radius,
            self._subject_brush_value,
        )
        self._subject_brush_last_point = image_point
        self.update()

    def _widget_point_to_image_point(self, point: QPoint) -> tuple[int, int]:
        scale_x = self._pixmap.width() / self._image_rect.width()
        scale_y = self._pixmap.height() / self._image_rect.height()

        image_x = int(round((point.x() - self._image_rect.left()) * scale_x))
        image_y = int(round((point.y() - self._image_rect.top()) * scale_y))
        image_x = max(0, min(self._pixmap.width() - 1, image_x))
        image_y = max(0, min(self._pixmap.height() - 1, image_y))
        return (image_x, image_y)

    @staticmethod
    def _opposite_crop_corner(crop_rect: QRect, handle: str) -> QPoint:
        opposites = {
            "top_left": crop_rect.bottomRight(),
            "top_right": crop_rect.bottomLeft(),
            "bottom_left": crop_rect.topRight(),
            "bottom_right": crop_rect.topLeft(),
        }
        return opposites[handle]

    def _widget_rect_to_image_box(
        self,
        widget_rect: QRect,
    ) -> tuple[int, int, int, int] | None:
        image_rect = self._current_image_rect()
        if not self._pixmap or self._pixmap.isNull() or image_rect.isNull():
            return None
        return self._widget_rect_to_image_box_for_rect(widget_rect, image_rect)

    def _widget_rect_to_image_box_for_rect(
        self,
        widget_rect: QRect,
        image_rect: QRect,
    ) -> tuple[int, int, int, int] | None:
        if not self._pixmap or self._pixmap.isNull() or image_rect.isNull():
            return None

        bounded = widget_rect.normalized().intersected(image_rect)
        if bounded.width() < 4 or bounded.height() < 4:
            return None

        scale_x = self._pixmap.width() / image_rect.width()
        scale_y = self._pixmap.height() / image_rect.height()

        left = int((bounded.left() - image_rect.left()) * scale_x)
        top = int((bounded.top() - image_rect.top()) * scale_y)
        right = int((bounded.right() + 1 - image_rect.left()) * scale_x)
        bottom = int((bounded.bottom() + 1 - image_rect.top()) * scale_y)

        left = max(0, min(self._pixmap.width() - 1, left))
        top = max(0, min(self._pixmap.height() - 1, top))
        right = max(1, min(self._pixmap.width(), right))
        bottom = max(1, min(self._pixmap.height(), bottom))

        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    def _image_box_to_widget_rect(self, box: tuple[int, int, int, int]) -> QRect:
        image_rect = self._current_image_rect()
        if not self._pixmap or self._pixmap.isNull() or image_rect.isNull():
            return QRect()

        left, top, right, bottom = box
        scale_x = image_rect.width() / self._pixmap.width()
        scale_y = image_rect.height() / self._pixmap.height()

        x1 = image_rect.left() + round(left * scale_x)
        y1 = image_rect.top() + round(top * scale_y)
        x2 = image_rect.left() + round(right * scale_x)
        y2 = image_rect.top() + round(bottom * scale_y)
        return QRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def _current_image_rect(self) -> QRect:
        return self._image_rect_for_zoom(self._zoom_factor, self._zoom_pan)

    def _clamp_to_image_rect(self, point: QPoint) -> QPoint:
        image_rect = self._current_image_rect()
        return QPoint(
            max(image_rect.left(), min(image_rect.right(), point.x())),
            max(image_rect.top(), min(image_rect.bottom(), point.y())),
        )

    def _refresh_cursor(self, position: QPoint | None = None) -> None:
        if self._pick_enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._subject_enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._crop_enabled:
            handle = self._crop_handle_at(position) if position is not None else None
            if handle in {"top_left", "bottom_right"}:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif handle in {"top_right", "bottom_left"}:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif (
                position is not None
                and self._current_crop_widget_rect()
                .intersected(self._crop_bounds_rect())
                .contains(position)
            ):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif (
                self._crop_fixed_to_view
                and self._can_pan_zoom()
                and (position is None or self._pan_bounds_rect().contains(position))
            ):
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._can_pan_zoom():
            if position is None or self._pan_bounds_rect().contains(position):
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.unsetCursor()
        else:
            self.unsetCursor()

    def _background_color_for_style(self) -> str:
        colors = {
            "white": "#FFFFFF",
            "dark": "#1F2937",
            "blue": "#DBEAFE",
            "yellow": "#FEF3C7",
        }
        return colors.get(self.background_style, "#FFFFFF")

    @staticmethod
    def _fill_if_positive(
        painter: QPainter,
        rect: QRect,
        color: QColor,
    ) -> None:
        if rect.width() > 0 and rect.height() > 0:
            painter.fillRect(rect, color)

    @staticmethod
    def _pil_to_pixmap(image: Image.Image) -> QPixmap:
        rgba = image.convert("RGBA")
        data = rgba.tobytes("raw", "RGBA")
        qimage = QImage(
            data,
            rgba.width,
            rgba.height,
            rgba.width * 4,
            QImage.Format.Format_RGBA8888,
        )
        return QPixmap.fromImage(qimage.copy())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Background Cleaner")
        self.resize(1280, 800)
        self.setMinimumSize(1040, 680)
        self.setAcceptDrops(True)

        self.source_path: Path | None = None
        self.loaded_image: Image.Image | None = None
        self.original_image: Image.Image | None = None
        self.processed_result: ProcessResult | None = None
        self.processed_result_source: str | None = None
        self.subject_detection_thread: ImageTaskThread | None = None
        self.crop_box: tuple[int, int, int, int] | None = None
        self.crop_applied = False
        self.subject_mask: Image.Image | None = None
        self.subject_brush_radius = 24
        self.photo_crop_box: tuple[int, int, int, int] | None = None
        self.photo_subject_mask: Image.Image | None = None
        self.background_color = PURE_GREEN
        self.background_mode = MODE_COLOR
        self.tolerance = DEFAULT_TOLERANCE
        self.lightness_threshold = DEFAULT_LIGHTNESS_THRESHOLD
        self.output_mode = OUTPUT_TRANSPARENT
        self.replacement_color = DEFAULT_REPLACEMENT_COLOR

        self.file_name_label = QLabel("No image loaded")
        self.file_name_label.setObjectName("MutedText")
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusText")
        self.file_meta_label = QLabel("Import a PNG or JPG image.")
        self.file_meta_label.setObjectName("MetaLabel")
        self.result_meta_label = QLabel("Process an image to enable export.")
        self.result_meta_label.setObjectName("MetaLabel")
        self.result_meta_label.setWordWrap(True)
        self.result_meta_label.setMaximumHeight(72)
        self.color_value_label = QLabel()
        self.color_value_label.setObjectName("MetaLabel")
        self.color_label = QLabel("Target color")
        self.color_label.setObjectName("MetaLabel")
        self.color_swatch = ColorSwatch("Click to choose the target background color")
        self.color_swatch.clicked.connect(self.choose_background_color)
        self.color_badge = QLabel()
        self.color_badge.setObjectName("Badge")
        self.tolerance_value_label = QLabel()
        self.tolerance_value_label.setObjectName("MetaLabel")
        self.lightness_value_label = QLabel()
        self.lightness_value_label.setObjectName("MetaLabel")
        self.mode_help_label = QLabel()
        self.mode_help_label.setObjectName("MutedText")
        self.mode_help_label.setWordWrap(True)
        self.replacement_color_value_label = QLabel()
        self.replacement_color_value_label.setObjectName("MetaLabel")
        self.replacement_color_swatch = ColorSwatch(
            "Click to choose the replacement background color"
        )
        self.replacement_color_swatch.clicked.connect(self.choose_replacement_color)
        self.crop_meta_label = QLabel("Import an image to crop it.")
        self.crop_meta_label.setObjectName("MetaLabel")
        self.crop_meta_label.setWordWrap(True)
        self.crop_select_button = QPushButton("Select Crop Area")
        self.crop_select_button.clicked.connect(self.toggle_crop_selection)
        self.crop_select_button.setEnabled(False)
        self.crop_apply_button = QPushButton("Crop Image")
        self.crop_apply_button.clicked.connect(self.crop_current_image)
        self.crop_apply_button.setEnabled(False)
        self.crop_reset_button = QPushButton("Reset Image")
        self.crop_reset_button.clicked.connect(self.reset_cropped_image)
        self.crop_reset_button.setEnabled(False)
        self.subject_meta_label = QLabel("Import an image to select its subject.")
        self.subject_meta_label.setObjectName("MetaLabel")
        self.subject_meta_label.setWordWrap(True)
        self.subject_select_button = QPushButton("Detect Subject")
        self.subject_select_button.clicked.connect(self.detect_subject)
        self.subject_select_button.setEnabled(False)
        self.subject_refine_button = QPushButton("Edit Subject Edge")
        self.subject_refine_button.clicked.connect(self.toggle_subject_selection)
        self.subject_refine_button.setEnabled(False)
        self.subject_brush_mode_combo = ScrollFriendlyComboBox()
        self.subject_brush_mode_combo.addItem("Add missing subject", 255)
        self.subject_brush_mode_combo.addItem("Remove leftover", 0)
        self.subject_brush_mode_combo.currentIndexChanged.connect(
            self.set_subject_brush_mode
        )
        self.subject_brush_mode_combo.setEnabled(False)
        self.subject_brush_value_label = QLabel(str(self.subject_brush_radius))
        self.subject_brush_value_label.setObjectName("MetaLabel")
        self.subject_brush_slider = ScrollFriendlySlider(Qt.Orientation.Horizontal)
        self.subject_brush_slider.setRange(4, 80)
        self.subject_brush_slider.setValue(self.subject_brush_radius)
        self.subject_brush_slider.valueChanged.connect(self.set_subject_brush_radius)
        self.subject_brush_slider.setEnabled(False)
        self.subject_apply_button = QPushButton("Apply Subject")
        self.subject_apply_button.clicked.connect(lambda: self.apply_subject_selection())
        self.subject_apply_button.setEnabled(False)
        self.subject_clear_button = QPushButton("Clear Subject")
        self.subject_clear_button.clicked.connect(self.clear_subject_mask)
        self.subject_clear_button.setEnabled(False)
        self.photo_crop_meta_label = QLabel(
            "Process the image before making an ID photo crop."
        )
        self.photo_crop_meta_label.setObjectName("MetaLabel")
        self.photo_crop_meta_label.setWordWrap(True)
        self.photo_size_combo = ScrollFriendlyComboBox()
        self.photo_size_combo.addItem("1 inch (295 x 413)", "1_inch")
        self.photo_size_combo.addItem("2 inch (413 x 579)", "2_inch")
        self.photo_size_combo.addItem("Custom", PHOTO_SIZE_CUSTOM)
        self.photo_size_combo.currentIndexChanged.connect(self.set_photo_size_mode)
        self.photo_width_input = QSpinBox()
        self.photo_width_input.setRange(20, 5000)
        self.photo_width_input.setValue(PHOTO_SIZE_1_INCH[0])
        self.photo_width_input.setSuffix(" px")
        self.photo_width_input.valueChanged.connect(self.set_custom_photo_size)
        self.photo_height_input = QSpinBox()
        self.photo_height_input.setRange(20, 5000)
        self.photo_height_input.setValue(PHOTO_SIZE_1_INCH[1])
        self.photo_height_input.setSuffix(" px")
        self.photo_height_input.valueChanged.connect(self.set_custom_photo_size)
        self.photo_crop_select_button = QPushButton("Select ID Photo Crop")
        self.photo_crop_select_button.clicked.connect(self.toggle_photo_crop_selection)
        self.photo_crop_select_button.setEnabled(False)
        self.photo_crop_apply_button = QPushButton("Apply ID Photo Crop")
        self.photo_crop_apply_button.clicked.connect(self.apply_photo_crop)
        self.photo_crop_apply_button.setEnabled(False)
        self.photo_crop_clear_button = QPushButton("Clear ID Photo Crop")
        self.photo_crop_clear_button.clicked.connect(self.clear_photo_crop_box)
        self.photo_crop_clear_button.setEnabled(False)

        self.original_canvas = ImageCanvas("Original preview", checkerboard=False)
        self.original_canvas.image_pixel_selected.connect(self.pick_background_color)
        self.original_canvas.crop_selection_changed.connect(self.set_crop_box)
        self.original_canvas.crop_selection_cleared.connect(self.clear_crop_box)
        self.original_canvas.subject_selection_changed.connect(self.set_subject_mask)
        self.original_canvas.subject_selection_cleared.connect(self.clear_subject_mask)
        self.processed_canvas = ImageCanvas("Cleaned preview", checkerboard=True)
        self.processed_canvas.set_zoom_enabled(True)
        self.processed_canvas.setToolTip(
            "Use the mouse wheel to zoom, then right-drag to move the preview"
        )
        self.processed_canvas.crop_selection_changed.connect(self.set_photo_crop_box)
        self.processed_canvas.crop_selection_cleared.connect(self.clear_photo_crop_box)

        self.drop_button = QPushButton("Drop image here or click")
        self.drop_button.setObjectName("DropButton")
        self.drop_button.clicked.connect(self.open_image_dialog)

        self.process_button = QPushButton("Apply Background")
        self.process_button.setObjectName("PrimaryButton")
        self.process_button.clicked.connect(self.process_current_image)
        self.process_button.setEnabled(False)

        self.mode_combo = ScrollFriendlyComboBox()
        self.mode_combo.addItem("Picked color", MODE_COLOR)
        self.mode_combo.addItem("Light/gray background", MODE_LIGHT_NEUTRAL)
        self.mode_combo.currentIndexChanged.connect(self.set_background_mode)

        self.preview_background_combo = ScrollFriendlyComboBox()
        self.preview_background_combo.addItem("Checkerboard", "checkerboard")
        self.preview_background_combo.addItem("Dark", "dark")
        self.preview_background_combo.addItem("Blue", "blue")
        self.preview_background_combo.addItem("Yellow", "yellow")
        self.preview_background_combo.setFixedWidth(116)
        self.preview_background_combo.currentIndexChanged.connect(
            self.set_result_preview_background
        )
        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.setToolTip("Restore the processed preview to its original fit")
        self.reset_zoom_button.clicked.connect(self.reset_processed_zoom)
        self.reset_zoom_button.setEnabled(False)
        self.processed_canvas.zoom_changed.connect(self.set_processed_zoom_state)

        self.output_mode_combo = ScrollFriendlyComboBox()
        self.output_mode_combo.addItem("Transparent", OUTPUT_TRANSPARENT)
        self.output_mode_combo.addItem("Solid color", OUTPUT_SOLID)
        self.output_mode_combo.currentIndexChanged.connect(self.set_output_mode)

        self.tolerance_slider = ScrollFriendlySlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setRange(0, 255)
        self.tolerance_slider.setValue(DEFAULT_TOLERANCE)
        self.tolerance_slider.valueChanged.connect(self.set_tolerance)

        self.lightness_slider = ScrollFriendlySlider(Qt.Orientation.Horizontal)
        self.lightness_slider.setRange(0, 255)
        self.lightness_slider.setValue(DEFAULT_LIGHTNESS_THRESHOLD)
        self.lightness_slider.valueChanged.connect(self.set_lightness_threshold)

        self.export_button = QPushButton("Export PNG")
        self.export_button.setObjectName("SuccessButton")
        self.export_button.clicked.connect(self.export_png)
        self.export_button.setEnabled(False)

        self.setCentralWidget(self._build_content())
        self._set_background_color(PURE_GREEN, invalidate_result=False)
        self._set_replacement_color(DEFAULT_REPLACEMENT_COLOR, invalidate_result=False)
        self._update_tolerance_display()
        self._update_lightness_display()
        self._refresh_mode_controls()
        self._refresh_output_controls()
        self._refresh_photo_crop_controls()
        self._update_status("Ready")

    def _build_content(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_top_bar())

        body = QHBoxLayout()
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(16)
        body.addWidget(self._build_import_panel(), 0)
        body.addWidget(self._build_preview_panel("Original", self.original_canvas), 1)
        body.addWidget(
            self._build_preview_panel(
                "Cleaned PNG",
                self.processed_canvas,
                self.preview_background_combo,
                self.reset_zoom_button,
            ),
            1,
        )
        body.addWidget(self._build_actions_panel(), 0)
        layout.addLayout(body, 1)

        layout.addWidget(self._build_status_bar())
        return root

    def _build_top_bar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("TopBar")
        frame.setFixedHeight(56)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(12)

        title = QLabel("Image Background Cleaner")
        title.setObjectName("AppTitle")

        layout.addWidget(title)
        layout.addWidget(self.color_badge)
        layout.addStretch(1)
        layout.addWidget(self.file_name_label)
        return frame

    def _build_import_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Panel")
        frame.setFixedWidth(260)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Import")
        title.setObjectName("SectionTitle")
        hint = QLabel("Load one PNG or JPG image, then choose how to process its background.")
        hint.setWordWrap(True)
        hint.setObjectName("MutedText")

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(self.drop_button)
        layout.addSpacing(8)
        layout.addWidget(QLabel("File"))
        layout.addWidget(self.file_meta_label)
        layout.addStretch(1)
        return frame

    def _build_preview_panel(
        self,
        title_text: str,
        canvas: ImageCanvas,
        control: QWidget | None = None,
        extra_control: QWidget | None = None,
    ) -> QFrame:
        frame = QFrame()
        frame.setObjectName("PreviewPanel")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel(title_text)
        title.setObjectName("PreviewTitle")

        header = QHBoxLayout()
        header.setSpacing(10)
        header.addWidget(title)
        header.addStretch(1)
        if control is not None:
            preview_label = QLabel("Preview bg")
            preview_label.setObjectName("MetaLabel")
            header.addWidget(preview_label)
            header.addWidget(control)
        if extra_control is not None:
            header.addWidget(extra_control)

        layout.addLayout(header)
        layout.addWidget(canvas, 1)
        return frame

    def _build_actions_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Panel")
        frame.setFixedWidth(320)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("SettingsScroll")
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMinimumHeight(0)
        scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )
        scroll_area.setWidgetResizable(True)

        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 8, 0)
        settings_layout.setSpacing(10)

        crop_title = QLabel("Crop")
        crop_title.setObjectName("SectionTitle")

        subject_title = QLabel("Subject")
        subject_title.setObjectName("SectionTitle")

        process_title = QLabel("Background")
        process_title.setObjectName("SectionTitle")

        process_hint = QLabel("Choose a removal mode for the background type.")
        process_hint.setWordWrap(True)
        process_hint.setObjectName("MutedText")

        mode_label = QLabel("Mode")
        mode_label.setObjectName("SectionTitle")

        color_row = QHBoxLayout()
        color_row.setSpacing(10)
        color_text_column = QVBoxLayout()
        color_text_column.setSpacing(4)
        color_text_column.addWidget(self.color_label)
        color_text_column.addWidget(self.color_value_label)
        color_row.addWidget(self.color_swatch)
        color_row.addLayout(color_text_column, 1)

        tolerance_header = QHBoxLayout()
        tolerance_header.setSpacing(8)
        tolerance_title = QLabel("Color tolerance")
        tolerance_title.setObjectName("SectionTitle")
        tolerance_header.addWidget(tolerance_title)
        tolerance_header.addStretch(1)
        tolerance_header.addWidget(self.tolerance_value_label)

        tolerance_hint = QLabel("Used by Picked color mode.")
        tolerance_hint.setWordWrap(True)
        tolerance_hint.setObjectName("MutedText")

        lightness_header = QHBoxLayout()
        lightness_header.setSpacing(8)
        lightness_title = QLabel("Light cutoff")
        lightness_title.setObjectName("SectionTitle")
        lightness_header.addWidget(lightness_title)
        lightness_header.addStretch(1)
        lightness_header.addWidget(self.lightness_value_label)

        lightness_hint = QLabel("Lower values remove more light gray or white background.")
        lightness_hint.setWordWrap(True)
        lightness_hint.setObjectName("MutedText")

        output_title = QLabel("Output")
        output_title.setObjectName("SectionTitle")

        photo_title = QLabel("ID Photo")
        photo_title.setObjectName("SectionTitle")

        photo_size_label = QLabel("Size")
        photo_size_label.setObjectName("SectionTitle")

        custom_photo_row = QHBoxLayout()
        custom_photo_row.setSpacing(8)
        custom_photo_row.addWidget(self.photo_width_input)
        custom_photo_row.addWidget(self.photo_height_input)

        subject_brush_header = QHBoxLayout()
        subject_brush_header.setSpacing(8)
        subject_brush_title = QLabel("Brush size")
        subject_brush_title.setObjectName("SectionTitle")
        subject_brush_header.addWidget(subject_brush_title)
        subject_brush_header.addStretch(1)
        subject_brush_header.addWidget(self.subject_brush_value_label)

        replacement_row = QHBoxLayout()
        replacement_row.setSpacing(10)
        replacement_text_column = QVBoxLayout()
        replacement_text_column.setSpacing(4)
        replacement_label = QLabel("Background color")
        replacement_label.setObjectName("MetaLabel")
        replacement_text_column.addWidget(replacement_label)
        replacement_text_column.addWidget(self.replacement_color_value_label)
        replacement_row.addWidget(self.replacement_color_swatch)
        replacement_row.addLayout(replacement_text_column, 1)

        result_title = QLabel("Result")
        result_title.setObjectName("SectionTitle")

        settings_layout.addWidget(crop_title)
        settings_layout.addWidget(self.crop_meta_label)
        settings_layout.addWidget(self.crop_select_button)
        settings_layout.addWidget(self.crop_apply_button)
        settings_layout.addWidget(self.crop_reset_button)
        settings_layout.addSpacing(6)
        settings_layout.addWidget(subject_title)
        settings_layout.addWidget(self.subject_meta_label)
        settings_layout.addWidget(self.subject_select_button)
        settings_layout.addWidget(self.subject_refine_button)
        settings_layout.addWidget(self.subject_brush_mode_combo)
        settings_layout.addLayout(subject_brush_header)
        settings_layout.addWidget(self.subject_brush_slider)
        settings_layout.addWidget(self.subject_apply_button)
        settings_layout.addWidget(self.subject_clear_button)
        settings_layout.addSpacing(6)
        settings_layout.addWidget(process_title)
        settings_layout.addWidget(process_hint)
        settings_layout.addWidget(mode_label)
        settings_layout.addWidget(self.mode_combo)
        settings_layout.addWidget(self.mode_help_label)
        settings_layout.addLayout(color_row)
        settings_layout.addSpacing(2)
        settings_layout.addLayout(tolerance_header)
        settings_layout.addWidget(self.tolerance_slider)
        settings_layout.addWidget(tolerance_hint)
        settings_layout.addSpacing(2)
        settings_layout.addLayout(lightness_header)
        settings_layout.addWidget(self.lightness_slider)
        settings_layout.addWidget(lightness_hint)
        settings_layout.addSpacing(2)
        settings_layout.addWidget(output_title)
        settings_layout.addWidget(self.output_mode_combo)
        settings_layout.addLayout(replacement_row)
        settings_layout.addSpacing(6)
        settings_layout.addWidget(photo_title)
        settings_layout.addWidget(self.photo_crop_meta_label)
        settings_layout.addWidget(photo_size_label)
        settings_layout.addWidget(self.photo_size_combo)
        settings_layout.addLayout(custom_photo_row)
        settings_layout.addWidget(self.photo_crop_select_button)
        settings_layout.addWidget(self.photo_crop_apply_button)
        settings_layout.addWidget(self.photo_crop_clear_button)
        settings_layout.addStretch(1)

        scroll_area.setWidget(settings)

        layout.addWidget(scroll_area, 1)
        layout.addWidget(self.process_button)
        layout.addWidget(result_title)
        layout.addWidget(self.result_meta_label)
        layout.addWidget(self.export_button)
        return frame

    def _build_status_bar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("StatusBar")
        frame.setFixedHeight(36)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.addWidget(self.status_label)
        return frame

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._is_subject_detection_running():
            event.ignore()
            return

        if self._first_supported_drop_path(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if self._is_subject_detection_running():
            event.ignore()
            self._update_status("Wait for subject detection to finish")
            return

        path = self._first_supported_drop_path(event)
        if not path:
            self._show_error("Unsupported file", "Drop a PNG, JPG, or JPEG image.")
            return

        event.acceptProposedAction()
        self.load_file(path)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if self._is_subject_detection_running():
            self._show_error(
                "Subject detection is running",
                "Wait for subject detection to finish before closing the app.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def open_image_dialog(self) -> None:
        if self._is_subject_detection_running():
            self._show_error(
                "Subject detection is running",
                "Wait for the current subject detection to finish before importing another image.",
            )
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Import Image",
            "",
            "Images (*.png *.jpg *.jpeg)",
        )
        if file_name:
            self.load_file(Path(file_name))

    def load_file(self, path: Path) -> None:
        if self._is_subject_detection_running():
            self._show_error(
                "Subject detection is running",
                "Wait for the current subject detection to finish before importing another image.",
            )
            return

        try:
            image = load_image(path)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not import image", str(exc))
            return

        self.source_path = path
        self.loaded_image = image.copy()
        self.original_image = image.copy()
        self.processed_result = None
        self.processed_result_source = None
        self.crop_box = None
        self.crop_applied = False
        self.subject_mask = None
        self.photo_subject_mask = None
        self.original_canvas.set_pick_enabled(False)
        self.original_canvas.set_crop_enabled(False)
        self.original_canvas.set_subject_enabled(False)
        self.original_canvas.clear_crop_selection()
        self.original_canvas.clear_subject_selection()

        self.file_name_label.setText(path.name)
        self._update_file_meta()
        self.crop_meta_label.setText(
            "Drag over the original preview to select an area."
        )
        self.result_meta_label.setText("Ready to process the background.")
        self.original_canvas.set_pixmap(self._pil_to_pixmap(image))
        self.processed_canvas.set_pixmap(None)
        self._reset_photo_crop_selection()
        self._refresh_process_button()
        self.crop_select_button.setText("Select Crop Area")
        self.crop_select_button.setEnabled(True)
        self.crop_apply_button.setEnabled(False)
        self.crop_reset_button.setEnabled(False)
        self.subject_meta_label.setText(
            "Detect the subject automatically, then edit the edge if needed."
        )
        self.subject_select_button.setText("Detect Subject")
        self.subject_select_button.setEnabled(True)
        self.subject_refine_button.setText("Edit Subject Edge")
        self.subject_refine_button.setEnabled(False)
        self.subject_brush_mode_combo.setEnabled(False)
        self.subject_brush_slider.setEnabled(False)
        self.subject_apply_button.setEnabled(False)
        self.subject_clear_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self._refresh_photo_crop_controls()
        self._refresh_mode_controls()
        self._update_status("Image imported")

    def process_current_image(self) -> None:
        if self.original_image is None or self._is_subject_detection_running():
            return
        if self.processed_result_source == "photo" and self.processed_result is not None:
            self.apply_photo_background()
            return
        if self.subject_mask is not None:
            self.apply_subject_selection(
                output_mode=OUTPUT_SOLID,
                replacement_color=self.background_color,
                result_mode="Subject mask background",
                applying_message=(
                    "Applying subject background...\n"
                    "Using the current subject background color."
                ),
                status_message=(
                    f"Replacing background with {self._color_to_hex(self.background_color)}"
                ),
                done_status="Subject background applied",
            )
            return

        self.process_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.process_button.setText("Removing...")
        self._update_status(self._processing_status_text())

        start = time.perf_counter()
        try:
            self.processed_result = remove_background(
                self.original_image,
                background_color=self.background_color,
                mode=self.background_mode,
                tolerance=self.tolerance,
                lightness_threshold=self.lightness_threshold,
                neutral_chroma_threshold=DEFAULT_NEUTRAL_CHROMA_THRESHOLD,
                output_mode=self.output_mode,
                replacement_color=self.replacement_color,
            )
            self.processed_result_source = "background"
        except Exception as exc:  # noqa: BLE001
            self.process_button.setText("Apply Background")
            self._refresh_process_button()
            self._refresh_export_button()
            self._show_error("Could not process image", str(exc))
            return

        elapsed = time.perf_counter() - start
        self._set_processed_preview(self.processed_result.image)
        if self.output_mode == OUTPUT_TRANSPARENT:
            self._select_result_preview_background("dark")
        self.result_meta_label.setText(
            f"Affected pixels: {self.processed_result.removed_pixels:,}\n"
            f"Mode: {self.mode_combo.currentText()}\n"
            f"{self._result_parameter_text()}\n"
            f"Output: {self._output_parameter_text()}"
        )
        self.process_button.setText("Apply Background")
        self._refresh_process_button()
        self.export_button.setEnabled(True)
        self._update_status(f"Background applied in {elapsed:.2f}s")

    def export_png(self) -> None:
        export_image = self._export_image()
        if export_image is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export PNG",
            self._default_export_name(),
            "PNG Image (*.png)",
        )
        if not file_name:
            return

        output_path = Path(file_name)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")

        try:
            save_png(export_image, output_path)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not export PNG", str(exc))
            return

        self._update_status(f"Exported {output_path.name}")
        QMessageBox.information(self, "Export complete", f"Saved PNG:\n{output_path}")

    def toggle_crop_selection(self) -> None:
        if self._is_subject_detection_running():
            self._show_error(
                "Subject detection is running",
                "Wait for detection to finish before changing the crop area.",
            )
            return

        if self.original_image is None:
            self._show_error("No image loaded", "Import an image before cropping.")
            return

        if self.original_canvas.is_crop_enabled():
            self._stop_crop_selection(clear_selection=True)
            self.clear_crop_box()
            self._update_status("Crop selection cancelled")
            return

        self.crop_box = None
        self.crop_apply_button.setEnabled(False)
        self._clear_subject_mask(discard_result=True)
        self.original_canvas.clear_crop_selection()
        self.original_canvas.set_crop_enabled(True)
        self.crop_select_button.setText("Cancel Selection")
        self.crop_meta_label.setText(
            "Drag over the original preview to select an area."
        )
        self._update_status("Drag on the original preview to select a crop area")

    def set_crop_box(self, left: int, top: int, right: int, bottom: int) -> None:
        self.crop_box = (left, top, right, bottom)
        width = right - left
        height = bottom - top
        self.crop_apply_button.setEnabled(True)
        self.crop_meta_label.setText(
            f"Selected area: {width} x {height}px\n"
            f"Top left: {left}, {top}\n"
            "Drag a corner handle to adjust."
        )
        self._update_status(f"Crop area selected: {width} x {height}px")

    def clear_crop_box(self) -> None:
        self.crop_box = None
        self.crop_apply_button.setEnabled(False)
        if self.original_image is None:
            self.crop_meta_label.setText("Import an image to crop it.")
        else:
            self.crop_meta_label.setText(
                "Drag over the original preview to select an area."
            )

    def crop_current_image(self) -> None:
        if (
            self.original_image is None
            or self.crop_box is None
            or self._is_subject_detection_running()
        ):
            return

        try:
            cropped_image = crop_image(self.original_image, self.crop_box)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not crop image", str(exc))
            return

        self.original_image = cropped_image
        self.crop_applied = True
        self.crop_box = None
        self._clear_subject_mask(discard_result=False)
        self.original_canvas.set_pixmap(self._pil_to_pixmap(cropped_image))
        self._stop_crop_selection(clear_selection=True)
        self._discard_result(
            "Image cropped. Export PNG directly or process the background."
        )
        self._update_file_meta()

        self.crop_meta_label.setText(
            f"Current image: {cropped_image.width} x {cropped_image.height}px\n"
            "Can export directly."
        )
        self.crop_apply_button.setEnabled(False)
        self.crop_reset_button.setEnabled(True)
        self._refresh_process_button()
        self._update_status(
            f"Cropped image to {cropped_image.width} x {cropped_image.height}px"
        )

    def reset_cropped_image(self) -> None:
        if self.loaded_image is None or self._is_subject_detection_running():
            return

        self.original_image = self.loaded_image.copy()
        self.crop_applied = False
        self.crop_box = None
        self._clear_subject_mask(discard_result=False)
        self.original_canvas.set_pixmap(self._pil_to_pixmap(self.original_image))
        self._stop_crop_selection(clear_selection=True)
        self._discard_result("Image restored. Ready to process the background.")
        self._update_file_meta()

        self.crop_meta_label.setText("Drag over the original preview to select an area.")
        self.crop_apply_button.setEnabled(False)
        self.crop_reset_button.setEnabled(False)
        self._refresh_process_button()
        self._update_status("Image restored to original size")

    def set_photo_size_mode(self, _index: int | None = None) -> None:
        self._refresh_photo_crop_controls()
        if self.processed_canvas.is_crop_enabled() and self.processed_result is not None:
            self._start_photo_crop_selection()

    def set_custom_photo_size(self, _value: int | None = None) -> None:
        self._refresh_photo_crop_controls()
        if (
            self.photo_size_combo.currentData() == PHOTO_SIZE_CUSTOM
            and self.processed_canvas.is_crop_enabled()
            and self.processed_result is not None
        ):
            self._start_photo_crop_selection()

    def toggle_photo_crop_selection(self) -> None:
        if self._is_subject_detection_running():
            self._show_error(
                "Subject detection is running",
                "Wait for detection to finish before cropping the target image.",
            )
            return
        if self.processed_result is None:
            self._show_error(
                "No target image",
                "Apply the subject or background first, then crop the target image.",
            )
            return

        if self.processed_canvas.is_crop_enabled():
            self.clear_photo_crop_box()
            self._update_status("ID photo crop cancelled")
            return

        self._start_photo_crop_selection()

    def set_photo_crop_box(self, left: int, top: int, right: int, bottom: int) -> None:
        self.photo_crop_box = (left, top, right, bottom)
        crop_width = right - left
        crop_height = bottom - top
        output_width, output_height = self._current_photo_output_size()
        self.photo_crop_meta_label.setText(
            f"Selected area: {crop_width} x {crop_height}px\n"
            f"Output: {output_width} x {output_height}px\n"
            "Drag inside the box to move it, or drag a corner to resize.\n"
            "Zoom or right-drag the image under the fixed frame to fine tune the crop."
        )
        self._refresh_photo_crop_controls()
        self._update_status(
            f"ID photo crop selected: {crop_width} x {crop_height}px"
        )

    def clear_photo_crop_box(self) -> None:
        self.photo_crop_box = None
        self.processed_canvas.set_crop_enabled(False)
        self.processed_canvas.set_crop_fixed_to_view(False)
        self.processed_canvas.set_crop_aspect_ratio(None)
        self.processed_canvas.clear_crop_selection()
        self.photo_crop_select_button.setText("Select ID Photo Crop")
        if self.processed_result is None:
            self.photo_crop_meta_label.setText(
                "Process the image before making an ID photo crop."
            )
        else:
            output_width, output_height = self._current_photo_output_size()
            self.photo_crop_meta_label.setText(
                f"Ready for {output_width} x {output_height}px output.\n"
                "Click Select ID Photo Crop to place a crop box on the target image."
            )
        self._refresh_photo_crop_controls()

    def apply_photo_crop(self) -> None:
        if self.processed_result is None or self.photo_crop_box is None:
            return

        output_size = self._current_photo_output_size()
        crop_box = self.processed_canvas.current_crop_image_box()
        if crop_box is None:
            self._show_error(
                "Could not crop ID photo",
                "Move the image so the fixed crop frame overlaps the photo.",
            )
            return
        self.photo_crop_box = crop_box
        crop_width = crop_box[2] - crop_box[0]
        crop_height = crop_box[3] - crop_box[1]

        try:
            photo_image = crop_and_resize_image(
                self.processed_result.image,
                crop_box,
                output_size,
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not crop ID photo", str(exc))
            return

        self.photo_subject_mask = self._subject_mask_for_photo_crop(
            crop_box,
            output_size,
        )
        self.processed_result = ProcessResult(
            image=photo_image,
            removed_pixels=self.processed_result.removed_pixels,
        )
        self.processed_result_source = "photo"
        self._set_processed_preview(photo_image, keep_photo_mask=True)
        self.result_meta_label.setText(
            f"Photo output: {output_size[0]} x {output_size[1]}px\n"
            f"Crop source: {crop_width} x {crop_height}px\n"
            f"Size: {self.photo_size_combo.currentText()}"
        )
        self.photo_crop_meta_label.setText(
            f"Current ID photo: {output_size[0]} x {output_size[1]}px.\n"
            "Export PNG to save this result."
        )
        self.export_button.setEnabled(True)
        self._refresh_photo_crop_controls()
        self._update_status(
            f"ID photo cropped to {output_size[0]} x {output_size[1]}px"
        )

    def apply_photo_background(self) -> None:
        if self.processed_result is None:
            return

        photo_image = self.processed_result.image
        photo_mask = self._current_photo_subject_mask()
        self.process_button.setEnabled(False)
        self.subject_apply_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.process_button.setText("Applying...")
        self.result_meta_label.setText(
            "Applying ID photo background...\nKeeping the current photo size."
        )
        self._update_status("Applying background to current ID photo")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        start = time.perf_counter()
        try:
            if photo_mask is not None:
                result = extract_subject_from_mask(
                    photo_image,
                    photo_mask,
                    output_mode=OUTPUT_SOLID,
                    replacement_color=self.background_color,
                )
                result_mode = "ID photo subject background"
                output_text = self._output_parameter_text(
                    OUTPUT_SOLID,
                    self.background_color,
                )
                next_photo_mask = photo_mask.copy()
            else:
                result = remove_background(
                    photo_image,
                    background_color=self.background_color,
                    mode=self.background_mode,
                    tolerance=self.tolerance,
                    lightness_threshold=self.lightness_threshold,
                    neutral_chroma_threshold=DEFAULT_NEUTRAL_CHROMA_THRESHOLD,
                    output_mode=self.output_mode,
                    replacement_color=self.replacement_color,
                )
                result_mode = f"ID photo {self.mode_combo.currentText()}"
                output_text = self._output_parameter_text()
                next_photo_mask = self._alpha_mask_from_image(result.image)
        except Exception as exc:  # noqa: BLE001
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            self.process_button.setText("Apply Background")
            self.subject_apply_button.setEnabled(self.subject_mask is not None)
            self._refresh_process_button()
            self._refresh_export_button()
            self._show_error("Could not apply photo background", str(exc))
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        elapsed = time.perf_counter() - start
        self.processed_result = result
        self.processed_result_source = "photo"
        self.photo_subject_mask = next_photo_mask
        self._set_processed_preview(result.image, keep_photo_mask=True)
        if photo_mask is None and self.output_mode == OUTPUT_TRANSPARENT:
            self._select_result_preview_background("dark")

        width, height = result.image.size
        lines = [
            f"Photo output: {width} x {height}px",
            f"Mode: {result_mode}",
            f"Background pixels: {result.removed_pixels:,}",
            f"Output: {output_text}",
        ]
        if self.photo_subject_mask is not None:
            lines.insert(
                3,
                f"Subject pixels: {self._foreground_pixels(self.photo_subject_mask):,}",
            )
        self.result_meta_label.setText("\n".join(lines))
        self.photo_crop_meta_label.setText(
            f"Current ID photo: {width} x {height}px.\n"
            "Background updated. Export PNG to save this result."
        )
        self.process_button.setText("Apply Background")
        self.subject_apply_button.setEnabled(self.subject_mask is not None)
        self._refresh_process_button()
        self._refresh_mode_controls()
        self.export_button.setEnabled(True)
        self._update_status(f"ID photo background applied in {elapsed:.2f}s")

    def _subject_mask_for_photo_crop(
        self,
        crop_box: tuple[int, int, int, int],
        output_size: tuple[int, int],
    ) -> Image.Image | None:
        if self.processed_result is None:
            return None

        source_mask: Image.Image | None = None
        if (
            self.subject_mask is not None
            and self.subject_mask.size == self.processed_result.image.size
        ):
            source_mask = self.subject_mask
        else:
            source_mask = self._alpha_mask_from_image(self.processed_result.image)

        if source_mask is None:
            return None

        try:
            return crop_and_resize_mask(source_mask, crop_box, output_size)
        except ValueError:
            return None

    def _current_photo_subject_mask(self) -> Image.Image | None:
        if self.processed_result is None:
            return None

        image_size = self.processed_result.image.size
        if (
            self.photo_subject_mask is not None
            and self.photo_subject_mask.size == image_size
        ):
            return self.photo_subject_mask.convert("L").copy()

        return self._alpha_mask_from_image(self.processed_result.image)

    @staticmethod
    def _alpha_mask_from_image(image: Image.Image) -> Image.Image | None:
        alpha = image.convert("RGBA").getchannel("A")
        if alpha.getextrema()[0] >= 255:
            return None
        return alpha

    def detect_subject(self) -> None:
        if self._is_subject_detection_running():
            return

        if self.original_image is None:
            self._show_error("No image loaded", "Import an image before detecting a subject.")
            return

        self._stop_crop_selection(clear_selection=True)
        self.clear_crop_box()
        self._stop_subject_selection(clear_selection=False)
        self.subject_select_button.setEnabled(False)
        self.subject_select_button.setText("Detecting...")
        self.subject_refine_button.setEnabled(False)
        self.subject_apply_button.setEnabled(False)
        self.subject_clear_button.setEnabled(False)
        self.subject_brush_mode_combo.setEnabled(False)
        self.subject_brush_slider.setEnabled(False)
        self.process_button.setEnabled(False)
        self.crop_select_button.setEnabled(False)
        self.drop_button.setEnabled(False)
        self.result_meta_label.setText(
            "Detecting subject...\nRunning local model and OpenCV fallback."
        )
        self.export_button.setEnabled(False)
        self._update_status("Detecting subject locally")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        image = self.original_image.copy()
        thread = ImageTaskThread(lambda: detect_subject_mask(image), self)
        thread.task_finished.connect(self._subject_detection_finished)
        thread.task_failed.connect(self._subject_detection_failed)
        thread.finished.connect(thread.deleteLater)
        self.subject_detection_thread = thread
        thread.start()

    def toggle_subject_selection(self) -> None:
        if self.original_image is None:
            self._show_error("No image loaded", "Import an image before refining a subject.")
            return
        if self.subject_mask is None:
            self._show_error(
                "No subject mask",
                "Detect the subject before refining its edge.",
            )
            return

        if self.original_canvas.is_subject_enabled():
            self._stop_subject_selection(clear_selection=False)
            self._update_status("Subject refinement stopped")
            return

        self._stop_crop_selection(clear_selection=True)
        self.clear_crop_box()
        self.original_canvas.set_subject_mask(self.subject_mask)
        self.original_canvas.set_subject_brush_value(
            self.subject_brush_mode_combo.currentData()
        )
        self.original_canvas.set_subject_brush_radius(self.subject_brush_radius)
        self.original_canvas.set_subject_enabled(True)
        self.subject_refine_button.setText("Stop Editing")
        self.subject_meta_label.setText(
            "Brush on the original preview to add missing subject or remove leftovers."
        )
        self._update_status("Brush on the original preview to edit the subject edge")

    def set_subject_mask(self, mask: object) -> None:
        if not isinstance(mask, Image.Image):
            self._clear_subject_mask(discard_result=False)
            return
        self._set_subject_mask(mask, invalidate_result=True)

    def clear_subject_mask(self) -> None:
        self._clear_subject_mask(discard_result=True)
        self._update_status("Subject mask cleared")

    def apply_subject_selection(
        self,
        *,
        output_mode: str | None = None,
        replacement_color: tuple[int, int, int] | None = None,
        result_mode: str = "Auto subject mask",
        applying_message: str = "Applying subject mask...\nGenerating the preview.",
        status_message: str = "Extracting selected subject",
        done_status: str = "Subject extracted",
    ) -> None:
        if self._is_subject_detection_running():
            self._update_status("Wait for subject detection to finish")
            return
        if self.original_image is None:
            self._show_error("No image loaded", "Import an image before applying a subject.")
            return
        if self.subject_mask is None:
            self._show_error(
                "No subject mask",
                "Detect the subject before applying it.",
            )
            return

        active_output_mode = self.output_mode if output_mode is None else output_mode
        active_replacement_color = (
            self.replacement_color if replacement_color is None else replacement_color
        )
        self.subject_apply_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.subject_apply_button.setText("Applying...")
        self.process_button.setText("Applying...")
        self.result_meta_label.setText(applying_message)
        self._update_status(status_message)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        start = time.perf_counter()
        try:
            self.processed_result = extract_subject_from_mask(
                self.original_image,
                self.subject_mask,
                output_mode=active_output_mode,
                replacement_color=active_replacement_color,
            )
            self.processed_result_source = "subject"
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self.subject_apply_button.setText("Apply Subject")
            self.process_button.setText("Apply Background")
            self.subject_apply_button.setEnabled(True)
            self._refresh_process_button()
            self._refresh_export_button()
            self._show_error("Could not apply subject selection", str(exc))
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        elapsed = time.perf_counter() - start
        self._set_processed_preview(self.processed_result.image)
        if active_output_mode == OUTPUT_TRANSPARENT:
            self._select_result_preview_background("dark")
        self.result_meta_label.setText(
            f"Background pixels: {self.processed_result.removed_pixels:,}\n"
            f"Mode: {result_mode}\n"
            f"Subject pixels: {self._subject_foreground_pixels():,}\n"
            "Output: "
            f"{self._output_parameter_text(active_output_mode, active_replacement_color)}"
        )
        self.subject_apply_button.setText("Apply Subject")
        self.process_button.setText("Apply Background")
        self.subject_apply_button.setEnabled(True)
        self._refresh_process_button()
        self._refresh_mode_controls()
        self.export_button.setEnabled(True)
        self._update_status(f"{done_status} in {elapsed:.2f}s")

    def enable_color_picker(self) -> None:
        if self.original_image is None:
            self._show_error("No image loaded", "Import an image before picking a color.")
            return

        self.original_canvas.set_pick_enabled(True)
        self._update_status("Click the background color in the original preview")

    def choose_background_color(self) -> None:
        title = (
            "Choose Subject Background Color"
            if self.subject_mask is not None
            else "Choose Target Background Color"
        )
        color = self._choose_rgb_color(
            current_color=self.background_color,
            title=title,
        )
        if color is None:
            return

        self.original_canvas.set_pick_enabled(False)
        self._apply_background_color_choice(color)
        if self.subject_mask is not None:
            self._update_status(
                f"Selected subject background RGB{color}. Apply Background to update."
            )
        else:
            self._update_status(f"Selected target color RGB{color}")

    def pick_background_color(self, image_x: int, image_y: int) -> None:
        if self.original_image is None:
            return

        red, green, blue, _alpha = self.original_image.getpixel((image_x, image_y))
        self.original_canvas.set_pick_enabled(False)
        self._apply_background_color_choice((red, green, blue))
        if self.subject_mask is not None:
            self._update_status(
                "Picked subject background "
                f"RGB({red}, {green}, {blue}). Apply Background to update."
            )
        else:
            self._update_status(
                f"Picked RGB({red}, {green}, {blue}) at {image_x}, {image_y}"
            )

    def set_background_mode(self, _index: int | None = None) -> None:
        self.background_mode = self.mode_combo.currentData()
        self._refresh_mode_controls()
        self._invalidate_background_result("Mode changed. Process the image again.")

    def set_tolerance(self, value: int) -> None:
        self.tolerance = value
        self._update_tolerance_display()
        self._invalidate_background_result("Tolerance changed. Process the image again.")

    def set_lightness_threshold(self, value: int) -> None:
        self.lightness_threshold = value
        self._update_lightness_display()
        self._invalidate_background_result("Light cutoff changed. Process the image again.")

    def set_result_preview_background(self, _index: int | None = None) -> None:
        self.processed_canvas.set_background_style(
            self.preview_background_combo.currentData()
        )

    def reset_processed_zoom(self) -> None:
        self.processed_canvas.reset_zoom()
        self._update_status("Preview zoom reset")

    def set_processed_zoom_state(self, zoom_factor: float) -> None:
        zoomed = zoom_factor > 1.001
        self.reset_zoom_button.setEnabled(zoomed)

    def set_output_mode(self, _index: int | None = None) -> None:
        self.output_mode = self.output_mode_combo.currentData()
        self._refresh_output_controls()
        self._invalidate_result("Output mode changed. Process the image again.")

    def set_subject_brush_mode(self, _index: int | None = None) -> None:
        value = self.subject_brush_mode_combo.currentData()
        self.original_canvas.set_subject_brush_value(value)

    def set_subject_brush_radius(self, value: int) -> None:
        self.subject_brush_radius = value
        self.subject_brush_value_label.setText(str(value))
        self.original_canvas.set_subject_brush_radius(value)

    def choose_replacement_color(self) -> None:
        color = self._choose_rgb_color(
            current_color=self.replacement_color,
            title="Choose Replacement Background Color",
        )
        if color is None:
            return

        self._set_replacement_color(color, invalidate_result=True)
        if self.output_mode != OUTPUT_SOLID:
            self._select_output_mode(OUTPUT_SOLID)

    def _choose_rgb_color(
        self,
        current_color: tuple[int, int, int],
        title: str,
    ) -> tuple[int, int, int] | None:
        color = QColorDialog.getColor(QColor(*current_color), self, title)
        if not color.isValid():
            return None
        return (color.red(), color.green(), color.blue())

    def _apply_background_color_choice(self, color: tuple[int, int, int]) -> None:
        self._set_background_color(color, invalidate_result=True)

        if self._is_light_neutral_color(color):
            self._select_background_mode(MODE_LIGHT_NEUTRAL)
        else:
            self._select_background_mode(MODE_COLOR)

    def _first_supported_drop_path(self, event: QDragEnterEvent | QDropEvent) -> Path | None:
        if not event.mimeData().hasUrls():
            return None

        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if is_supported_image(path):
                return path
        return None

    def _update_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _show_error(self, title: str, message: str) -> None:
        self._update_status(message)
        QMessageBox.warning(self, title, message)

    def _set_processed_preview(
        self,
        image: Image.Image,
        *,
        keep_photo_mask: bool = False,
    ) -> None:
        if not keep_photo_mask:
            self.photo_subject_mask = None
        self.processed_canvas.set_pixmap(self._pil_to_pixmap(image))
        self._reset_photo_crop_selection()

    def _start_photo_crop_selection(self) -> None:
        if self.processed_result is None:
            return

        output_width, output_height = self._current_photo_output_size()
        aspect_ratio = output_width / output_height
        crop_box = self._default_photo_crop_box(
            self.processed_result.image.size,
            aspect_ratio,
            preferred_box=self.processed_canvas.visible_image_box(),
        )

        self._stop_crop_selection(clear_selection=True)
        self.clear_crop_box()
        self._stop_subject_selection(clear_selection=False)
        self.processed_canvas.set_crop_fixed_to_view(True)
        self.processed_canvas.set_crop_aspect_ratio(aspect_ratio)
        self.processed_canvas.set_crop_enabled(True)
        self.processed_canvas.set_crop_box(crop_box, notify=True)
        self.photo_crop_select_button.setText("Cancel ID Photo Crop")
        self._refresh_photo_crop_controls()
        self._update_status(
            f"Adjust the ID photo crop for {output_width} x {output_height}px output"
        )

    def _reset_photo_crop_selection(self) -> None:
        self.photo_crop_box = None
        self.processed_canvas.set_crop_enabled(False)
        self.processed_canvas.set_crop_fixed_to_view(False)
        self.processed_canvas.set_crop_aspect_ratio(None)
        self.processed_canvas.clear_crop_selection()
        self.photo_crop_select_button.setText("Select ID Photo Crop")
        if self.processed_result is None:
            self.photo_crop_meta_label.setText(
                "Process the image before making an ID photo crop."
            )
        else:
            output_width, output_height = self._current_photo_output_size()
            self.photo_crop_meta_label.setText(
                f"Ready for {output_width} x {output_height}px output.\n"
                "Click Select ID Photo Crop to place a crop box on the target image."
            )
        self._refresh_photo_crop_controls()

    def _refresh_photo_crop_controls(self) -> None:
        has_result = (
            self.processed_result is not None
            and not self._is_subject_detection_running()
        )
        custom_mode = self.photo_size_combo.currentData() == PHOTO_SIZE_CUSTOM
        self.photo_width_input.setEnabled(custom_mode)
        self.photo_height_input.setEnabled(custom_mode)
        self.photo_crop_select_button.setEnabled(has_result)
        self.photo_crop_apply_button.setEnabled(has_result and self.photo_crop_box is not None)
        self.photo_crop_clear_button.setEnabled(
            has_result
            and (self.photo_crop_box is not None or self.processed_canvas.is_crop_enabled())
        )

    def _current_photo_output_size(self) -> tuple[int, int]:
        preset = self.photo_size_combo.currentData()
        if preset == "1_inch":
            return PHOTO_SIZE_1_INCH
        if preset == "2_inch":
            return PHOTO_SIZE_2_INCH
        return (self.photo_width_input.value(), self.photo_height_input.value())

    @staticmethod
    def _default_photo_crop_box(
        image_size: tuple[int, int],
        aspect_ratio: float,
        preferred_box: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int, int, int]:
        image_width, image_height = image_size
        if preferred_box is None:
            bounds_left, bounds_top, bounds_right, bounds_bottom = (
                0,
                0,
                image_width,
                image_height,
            )
        else:
            left, top, right, bottom = preferred_box
            bounds_left = max(0, min(image_width - 1, left))
            bounds_top = max(0, min(image_height - 1, top))
            bounds_right = max(bounds_left + 1, min(image_width, right))
            bounds_bottom = max(bounds_top + 1, min(image_height, bottom))

        bounds_width = bounds_right - bounds_left
        bounds_height = bounds_bottom - bounds_top
        max_width = max(1, round(bounds_width * 0.9))
        max_height = max(1, round(bounds_height * 0.9))

        if max_width / max_height > aspect_ratio:
            crop_height = max_height
            crop_width = max(1, round(crop_height * aspect_ratio))
        else:
            crop_width = max_width
            crop_height = max(1, round(crop_width / aspect_ratio))

        crop_width = min(crop_width, bounds_width)
        crop_height = min(crop_height, bounds_height)
        left = bounds_left + max(0, (bounds_width - crop_width) // 2)
        top = bounds_top + max(0, (bounds_height - crop_height) // 2)
        return (left, top, left + crop_width, top + crop_height)

    def _is_subject_detection_running(self) -> bool:
        return (
            self.subject_detection_thread is not None
            and self.subject_detection_thread.isRunning()
        )

    def _finish_subject_detection_task(self) -> None:
        if QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self.subject_detection_thread = None
        has_image = self.original_image is not None
        self.drop_button.setEnabled(True)
        self._refresh_process_button()
        self.crop_select_button.setEnabled(has_image)

    def _restore_subject_controls(self) -> None:
        has_image = self.original_image is not None
        has_mask = self.subject_mask is not None
        self.subject_select_button.setText("Detect Again" if has_mask else "Detect Subject")
        self.subject_select_button.setEnabled(has_image)
        self.subject_refine_button.setEnabled(has_mask)
        self.subject_brush_mode_combo.setEnabled(has_mask)
        self.subject_brush_slider.setEnabled(has_mask)
        self.subject_apply_button.setEnabled(has_mask)
        self.subject_clear_button.setEnabled(has_mask)
        self._refresh_export_button()

    def _subject_detection_finished(self, result: object, elapsed: float) -> None:
        self._finish_subject_detection_task()
        if not isinstance(result, Image.Image):
            self.result_meta_label.setText(
                "Subject detection failed.\nNo usable mask was returned."
            )
            self._restore_subject_controls()
            self._show_error(
                "Could not detect subject",
                "The detector did not return a usable mask.",
            )
            return

        self._set_subject_mask(result, invalidate_result=False)
        self.subject_select_button.setText("Detect Again")
        self.subject_select_button.setEnabled(True)
        self._update_status(f"Subject detected in {elapsed:.2f}s")
        self.apply_subject_selection()

    def _subject_detection_failed(self, message: str) -> None:
        self._finish_subject_detection_task()
        self.result_meta_label.setText(
            "Subject detection failed.\nTry detecting again or use background mode."
        )
        self._restore_subject_controls()
        self._show_error("Could not detect subject", message)

    def _stop_crop_selection(self, clear_selection: bool) -> None:
        self.original_canvas.set_crop_enabled(False)
        if clear_selection:
            self.original_canvas.clear_crop_selection()
        self.crop_select_button.setText("Select Crop Area")

    def _stop_subject_selection(self, clear_selection: bool) -> None:
        self.original_canvas.set_subject_enabled(False)
        if clear_selection:
            self.original_canvas.clear_subject_selection()
        self.subject_refine_button.setText("Edit Subject Edge")

    def _set_subject_mask(
        self,
        mask: Image.Image,
        invalidate_result: bool,
    ) -> None:
        self.subject_mask = mask.convert("L").copy()
        self.original_canvas.set_subject_mask(self.subject_mask)
        self.subject_select_button.setText("Detect Again")
        self.subject_select_button.setEnabled(self.original_image is not None)
        self.subject_refine_button.setEnabled(True)
        self.subject_brush_mode_combo.setEnabled(True)
        self.subject_brush_slider.setEnabled(True)
        self.subject_apply_button.setEnabled(True)
        self.subject_clear_button.setEnabled(True)
        self._refresh_process_button()
        self._refresh_mode_controls()

        if invalidate_result and self.processed_result_source == "subject":
            self._discard_result("Subject mask changed. Apply it again.")

        self.subject_meta_label.setText(
            f"Auto mask ready: {self._subject_foreground_pixels():,} subject pixels.\n"
            "Use Edit Subject Edge if a small area needs correction."
        )

    def _clear_subject_mask(self, discard_result: bool) -> None:
        self.subject_mask = None
        self.original_canvas.set_subject_enabled(False)
        self.original_canvas.clear_subject_selection()
        has_image = self.original_image is not None
        self.subject_select_button.setText("Detect Subject")
        self.subject_select_button.setEnabled(has_image)
        self.subject_refine_button.setText("Edit Subject Edge")
        self.subject_refine_button.setEnabled(False)
        self.subject_brush_mode_combo.setEnabled(False)
        self.subject_brush_slider.setEnabled(False)
        self.subject_apply_button.setEnabled(False)
        self.subject_clear_button.setEnabled(False)
        self._refresh_process_button()
        self._refresh_mode_controls()

        if has_image:
            self.subject_meta_label.setText(
                "Detect the subject automatically, then edit the edge if needed."
            )
        else:
            self.subject_meta_label.setText("Import an image to select its subject.")

        if discard_result and self.processed_result_source == "subject":
            self._discard_result("Subject selection cleared. Ready to process again.")

    def _subject_foreground_pixels(self) -> int:
        if self.subject_mask is None:
            return 0
        return self._foreground_pixels(self.subject_mask)

    @staticmethod
    def _foreground_pixels(mask: Image.Image) -> int:
        histogram = mask.convert("L").histogram()
        return sum(histogram[128:])

    def _discard_result(self, message: str) -> None:
        self.processed_result = None
        self.processed_result_source = None
        self.photo_subject_mask = None
        self.processed_canvas.set_pixmap(None)
        self.result_meta_label.setText(message)
        self._reset_photo_crop_selection()
        self._refresh_export_button()

    def _refresh_export_button(self) -> None:
        self.export_button.setEnabled(self._export_image() is not None)

    def _refresh_process_button(self) -> None:
        self.process_button.setEnabled(
            self.original_image is not None
            and not self._is_subject_detection_running()
        )

    def _export_image(self) -> Image.Image | None:
        if self.processed_result is not None:
            return self.processed_result.image
        if self.crop_applied and self.original_image is not None:
            return self.original_image
        return None

    def _default_export_name(self) -> str:
        if self.processed_result is not None:
            suffix = {
                "subject": "subject",
                "photo": "photo",
            }.get(self.processed_result_source or "", "cleaned")
        else:
            suffix = "cropped"

        if self.source_path is None:
            return f"{suffix}.png"

        return f"{self.source_path.stem}_{suffix}.png"

    def _update_file_meta(self) -> None:
        if self.source_path is None or self.original_image is None:
            self.file_meta_label.setText("Import a PNG or JPG image.")
            return

        lines = [
            self.source_path.name,
            f"{self.original_image.width} x {self.original_image.height}px",
            self.source_path.suffix.upper()[1:],
        ]
        if (
            self.loaded_image is not None
            and (
                self.original_image.width != self.loaded_image.width
                or self.original_image.height != self.loaded_image.height
            )
        ):
            lines.append(
                f"Original: {self.loaded_image.width} x {self.loaded_image.height}px"
            )
        self.file_meta_label.setText("\n".join(lines))

    def _set_background_color(
        self,
        color: tuple[int, int, int],
        invalidate_result: bool,
    ) -> None:
        self.background_color = color
        self._update_color_display()

        if invalidate_result:
            self._invalidate_background_result("Color changed. Process the image again.")

    def _set_replacement_color(
        self,
        color: tuple[int, int, int],
        invalidate_result: bool,
    ) -> None:
        self.replacement_color = color
        self._update_replacement_color_display()

        if invalidate_result:
            self._invalidate_result("Replacement color changed. Process the image again.")

    def _invalidate_result(self, message: str) -> None:
        if self.processed_result is not None:
            self._discard_result(message)

    def _invalidate_background_result(self, message: str) -> None:
        if self.processed_result_source == "background":
            self._discard_result(message)
        self._refresh_process_button()

    def _update_color_display(self) -> None:
        red, green, blue = self.background_color
        hex_color = self._color_to_hex(self.background_color)
        self.color_value_label.setText(f"RGB({red}, {green}, {blue})\n{hex_color}")
        self.color_badge.setText(f"RGB({red}, {green}, {blue})")
        self.color_swatch.setStyleSheet(
            "QLabel#ColorSwatch {"
            f"background: {hex_color};"
            "border: 1px solid #D0D5DD;"
            "border-radius: 6px;"
            "}"
            "QLabel#ColorSwatch:hover {"
            "border: 2px solid #2563EB;"
            "}"
        )

    def _update_replacement_color_display(self) -> None:
        red, green, blue = self.replacement_color
        hex_color = self._color_to_hex(self.replacement_color)
        self.replacement_color_value_label.setText(
            f"RGB({red}, {green}, {blue})\n{hex_color}"
        )
        self.replacement_color_swatch.setStyleSheet(
            "QLabel#ColorSwatch {"
            f"background: {hex_color};"
            "border: 1px solid #D0D5DD;"
            "border-radius: 6px;"
            "}"
            "QLabel#ColorSwatch:hover {"
            "border: 2px solid #2563EB;"
            "}"
        )

    def _update_tolerance_display(self) -> None:
        self.tolerance_value_label.setText(str(self.tolerance))

    def _update_lightness_display(self) -> None:
        self.lightness_value_label.setText(str(self.lightness_threshold))

    def _refresh_mode_controls(self) -> None:
        color_mode = self.background_mode == MODE_COLOR
        self.tolerance_slider.setEnabled(color_mode)
        self.lightness_slider.setEnabled(not color_mode)

        if self.subject_mask is not None:
            self.color_label.setText("Subject background")
            self.color_swatch.setToolTip(
                "Click to choose the replacement color for the detected subject"
            )
            self.mode_help_label.setText(
                "Subject result is active. Apply Background uses the subject mask and fills the background with this color. Preview bg only affects transparent preview."
            )
        elif color_mode:
            self.color_label.setText("Target color")
            self.color_swatch.setToolTip("Click to choose the target background color")
            self.mode_help_label.setText(
                "Removes pixels near the picked RGB color. Good for green or blue screens."
            )
        else:
            self.color_label.setText("Target color")
            self.color_swatch.setToolTip("Click to choose the target background color")
            self.mode_help_label.setText(
                "Removes bright low-saturation pixels. Good for white, gray, or checker backgrounds behind stamps."
            )
        self._refresh_process_button()

    def _refresh_output_controls(self) -> None:
        self.replacement_color_swatch.set_click_enabled(True)
        self.replacement_color_value_label.setEnabled(True)

    def _select_background_mode(self, mode: str) -> None:
        index = self.mode_combo.findData(mode)
        if index >= 0 and index != self.mode_combo.currentIndex():
            self.mode_combo.setCurrentIndex(index)
        else:
            self.background_mode = mode
            self._refresh_mode_controls()

    def _select_output_mode(self, mode: str) -> None:
        index = self.output_mode_combo.findData(mode)
        if index >= 0 and index != self.output_mode_combo.currentIndex():
            self.output_mode_combo.setCurrentIndex(index)
        else:
            self.output_mode = mode
            self._refresh_output_controls()

    def _select_result_preview_background(self, style: str) -> None:
        index = self.preview_background_combo.findData(style)
        if index >= 0 and index != self.preview_background_combo.currentIndex():
            self.preview_background_combo.setCurrentIndex(index)
        else:
            self.processed_canvas.set_background_style(style)

    def _processing_status_text(self) -> str:
        if self.background_mode == MODE_LIGHT_NEUTRAL:
            return f"Removing light neutral background with cutoff {self.lightness_threshold}"
        return (
            f"Detecting pixels near RGB{self.background_color} "
            f"with tolerance {self.tolerance}"
        )

    def _result_parameter_text(self) -> str:
        if self.background_mode == MODE_LIGHT_NEUTRAL:
            return f"Light cutoff: {self.lightness_threshold}"
        return f"Tolerance: {self.tolerance}"

    def _output_parameter_text(
        self,
        output_mode: str | None = None,
        replacement_color: tuple[int, int, int] | None = None,
    ) -> str:
        active_output_mode = self.output_mode if output_mode is None else output_mode
        active_replacement_color = (
            self.replacement_color if replacement_color is None else replacement_color
        )
        if active_output_mode == OUTPUT_SOLID:
            return f"Solid {self._color_to_hex(active_replacement_color)}"
        return "Transparent PNG with alpha"

    @staticmethod
    def _is_light_neutral_color(color: tuple[int, int, int]) -> bool:
        red, green, blue = color
        return max(color) >= 180 and max(color) - min(color) <= 35

    @staticmethod
    def _color_to_hex(color: tuple[int, int, int]) -> str:
        return "#{:02X}{:02X}{:02X}".format(*color)

    @staticmethod
    def _pil_to_pixmap(image: Image.Image) -> QPixmap:
        rgba = image.convert("RGBA")
        data = rgba.tobytes("raw", "RGBA")
        qimage = QImage(
            data,
            rgba.width,
            rgba.height,
            rgba.width * 4,
            QImage.Format.Format_RGBA8888,
        )
        return QPixmap.fromImage(qimage.copy())
