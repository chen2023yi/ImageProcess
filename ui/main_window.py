from __future__ import annotations

import time
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QRect, QSize, Qt, Signal
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
    is_supported_image,
    load_image,
    remove_background,
    save_png,
)


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


class ImageCanvas(QWidget):
    image_pixel_selected = Signal(int, int)

    def __init__(self, empty_text: str, checkerboard: bool = False) -> None:
        super().__init__()
        self.empty_text = empty_text
        self.background_style = "checkerboard" if checkerboard else "white"
        self._pixmap: QPixmap | None = None
        self._image_rect = QRect()
        self._pick_enabled = False
        self.setMinimumSize(320, 360)

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self._image_rect = QRect()
        self.update()

    def set_pick_enabled(self, enabled: bool) -> None:
        self._pick_enabled = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()

    def set_background_style(self, style: str) -> None:
        self.background_style = style
        self.update()

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

        available = self.rect().adjusted(24, 24, -24, -24).size()
        scaled = self._pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._image_rect = QRect(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(x, y, scaled)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
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

    def _background_color_for_style(self) -> str:
        colors = {
            "white": "#FFFFFF",
            "dark": "#1F2937",
            "blue": "#DBEAFE",
            "yellow": "#FEF3C7",
        }
        return colors.get(self.background_style, "#FFFFFF")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Background Cleaner")
        self.resize(1280, 800)
        self.setMinimumSize(1040, 680)
        self.setAcceptDrops(True)

        self.source_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.processed_result: ProcessResult | None = None
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

        self.original_canvas = ImageCanvas("Original preview", checkerboard=False)
        self.original_canvas.image_pixel_selected.connect(self.pick_background_color)
        self.processed_canvas = ImageCanvas("Cleaned preview", checkerboard=True)

        self.drop_button = QPushButton("Drop image here or click")
        self.drop_button.setObjectName("DropButton")
        self.drop_button.clicked.connect(self.open_image_dialog)

        self.process_button = QPushButton("Apply Background")
        self.process_button.setObjectName("PrimaryButton")
        self.process_button.clicked.connect(self.process_current_image)
        self.process_button.setEnabled(False)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Picked color", MODE_COLOR)
        self.mode_combo.addItem("Light/gray background", MODE_LIGHT_NEUTRAL)
        self.mode_combo.currentIndexChanged.connect(self.set_background_mode)

        self.preview_background_combo = QComboBox()
        self.preview_background_combo.addItem("Checkerboard", "checkerboard")
        self.preview_background_combo.addItem("Dark", "dark")
        self.preview_background_combo.addItem("Blue", "blue")
        self.preview_background_combo.addItem("Yellow", "yellow")
        self.preview_background_combo.setFixedWidth(116)
        self.preview_background_combo.currentIndexChanged.connect(
            self.set_result_preview_background
        )

        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItem("Transparent", OUTPUT_TRANSPARENT)
        self.output_mode_combo.addItem("Solid color", OUTPUT_SOLID)
        self.output_mode_combo.currentIndexChanged.connect(self.set_output_mode)

        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setRange(0, 255)
        self.tolerance_slider.setValue(DEFAULT_TOLERANCE)
        self.tolerance_slider.valueChanged.connect(self.set_tolerance)

        self.lightness_slider = QSlider(Qt.Orientation.Horizontal)
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
        color_label = QLabel("Target color")
        color_label.setObjectName("MetaLabel")
        color_text_column.addWidget(color_label)
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
        if self._first_supported_drop_path(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        path = self._first_supported_drop_path(event)
        if not path:
            self._show_error("Unsupported file", "Drop a PNG, JPG, or JPEG image.")
            return

        event.acceptProposedAction()
        self.load_file(path)

    def open_image_dialog(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Import Image",
            "",
            "Images (*.png *.jpg *.jpeg)",
        )
        if file_name:
            self.load_file(Path(file_name))

    def load_file(self, path: Path) -> None:
        try:
            image = load_image(path)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not import image", str(exc))
            return

        self.source_path = path
        self.original_image = image
        self.processed_result = None
        self.original_canvas.set_pick_enabled(False)

        self.file_name_label.setText(path.name)
        self.file_meta_label.setText(
            f"{path.name}\n{image.width} x {image.height}px\n{path.suffix.upper()[1:]}"
        )
        self.result_meta_label.setText("Ready to process the background.")
        self.original_canvas.set_pixmap(self._pil_to_pixmap(image))
        self.processed_canvas.set_pixmap(None)
        self.process_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self._update_status("Image imported")

    def process_current_image(self) -> None:
        if self.original_image is None:
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
        except Exception as exc:  # noqa: BLE001
            self.process_button.setText("Apply Background")
            self.process_button.setEnabled(True)
            self._show_error("Could not process image", str(exc))
            return

        elapsed = time.perf_counter() - start
        self.processed_canvas.set_pixmap(self._pil_to_pixmap(self.processed_result.image))
        if self.output_mode == OUTPUT_TRANSPARENT:
            self._select_result_preview_background("dark")
        self.result_meta_label.setText(
            f"Affected pixels: {self.processed_result.removed_pixels:,}\n"
            f"Mode: {self.mode_combo.currentText()}\n"
            f"{self._result_parameter_text()}\n"
            f"Output: {self._output_parameter_text()}"
        )
        self.process_button.setText("Apply Background")
        self.process_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self._update_status(f"Background applied in {elapsed:.2f}s")

    def export_png(self) -> None:
        if self.processed_result is None:
            return

        default_name = "cleaned.png"
        if self.source_path:
            default_name = f"{self.source_path.stem}_cleaned.png"

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export PNG",
            default_name,
            "PNG Image (*.png)",
        )
        if not file_name:
            return

        output_path = Path(file_name)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")

        try:
            save_png(self.processed_result.image, output_path)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not export PNG", str(exc))
            return

        self._update_status(f"Exported {output_path.name}")
        QMessageBox.information(self, "Export complete", f"Saved PNG:\n{output_path}")

    def enable_color_picker(self) -> None:
        if self.original_image is None:
            self._show_error("No image loaded", "Import an image before picking a color.")
            return

        self.original_canvas.set_pick_enabled(True)
        self._update_status("Click the background color in the original preview")

    def choose_background_color(self) -> None:
        color = self._choose_rgb_color(
            current_color=self.background_color,
            title="Choose Target Background Color",
        )
        if color is None:
            return

        self.original_canvas.set_pick_enabled(False)
        self._apply_background_color_choice(color)
        self._update_status(f"Selected target color RGB{color}")

    def pick_background_color(self, image_x: int, image_y: int) -> None:
        if self.original_image is None:
            return

        red, green, blue, _alpha = self.original_image.getpixel((image_x, image_y))
        self.original_canvas.set_pick_enabled(False)
        self._apply_background_color_choice((red, green, blue))
        self._update_status(f"Picked RGB({red}, {green}, {blue}) at {image_x}, {image_y}")

    def set_background_mode(self, _index: int | None = None) -> None:
        self.background_mode = self.mode_combo.currentData()
        self._refresh_mode_controls()
        self._invalidate_result("Mode changed. Process the image again.")

    def set_tolerance(self, value: int) -> None:
        self.tolerance = value
        self._update_tolerance_display()
        self._invalidate_result("Tolerance changed. Process the image again.")

    def set_lightness_threshold(self, value: int) -> None:
        self.lightness_threshold = value
        self._update_lightness_display()
        self._invalidate_result("Light cutoff changed. Process the image again.")

    def set_result_preview_background(self, _index: int | None = None) -> None:
        self.processed_canvas.set_background_style(
            self.preview_background_combo.currentData()
        )

    def set_output_mode(self, _index: int | None = None) -> None:
        self.output_mode = self.output_mode_combo.currentData()
        self._refresh_output_controls()
        self._invalidate_result("Output mode changed. Process the image again.")

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

    def _set_background_color(
        self,
        color: tuple[int, int, int],
        invalidate_result: bool,
    ) -> None:
        self.background_color = color
        self._update_color_display()

        if invalidate_result:
            self._invalidate_result("Color changed. Process the image again.")

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
            self.processed_result = None
            self.processed_canvas.set_pixmap(None)
            self.export_button.setEnabled(False)
            self.result_meta_label.setText(message)

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

        if color_mode:
            self.mode_help_label.setText(
                "Removes pixels near the picked RGB color. Good for green or blue screens."
            )
        else:
            self.mode_help_label.setText(
                "Removes bright low-saturation pixels. Good for white, gray, or checker backgrounds behind stamps."
            )

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

    def _output_parameter_text(self) -> str:
        if self.output_mode == OUTPUT_SOLID:
            return f"Solid {self._color_to_hex(self.replacement_color)}"
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
