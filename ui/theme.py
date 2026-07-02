from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


APP_STYLESHEET = """
QMainWindow {
    background: #F5F7FA;
}

QWidget {
    color: #172033;
    font-size: 13px;
}

QFrame#TopBar,
QFrame#StatusBar,
QFrame#Panel,
QFrame#PreviewPanel {
    background: #FFFFFF;
    border: 1px solid #D9DEE7;
    border-radius: 8px;
}

QFrame#TopBar {
    border-radius: 0;
    border-left: 0;
    border-right: 0;
    border-top: 0;
}

QFrame#StatusBar {
    border-radius: 0;
    border-left: 0;
    border-right: 0;
    border-bottom: 0;
}

QLabel#AppTitle {
    font-size: 15px;
    font-weight: 600;
}

QLabel#SectionTitle,
QLabel#PreviewTitle {
    color: #172033;
    font-size: 13px;
    font-weight: 600;
}

QLabel#MutedText,
QLabel#MetaLabel,
QLabel#StatusText {
    color: #667085;
    font-size: 12px;
}

QLabel#Badge {
    background: #ECFDF3;
    border: 1px solid #ABEFC6;
    border-radius: 6px;
    color: #067647;
    font-size: 12px;
    padding: 4px 8px;
}

QPushButton {
    background: #FFFFFF;
    border: 1px solid #D0D5DD;
    border-radius: 6px;
    color: #172033;
    font-size: 13px;
    font-weight: 600;
    min-height: 34px;
    padding: 8px 12px;
}

QPushButton:hover {
    background: #F8FAFC;
    border-color: #B8C0CC;
}

QPushButton:pressed {
    background: #EEF2F6;
}

QPushButton:disabled {
    background: #F2F4F7;
    border-color: #E4E7EC;
    color: #98A2B3;
}

QPushButton#PrimaryButton {
    background: #2563EB;
    border-color: #2563EB;
    color: #FFFFFF;
}

QPushButton#PrimaryButton:hover {
    background: #1D4ED8;
    border-color: #1D4ED8;
}

QPushButton#PrimaryButton:disabled {
    background: #F2F4F7;
    border-color: #E4E7EC;
    color: #98A2B3;
}

QPushButton#SuccessButton {
    background: #16A34A;
    border-color: #16A34A;
    color: #FFFFFF;
}

QPushButton#SuccessButton:hover {
    background: #15803D;
    border-color: #15803D;
}

QPushButton#SuccessButton:disabled {
    background: #F2F4F7;
    border-color: #E4E7EC;
    color: #98A2B3;
}

QPushButton#DropButton {
    border-style: dashed;
    border-color: #98A2B3;
    color: #344054;
    min-height: 64px;
}

QLineEdit,
QComboBox {
    background: #FFFFFF;
    border: 1px solid #D0D5DD;
    border-radius: 6px;
    color: #172033;
    min-height: 32px;
    padding: 6px 8px;
}

QComboBox:disabled {
    background: #F2F4F7;
    color: #98A2B3;
}

QComboBox QAbstractItemView {
    background: #FFFFFF;
    border: 1px solid #D0D5DD;
    border-radius: 6px;
    color: #172033;
    outline: 0;
    padding: 4px;
    selection-background-color: #EFF6FF;
    selection-color: #172033;
}

QComboBox QAbstractItemView::item {
    background: #FFFFFF;
    color: #172033;
    min-height: 28px;
    padding: 6px 8px;
}

QComboBox QAbstractItemView::item:hover,
QComboBox QAbstractItemView::item:selected {
    background: #EFF6FF;
    color: #172033;
}

QScrollArea#SettingsScroll {
    background: transparent;
    border: 0;
}

QScrollArea#SettingsScroll QWidget {
    background: transparent;
}

QScrollBar:vertical {
    background: transparent;
    margin: 0;
    width: 8px;
}

QScrollBar::handle:vertical {
    background: #CBD5E1;
    border-radius: 4px;
    min-height: 32px;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

QSlider::groove:horizontal {
    background: #E4E7EC;
    border-radius: 3px;
    height: 6px;
}

QSlider::sub-page:horizontal {
    background: #2563EB;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 1px solid #98A2B3;
    border-radius: 8px;
    height: 16px;
    margin: -5px 0;
    width: 16px;
}

QMessageBox {
    background: #FFFFFF;
}
"""


def apply_theme(app: QApplication) -> None:
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)
