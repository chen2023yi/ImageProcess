from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    apply_theme(app)

    window = MainWindow()
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
