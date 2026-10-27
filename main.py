"""Audio Trimmer. Run with: python main.py [optional audio file]"""
import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def dark_palette() -> QPalette:
    palette = QPalette()
    colors = {
        QPalette.Window: "#141416",
        QPalette.WindowText: "#e6e6e6",
        QPalette.Base: "#0f1012",
        QPalette.AlternateBase: "#1b1c1f",
        QPalette.ToolTipBase: "#24262b",
        QPalette.ToolTipText: "#e6e6e6",
        QPalette.Text: "#e6e6e6",
        QPalette.Button: "#24262b",
        QPalette.ButtonText: "#e6e6e6",
        QPalette.Highlight: "#ffc940",
        QPalette.HighlightedText: "#141416",
        QPalette.Link: "#3d9bff",
        QPalette.PlaceholderText: "#6b6f76",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor("#5d6066"))
    return palette


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Trimmer")
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        window.load(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
