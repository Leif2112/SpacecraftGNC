import sys
import os
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPlainTextEdit
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtCore import Qt
from ascii_globe.map import generate_ascii_map

track = [
    (-160 + i * 10, 10 * np.sin(i * 0.2))  # Fake sinusoidal LEO path
    for i in range(36)
]



class MapModule(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASCII Map Viewer")

        # === Font Setup ===
        font = QFont("Courier New", 4)
        font.setStyleHint(QFont.StyleHint.Monospace)
        metrics = QFontMetrics(font)
        char_width = metrics.horizontalAdvance("M")
        char_height = metrics.height()

        # === Map Generation with Squished Vertical ===
        map_width = 480
        map_height = 240
        lat_scale = 0.45
        shapefile_path = os.path.join("data", "ne_10m_admin_0_countries_lakes.shp")

        for lon, lat in track:
            print(f"TRACK POINT: lon={lon:.2f}, lat={lat:.2f}")

        ascii_map = generate_ascii_map(
            shapefile_path=shapefile_path,
            width=map_width,
            height=map_height,
            land_char="_",
            lat_scale=lat_scale,
            track_points=track
        )

        lines = ascii_map.splitlines()
        max_line_length = max(len(line) for line in lines)
        line_count = len(lines)

        # === Clamp Window to Actual Map Size ===
        window_width = max_line_length * char_width + 12
        window_height = line_count * char_height + 30
        self.setFixedSize(window_width, window_height)

        # === Text Display ===
        self.text_display = QPlainTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text_display.setFont(font)
        self.text_display.setStyleSheet("""
            QPlainTextEdit {
                background-color: black;
                color: #FFFFFF;
                border: none;
            }
        """)
        self.setCentralWidget(self.text_display)
        self.text_display.setPlainText(ascii_map)


def main():
    app = QApplication(sys.argv)
    window = MapModule()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
