import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPlainTextEdit,
    QAction, QFileDialog, QMessageBox
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont

from ascii_globe.map import generate_ascii_map

class AsciiMapWindow(QMainWindow):
    def __init__(self, shapefile_path=None, width=240, height=120, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ASCII Globe Viewer")
        self.resize(800, 600)

        # Central text widget for ASCII map
        self.viewer = QPlainTextEdit(self)
        self.viewer.setReadOnly(True)
        self.viewer.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.viewer.setFont(QFont("Courier", 8))
        self.setCentralWidget(self.viewer)

        # Default map settings
        self.shapefile_path = shapefile_path
        self.map_width = width
        self.map_height = height

        # Timer for periodic redraw (e.g., dynamic data)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_map)
        self.timer.start(1000)  # update every 1 second

        # Menu bar
        self._create_menu()
        if self.shapefile_path:
            self.update_map()

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Shapefile...", self)
        open_action.setStatusTip("Load a new world shapefile")
        open_action.triggered.connect(self.open_shapefile)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def open_shapefile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select World Shapefile", "", "Shapefiles (*.shp)"
        )
        if path:
            self.shapefile_path = path
            try:
                self.update_map()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load shapefile:\n{e}")

    def update_map(self):
        if not self.shapefile_path:
            # Nothing to draw yet
            return
        try:
            ascii_text = generate_ascii_map(
                self.shapefile_path,
                width=self.map_width,
                height=self.map_height
            )
            # Display with padding trimmed to window size
            self.viewer.setPlainText(ascii_text)
        except Exception as e:
            # Show errors in a message box
            self.timer.stop()
            QMessageBox.critical(self, "Rendering Error", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Optionally pass a default shapefile path here
    window = AsciiMapWindow(shapefile_path="path/to/world.shp", width=200, height=80)
    window.show()
    sys.exit(app.exec_())
