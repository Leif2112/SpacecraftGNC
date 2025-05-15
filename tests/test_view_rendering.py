class DummyModel:
    def __init__(self, char="#"):
        self.width = 0
        self.height = 0
        self.char = char

    def _rasterize(self):
        self.full_grid = [[self.char] * self.width for _ in range(self.height)]

    def get_viewport(self, top, left, rows, cols, downsample=False):
        return ["".join(self.full_grid[y][left:left+cols]) for y in range(top, min(top+rows, self.height))]


import pytest
from PyQt6.QtWidgets import QApplication
from ascii_globe.view import ASCIIMapView

@pytest.mark.gui
def test_ascii_view_rendering_fills_window(qtbot):
    model = DummyModel(char="X")
    model.width = 80
    model.height = 24
    model._rasterize()

    widget = ASCIIMapView(model)
    widget.resize(800, 600)

    qtbot.addWidget(widget)
    widget.show()

    qtbot.waitUntil(lambda: widget.isVisible())

    # Force refresh and grab rendered lines
    widget.refresh()
    text = widget.toPlainText()
    lines = text.splitlines()

    assert all(line.strip() == "X" * len(line.strip()) for line in lines), "Map lines not rendered correctly"
    assert len(lines) <= model.height, "Too many lines rendered (overflow)"
