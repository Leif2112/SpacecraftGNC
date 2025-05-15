# File: ascii_globe/model.py
from typing import List
import numpy as np
import geopandas as gpd
import shapely.vectorized as sv
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

class AsciiMapModel:
    """
    Handles loading, simplifying, and rasterizing world geometry into an ASCII grid.
    """
    def __init__(
        self,
        shapefile_path: str,
        width: int,
        height: int,
        land_char: str = "●"
    ) -> None:
        self.shapefile_path = shapefile_path
        self.width = width
        self.height = height
        self.land_char = land_char
        self.full_grid: np.ndarray = np.full((self.height, self.width), " ", dtype='<U1')

        # Precompute lon/lat grid for vectorized operations
        xs = np.linspace(-180.0, 180.0, self.width)
        ys = np.linspace(90.0, -90.0, self.height)
        self.lon_grid, self.lat_grid = np.meshgrid(xs, ys)

        self._load_and_simplify()
        self._rasterize()

    def _load_and_simplify(self) -> None:
        world = gpd.read_file(self.shapefile_path)
        simplified = world.geometry.simplify(tolerance=0.1, preserve_topology=True)
        self.geometry: MultiPolygon = unary_union(simplified)

    def _rasterize(self) -> None:
        # Regenerate coordinate grid based on current resolution
        xs = np.linspace(-180.0, 180.0, self.width)
        ys = np.linspace(90.0, -90.0, self.height)
        self.lon_grid, self.lat_grid = np.meshgrid(xs, ys)

        self.full_grid = np.full((self.height, self.width), " ", dtype="<U1")

        # Fill interior using vectorized contains
        mask = sv.contains(self.geometry, self.lon_grid, self.lat_grid)
        self.full_grid[mask] = self.land_char

        # Draw polygon outlines
        polys = [self.geometry] if isinstance(self.geometry, Polygon) else list(self.geometry.geoms)
        for poly in polys:
            coords = np.array(poly.exterior.coords)
            xs_idx = np.clip(((coords[:, 0] + 180.0) / 360.0 * (self.width - 1)).round().astype(int), 0, self.width - 1)
            ys_idx = np.clip(((90.0 - coords[:, 1]) / 180.0 * (self.height - 1)).round().astype(int), 0, self.height - 1)
            for (x0, y0), (x1, y1) in zip(zip(xs_idx, ys_idx), zip(xs_idx[1:], ys_idx[1:])):
                self._draw_line(x0, y0, x1, y1)


    def _draw_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self.full_grid[y0, x0] = self.land_char
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def get_viewport(
        self,
        top: int,
        left: int,
        rows: int,
        cols: int,
        downsample: bool = False
    ) -> List[str]:
        grid = self.full_grid
        if downsample:
            row_factor = max(1, grid.shape[0] // rows)
            col_factor = max(1, grid.shape[1] // cols)
            small = grid[::row_factor, ::col_factor]
            small = small[:rows, :cols]
            return ["".join(row).ljust(cols) for row in small]

        sub = grid[top:top+rows, left:left+cols]
        result = []
        for r in range(rows):
            if r < sub.shape[0]:
                line = sub[r]
                if sub.shape[1] < cols:
                    line = np.pad(line, (0, cols-sub.shape[1]), constant_values=' ')
                result.append("".join(line))
            else:
                result.append(' ' * cols)
        return result