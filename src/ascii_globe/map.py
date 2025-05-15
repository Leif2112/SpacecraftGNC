# ascii_globe/map.py
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def lonlat_to_grid(lon, lat, width, height, lat_scale=1):
    """Convert lon/lat to grid X/Y using equirectangular projection with vertical compression."""
    x = int((lon + 180.0) / 360.0 * width)
    y = int((90.0 - lat) / 180.0 * height * lat_scale)
    return x, y


def draw_line(grid, x0, y0, x1, y1, char="●"):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    max_y, max_x = grid.shape

    while True:
        if 0 <= x0 < max_x and 0 <= y0 < max_y:
            grid[y0, x0] = char
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def fill_polygon(grid, polygon, width, height, fill_char="●", lat_scale=1):
    minx, miny, maxx, maxy = polygon.bounds
    scaled_height = int(height * lat_scale)

    x0, y0 = lonlat_to_grid(minx, maxy, width, height, lat_scale)
    x1, y1 = lonlat_to_grid(maxx, miny, width, height, lat_scale)

    # Clamp to grid size using actual grid dimensions
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, scaled_height - 1))
    y1 = max(0, min(y1, scaled_height - 1))

    for y in range(min(y0, y1), max(y0, y1) + 1):
        for x in range(min(x0, x1), max(x0, x1) + 1):
            lon = x / width * 360.0 - 180.0
            lat = 90.0 - y / scaled_height * 180.0
            if polygon.contains(Point(lon, lat)):
                grid[y, x] = fill_char


def overlay_ground_track(grid, track_points, width, height, lat_scale=0.45, symbol="▲"):
    count = 0
    for lon, lat in track_points:
        x, y = lonlat_to_grid(lon, lat, width, height, lat_scale)
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            grid[y, x] = symbol
            count += 1
    print(f"DEBUG: Plotted {count} track points")   

def generate_ascii_map(
    shapefile_path: str,
    width: int = 240,
    height: int = 120,
    land_char: str = "●",
    lat_scale: float = 0.45,
    track_points: list[tuple[float, float]] = None
) -> str:
    scaled_height = int(height * lat_scale)
    grid = np.full((scaled_height, width), " ", dtype=str)

    # Load and draw countries
    world = gpd.read_file(shapefile_path)
    world["geometry"] = world["geometry"].simplify(tolerance=0.1, preserve_topology=True)

    for _, country in world.iterrows():
        geom = country.geometry
        if geom is None:
            continue

        polygons = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polygons:
            fill_polygon(grid, poly, width, height, land_char, lat_scale)
            try:
                coords = list(poly.exterior.coords)
                for i in range(len(coords) - 1):
                    x0, y0 = lonlat_to_grid(*coords[i], width, height, lat_scale)
                    x1, y1 = lonlat_to_grid(*coords[i + 1], width, height, lat_scale)
                    if 0 <= y0 < grid.shape[0] and 0 <= y1 < grid.shape[0]:
                        draw_line(grid, x0, y0, x1, y1, land_char)
            except Exception:
                continue

    # === Overlay satellite track
    if track_points:
        overlay_ground_track(grid, track_points, width, height, lat_scale, symbol="▲")

    ascii_lines = ["".join(row) for row in grid]
    while ascii_lines and ascii_lines[-1].strip() == "":
        ascii_lines.pop()
    return "\n".join(ascii_lines)

