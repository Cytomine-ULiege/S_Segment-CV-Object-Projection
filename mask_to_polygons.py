# -*- coding: utf-8 -*-


import numpy as np
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, MultiPolygon


def geom_as_list(geometry):
    """Return the list of sub-polygon a polygon is made up of"""
    if geometry.geom_type == "Polygon":
        return [geometry]
    elif geometry.geom_type == "MultiPolygon":
        return geometry.geoms


def linear_ring_is_valid(ring):
    points = set([(x, y) for x, y in ring.coords])
    return len(points) >= 3


def fix_geometry(geometry):
    """Attempts to fix an invalid geometry (from https://goo.gl/nfivMh)"""
    try:
        return geometry.buffer(0)
    except ValueError:
        pass

    polygons = geom_as_list(geometry)

    fixed_polygons = list()
    for i, polygon in enumerate(polygons):
        if not linear_ring_is_valid(polygon.exterior):
            continue

        interiors = []
        for ring in polygon.interiors:
            if linear_ring_is_valid(ring):
                interiors.append(ring)

        fixed_polygon = Polygon(polygon.exterior, interiors)

        try:
            fixed_polygon = fixed_polygon.buffer(0)
        except ValueError:
            continue

        fixed_polygons.extend(geom_as_list(fixed_polygon))

    if len(fixed_polygons) > 0:
        return MultiPolygon(fixed_polygons)
    else:
        return None


def flatten_geoms(geoms):
    """Flatten (possibly nested) multipart geometry."""
    geometries = []
    for g in geoms:
        if hasattr(g, "geoms"):
            geometries.extend(flatten_geoms(g))
        else:
            geometries.append(g)
    return geometries


def mask_to_objects_2d(mask, background=0, offset=None, flatten_collection=True):
    """Convert 2D (binary or label) mask to polygons. Generates borders fitting in the objects.
    Parameters
    ----------
    mask: ndarray
        2D mask array. Expected shape: (height, width).
    background: int
        Value used for encoding background pixels.
    offset: tuple (optional, default: None)
        (x, y) coordinate offset to apply to all the extracted polygons.
    flatten_collection: bool
        True for flattening geometry collections into individual geometries.
    Returns
    -------
    extracted: list of Geometry
        Each geometry represent an object from the image.
    """
    if mask.ndim != 2:
        raise ValueError("Cannot handle image with ndim different from 2 ({} dim. given).".format(mask.ndim))
    if offset is None:
        offset = (0, 0)
    exclusion = np.logical_not(mask == background)
    affine = Affine(1, 0, offset[0], 0, 1, offset[1])
    geometries = list()
    for gjson, _ in shapes(mask.copy(), mask=exclusion, transform=affine):
        polygon = shape(gjson)

        # fixing polygon
        if not polygon.is_valid:  # attempt to fix
            polygon = fix_geometry(polygon)
        if not polygon.is_valid:  # could not be fixed
            continue

        if not hasattr(polygon, "geoms") or not flatten_collection:
            geometries.append(polygon)
        else:
            for curr in flatten_geoms(polygon.geoms):
                geometries.append(curr)
    return geometries
