# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2020. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from cytomine import CytomineJob
from cytomine.models import ImageInstanceCollection, AnnotationCollection, Annotation
from cytomine.models._utilities.parallel import generic_parallel
from shapely.affinity import affine_transform
from skimage import img_as_uint
from skimage.filters import *
from PIL import Image

__author__ = "Rubens Ulysse <urubens@uliege.be>"
__copyright__ = "Copyright 2010-2020 University of Liège, Belgium, https://uliege.cytomine.org/"

from mask_to_polygons import mask_to_objects_2d


def change_referential(p, height):
    return affine_transform(p, [1, 0, 0, -1, 0, height])


def _get_filter(name):
    if name == 'isodata':
        return threshold_isodata
    elif name == 'otsu':
        return threshold_otsu
    elif name == 'li':
        return threshold_li
    elif name == 'yen':
        return threshold_yen
    else:
        raise ValueError("Filter {} is not found".format(name))


def _get_window_from_tile(tile, image, tile_size):
    tile_x, tile_y = tile
    x = tile_x * tile_size
    y = tile_y * tile_size
    width = min(tile_size, image.width - x)
    height = min(tile_size, image.height - y)
    return x, y, width, height


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(progress=1, statusComment="Initialisation")
        cj.log(str(cj.parameters))

        term_ids = [cj.parameters.cytomine_id_term] if hasattr(cj.parameters, "cytomine_id_term") else None

        image_ids = [int(image_id) for image_id in cj.parameters.cytomine_id_images.split(",")]
        images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
        images = [image for image in images if image.id in image_ids]

        use_global_threshold = True  # TODO
        tile_size = cj.parameters.tile_size
        filter_func = _get_filter(cj.parameters.filter)
        projection = cj.parameters.projection
        if projection not in ('min', 'max', 'average'):
            raise ValueError("Projection {} is not found".format(projection))

        cj.log("Filter: {}".format(cj.parameters.filter))
        cj.log("Projection: {}".format(projection))
        for image in cj.monitor(images, prefix="Running detection on image", start=5, end=99):
            def worker_tile_func(tile):
                x, y, width, height = _get_window_from_tile(tile, image, tile_size)
                dest = os.path.join("/tmp", "{}-{}-{}.png".format(image.id, tile[0], tile[1]))
                image.window(x, y, width, height, dest_pattern=dest, projection=projection, override=False)
                window = np.asarray(Image.open(dest))
                threshold = filter_func(window)
                return window, threshold

            cj.log("Get tiles for image {}".format(image.instanceFilename))
            tiles = []
            x_tiles = int(np.ceil(image.width / tile_size))
            y_tiles = int(np.ceil(image.height / tile_size))
            for x in range(x_tiles):
                for y in range(y_tiles):
                    tiles.append((x, y))

            results = generic_parallel(tiles, worker_tile_func)
            data = []
            for result in results:
                tile, output = result
                window, threshold = output
                data.append((tile, window, threshold))

            thresholds = [t for _, _, t in data]
            mean_threshold = int(np.mean(thresholds))
            cj.log("Mean threshold is {}".format(mean_threshold))

            def worker_annotations_func(data):
                tile, window, threshold = data
                if use_global_threshold:
                    threshold = mean_threshold
                x, y, _, _ = _get_window_from_tile(tile, image, tile_size)
                filtered = img_as_uint(window > threshold)
                geometries = mask_to_objects_2d(filtered, offset=(x, y))
                ac = AnnotationCollection()
                for geometry in geometries:
                    if geometry.area > cj.parameters.min_area:
                        ac.append(Annotation(location=change_referential(geometry, image.height).wkt,
                                             id_image=image.id, id_terms=term_ids))

                ac.save()

            cj.log("Extract annotations from filtered image {}".format(image.instanceFilename))
            results = generic_parallel(data, worker_annotations_func)

        cj.job.update(statusComment="Finished.", progress=100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
