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

import numpy as np
from cytomine import CytomineJob
from cytomine.models import ImageInstanceCollection, AnnotationCollection, Annotation
from cytomine.models._utilities.parallel import generic_parallel
from shapely.affinity import affine_transform
from skimage import img_as_uint
from skimage.filters import *
from sldc import SemanticMerger
from mask_to_polygons import mask_to_objects_2d
from sldc_adapter import CytomineProjectionSlide, CytomineProjectionTileBuilder

__author__ = "Rubens Ulysse <urubens@uliege.be>"
__copyright__ = "Copyright 2010-2020 University of Liège, Belgium, https://uliege.cytomine.org/"


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
                window = tile.np_image
                threshold = filter_func(window)
                return window, threshold

            cj.log("Get tiles for image {}".format(image.instanceFilename))
            sldc_image = CytomineProjectionSlide(image, projection)
            tile_builder = CytomineProjectionTileBuilder("/tmp")
            topology = sldc_image.tile_topology(tile_builder, tile_size, tile_size, 5)

            results = generic_parallel(topology, worker_tile_func)
            thresholds = list()
            for result in results:
                tile, output = result
                window, threshold = output
                thresholds.append(threshold)

            global_threshold = int(np.mean(thresholds))
            cj.log("Mean threshold is {}".format(global_threshold))

            def worker_annotations_func(tile):
                filtered = img_as_uint(tile.np_image > global_threshold)
                return mask_to_objects_2d(filtered, offset=tile.abs_offset)

            cj.log("Extract annotations from filtered tiles for image {}".format(image.instanceFilename))
            results = generic_parallel(topology, worker_annotations_func)
            ids, geometries = list(), list()
            for result in results:
                tile, tile_geometries = result
                # Workaround for slow SemanticMerger but geometries shouldn't be filtered at this stage.
                tile_geometries = [g for g in tile_geometries if g.area > cj.parameters.min_area]
                ids.append(tile.identifier)
                geometries.append(tile_geometries)

            cj.log("Merge annotations from filtered tiles for image {}".format(image.instanceFilename))
            merged_geometries = SemanticMerger(tolerance=1).merge(ids, geometries, topology)
            cj.log("{} merged geometries".format(len(merged_geometries)))
            ac = AnnotationCollection()
            for geometry in merged_geometries:
                if geometry.area > cj.parameters.min_area:
                    ac.append(Annotation(location=change_referential(geometry, image.height).wkt,
                                         id_image=image.id, id_terms=term_ids))
            ac.save()

        cj.job.update(statusComment="Finished.", progress=100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
