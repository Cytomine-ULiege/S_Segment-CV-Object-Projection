import os

from PIL import Image
import numpy as np
from sldc import TileExtractionException, alpha_rasterize
from sldc_cytomine import CytomineSlide, CytomineTile, CytomineTileBuilder


class CytomineProjectionSlide(CytomineSlide):
    def __init__(self, img_instance, projection):
        self._img_instance = img_instance
        self._projection = projection

    @property
    def projection(self):
        return self._projection

    @property
    def np_image(self):
        raise NotImplementedError("Disabled due to the too heavy size of the images")

    @property
    def channels(self):
        return 1


class CytomineProjectionTile(CytomineTile):
    def __init__(self, working_path, parent, offset, width, height, tile_identifier=None, **kwargs):
        super().__init__(working_path, parent, offset, width, height, tile_identifier)

    @property
    def np_image(self):
        try:
            image_instance = self.base_image.image_instance
            x, y, width, height = self.abs_offset_x, self.abs_offset_y, self.width, self.height

            # check if the tile was cached
            cache_filename_format = "{id}-{x}-{y}-{w}-{h}.png"
            cache_filename = cache_filename_format.format(id=image_instance.id, x=x, y=y, w=width, h=height)
            cache_path = os.path.join(self._working_path, cache_filename)
            if not os.path.exists(cache_path):
                if not image_instance.window(x=x, y=y, w=width, h=height, dest_pattern=cache_path,
                                             projection=self.base_image.projection):
                    raise TileExtractionException("Cannot fetch tile at for "
                                                  "'{}'.".format(cache_filename_format.split(".", 1)[0]))

            # load image
            np_array = np.asarray(Image.open(cache_path))
            if np_array.shape[1] != width or np_array.shape[0] != height:
                raise TileExtractionException("Fetched image has invalid size : {} instead "
                                              "of {}".format(np_array.shape, (width, height, self.channels)))
            return np_array
        except IOError as e:
            raise TileExtractionException(str(e))

    @property
    def channels(self):
        return 1


class CytomineProjectionTileBuilder(CytomineTileBuilder):
    def __init__(self, working_path):
        super().__init__(working_path)

    def build(self, image, offset, width, height, **kwargs):
        return CytomineProjectionTile(self._working_path, image, offset, width, height)
