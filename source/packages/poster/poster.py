import os
from logging import Logger

from PIL import Image


def Poster(
    logger: Logger,
    in_db_path: str,
    out_db_path: str,
    discrete_levels: int,
    n_threads: int,
):
    """This function posterize all images in a directory.

    :emphasis:`params`
        - :attr:`logger` (:type:`Logger`): logger object
        - :attr:`in_db_path` (:type:`str`): input database path
        - :attr:`out_db_path` (:type:`str`): output database path
        - :attr:`discrete_levels` (:type:`int`): number of discrete levels
        - :attr:`n_threads` (:type:`int`): number of threads

    """

    for dirname, _, filenames in os.walk(in_db_path):
        for filename in filenames:
            if filename.endswith(".pgm"):
                logger.info(f"Processing file: {filename}")
                image = Image.open(os.path.join(dirname, filename)).convert("L")
                image = image.quantize(discrete_levels, kmeans=discrete_levels)
                image.save(os.path.join(out_db_path, dirname, filename))
