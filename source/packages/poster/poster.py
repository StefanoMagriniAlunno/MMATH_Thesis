import os
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from typing import List

from PIL import Image
from tqdm import tqdm


def poster(
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

    if not os.path.exists(in_db_path):
        raise ValueError(f"input database {in_db_path} does not exist")
    if not os.path.exists(out_db_path):
        raise ValueError(f"output database {out_db_path} does not exist")

    # rilevo tutti i file da convertire come path relative
    files: List[str] = []
    for dirname, _, filenames in os.walk(in_db_path):
        for filename in filenames:
            if filename.endswith(".pgm"):
                rel_path = os.path.relpath(dirname, in_db_path)
                files.append(os.path.join(rel_path, filename))
    logger.info(f"Detected {len(files)} files in {in_db_path}")

    # preparo la cartella di destinazione
    for dirname, _, filenames in os.walk(in_db_path):
        for filename in filenames:
            if filename.endswith(".pgm"):
                rel_path = os.path.relpath(dirname, in_db_path)
                if not os.path.exists(os.path.join(out_db_path, rel_path)):
                    os.mkdir(os.path.join(out_db_path, rel_path))
                    break

    def process_file(file, in_db_path, out_db_path, discrete_levels, logger):
        try:
            rel_path = os.path.relpath(os.path.join(in_db_path, file))
            logger.info(f"Processing file: {rel_path}")
            img = Image.open(os.path.join(in_db_path, file)).convert("L")
            img = img.quantize(discrete_levels).convert("L")
            img.save(os.path.join(out_db_path, file), format="PPM")
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda file: process_file(
                        file, in_db_path, out_db_path, discrete_levels, logger
                    ),
                    files,
                ),
                desc="poster",
                leave=False,
                total=len(files),
            )
        )
