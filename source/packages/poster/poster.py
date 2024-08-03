import os
from logging import Logger
from typing import List

import joblib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm


def models(
    logger: Logger,
    in_db_path: str,
    out_db_path: str,
    discrete_levels: int,
):
    """This function posterises all images and saving the kmeans models.

    :params logger: logger object
    :type logger: Logger
    :params in_db_path: input database path
    :type in_db_path: str
    :params out_db_path: output database path
    :type out_db_path: str
    :params discrete_levels: number of discrete levels
    :type discrete_levels: int
    :params n_threads: number of threads
    :type n_threads: int

    """

    if not os.path.exists(in_db_path):
        raise ValueError(f"input database {in_db_path} does not exist")
    if not os.path.exists(out_db_path):
        raise ValueError(f"output database {out_db_path} does not exist")

    # rilevo tutti i file da convertire come path relative
    files: List[str] = []
    for dirname, _, filenames in os.walk(in_db_path):
        for filename in filenames:
            rel_path = os.path.relpath(dirname, in_db_path)
            files.append(os.path.join(rel_path, filename))
    logger.info(f"Detected {len(files)} files in {in_db_path}")

    # preparo la cartella di destinazione
    for dirname, _, filenames in os.walk(in_db_path):
        for filename in filenames:
            rel_path = os.path.relpath(dirname, in_db_path)
            if not os.path.exists(os.path.join(out_db_path, rel_path)):
                os.mkdir(os.path.join(out_db_path, rel_path))
                break

    for file in tqdm(files, "poster", leave=False):
        logger.info(f"Processing file: {file}")
        img_array = (
            np.array(
                Image.open(os.path.join(in_db_path, file)).convert("L"), dtype=float
            )
            / 255.0
        )
        # eseguo la clusterizzazione
        max_iter = 50
        tol = 1e-2
        kmeans = KMeans(n_clusters=discrete_levels, max_iter=max_iter, tol=tol)
        kmeans.fit(img_array.flatten().reshape(-1, 1))
        if kmeans.n_iter_ == max_iter:
            logger.warning(
                f"during posterisation of image {file}, the kmeans algorithm reach max number of iterations ({max_iter}) with tollerance {kmeans.inertia_}"
            )
        # salvo il modello
        joblib.dump(kmeans, os.path.join(out_db_path, file + ".pkl"))
        logger.info(f"KMeans model saved at {os.path.join(out_db_path, file + '.pkl')}")
