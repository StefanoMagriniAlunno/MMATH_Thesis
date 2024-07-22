import os
from logging import Logger
from typing import List

import numpy as np
import torch
from PIL import Image
from torch import device
from tqdm import tqdm


def cleaner(
    logger: Logger,
    in_db_path: str,
    out_db_path: str,
    method: str,
    descending: bool,
    percentile: float,
    device: device = "cpu",
):
    """This function clean all images in a directory.

    :emphasis:`params`
        - :attr:`logger` (:type:`Logger`): logger object
        - :attr:`in_db_path` (:type:`str`): input database path
        - :attr:`out_db_path` (:type:`str`): output database path
        - :attr:`method` (:type:`str`): method to clean the images (svd, fft, static)
        - :attr:`descending` (:type:`bool`): descending order
        - :attr:`percentile` (:type:`float`): percentile value
        - :attr:`device` (:type:`device`): device to use (default: 'cpu')

    :emphasis:`raises`
        - :exc:`ValueError`: input database does not exist
        - :exc:`ValueError`: output database does not exist
        - :exc:`ValueError`: method not supported

    """

    if not os.path.exists(in_db_path):
        raise ValueError(f"input database {in_db_path} does not exist")
    if not os.path.exists(out_db_path):
        raise ValueError(f"output database {out_db_path} does not exist")

    if method not in ["svd", "fft", "static"]:
        raise ValueError(f"method {method} not supported")

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

    if method == "svd":
        for rel_path in tqdm(files, "svd", leave=False):
            logger.info(f"Processing file: {rel_path}")
            image = Image.open(os.path.join(in_db_path, rel_path)).convert("L")
            image_matrix = np.array(image, dtype=float)
            image_tensor = (
                torch.tensor(image_matrix, dtype=torch.float32, device=device) / 255.0
            )
            image_tensor = 1 - image_tensor

            U, S, V = torch.svd(image_tensor)

            threshold = torch.quantile(S, percentile)

            if descending:
                S = torch.where(S > threshold, S, torch.tensor(0.0))
            else:
                S = torch.where(S < threshold, S, torch.tensor(0.0))

            image_tensor = torch.mm(U, torch.mm(torch.diag(S), V.t()))

            # normalize in [0, 1]
            image_tensor = (image_tensor - torch.min(image_tensor)) / (
                torch.max(image_tensor) - torch.min(image_tensor)
            )

            image = Image.fromarray(
                ((1 - image_tensor) * 255.0).type(torch.uint8).cpu().numpy()
            )
            image.save(os.path.join(out_db_path, rel_path))

    elif method == "fft":
        for rel_path in tqdm(files, "fft", leave=False):
            logger.info(f"Processing file: {rel_path}")
            image = Image.open(os.path.join(in_db_path, rel_path)).convert("L")
            image_matrix = np.array(image, dtype=float) / 255.0
            image_tensor = torch.tensor(
                image_matrix, dtype=torch.float32, device=device
            )
            image_tensor = 1 - image_tensor

            fft_image = torch.fft.fft2(image_tensor, norm="ortho")
            fft_image_amplitude = torch.abs(fft_image)

            fft_image_amplitude_flatten = fft_image_amplitude.flatten()
            _, indices = torch.sort(fft_image_amplitude_flatten, descending=True)
            threshold = fft_image_amplitude_flatten[
                indices[int(percentile * len(fft_image_amplitude_flatten))]
            ]

            if descending:
                mask = fft_image_amplitude > threshold
            else:
                mask = fft_image_amplitude < threshold
            fft_image = fft_image * mask

            image_tensor = torch.fft.ifft2(fft_image, norm="ortho").real

            # normalize
            image_tensor = (image_tensor - torch.min(image_tensor)) / (
                torch.max(image_tensor) - torch.min(image_tensor)
            )

            image = Image.fromarray(
                ((1 - image_tensor) * 255.0).type(torch.uint8).cpu().numpy()
            )
            image.save(os.path.join(out_db_path, rel_path))
