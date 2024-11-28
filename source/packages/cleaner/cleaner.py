import logging
import os
import typing

import numpy
import torch
import tqdm
from PIL import Image


def fft(
    logger: logging.Logger,
    in_db_path: str,
    out_db_path: str,
    preserve: bool,
    percentile: float,
    device: torch.device = "cpu",
):
    """This function cleans all images in a directory using FFT.
    In particular, this function considers significant the frequencies with high amplitude.

    :param logger: logging.Logger object
    :type logger: logging.Logger
    :param in_db_path: Input database path
    :type in_db_path: str
    :param out_db_path: Output database path
    :type out_db_path: str
    :param preserve: Descending order
    :type preserve: bool
    :param percentile: Percentile value
    :type percentile: float
    :param device: Device to use (default: 'cpu')
    :type device: str

    :raises ValueError: If the input database does not exist
    :raises ValueError: If the output database does not exist
    :raises ValueError: If the method is not supported

    :usage:

    If you want to preserve the first 0.01% significant part of the image:

    .. code-block:: python

        cleaner.fft(logger, "path/to/in_db", "path/to/out_db", True, 0.0001, device)

    If you want to remove the first 0.01% significant part of the image:

    .. code-block:: python

        cleaner.fft(logger, "path/to/in_db", "path/to/out_db", False, 0.0001, device)

    .. :no-index:
    """

    if not os.path.exists(in_db_path):
        raise ValueError(f"input database {in_db_path} does not exist")
    if not os.path.exists(out_db_path):
        raise ValueError(f"output database {out_db_path} does not exist")

    # rilevo tutti i file da convertire come path relative
    files: typing.List[str] = []
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

    for rel_path in tqdm.tqdm(files, "fft", leave=False):
        logger.info(f"Processing file: {rel_path}")
        image = Image.open(os.path.join(in_db_path, rel_path)).convert("L")
        image_matrix = numpy.array(image, dtype=float) / 255.0
        image_tensor = torch.tensor(image_matrix, dtype=torch.float32, device=device)

        # analizzo le frequenze
        image_tensor_mean = torch.mean(image_tensor)
        image_tensor = image_tensor - image_tensor_mean
        fft_image = torch.fft.fft2(image_tensor, norm="ortho")
        fft_image_amplitude = torch.abs(fft_image)

        fft_image_amplitude_flatten = fft_image_amplitude.flatten()
        _, indices = torch.sort(fft_image_amplitude_flatten, descending=True)
        threshold = fft_image_amplitude_flatten[
            indices[int(percentile * len(fft_image_amplitude_flatten))]
        ]

        if preserve:
            mask = fft_image_amplitude > threshold
        else:
            mask = fft_image_amplitude < threshold

        fft_image = fft_image * mask

        image_tensor = torch.fft.ifft2(fft_image, norm="ortho").real

        # normalize image
        image_tensor = (image_tensor - torch.min(image_tensor)) / (
            torch.max(image_tensor) - torch.min(image_tensor)
        )
        # count how many pixels are white in original image
        Wpercentile = numpy.count_nonzero(image_matrix >= 0.8) / image_matrix.size
        # count how many pixels are black in original image
        Bpercentile = numpy.count_nonzero(image_matrix <= 0.2) / image_matrix.size
        # in image_tensor the first Wpercentile pixels became white
        # and the first Bpercentile pixels became black
        thresholdW = torch.quantile(image_tensor, 1 - Wpercentile)
        thresholdB = torch.quantile(image_tensor, Bpercentile)
        # normalize image again to have thresholdW in 1 and thresholdB in 0
        image_tensor = (image_tensor - thresholdB) / (thresholdW - thresholdB)
        image_tensor = image_tensor.clamp(0.0, 1.0)

        image = Image.fromarray((image_tensor * 255.0).type(torch.uint8).cpu().numpy())
        image.save(os.path.join(out_db_path, rel_path), format="PPM")
