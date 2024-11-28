import argparse
import logging
import os
import random
import typing

import numpy
import pandas
import pycuda.autoinit
import pycuda.driver
import torch
import tqdm
from packages import cleaner, clustering, common, distance, synthesis


def main_comparing(
    logger: logging.Logger,
    synthetized_path: str,
    n_tiles: int,
    n_clusters: int,
    fcm_tollerance: float,
):

    pycuda.driver.Device(0)
    context = pycuda.driver.Context.get_current()

    # Imposta la modalità di calcolo su EXCLUSIVE_PROCESS
    context.set_cache_config(pycuda.driver.func_cache.PREFER_SHARED)

    # leggo tuttti i file da esaminare
    files: typing.List[str] = []
    for dirname, _, filenames in os.walk(synthetized_path):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    logger.info(f"Detected {len(files)} files in {synthetized_path}")

    # make a dataframe of float values with shape filesxfiles, it is inizialized with nan values
    # if exists r"./data/distances.csv" load the csv file
    if os.path.exists(r"./data/distances.csv"):
        data_frame = pandas.read_csv(r"./data/distances.csv", index_col=0).astype(
            numpy.float32
        )
    else:
        data_frame = pandas.DataFrame(
            numpy.full((len(files), len(files)), numpy.nan)
        ).astype(numpy.float32)
        data_frame.columns = files
        data_frame.index = files
        # inizializzo a 0 tutta la diagonale
        for file in files:
            data_frame.loc[file, file] = 0.0
        data_frame.to_csv(r"./data/distances.csv", float_format="%.16f")

    for work_1_index in tqdm.tqdm(range(len(files)), "clustering", leave=False):
        work_2_indices = [
            i
            for i in range(len(files))
            if (i - work_1_index + (i >= work_1_index)) % 2 == 0
        ]

        # ? clustering of graph described by dataframe (using N clusters)
        # ? sort work_1_index using size of the clusters
        # ? take next work_1_index using a work from a little cluester

        # remove from work_2_indices the index with "already computed" files
        work_2_indices_new = []
        for work_2_index in work_2_indices:
            work_1 = files[work_1_index]
            work_2 = files[work_2_index]
            if numpy.isnan(data_frame.loc[work_1, work_2]):
                work_2_indices_new.append(work_2_index)
            else:
                logger.info(f"{work_1} and {work_2} already computed")
        work_2_indices = work_2_indices_new

        # work_2 reshuffling
        random.shuffle(work_2_indices)

        for work_2_index in tqdm.tqdm(work_2_indices, files[work_1_index], leave=False):
            work_1 = files[work_1_index]
            work_2 = files[work_2_index]
            if numpy.isnan(data_frame.loc[work_1, work_2]):
                logger.info(f"Processing {work_1} and {work_2}...")
                # estraggo le sintesi interessate
                with open(work_1, "br") as f:
                    values = f.read()
                synth_1 = (
                    numpy.frombuffer(values, dtype=numpy.uint8)
                    .reshape(-1, n_tiles * n_tiles)
                    .astype(numpy.float32)
                    / 255.0
                )
                with open(work_2, "br") as f:
                    values = f.read()
                synth_2 = (
                    numpy.frombuffer(values, dtype=numpy.uint8)
                    .reshape(-1, n_tiles * n_tiles)
                    .astype(numpy.float32)
                    / 255.0
                )
                # unisco le due sintesi in un' unica matrice
                synth_merge = numpy.vstack((synth_1, synth_2))
                # salvo la matrice in un file temporaneo come float32 binario (i dati da clusterizzare)
                with open(r"./temp/synth_merge", "bw") as f:
                    f.write(synth_merge.tobytes())
                # estraggo un campione di n_clusters righe da synth_merge
                synth_sample = numpy.random.choice(
                    synth_merge.shape[0], n_clusters, replace=False
                )
                synth_sample = synth_merge[synth_sample]
                # aggiungo del noise
                synth_sample += numpy.random.normal(0, 0.01, synth_sample.shape)
                # salvo il campione in un file temporaneo come float32 binario (i centroidi iniziali)
                with open(r"./temp/synth_sample", "bw") as f:
                    f.write(synth_sample.tobytes())

                # costruisco il campione dei pesi, tutti uguali per i rispettivi synth
                synth_weights = numpy.ones((synth_merge.shape[0]), dtype=numpy.float32)
                # mi chiedo quale sia la sintesi con più dati
                synth_weights[: synth_1.shape[0]] = (
                    numpy.ones((synth_1.shape[0]), dtype=numpy.float32)
                    / synth_1.shape[0]
                )
                synth_weights[synth_1.shape[0] :] = (
                    numpy.ones((synth_2.shape[0]), dtype=numpy.float32)
                    / synth_2.shape[0]
                )
                # salvo i pesi in un file temporaneo come float32 binario
                with open(r"./temp/synth_weights", "bw") as f:
                    f.write(synth_weights.tobytes())

                # libero la ram (fondamentale per evitare memory error)
                del values
                del synth_1
                del synth_2
                del synth_merge  # ora presente in ./temp/synth_merge
                del synth_sample  # ora presente in ./temp/synth_sample
                del synth_weights  # ora presente in ./temp/synth_weights

                # eseguo il clustering fcm
                try:
                    clustering.fcm(
                        logger,
                        r"./temp/synth_merge",
                        r"./temp/synth_weights",
                        r"./temp/synth_sample",
                        r"./temp/centroids",
                        n_tiles * n_tiles,
                        fcm_tollerance,
                        r"./logs/fcm.log",
                    )
                except SyntaxError:
                    logger.critical("Implementation error!")
                    exit()
                except ValueError:
                    logger.error("Unvalid inputs")
                    exit()
                except Exception:
                    logger.error("Unexpected error")
                    exit()

                # uso i centroidi in "./temp/centroids" per calcolare la distanza tra i due synth
                # load data
                with open(work_1, "br") as f:
                    values = f.read()
                synth_1 = (
                    numpy.frombuffer(values, dtype=numpy.uint8)
                    .reshape(-1, n_tiles * n_tiles)
                    .astype(numpy.float32)
                    / 255.0
                )
                with open(work_2, "br") as f:
                    values = f.read()
                synth_2 = (
                    numpy.frombuffer(values, dtype=numpy.uint8)
                    .reshape(-1, n_tiles * n_tiles)
                    .astype(numpy.float32)
                    / 255.0
                )
                weights_1 = (
                    numpy.ones((synth_1.shape[0]), dtype=numpy.float32)
                    / synth_1.shape[0]
                )
                weights_2 = (
                    numpy.ones((synth_2.shape[0]), dtype=numpy.float32)
                    / synth_2.shape[0]
                )
                with open(r"./temp/centroids", "br") as f:
                    values = f.read()
                centroids = numpy.frombuffer(values, dtype=numpy.float32).reshape(
                    n_clusters, n_tiles * n_tiles
                )
                # compute distance
                try:
                    dist = distance.compute_distance(
                        logger,
                        synth_1,
                        synth_2,
                        weights_1,
                        weights_2,
                        centroids,
                    )
                except SyntaxError:
                    logger.critical("Implementation error!")
                    exit()
                except ValueError:
                    logger.error("Unvalid inputs")
                    exit()
                except Exception:
                    logger.error("Unexpected error")
                    exit()
                else:
                    logger.info(f"Computed distance between {work_1}, {work_2}: {dist}")
                    # save output in the dataframe
                    data_frame.loc[work_1, work_2] = dist
                    data_frame.loc[work_2, work_1] = dist
                    # save the datadrame in ./distances.csv
                    data_frame.to_csv(r"./data/distances.csv", float_format="%.16f")
            else:
                logger.info(f"{work_1} and {work_2} already computed")


def main_synthesis(
    logger: logging.Logger,
    db_preprocessed_path: str,
    synthetized_path: str,
    n_tiles: int,
):
    try:
        synthesis.synthetizer(
            logger,
            db_preprocessed_path,
            synthetized_path,
            r"./temp/synthesis_list.log",
            n_tiles,
            8,
            r"./logs/synthesis.log",
        )
    except SyntaxError:
        logger.critical("Implementation error!")
        raise
    except ValueError:
        logger.error("Unvalid inputs")
        raise
    except Exception:
        logger.error("Unexcpected error")
        raise
    logger.info("Synthesis completed!")


def main_cleaning(
    logger: logging.Logger,
    device: torch.device,
    db_path: str,
    db_preprocessed_path: str,
):
    try:
        cleaner.fft(
            logger, db_path, db_preprocessed_path, False, 0.0005, device
        )  # remove best 0.1%
    except ValueError:
        logger.error("Unvalid inputs")
        raise
    except Exception:
        logger.error("Unexpected error")
        raise
    logger.info("Cleaning completed!")


def main():
    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument(
        "--cleaning", action="store_true", help="Start the cleaning process"
    )
    parser.add_argument(
        "--synthesis", action="store_true", help="Start the synthesis process"
    )
    parser.add_argument(
        "--comparing", action="store_true", help="Start the comparing process"
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger = common.main(r"./logs/dev.log")

    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        exit()
    device = torch.device("cuda")
    pycuda_device = pycuda.driver.Device(0)
    if device.type == "cpu":
        logger.warning("CUDA is not available, using CPU")

    # shutil.rmtree("data/out")
    db_path = r"./data/db/cutted_set"
    n_tiles = 4
    n_centroids = 1_024
    fcm_tollerance = 0.1
    if (
        n_centroids
        > pycuda_device.get_attributes()[
            pycuda.driver.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR
        ]
    ):
        logger.error("Too many centroids")
        exit()

    # pulisco le immagini con fft
    db_preprocessed_path = r"./data/.out/preprocessed"
    os.makedirs(db_preprocessed_path, exist_ok=True)
    # eseguo la sintesi delle immagini
    synthetized_path = r"./data/.out/synthetized"
    os.makedirs(synthetized_path, exist_ok=True)

    if args.cleaning:
        main_cleaning(
            logger,
            device,
            db_path,
            db_preprocessed_path,
        )
    if args.synthesis:
        main_synthesis(
            logger,
            db_preprocessed_path,
            synthetized_path,
            n_tiles,
        )
    if args.comparing:
        main_comparing(logger, synthetized_path, n_tiles, n_centroids, fcm_tollerance)


if __name__ == "__main__":
    main()
