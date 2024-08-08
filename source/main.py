import argparse
import os
from typing import List

import numpy as np
import pycuda.autoinit
import pycuda.driver
import torch
from packages import cleaner, clustering, common, synthesis
from tqdm import tqdm


def main_comparing(logger, synthetized_path, n_tiles, n_clusters, fcm_tollerance):

    pycuda.driver.Device(0)
    context = pycuda.driver.Context.get_current()

    # Imposta la modalità di calcolo su EXCLUSIVE_PROCESS
    context.set_cache_config(pycuda.driver.func_cache.PREFER_SHARED)

    # leggo tuttti i file da esaminare
    files: List[str] = []
    for dirname, _, filenames in os.walk(synthetized_path):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    logger.info(f"Detected {len(files)} files in {synthetized_path}")

    for work_1_index in tqdm(range(len(files)), "clustering", leave=False):
        work_2_indices = [
            i
            for i in range(len(files))
            if (i - work_1_index + (i >= work_1_index)) % 2 == 0
        ]
        for work_2_index in tqdm(work_2_indices, files[work_1_index], leave=False):
            work_1 = files[work_1_index]
            work_2 = files[work_2_index]
            logger.info(f"Processing {work_1} and {work_2}...")
            # estraggo le sintesi interessate
            with open(work_1, "br") as f:
                values = f.read()
            synth_1 = (
                np.frombuffer(values, dtype=np.uint8)
                .reshape(-1, n_tiles * n_tiles)
                .astype(np.float32)
                / 255.0
            )
            with open(work_2, "br") as f:
                values = f.read()
            synth_2 = (
                np.frombuffer(values, dtype=np.uint8)
                .reshape(-1, n_tiles * n_tiles)
                .astype(np.float32)
                / 255.0
            )
            # unisco le due sintesi in un unica matrice
            synth_merge = np.vstack((synth_1, synth_2))
            # salvo la matrice in un file temporaneo come float32 binario (i dati da clusterizzare)
            with open("temp/synth_merge", "bw") as f:
                f.write(synth_merge.tobytes())
            # estraggo un campione di n_clusters righe da synth_merge
            synth_sample = np.random.choice(
                synth_merge.shape[0], n_clusters, replace=False
            )
            synth_sample = synth_merge[synth_sample]
            # salvo il campione in un file temporaneo come float32 binario (i centroidi iniziali)
            with open("temp/synth_sample", "bw") as f:
                f.write(synth_sample.tobytes())

            # costruisco il campione dei pesi, tutti uguali per i rispettivi synth
            synth_weights = np.ones((synth_merge.shape[0]), dtype=np.float32)
            # mi chiedo quale sia la sintesi con più dati
            if synth_1.shape[0] > synth_2.shape[0]:
                # chi ha meno dati, ha peso 1.0
                synth_weights[synth_1.shape[0] :] = 1.0
                # chi ha peso maggiore, ha peso il rapporto tra il numero di dati
                synth_weights[: synth_1.shape[0]] = synth_1.shape[0] / synth_2.shape[0]
            else:
                synth_weights[: synth_1.shape[0]] = 1.0
                synth_weights[synth_1.shape[0] :] = synth_2.shape[0] / synth_1.shape[0]
            # salvo i pesi in un file temporaneo come float32 binario
            with open("temp/synth_weights", "bw") as f:
                f.write(synth_weights.tobytes())

            # libero la ram (fondamentale per evitare memory error)
            del synth_1
            del synth_2
            del synth_merge
            del synth_sample
            del synth_weights
            # eseguo il clustering fcm
            try:
                clustering.fcm(
                    logger,
                    "temp/synth_merge",
                    "temp/synth_weights",
                    "temp/synth_sample",
                    "temp/centroids",
                    n_tiles * n_tiles,
                    fcm_tollerance * (n_tiles * n_tiles),
                    "logs/fcm.log",
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
            # todo: trovare un modo di riportare la matrice delle memberships con safe memory
            # note: prova a usare torch con il kernel usato in C++ adattato con pycuda (che dovrebbe poter compilare i kernel)
            # anche se il costo computazionale è inevitabilmente superiore, questo aiuta in una implementazione più sicura


def main_synthesis(
    logger, db_preprocessed_path, synthetized_path, n_tiles, n_centroids, fcm_tollerance
):
    try:
        synthesis.synthetizer(
            logger,
            db_preprocessed_path,
            synthetized_path,
            "temp/synthesis_list.log",
            n_tiles,
            8,
            "logs/synthesis.log",
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
    main_comparing(logger, synthetized_path, n_tiles, n_centroids, fcm_tollerance)


def main_cleaning(
    logger,
    device,
    db_path,
    db_preprocessed_path,
    synthetized_path,
    n_tiles,
    n_centroids,
    fcm_tollerance,
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
    main_synthesis(
        logger,
        db_preprocessed_path,
        synthetized_path,
        n_tiles,
        n_centroids,
        fcm_tollerance,
    )


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
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

    logger = common.main(r"logs/dev.log")

    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        exit()
    device = torch.device("cuda")
    pycuda_device = pycuda.driver.Device(0)
    if device.type == "cpu":
        logger.warning("CUDA is not available, using CPU")

    # shutil.rmtree("data/out")
    db_path = "data/db/cutted_set/Author1"
    n_tiles = 4
    n_centroids = 1024
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
    db_preprocessed_path = "data/out/preprocessed"
    os.makedirs(db_preprocessed_path, exist_ok=True)
    # eseguo la sintesi delle immagini
    synthetized_path = "data/out/synthetized"
    os.makedirs(synthetized_path, exist_ok=True)

    if args.cleaning:
        main_cleaning(
            logger,
            device,
            db_path,
            db_preprocessed_path,
            synthetized_path,
            n_tiles,
            n_centroids,
            fcm_tollerance,
        )
    elif args.synthesis:
        main_synthesis(
            logger,
            db_preprocessed_path,
            synthetized_path,
            n_tiles,
            n_centroids,
            fcm_tollerance,
        )
    elif args.comparing:
        main_comparing(logger, synthetized_path, n_tiles, n_centroids, fcm_tollerance)
    else:
        logger.error("No process selected")


if __name__ == "__main__":
    main()
