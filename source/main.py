import os
from typing import List

import numpy as np
import torch
from packages import cleaner, clustering, common, synthesis
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = common.main(r"logs/dev.log")
    if device.type == "cpu":
        logger.warning("CUDA is not available, using CPU")

    # shutil.rmtree("data/out")
    db_path = "data/db/cutted_set/Author1"
    n_tiles = 6
    n_clusters = 10000

    # pulisco le immagini con fft
    db_preprocessed_path = "data/out/preprocessed"
    os.makedirs(db_preprocessed_path, exist_ok=True)

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

    # eseguo la sintesi delle immagini
    synthetized_path = "data/out/synthetized"
    os.makedirs(synthetized_path, exist_ok=True)

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
            # libero la ram (fondamentale per evitare memory error)
            del synth_1
            del synth_2
            del synth_merge
            del synth_sample
            # eseguo il clustering fcm
            try:
                clustering.fcm(
                    logger,
                    "temp/synth_merge",
                    "temp/synth_sample",
                    "temp/centroids",
                    n_tiles * n_tiles,
                    0.1,
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
