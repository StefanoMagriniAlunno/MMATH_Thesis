import os
import shutil

import numpy as np
import torch
from packages import cleaner, common, poster, synthesis
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = common.main(r"logs/dev.log")

    shutil.rmtree("data/out")
    db_path = "data/db/cutted_set/Author2"
    n_tiles = 6

    # pulisco le immagini con fft
    db_preprocessed_path = "data/out/preprocessed"
    os.makedirs(db_preprocessed_path, exist_ok=True)

    try:
        cleaner.fft(
            logger, db_path, db_preprocessed_path, False, 0.001, device
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
            "temp/synth.log",
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

    # trasformo ogni sintesi in una matrice numpy di float che salvo in un file numpy
    for dirname, _, filenames in os.walk(synthetized_path):
        rel_path = os.path.relpath(dirname, synthetized_path)
        for filename in tqdm(filenames, rel_path, leave=False):
            with open(os.path.join(dirname, filename), "br") as f:
                values = f.read()
            matrix = (
                np.frombuffer(values, dtype=np.uint8).reshape(-1, n_tiles).astype(float)
            )
            # rimuovo il vecchio file
            os.remove(os.path.join(dirname, filename))
            # savlo il nuovo file
            np.save(os.path.join(dirname, filename), matrix)
        logger.info(f"{rel_path} processed")

    # eseguo le posterizzazioni delle immagini
    db_posterised_path = "data/out/posterised"
    os.makedirs(db_posterised_path, exist_ok=True)
    poster.models(logger, db_path, db_posterised_path, 64)
    logger.info("posterisation completed")

    # per ogni opera propongo un insieme
