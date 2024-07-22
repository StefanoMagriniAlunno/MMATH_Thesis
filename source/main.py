import os

import torch
from packages import cleaner, common

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = common.main(r"logs/dev.log")

    db_path_test = "data/db/Stefano"

    # pulisco le immagini con fft
    db_path_out = "data/db/Stefano_fft"
    if not os.path.exists(db_path_out):
        os.mkdir(db_path_out)

    try:
        cleaner.cleaner(logger, db_path_test, db_path_out, "fft", False, 0.0001, device)
    except ValueError:
        logger.error("Input non validi")
        raise
    except Exception:
        logger.error("Unexpected error")
        raise

    # pulisco le immagini con svd
    db_path_out = "data/db/Stefano_svd"
    if not os.path.exists(db_path_out):
        os.mkdir(db_path_out)

    try:
        cleaner.cleaner(logger, db_path_test, db_path_out, "svd", False, 0.01, device)
    except ValueError:
        logger.error("Input non validi")
        raise
    except Exception:
        logger.error("Unexpected error")
        raise
