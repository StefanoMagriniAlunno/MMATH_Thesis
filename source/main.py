import os

import torch
from packages import cleaner, common, poster

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = common.main(r"logs/dev.log")

    db_path_test = "data/db/Stefano"

    # pulisco le immagini con fft
    db_path_out_fft = "data/db/Stefano_fft"
    if not os.path.exists(db_path_out_fft):
        os.mkdir(db_path_out_fft)

    try:
        cleaner.fft(
            logger, db_path_test, db_path_out_fft, False, 0.001, device
        )  # remove best 0.1%
    except ValueError:
        logger.error("Input non validi")
        raise
    except Exception:
        logger.error("Unexpected error")
        raise

    # procedo con la posterizzazione
    db_path_out_posterized = "data/db/Stefano_posterized"
    if not os.path.exists(db_path_out_posterized):
        os.mkdir(db_path_out_posterized)

    try:
        poster.poster(logger, db_path_out_fft, db_path_out_posterized, 32, 8)
    except ValueError:
        logger.error("Input non validi")
        raise
    except Exception:
        logger.error("Unexpected error")
        raise
