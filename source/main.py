import os

import torch
from packages import cleaner, common, synthesis

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = common.main(r"logs/dev.log")

    db_path_test = "data/db/Stefano"

    # pulisco le immagini con fft
    db_path_out_preprocessed = "data/out/preprocessed"
    os.makedirs(db_path_out_preprocessed, exist_ok=True)

    try:
        cleaner.fft(
            logger, db_path_test, db_path_out_preprocessed, False, 0.001, device
        )  # remove best 0.1%
    except ValueError:
        logger.error("Unvalid inputs")
        raise
    except Exception:
        logger.error("Unexpected error")
        raise

    # aumento il contrasto

    """
    cleaner.contrast(
        logger, db_path_out_preprocessed, 0.1, 1.0, 2.0, device
    )
    """

    # eseguo la sintesi delle immagini
    db_path_out_synthetized = "data/out/synthetized"
    os.makedirs(db_path_out_synthetized, exist_ok=True)

    try:
        synthesis.synthetizer(
            logger,
            db_path_out_preprocessed,
            db_path_out_synthetized,
            "temp/synth.log",
            "logs/synthesis.log",
            6,
            8,
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
