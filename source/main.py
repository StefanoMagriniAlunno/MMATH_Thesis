import os

import torch
from packages import cleaner, common, synthesis

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = common.main(r"logs/dev.log")

    db_path_test = "data/db/Stefano"

    # pulisco le immagini con fft
    db_path_out_fft = "data/out/fft"
    os.makedirs(db_path_out_fft, exist_ok=True)

    try:
        cleaner.fft(
            logger, db_path_test, db_path_out_fft, False, 0.001, device
        )  # remove best 0.1%
    except ValueError:
        logger.error("Unvalid inputs")
        raise
    except Exception:
        logger.error("Unexpected error")
        raise

    # eseguo la sintesi delle immagini
    db_path_out_synth = "data/out/synth"
    os.makedirs(db_path_out_synth, exist_ok=True)

    try:
        synthesis.synth(
            logger,
            db_path_out_fft,
            db_path_out_synth,
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
