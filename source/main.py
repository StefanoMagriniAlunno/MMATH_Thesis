import os
import shutil

import torch
from packages import cleaner, common, synthesis

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = common.main(r"logs/dev.log")

    db_path = "data/db/cutted_set"

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
    for folder in os.listdir(db_preprocessed_path):
        author_path_input = os.path.join(db_preprocessed_path, folder)
        author_path_output = os.path.join(synthetized_path, folder)
        os.makedirs(author_path_output, exist_ok=True)
        logger.info(f"Processing {author_path_input}...")
        try:
            synthesis.synthetizer(
                logger,
                author_path_input,
                author_path_output,
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
        # comprimo la cartella author_path_output
        logger.info(f"Compressing {author_path_output}...")
        shutil.make_archive(
            author_path_output, "zip", author_path_output, logger=logger
        )
        # elimino la vecchia directory
        shutil.rmtree(author_path_output)
    logger.info("Synthesis completed!")
