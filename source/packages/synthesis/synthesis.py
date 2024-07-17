import os
from logging import Logger
from typing import List

import synthesis


def synth(
    logger: Logger,
    in_db_path: str,
    out_db_path: str,
    list_file_path: str,
    log_file_path: str,
):
    # aggiungo la path completa
    in_db_completepath = os.path.join(os.getcwd(), in_db_path)
    out_db_completepath = os.path.join(os.getcwd(), out_db_path)
    list_file_completepath = os.path.join(os.getcwd(), list_file_path)
    log_file_completepath = os.path.join(os.getcwd(), log_file_path)

    if not os.path.exists(in_db_completepath):
        raise ValueError(f"input database {in_db_completepath} does not exist")
    if not os.path.exists(out_db_completepath):
        raise ValueError(f"output database {out_db_completepath} does not exist")

    if not os.path.exists(os.path.dirname(list_file_completepath)):
        raise ValueError(
            f"directory of list file {list_file_completepath} does not exist"
        )
    if not os.path.exists(os.path.dirname(log_file_completepath)):
        raise ValueError(
            f"directory of list file {log_file_completepath} does not exist"
        )

    # creo il file di log come file vuoto
    with open(log_file_completepath, "w", encoding="utf-8") as f:
        pass

    # creo il file con la lista delle immagini
    list_file: List[str] = []
    for dirpath, _, filenames in os.walk(in_db_completepath):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(file_path, start=in_db_completepath)
            list_file.append(relative_path)
    with open(list_file_completepath, "w") as f:
        for file_path in list_file:
            f.write(file_path + "\n")

    # chiamo la funzione C
    try:
        synthesis.wrapper(
            in_db_completepath,
            out_db_completepath,
            list_file_completepath,
            log_file_completepath,
        )
    except SyntaxError as e:
        logger.critical("SyntaxError detected in synthesis.wrapper")
        raise e
    except IOError as e:
        raise Exception("IOError detected in synthesis.wrapper") from e
    except MemoryError as e:
        raise Exception("MemoryError detected in synthesis.wrapper") from e
    except Exception:
        raise
