import os
from logging import Logger
from typing import List

import libsynthesis


def synth(
    logger: Logger | None,
    in_db_path: str,
    out_db_path: str,
    list_file_path: str,
    log_file_path: str,
    n_tailes: int,
    n_threads: int,
):
    """This function synthesizes all images in a directory.

    :emphasis:`params`
        - :attr:`logger` (:type:`Logger` | :type:`None`): logger object
        - :attr:`in_db_path` (:type:`str`): input database path
        - :attr:`out_db_path` (:type:`str`): output database path
        - :attr:`list_file_path` (:type:`str`): path to the list of all files in the input database
        - :attr:`log_file_path` (:type:`str`): path to the log file of synthesis function
        - :attr:`n_threads` (:type:`int`): number of threads

    :emphasis:`raises`
        - :exc:`SyntaxError`: SyntaxError detected in synthesis.wrapper
        - :exc:`ValueError`: input database does not exist
        - :exc:`ValueError`: output database does not exist
        - :exc:`ValueError`: directory of list file does not exist
        - :exc:`ValueError`: directory of log file does not exist
        - :exc:`Exception`: ValueError detected in synthesis.wrapper
        - :exc:`Exception`: IOError detected in synthesis.wrapper
        - :exc:`Exception`: MemoryError detected in synthesis.wrapper
        - :exc:`Exception`: Unexpected error
    """
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

    with open(log_file_completepath, "w", encoding="utf-8") as f:
        pass

    # ricreo l'albero delle directory di in_db_completepath in out_db_completepath
    for dirname, _, filenames in os.walk(in_db_completepath):
        relative_path = os.path.relpath(dirname, in_db_completepath)
        if not os.path.exists(os.path.join(out_db_completepath, relative_path)):
            os.mkdir(os.path.join(out_db_completepath, relative_path))

    list_file: List[str] = []
    for dirpath, _, filenames in os.walk(in_db_completepath):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(file_path, start=in_db_completepath)
            list_file.append(relative_path)
    with open(list_file_completepath, "w") as f:
        for file_path in list_file:
            f.write(file_path + "\n")

    try:
        libsynthesis.wrapper(
            in_db_completepath,
            out_db_completepath,
            list_file_completepath,
            log_file_completepath,
            n_tailes,
            n_threads,
        )
    except SyntaxError:
        if logger:
            logger.critical("SyntaxError detected in synthesis.wrapper")
        else:
            print("SyntaxError detected in synthesis.wrapper")
        raise
    except ValueError as e:
        raise Exception("ValueError detected in synthesis.wrapper") from e
    except IOError as e:
        raise Exception("IOError detected in synthesis.wrapper") from e
    except MemoryError as e:
        raise Exception("MemoryError detected in synthesis.wrapper") from e
    except Exception:
        if logger:
            logger.error("Unexpected error")
        else:
            print("Unexpected error")
        raise
