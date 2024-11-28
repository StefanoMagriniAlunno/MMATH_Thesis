import logging
import os
import typing

import libsynthesis


def synthetizer(
    logger: logging.Logger,
    in_db_path: str,
    out_db_path: str,
    list_file_path: str,
    n_tiles: int,
    n_threads: int,
    log_file_path: str,
):
    """This function synthesizes all images in a directory.

    :param logger: logging.Logger object
    :type logger: logging.Logger
    :param in_db_path: input database relative path
    :type in_db_path: str
    :param out_db_path: output database relative path
    :type out_db_path: str
    :param list_file_path: relative path to the list of all files in the input database
    :type list_file_path: str
    :param n_tiles: size of tiles
    :type n_tiles: int
    :param n_threads: number of threads
    :type n_threads: int
    :param log_file_path: relative path to the log file of synthesis function
    :type log_file_path: str

    :raises SyntaxError: SyntaxError detected in synthesis.wrapper
    :raises ValueError: values are not valid
    :raises OSError: IOError detected in synthesis.wrapper
    :raises MemoryError: MemoryError detected in synthesis.wrapper

    :usage:

    If you want to synthesize all images in a directory:

    .. code-block:: python

        try:
            synthesis.synthetizer(
                logger,
                "path/to/in_db",
                "path/to/out_db",
                "path/to/list_file",
                "path/to/log_file",
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

    In this example, you can monitor the progress of the synthesis
    process by checking the log file at "path/to/log_file".

    :pseudo code:

    .. code-block:: none

        FUNCTION synthetizer(logger, in_db_path, out_db_path, list_file_path, log_file_path, n_tailes, n_threads)
            # Check if input and output databases exist
            IF NOT exists(in_db_path) THEN
                RAISE ValueError("Input database does not exist")
            END IF
            IF NOT exists(out_db_path) THEN
                RAISE ValueError("Output database does not exist")
            END IF
            IF NOT exists(dirname(list_file_path)) THEN
                RAISE ValueError("Directory of list file does not exist")
            END IF
            IF NOT exists(dirname(log_file_path)) THEN
                RAISE ValueError("Directory of log file does not exist")
            END IF

            # Create log file
            CREATE log_file_path

            # Create directories in out_db_path
            FOR each directory IN in_db_path
                relative_path = relative(directory, in_db_path)
                IF NOT exists(join(out_db_path, relative_path)) THEN
                    CREATE join(out_db_path, relative_path)
                END IF
            END FOR

            # Create list file
            list_file = []

            FOR each directory IN in_db_path
                FOR each file IN directory
                    relative_path = relative(file, in_db_path)
                    list_file.append(relative_path)
                END FOR
            END FOR

            CREATE list_file_path
            FOR each file IN list_file
                WRITE file TO list_file_path
            END FOR

            # Call synthesis.wrapper
            CALL libsynthesis.wrapper(in_db_path, out_db_path, list_file_path, log_file_path, n_tailes, n_threads)
        END FUNCTION

    .. :no-index:
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

    list_file: typing.List[str] = []
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
            n_tiles,
            n_threads,
            log_file_completepath,
        )
    except SyntaxError as e:
        logger.critical(e)
        raise
    except ValueError as e:
        logger.error(e)
        raise
    except OSError as e:
        logger.error(e)
        raise
    except MemoryError as e:
        logger.error(e)
        raise
    except Exception:
        logger.error("Unexpected error")
        raise
