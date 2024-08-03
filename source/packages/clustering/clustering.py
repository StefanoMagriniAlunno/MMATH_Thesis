import os
from logging import Logger

import libclustering


def fcm(
    logger: Logger | None,
    datafile_path: str,
    centroids_path: str,
    outfile_path: str,
    n_tiles: str,
    log_file_path: str,
):
    """This function synthesizes all images in a directory.

    :param logger: Logger object
    :type logger: Logger|None
    :param datafile_path: file with all data
    :type datafile_path: str
    :param centroids_path: file with the initial centroids
    :type outfile_path: str
    :param outfile_path: file that will have computed centroids
    :type outfile_path: str
    :param n_tiles: size of tiles
    :type n_tiles: int
    :param log_file_path: relative path to the log file of fcm function
    :type log_file_path: str


    :raises SyntaxError: SyntaxError detected in synthesis.wrapper

    :usage:



    .. code-block:: python

        try:

        except SyntaxError:
            logger.critical("Implementation error!")
            raise

    In this example, you can monitor the progress of the synthesis
    process by checking the log file at "path/to/log_file".

    :pseudo code:

    .. code-block:: none


    .. :no-index:
    """
    datafile_completepath = os.path.join(os.getcwd(), datafile_path)
    centroids_completepath = os.path.join(os.getcwd(), centroids_path)
    outfile_completepath = os.path.join(os.getcwd(), outfile_path)
    log_file_completepath = os.path.join(os.getcwd(), log_file_path)

    if not os.path.exists(datafile_completepath):
        raise ValueError(f"input database {datafile_completepath} does not exist")
    if not os.path.exists(centroids_completepath):
        raise ValueError(f"output database {centroids_completepath} does not exist")

    if not os.path.exists(os.path.dirname(outfile_completepath)):
        raise ValueError(
            f"directory of list file {outfile_completepath} does not exist"
        )
    if not os.path.exists(os.path.dirname(log_file_completepath)):
        raise ValueError(
            f"directory of list file {log_file_completepath} does not exist"
        )

    try:
        libclustering.fcmwrapper(
            datafile_completepath,
            centroids_completepath,
            outfile_completepath,
            n_tiles,
            log_file_completepath,
        )
    except SyntaxError:
        if logger:
            logger.critical("SyntaxError detected in synthesis.wrapper")
        else:
            print("SyntaxError detected in synthesis.wrapper")
        raise
    except Exception:
        if logger:
            logger.error("Unexpected error")
        else:
            print("Unexpected error")
        raise
