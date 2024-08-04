import os
from logging import Logger

import libclustering


def fcm(
    logger: Logger,
    datafile_path: str,
    centroids_path: str,
    outfile_path: str,
    n_dimensions: int,
    tollerance: float,
    log_file_path: str,
):
    """This function performs the Fuzzy C-Means clustering algorithm.

    :param logger: Logger object
    :type logger: Logger
    :param datafile_path: file with all data
    :type datafile_path: str
    :param centroids_path: file with the initial centroids
    :type outfile_path: str
    :param outfile_path: file that will have computed centroids
    :type outfile_path: str
    :param n_dimensions: number of dimensions
    :type n_dimensions: int
    :param tollerance: tollerance
    :type tollerance: float
    :param log_file_path: relative path to the log file of fcm function
    :type log_file_path: str

    :raises SyntaxError: SyntaxError detected in fcm.wrapper
    :raises ValueError: input values are not valid
    :raises OSError: IOError detected in fcm.wrapper
    :raises RuntimeError: Device error detected in fcm.wrapper

    :usage:

    If you want compute the centroids of the images in the directory "path/to/datafile",
    you can use the following code:

    .. code-block:: python

        try:
            fcm(
                logger,  # logger object
                "path/to/datafile",  # file with all data
                "path/to/centroids",  # file with the initial centroids
                "path/to/outfile",  # file that will have computed centroids
                10,  # number of dimensions
                "path/to/log_file",  # relative path to the log file of fcm function
                0.0005,  # tollerance
            )
        except SyntaxError:
            logger.critical("SyntaxError detected in fcm.wrapper")
            raise
        except ValueError as e:
            logger.error(e)
            raise
        except OSError as e:
            logger.error(e)
            raise
        except RuntimeError as e:
            logger.error(e)
            raise
        except Exception as e:
            logger.error(e)
            raise

    In this example, you can monitor the progress of the Fuzzy C-Means clustering algorithm
    in the log file "path/to/log_file".

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
            n_dimensions,
            tollerance,
            log_file_completepath,
        )
    except SyntaxError:
        logger.critical("SyntaxError detected in synthesis.wrapper")
        raise
    except ValueError as e:
        logger.error(e)
        raise
    except OSError as e:
        logger.error(e)
        raise
    except RuntimeError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.error(e)
        raise
