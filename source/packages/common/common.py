import logging
import os
import tempfile
from logging import Logger
from typing import IO


class LogBase:
    """Base class for logging.
    When inherited, this class logs the initialization and deletion of the class.

    :emphasis:`usage`

    >>> class my_class(LogBase):
    ...     def __init__(self, logger: Logger):
    ...         super().__init__(logger)
    ...         self.logger.info("Hello, world!")
    ...     def __del__(self):
    ...         super().__del__()
    ...         logger.info("Goodbye, world!")
    ...
    >>> my_instance = my_class()
    >>> del my_instance

    """

    def __init__(self, logger: Logger):
        logger.debug(f"{self.__class__.__name__}.__init__()")
        self.logger = logger

    def __del__(self):
        self.logger.debug(f"{self.__class__.__name__}.__del__()")


def main(log_path: str) -> Logger:
    """Call this function to get the logger"""
    logging.basicConfig(
        filename=os.path.join(os.getcwd(), log_path),
        filemode="w",
        format="%(asctime)-16s | %(processName)-16s %(threadName)-16s | %(levelname)-8s | %(pathname)s %(lineno)d : %(message)s",
        datefmt="%Y%m%d_%H%M%S",
        level=logging.DEBUG,
    )
    return logging.getLogger(__name__)


def tempgen(directory: str = "./temp") -> IO[bytes]:
    """This function manage temporary files in a directory

    :emphasis:`params`
        - :attr:`directory` (str, optional): directory with temporary files (default `./temp`).

    :emphasis:`returns`
        - File name of the binary temporary file

    :emphasis:`raise`
        - :exc:`Exception` directory not found

    """
    if not os.path.exists(directory):
        raise Exception(f"Directory '{directory}' not found.")
    return tempfile.NamedTemporaryFile(dir=directory)
