import logging
import logging.config
import os
import sys

from utils.utils import make_dirs_if_not_present
from .timer import Timer


class Logger:
    """
    """

    def __init__(self, path, log_level=None, name=None):
        """

        Args:
            log_level: 
            name: 
        """
        self.logger = None
        self.timer = Timer()

        self.log_filename = "train_"
        self.log_filename += self.timer.get_time()
        self.log_filename += ".log"

        self.log_folder = os.path.join(path, 'logs/')
        make_dirs_if_not_present(self.log_folder)

        self.log_filename = os.path.join(self.log_folder, self.log_filename)

        logging.captureWarnings(True)

        if not name:
            name = __name__

        self.logger = logging.getLogger(name)

        # Set level
        if log_level is None:
            level = 'INFO'
        else:
            level = log_level
        self.logger.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d-%H:%M:%S",
        )

        # Add handlers
        file_hdl = logging.FileHandler(self.log_filename)
        file_hdl.setFormatter(formatter)
        self.logger.addHandler(file_hdl)
        # logging.getLogger('py.warnings').addHandler(file_hdl)
        cons_hdl = logging.StreamHandler(sys.stdout)
        cons_hdl.setFormatter(formatter)
        self.logger.addHandler(cons_hdl)

    def get_logger(self):
        """
        Returns:
        """
        return self.logger




def logger():
    logger = None


    log_filename = "train_" + Timer().get_time() + ".log"


    log_folder = os.path.join(path, 'logs/')
    make_dirs_if_not_present(log_folder)

    log_filename = os.path.join(log_folder, log_filename)

    logging.captureWarnings(True)

    if not name:
        name = __name__

    logger = logging.getLogger(name)

    # Set level

    logger.setLevel(getattr(logging,'INFO'))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d-%H:%M:%S",
    )

    # Add handlers
    file_hdl = logging.FileHandler(log_filename)
    file_hdl.setFormatter(formatter)
    logger.addHandler(file_hdl)
    # logging.getLogger('py.warnings').addHandler(file_hdl)
    cons_hdl = logging.StreamHandler(sys.stdout)
    cons_hdl.setFormatter(formatter)
    logger.addHandler(cons_hdl)