import logging
import os
import time


def logger_setup(mode):
    """
    Set up deepquest logs.
    :param mode: Mode of operation to be appended to log file name (string)
    :return: logger and logging objects
    """
    logFileName = 'logs/%s-deepquest-%s.log' % (time.strftime("%Y-%m-%d-%H-%M-%S"), mode)
    if not os.path.exists(os.path.dirname(logFileName)):
        os.makedirs(os.path.dirname(logFileName))
    logging.basicConfig(level=logging.INFO)
    fileh = logging.FileHandler(logFileName, 'w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)

    logger = logging.getLogger()  # root logger
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(fileh)      # set the new handler
    logger.addHandler(logging.StreamHandler())

    return logger, logging
