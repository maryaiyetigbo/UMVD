import logging
import os
from os import path as osp
import datetime


def init_logging(log=True, log_file=None, resume_training=True):
    handlers = [logging.StreamHandler()]
    if log and log_file is not None:
        mode = "a" if os.path.isfile(resume_training) else "w"
        handlers.append(logging.FileHandler(log_file, mode=mode))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info("COMMAND: %s" % " ".join(sys.argv))
    # logging.info("Arguments: {}".format(vars(args)))




def get_logger(logger_name='decay', log_level=logging.INFO, log_folder=None, sequence=None):
    """
    CODE MODIFIED FROM: https://github.com/jiangyitong/RCD/blob/main/basicsr/utils/logger.py
    """
    initialized_logger = {}
    if log_folder is not None:
        os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f'atrain_{sequence}.log')
    logger = logging.getLogger(logger_name)
    format_str = "[%(asctime)s]:  %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(log_level)
    # add file handler
    file_handler = logging.FileHandler(log_file, 'a')
    file_handler.setFormatter(logging.Formatter(format_str, datefmt="%m-%d-%Y %H:%M:%S"))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
        
    initialized_logger[logger_name] = True
    return logger



def close_logger_handlers(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)



def get_filename():
    ct=datetime.datetime.now()
    logname = f"{ct.month:02d}-{ct.day:02d}_{ct.hour:02d}:{ct.minute:02d}"
    return logname
