import logging
import os

logging_name = ""

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger(logging_name)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

console_handler.setFormatter(CustomFormatter())
logger.addHandler(console_handler)

env_name = ""

try:
    if env_name == "":
        raise KeyError("env_name variable set to '' in __init__")
    home_path = os.environ[env_name]
    file_handler = logging.FileHandler(
        os.path.join(home_path, "{}_log.txt".format(logging_name)))
    file_handler.setFormatter(CustomFormatter())
    file_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.debug("Logging file successfully identified")
except KeyError:
    logger.warning(
        "{} environment variable not set. Logging to file will not be performed".format(env_name))

