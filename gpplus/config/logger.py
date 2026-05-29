import logging
import sys

# Create a package-wide logger
logger = logging.getLogger("gpplus")
logger.setLevel(logging.CRITICAL)  # Default: Disable logging unless configured by the user

# Prevent log duplication if a handler already exists
if not logger.handlers:
    handler = logging.NullHandler()  # Default: No output
    logger.addHandler(handler)


def configure_logger(level=logging.INFO, log_to_file=None):
    """
    Configures the package-wide logger.

    Parameters:
    - level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    - log_to_file (str | None): File path to log output (if None, logs to stdout)
    """
    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    if log_to_file:
        handler = logging.FileHandler(log_to_file)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    logger.info(
        f"Logger configured: level={logging.getLevelName(level)}, output={'file' if log_to_file else 'console'}"
    )


# Ensure that if the logger has never been configured, we set it up with a default format
if not logger.hasHandlers():
    configure_logger(logging.WARNING)
