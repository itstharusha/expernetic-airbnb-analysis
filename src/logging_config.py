import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with a standard format.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers are present to avoid duplication
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs if root logger is also configured
        logger.propagate = False

    return logger
