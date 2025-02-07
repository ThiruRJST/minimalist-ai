import logging
import os

os.makedirs("../logs", exist_ok=True)

str_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=str_format, handlers=[
    logging.FileHandler("../logs/logfile.log")
])

logger = logging.getLogger(__name__)
