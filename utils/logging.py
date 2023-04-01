import logging
import time

timestr = time.strftime("%Y%m%d")

logging.basicConfig(filename=f'logs/{timestr}.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  # noqa: E501
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

logger.addHandler(logging.StreamHandler())
