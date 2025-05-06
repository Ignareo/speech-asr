# utils/logger.py

import logging
import sys

def setup_logger(level=logging.INFO):
    logger = logging.getLogger()  # root logger
    logger.setLevel(level)

    # 清除旧的 handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # 新增一个新的 StreamHandler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    # 强制确保 propagation 打开
    logger.propagate = True