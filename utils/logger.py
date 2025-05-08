# utils/logger.py

import logging
import sys
import os
import platform
from datetime import datetime

# 在Windows平台上启用ANSI颜色支持
if platform.system() == 'Windows':
    try:
        import colorama
        colorama.init()
    except ImportError:
        # 如果没有安装colorama，尝试使用Windows API启用ANSI
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            # 如果都失败，则在下面使用时会禁用颜色
            pass

# ANSI颜色代码
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    
    # 前景色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 背景色
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # 检查是否支持颜色
    @staticmethod
    def supports_color():
        """检查终端是否支持颜色"""
        plat = platform.system()
        supported_platform = plat != 'Windows' or 'ANSICON' in os.environ or 'WT_SESSION' in os.environ
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        if not supported_platform or not is_a_tty:
            return False
        return True
    
    @staticmethod
    def disable_colors():
        """禁用所有颜色代码"""
        for name in dir(Colors):
            if name.isupper() and not name.startswith('__') and isinstance(getattr(Colors, name), str):
                setattr(Colors, name, '')
        return Colors

class ColoredFormatter(logging.Formatter):
    """自定义日志格式化器，支持彩色输出"""
    
    FORMATS = {
        logging.DEBUG: Colors.DIM + '%(asctime)s - ' + Colors.BLUE + '%(levelname)s' + Colors.RESET + Colors.DIM + ' - %(message)s' + Colors.RESET,
        logging.INFO: '%(asctime)s - ' + Colors.GREEN + '%(levelname)s' + Colors.RESET + ' - %(message)s',
        logging.WARNING: '%(asctime)s - ' + Colors.YELLOW + '%(levelname)s' + Colors.RESET + ' - %(message)s',
        logging.ERROR: '%(asctime)s - ' + Colors.RED + '%(levelname)s' + Colors.RESET + ' - %(message)s',
        logging.CRITICAL: Colors.BG_RED + '%(asctime)s - %(levelname)s - %(message)s' + Colors.RESET,
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(level=logging.INFO, enable_color=True):
    """设置日志配置
    
    Args:
        level: 日志级别
        enable_color: 是否启用彩色输出
    """
    # 检查颜色支持
    if enable_color and not Colors.supports_color():
        # 如果不支持颜色但要求启用，输出一个警告
        sys.stderr.write("Warning: Terminal does not support color output. Color disabled.\n")
        enable_color = False
    
    # 如果不启用颜色，禁用所有颜色代码
    if not enable_color:
        Colors.disable_colors()
    
    logger = logging.getLogger()  # root logger
    logger.setLevel(level)

    # 清除旧的 handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # 新增一个新的 StreamHandler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # 根据是否启用彩色选择不同的格式化器
    if enable_color:
        ch.setFormatter(ColoredFormatter())
    else:
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                         datefmt='%Y-%m-%d %H:%M:%S'))
    
    logger.addHandler(ch)

    # 强制确保 propagation 打开
    logger.propagate = True
    
    return logger