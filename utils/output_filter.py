# utils/output_filter.py

import sys
import re
import threading
from io import StringIO
import logging

class OutputFilter:
    """捕获和过滤标准输出，用于减少重复的进度条输出"""
    
    def __init__(self, filter_patterns=None):
        self.terminal = sys.stdout
        self.buffer = StringIO()
        self.lock = threading.Lock()
        self.last_line = ""
        self.capturing = False
        
        # 默认过滤模式
        self.filter_patterns = filter_patterns or [
            # 过滤重复的进度条
            r"rtf_avg:.*\d+%\|█+\|.*\[\d+\/\d+",  # 匹配进度条
            # 过滤模型加载的冗长日志
            r"Loading pretrained params from",
            r"Loading ckpt:",
            r"scope_map:",
            r"excludes:",
            r"model_conf is not provided",
            r"download models from model hub",
            r"Building VAD model",
            # 过滤只显示型号和设备的行
            r"funasr version: \d+\.\d+\.\d+",
        ]
        
        # 保留次数计数
        self.counter = {}
        # 进度条的特殊处理 - 只保留奇数行或偶数行取决于需求
        self.rtf_counter = 0
    
    def start(self):
        """开始捕获标准输出"""
        if not self.capturing:
            self.capturing = True
            sys.stdout = self
    
    def stop(self):
        """停止捕获，恢复标准输出"""
        if self.capturing:
            sys.stdout = self.terminal
            self.capturing = False
    
    def should_filter(self, string):
        """判断一行内容是否应该被过滤"""
        
        # 进度条特殊处理（只显示一半）
        if re.search(r"rtf_avg:.*\d+%\|█+\|", string):
            self.rtf_counter += 1
            # 只保留偶数行的进度条
            return self.rtf_counter % 2 == 1
        
        # 处理其他过滤模式
        for pattern in self.filter_patterns:
            if re.search(pattern, string):
                # 匹配此模式的第一次出现
                key = pattern
                self.counter[key] = self.counter.get(key, 0) + 1
                
                # 对于模型加载信息，我们只显示一次
                if "Loading" in pattern or "model" in pattern:
                    return True
                
                return False
        
        # 默认不过滤
        return False
    
    def write(self, string):
        """处理写入到标准输出的文本"""
        with self.lock:
            # 检查是否需要过滤
            if self.should_filter(string):
                # 仍然保存到缓冲区，但不写入终端
                self.buffer.write(string)
                return
            
            # 写入到原始终端
            self.terminal.write(string)
            self.terminal.flush()
            
            # 保存到缓冲区
            self.buffer.write(string)
            self.last_line = string
    
    def flush(self):
        """刷新输出"""
        with self.lock:
            self.terminal.flush()
            self.buffer.flush()
    
    def get_content(self):
        """获取捕获的内容"""
        with self.lock:
            return self.buffer.getvalue()

# 创建全局过滤器实例
output_filter = OutputFilter() 