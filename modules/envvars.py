#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
处理环境变量的工具。

作者: CoolCLK
"""

import os

class EnvironmentVariable:
    """处理环境变量的通用类"""

    __key = None

    def __init__(self, key):
        """
        初始化一个环境变量工具。

        :param key: 键名
        :type key: str
        """
        self.__key = key

    def __setvalue(self, value):
        """
        设定环境变量的值。

        :param value: 值
        :type value: str
        """
        os.environ[self.__key] = value

class __TensorFlow(EnvironmentVariable):
    """处理 tensorflow 环境变量"""

    def __init__(self):
        super().__init__('TF_CPP_MIN_LOG_LEVEL')

    def set_min_log_level(self, level: int):
        """
        设定日志等级。

        :param level: 日志等级，仅限于 0~3 。
        :type level: int
        """
        if level < 0 or level > 3:
            raise ValueError("Value out of range.")
        self.__setvalue(str(level))

class __PyTorch():
    """处理 pytorch 环境变量"""

    __cuda_alloc_conf = EnvironmentVariable('PYTORCH_CUDA_ALLOC_CONF')
    __cuda_alloc_conf_max_split_size_mb = 0
    __cuda_alloc_conf_expandable_segments = False

    def set_max_split_size_mb(self, value: int):
        """
        设定最大分区大小。

        :param value: 以 MB 为单位。
        :type value: int
        """
        if value <= 0:
            raise ValueError("Value out of range.")
        self.__cuda_alloc_conf_expandable_segments = value
        self.__buildvalue__()

    def set_expandable_segments(self, enable: bool):
        """
        启用动态分区。

        :param enable: 以 MB 为单位。
        :type enable: bool
        """
        self.__cuda_alloc_conf_expandable_segments = enable
        self.__buildvalue__()
    
    def __buildvalue__(self):
        confs = []
        if self.__cuda_alloc_conf_max_split_size_mb > 0:
            confs += "max_split_size_mb:%s" % self.__cuda_alloc_conf_max_split_size_mb
        if self.__cuda_alloc_conf_expandable_segments > 0:
            confs += "expandable_segments:%s" % self.__cuda_alloc_conf_expandable_segments
        self.__cuda_alloc_conf.__setvalue(','.join(confs))

tensorflow = __TensorFlow()
pytroch = __PyTorch()