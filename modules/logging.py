#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
日志工具。

依赖库: 
- colorama==0.4.6
作者: CoolCLK
"""

import logging
import colorama

def apply_format(level = logging.INFO):
    """
    设定全局 Logger 格式。

    :param name: 名称
    :type name: str
    :return: 对于的 Logger
    :rtype: logging.Logger
    """
    logging.basicConfig(
        level = level,
        format = colorama.Fore.LIGHTBLACK_EX + '[' + colorama.Fore.CYAN + 'turn-live-photos' + colorama.Fore.LIGHTBLACK_EX + '/' + colorama.Fore.LIGHTBLUE_EX + '%(name)s' + colorama.Fore.LIGHTBLACK_EX + '] [' + colorama.Fore.GREEN + '%(asctime)s' + colorama.Fore.LIGHTBLACK_EX + '] [' + colorama.Fore.RED + '%(levelname)s' + colorama.Fore.LIGHTBLACK_EX + '] ' + colorama.Fore.RESET + '%(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

def get_logger(name: str, level = logging.INFO):
    """
    获取一个带格式的 Logger 。

    :param name: 名称
    :type name: str
    :return: 对于的 Logger
    :rtype: logging.Logger
    """
    logging.basicConfig(
        level = level,
        format = colorama.Fore.LIGHTBLACK_EX + '[' + colorama.Fore.CYAN + 'turn-live-photos' + colorama.Fore.LIGHTBLACK_EX + '/' + colorama.Fore.LIGHTBLUE_EX + '%(name)s' + colorama.Fore.LIGHTBLACK_EX + '] [' + colorama.Fore.GREEN + '%(asctime)s' + colorama.Fore.LIGHTBLACK_EX + '] [' + colorama.Fore.RED + '%(levelname)s' + colorama.Fore.LIGHTBLACK_EX + '] ' + colorama.Fore.RESET + '%(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    return logger