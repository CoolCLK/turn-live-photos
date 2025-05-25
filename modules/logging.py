import logging
import colorama

def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format=colorama.Fore.LIGHTBLACK_EX + '[' + colorama.Fore.CYAN + 'turn-live-photos' + colorama.Fore.LIGHTBLACK_EX + '/' + colorama.Fore.LIGHTBLUE_EX + '%(name)s' + colorama.Fore.LIGHTBLACK_EX + '] [' + colorama.Fore.GREEN + '%(asctime)s' + colorama.Fore.LIGHTBLACK_EX + '] [' + colorama.Fore.RED + '%(levelname)s' + colorama.Fore.LIGHTBLACK_EX + '] ' + colorama.Fore.RESET + '%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    return logger