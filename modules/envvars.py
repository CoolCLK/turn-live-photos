import os

class __EnvironmentVariable:
    __key = None

    def __init__(self, key):
        self.__key = key

    def __setvalue(self, value):
        os.environ[self.__key] = value

class __TensorFlow(__EnvironmentVariable):
    def __init__(self):
        super().__init__('TF_CPP_MIN_LOG_LEVEL')

    def set_min_log_level(self, level: int):
        if level < 0 or level > 3:
            raise ValueError("Value out of range.")
        self.__setvalue(str(level))

class __PyTorch():
    __cuda_alloc_conf = __EnvironmentVariable('PYTORCH_CUDA_ALLOC_CONF')
    __cuda_alloc_conf_max_split_size_mb = 0
    __cuda_alloc_conf_expandable_segments = False

    def set_max_split_size_mb(self, value: int):
        if value <= 0:
            raise ValueError("Value out of range.")
        self.__cuda_alloc_conf_expandable_segments = value
        self.__buildvalue__()

    def set_expandable_segments(self, enable: bool):
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