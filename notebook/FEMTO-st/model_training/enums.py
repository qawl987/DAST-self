from enum import Enum

class NormType(Enum):
    NO_NORM = 0
    BATCH_NORM = 1
    LAYER_NORM = 2
    
class TrainType(Enum):
    DL = 0
    TL = 1