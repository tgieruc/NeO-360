from .nerds360 import NeRDS360
from .nerds360_ae import NeRDS360_AE
from .carla import Carla
dataset_dict = {
    "nerds360": NeRDS360,
    "nerds360_ae": NeRDS360_AE,
    "carla": Carla,
}