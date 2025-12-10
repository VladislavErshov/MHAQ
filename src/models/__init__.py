import sys
from torchvision.models import resnet18
from .cls.resnet.resnet_cifar10 import resnet18_cifar10
from .cls.resnet.resnet_cifar import resnet20_cifar10
from .cls.resnet.resnet_cifar import resnet20_cifar10_new
from .cls.resnet.resnet_cifar import resnet20_cifar100
from .cls.resnet.resnet_cifar10 import resnet50_cifar10
from .od import yolo_v11
from .od.yolo_v11 import yolo_v11_n

sys.modules["nets.nn"] = yolo_v11
sys.modules["nets"] = yolo_v11

__all__ = ["resnet18_cifar10", "resnet18", "resnet20_cifar10", "resnet20_cifar10_new", "resnet20_cifar100", "resnet34_cifar10", "resnet50_cifar10", "yolo_v11_n"]