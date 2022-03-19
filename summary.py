#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from torchstat import stat

from nets.hrnet import HRnet

if __name__ == "__main__":
    model = HRnet(num_classes=21, backbone='hrnetv2_w32', pretrained=False)
    stat(model, (3, 480, 480))
