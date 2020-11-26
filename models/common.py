# This file contains modules common to various models
import math

import torch
import torch.nn as nn
from utils.general import non_max_suppression
from utils.activations import MemoryEfficientMish


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=None):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=None):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act == "mish":
            self.act = MemoryEfficientMish()
        elif act == "hardswish":
            self.act = nn.Hardswish()
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act == None:
            self.act = nn.Identity()
        #self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class AddCoords(nn.Module):
    # add Coordinate
    def __init__(self):
        super(AddCoords,self).__init__()
    def forward(self, input_tensor):
        """
        Args:
            input_tensor:shape(batch,channel,x_dim,y_dim)
        """
        batch_size,_,x_dim,y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1,y_dim,1)
        yy_channel = torch.arange(y_dim).repeat(1,x_dim,1).transpose(1,2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size,1,1,1).transpose(2,3)
        yy_channel = yy_channel.repeat(batch_size,1,1,1).transpose(2,3)

        # xx_channel.detach()
        # yy_channel.detach()

        ret = torch.cat([input_tensor,xx_channel.type_as(input_tensor),yy_channel.type_as(input_tensor)],dim=1) # dim = 1 -> add channel
        return ret

class CoordConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=None):
        super(CoordConv,self).__init__()
        self.addcoords = AddCoords()
        c_ = c1+2
        self.conv = nn.Conv2d(c_, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act == "mish":
            self.act = MemoryEfficientMish()
        elif act == "hardswish":
            self.act = nn.Hardswish()
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act == None:
            self.act = nn.Identity()
        # self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(self.addcoords(x))))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=None):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCoord(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=None):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckCoord, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.CoordConv = CoordConv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.CoordConv(x)) if self.add else self.cv2(self.CoordConv(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=None):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class BottleneckCSPCoord(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=None):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPCoord, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1_coord = CoordConv(c1, c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4_coord = CoordConv(2 * c_, c2, 1, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[BottleneckCoord(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1_coord(x)))
        y2 = self.cv2(x)
        return self.cv4_coord(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), act=None):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1,act=act)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=None):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1, act=None):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.3  # confidence threshold
    iou = 0.6  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, dimension=1):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
