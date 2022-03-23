import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from torch import Tensor

class BottleneckDecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, pad_num=1, dil_num=1, dropRate=0.0):
        super(BottleneckDecoderBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.InstanceNorm2d(in_planes + 32)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.bn3 = nn.InstanceNorm2d(in_planes + 2*32)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.bn4 = nn.InstanceNorm2d(in_planes + 3*32)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.bn5 = nn.InstanceNorm2d(in_planes + 4*32)
        self.relu5 = nn.LeakyReLU(inplace=True)
        self.bn6 = nn.InstanceNorm2d(in_planes + 5*32)
        self.relu6= nn.LeakyReLU(inplace=True)
        self.bn7 = nn.InstanceNorm2d(inter_planes)
        self.relu7= nn.LeakyReLU(inplace=True)
        self.bn8 = nn.InstanceNorm2d(in_planes+out_planes)
        self.relu8= nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=pad_num, dilation=dil_num, bias=False)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=pad_num, dilation=dil_num, bias=False)
        self.conv3 = nn.Conv2d(in_planes + 2*32, 32, kernel_size=3, stride=1,
                               padding=pad_num, dilation=dil_num, bias=False)
        self.conv4 = nn.Conv2d(in_planes + 3*32, 32, kernel_size=3, stride=1,
                               padding=pad_num, dilation=dil_num, bias=False)
        self.conv5 = nn.Conv2d(in_planes + 4*32, 32, kernel_size=3, stride=1,
                               padding=pad_num, dilation=dil_num, bias=False)
        self.conv6 = nn.Conv2d(in_planes + 5*32, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv7 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=pad_num, dilation=dil_num, bias=False)
        self.conv8 = nn.Conv2d(in_planes+out_planes,out_planes,1,1)
        self.droprate = dropRate

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        out = self.conv7(self.relu7(self.bn7(out6)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        #out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        output = self.conv8(self.relu8(self.bn8(torch.cat([x, out], 1))))
        return output+x



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=pad_num, dilation=dil_num, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=pad_num, dilation=dil_num, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=pad_num, dilation=dil_num, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = nn.Dropout2d(0.1)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out1 = self.droprate(out)
        return F.upsample_nearest(out1, scale_factor=2)



class refineblock(nn.Module):
    def __init__(self, in_planes,OUTPUT_C):
        super(refineblock, self).__init__()

        self.conv_refin = nn.Conv2d(in_planes, 20, 3, 1, 1)
        self.tanh = nn.Tanh()
        
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine3 = nn.Conv2d(20 + 4, 20, kernel_size=3, stride=1, padding=1)
        ##
        self.refine4 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.refine5 = nn.Conv2d(20, 20, kernel_size=7, stride=1, padding=3)
        self.refine6 = nn.Conv2d(20, OUTPUT_C, kernel_size=7, stride=1, padding=3)
        ##
        self.upsample = F.upsample
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):

        x9 = self.relu(self.conv_refin(x))


        shape_out = x9.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out, mode='bilinear',align_corners=True)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out, mode='bilinear',align_corners=True)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out, mode='bilinear',align_corners=True)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out, mode='bilinear',align_corners=True)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)

        dehaze = self.tanh(self.refine3(dehaze))
        dehaze = self.relu(self.refine4(dehaze))
        dehaze = self.relu(self.refine5(dehaze))     
        
        dehaze = self.refine6(dehaze)
        return dehaze

class Transiondown(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(Transiondown,self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(in_planes,out_planes,3,1,1)
        self.down = nn.AvgPool2d(3,2,1)

    def forward(self,x):
        out = self.conv0(self.relu(self.bn1(x)))
        return self.down(out)


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNetOriginal(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), concat_channels=(0,0,0,0),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseNetOriginal, self).__init__()

        # First convolution
        self.add_module('conv0', nn.Conv2d(6, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False))
        self.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.add_module('relu0', nn.ReLU(inplace=True))
        num_init_features += concat_channels[0]
        self.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if len(concat_channels)  > i+1:
                num_features += concat_channels[i+1]
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.add_module('norm5', nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x ,skip0=None, skip1=None, skip2=None, skip3=None):
        # x1 = self.relu0(self.norm0(self.conv0(x)))
        # x1 = torch.cat([x1,skip0],dim=1)
        # #x2 = self.pool0(x1)
        # #x2 = torch.cat([x2,skip1],dim=1)
        # x2 = self.transition1(self.denseblock1(x1))
        # x2 = torch.cat([x2,skip1],dim=1)
        # x3 = self.transition2(self.denseblock2(x2))
        # #x4 = torch.cat([x4,skip3],dim=1)
        # #x5 = self.transition3(self.denseblock3(x4))
        # x4 = self.norm5(self.denseblock3(x3))

        x1 = self.relu0(self.norm0(self.conv0(x)))
        x1 = torch.cat([x1,skip0],dim=1)
        x2 = self.pool0(x1)
        x2 = torch.cat([x2,skip1],dim=1)
        x3 = self.transition1(self.denseblock1(x2))
        x3 = torch.cat([x3,skip2],dim=1)
        x4 = self.transition2(self.denseblock2(x3))
        x4 = torch.cat([x4,skip3],dim=1)
        x5 = self.transition3(self.denseblock3(x4))
        x6 = self.norm5(self.denseblock4(x5))
        
        return x1,x2,x3,x4,x6#x1,x2,x4

