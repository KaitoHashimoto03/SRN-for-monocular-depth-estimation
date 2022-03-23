# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math
import numpy as np

from collections import namedtuple

#from model.srn_dil import *

import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
from model.module import *
from model.convlstm import ConvLSTMCell
from model.aspp import ASPP_block


# This sets the batch norm layers in pytorch as if {'is_training': False, 'scale': True} in tensorflow
def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()                      # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        #print(m.name)
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0 

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = torch_nn_func.interpolate(x, scale_factor=self.ratio, mode='bilinear')
        out = self.conv(up_x)
        out = self.elu(out)
        return out

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        #interpolate x from x.size to larger size(concat_with.shape=(C,H,W))
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )


# class reduction_1x1(nn.Sequential):
#     def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
#         super(reduction_1x1, self).__init__()        
#         self.max_depth = max_depth
#         self.is_final = is_final
#         self.sigmoid = nn.Sigmoid()
#         self.reduc = torch.nn.Sequential()
        
#         while num_out_filters >= 4:
#             if num_out_filters < 8:
#                 if self.is_final:
#                     self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
#                                                                                  kernel_size=1, stride=1, padding=0),
#                                                                        nn.Sigmoid()))
#                 else:
#                     self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
#                                                                           kernel_size=1, stride=1, padding=0))
#                 break
#             else:
#                 self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
#                                       torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
#                                                                     bias=False, kernel_size=1, stride=1, padding=0),
#                                                           nn.ELU()))

#             num_in_filters = num_out_filters
#             num_out_filters = num_out_filters // 2
    
#     def forward(self, net):
#         net = self.reduc.forward(net)
#         if not self.is_final:
#             theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
#             phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
#             dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
#             n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
#             n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
#             n3 = torch.cos(theta).unsqueeze(1)
#             n4 = dist.unsqueeze(1)
#             net = torch.cat([n1, n2, n3, n4], dim=1)
        
#         return net

class encoder(nn.Module):
    def __init__(self, params):
        super(encoder, self).__init__()
        self.params = params
        import torchvision.models as models
        if params.encoder == 'densenet121_bts':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif params.encoder == 'densenet161_bts':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif params.encoder == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext50_bts':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext101_bts':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        features = [x]
        skip_feat = [x]
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
        
        return skip_feat
    

class BtsModel(nn.Module):
    def __init__(self, params):
        super(BtsModel, self).__init__()
        self.params = params
        # self.encoder = encoder(params)
        # self.decoder = bts(params, self.encoder.feat_out_channels, params.bts_size)

        self.conv0 = nn.Conv2d(3,1,3,1,1)
        self.elu0 = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        self.srn = SRN(params)
        #self.srn.decode.apply(weights_init_xavier)

    def forward(self, x, focal):
        # depths = []
        # outputs = self.srn(x)
        # for i in range(3):
        #     depths.append(self.params.max_depth * self.sigmoid(self.conv0(self.elu0(outputs[i]))))
        x = self.sigmoid(self.conv0(self.elu0(self.srn(x)[-1])))
        final_depth = self.params.max_depth * x
        return final_depth

class DecodeModel(nn.Module):
    def __init__(self,half=False):
        super(DecodeModel,self).__init__()
        feat_out_channels = [64, 64, 128, 256, 1024]#densenet-161[96, 96, 192, 384, 2208]
        en_out_channels = [128,192,320,608,1200]#[160,256,416,784,1288]#[128,192,320,608,1200]
        num_features = 512
        self.upconv5    = upconv(feat_out_channels[4]+en_out_channels[4], num_features)
        self.bn5        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        
        self.conv5      = torch.nn.Sequential(nn.Conv2d(num_features + en_out_channels[3], num_features, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.upconv4    = upconv(num_features, num_features // 2)
        self.bn4        = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4      = torch.nn.Sequential(nn.Conv2d(num_features // 2 + en_out_channels[2], num_features // 2, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.bn4_2      = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.upconv3    = upconv(num_features // 2, num_features // 4)
        self.bn3        = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3      = torch.nn.Sequential(nn.Conv2d(num_features // 4 + en_out_channels[1], num_features // 4, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.bn3_2      = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)


        ############# 256-256  ##############
        # self.conv0 = nn.Conv2d(3 + 3,64,7,2,3)
        # self.norm0 = nn.InstanceNorm2d(64)
        # self.relu0 = nn.LeakyReLU(inplace=True)

        # self.dense_block1 = BottleneckDecoderBlock(64,64)#,pad_num=2,dil_num=2)
        # self.trans_block1 = Transiondown(64,64)
    
        # self.dense1 = BottleneckDecoderBlock(64,64)#,pad_num=2,dil_num=2)
        # self.dense2 = BottleneckDecoderBlock(64,64)#,pad_num=4,dil_num=4)
        self.DenseNetOriginal = DenseNetOriginal(32,(6,12,24,16),feat_out_channels[0:4])
        
        self.lstm = ConvLSTMCell(64,64,(3,3),bias=True)

        self.d_conv4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=1)
        self.up1 = UpSample(64+en_out_channels[0], 32)
        self.up2 = UpSample(skip_input=32 + 6, output_features=16)
        self.d_conv5 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.elu = nn.ELU()
        self.bn_1       = nn.BatchNorm2d(64, momentum=0.01, affine=True, eps=1.1e-5)
        self.bn_2       = nn.BatchNorm2d(64+128, momentum=0.01, affine=True, eps=1.1e-5)
        self.aspp = ASPP_block(128,64,[3,6,12,18,24])

        self.sigmoid = nn.Sigmoid()

    def forward(self,x,state=None,features=None,i=None):
        skip0, skip1, skip2, skip3 = features[1], features[2], features[3], features[4]
        dense_features = torch.nn.ReLU()(features[5])

        e1,e2,e3,e4,e5 = self.DenseNetOriginal(x,skip0,skip1, skip2, skip3) #_C 128,192,320,608,1200 160,256,416,784,1288
        #print(e1.size(),e2.size(),e3.size())
        #print(e1.size(),e2.size(),e3.size(),e4.size(),e5.size())
        dense_features = torch.cat([dense_features,e5],dim=1)

        upconv5 = self.upconv5(dense_features) # H/16
        upconv5 = self.bn5(upconv5)
        #print(dense_features.size(),upconv5.size(),skip3.size())
        concat5 = torch.cat([upconv5, e4], dim=1)#concat5 = torch.cat([upconv5, e4], dim=1)
        iconv5 = self.conv5(concat5)

        #iconv5,h = self.lstm(iconv5,state)
        
        upconv4 = self.upconv4(iconv5) # H/8
        upconv4 = self.bn4(upconv4)
        #print(iconv5.size(),upconv4.size(),skip2.size())
        concat4 = torch.cat([upconv4, e3], dim=1)#concat4 = torch.cat([upconv4, e3], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)

        upconv3 = self.upconv3(iconv4) # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, e2], dim=1)
        iconv3 = self.conv3(concat3)
        iconv3 = self.bn3_2(iconv3)
        x3 = iconv3

        #print("x:",x.size())
        # x0 = self.relu0(self.norm0(self.conv0(x))) #H/2
        # x01 = self.dense_block1(x0)
        # x1 = self.trans_block1(x01) #H/4
        # x2 = self.dense1(x1)
        # x3 = self.dense2(x2)
        # x3 = torch.cat([x3,iconv3],dim=1)

        # x1a = self.bn_1(self.elu(x1))#ASPP
        # x3a = self.bn_2(self.elu(x3))
        # xa = torch.cat([x1a, x3a], dim=1)

        if i==0:
            self.aspp.change_dilation_rates([3,6,9,12,15])
        elif i==1: 
            self.aspp.change_dilation_rates([6,12,18,24,30])
        elif i==2:
            self.aspp.change_dilation_rates([12,24,36,48,60])
        xa = self.aspp(x3)

        #print("x3:",x3.size())
        x31,h = self.lstm(xa,state)
        #h = None
        #x31 = xa

        #x4 = self.d_conv4(x31)

        #skip_1of2s = torch.cat([x01,skip0],dim=1) #_C=64+64
        #e1 = torch.cat([e1, skip0], dim=1)
        x4 = self.d_conv4(x31)
        x42 = self.up1(x4,e1) #H/2
        x5 = self.up2(x42,x) #H/1
        result = self.d_conv5(x5)

        return result,h

class SRN(nn.Module):
    def __init__(self,params):
        super(SRN,self).__init__()
        self.encoder = encoder(params)
        for name,param in self.encoder.named_parameters():
            if any(x in name for x in ['conv0', 'norm']):
                param.requires_grad = False
        self.decode = DecodeModel()
        self.decode.apply(weights_init_xavier)
        #self.down = nn.AvgPool2d(3,2,1)
        self.input_size = np.tile(np.array([params.input_height,params.input_width]),(3,1))
        self.input_size[1,:] = self.input_size[0,:]//64*32
        self.input_size[2,:] = self.input_size[1,:]//64*32
        self.state_size = self.input_size//4
        self.input_size = self.input_size.tolist()
        self.state_size = self.state_size.tolist()
        down1 = nn.AdaptiveAvgPool2d((self.input_size[1][0],self.input_size[1][1]))
        down2 = nn.AdaptiveAvgPool2d((self.input_size[2][0],self.input_size[2][1]))
        self.down = (down1,down2)

        self.up = nn.functional.interpolate

    def forward(self,x):
        if x.size()[2] != self.input_size[0][0]:
            self.input_size = np.tile(np.array([x.size()[2],x.size()[3]]),(3,1))
            self.input_size[1,:] = self.input_size[0,:]//64*32
            self.input_size[2,:] = self.input_size[1,:]//64*32
            self.state_size = self.input_size//4
            self.input_size = self.input_size.tolist()
            self.state_size = self.state_size.tolist()
            down1 = nn.AdaptiveAvgPool2d((self.input_size[1][0],self.input_size[1][1]))
            down2 = nn.AdaptiveAvgPool2d((self.input_size[2][0],self.input_size[2][1]))
            self.down = (down1,down2)
        pred_input = self.down[1](self.down[0](x))
        b,_,h,w = pred_input.size()
        state = (torch.zeros(b,64,h//4,w//4,device=self.decode.DenseNetOriginal.conv0.weight.device),torch.zeros(b,64,h//4,w//4,device=self.decode.DenseNetOriginal.conv0.weight.device))
        input_imgs = [0]*3
        results = []
        for j in range(3):
            if j == 0:
                input_imgs[2-j] = x
            else:
                input_imgs[2-j] = self.down[j-1](input_imgs[2-j+1])

        for i in range(3):
            #print(i,input_imgs[i].size(),pred_input.size())
            im_input = torch.cat([input_imgs[i],pred_input],dim=1)

            #print("im_input:",im_input.size())
            #print("state:",state[0].size(),state[1].size())
            skip_feat = self.encoder(im_input[:,0:3,:,:])
            output,new_state = self.decode(im_input,state,skip_feat,i)

            # out1 = self.up(output,scale_factor=2.0,mode="nearest")
            if i<2:
                out1 = self.up(output,size=self.input_size[1-i],mode="nearest")
                pred_input = out1.clone().detach()
                state = self.up(new_state,size=self.state_size[1-i],mode="nearest")

            results.append(output)


        return tuple(results)
    