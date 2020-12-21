import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
from model.module import *
from model.convlstm import ConvLSTMCell
from model.aspp import ASPP_block

OUTPUT_C = 3 #3

class DecodeModel(nn.Module):
    def __init__(self,half=False):
        super(DecodeModel,self).__init__()
        ############# 256-256  ##############

        self.conv0 = nn.Conv2d(3 + OUTPUT_C,64,7,2,3)
        self.norm0 = nn.InstanceNorm2d(64)
        self.relu0 = nn.LeakyReLU(inplace=True)

        self.dense_block1 = BottleneckDecoderBlock(64,64)#,pad_num=2,dil_num=2)
        self.trans_block1 = Transiondown(64,64)
    
        self.dense1 = BottleneckDecoderBlock(64,64)#,pad_num=2,dil_num=2)
        self.dense2 = BottleneckDecoderBlock(64,64)#,pad_num=4,dil_num=4)
        
        self.lstm = ConvLSTMCell(64,64,(3,3),bias=True)
        self.dense3 = BottleneckDecoderBlock(64,64)#,pad_num=8,dil_num=8)
        self.dense4 = BottleneckDecoderBlock(64,64)#,pad_num=8,dil_num=8)

        self.trans4 = TransitionBlock(64,64)
        self.half = half
         
        self.dense5 = BottleneckDecoderBlock(128,128)
        self.trans5 = TransitionBlock(128,64)
        self.conv1 = nn.Conv2d(64,16,3,1,2,2)
        self.norm1 = nn.InstanceNorm2d(16)

        self.refine = refineblock(16+3,OUTPUT_C)

        self.aspp = ASPP_block(64,64,64)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x,state=None):
        #print("x:",x.size())
        x0 = self.relu0(self.norm0(self.conv0(x)))
        x01 = self.dense_block1(x0)
        x1 = self.trans_block1(x01)
        x2 = self.dense1(x1)
        x3 = self.dense2(x2)
        xa = self.aspp(x3,x1)
        #print("x3:",x3.size())
        x31,h = self.lstm(xa,state)
        x4 = self.dense3(x31)
        x5 = self.trans4(self.dense4(x4))

        x51 = torch.cat([x5,x01],dim=1)
        x6 = self.trans5(self.dense5(x51))
        x6 = self.relu0(self.norm1(self.conv1(x6)))

        x61 = torch.cat([x6,x[:,0:3,:,:]],1)
        dehaze = self.refine(x61)

        #dehaze = self.sigmoid(dehaze)        

        return dehaze,h


class SRN(nn.Module):
    def __init__(self):
        super(SRN,self).__init__()
        self.decode = DecodeModel()
        self.down = nn.AvgPool2d(3,2,1)

        self.up = F.interpolate

        # self.conv = nn.Conv2d(3,1,3,1,1)
        # self.bn = nn.InstanceNorm2d(1)
        # self.relu = nn.LeakyReLU(inplace=True)
    def forward(self,x):
        pred_input = self.down(self.down(x))
        #pred_input = self.down(self.down(self.relu(self.bn(self.conv(x)))))
        b,_,h,w = pred_input.size()
        state = (torch.zeros(b,64,h//4,w//4,device=self.decode.conv0.weight.device),torch.zeros(b,64,h//4,w//4,device=self.decode.conv0.weight.device))
        input_imgs = [0]*3
        results = []
        for j in range(3):
            if j == 0:
                input_imgs[2-j] = x
            else:
                input_imgs[2-j] = self.down(input_imgs[2-j+1])

        for i in range(3):
            im_input = torch.cat([input_imgs[i],pred_input],dim=1)

            # print("im_input:",im_input.size())
            # print("state:",state[0].size(),state[1].size())
            output,new_state = self.decode(im_input,state)


            out1 = self.up(output,scale_factor=2.0,mode="nearest")
            pred_input = out1.clone().detach()
            state = self.up(new_state,scale_factor=2.0,mode="nearest")

            results.append(output)


        return tuple(results)
