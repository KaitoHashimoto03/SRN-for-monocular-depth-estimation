import torch
import torch.nn as nn
import torch.nn.functional as F


class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True, eps=1.1e-5))
        
        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, bias=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels*2, momentum=0.01, affine=True, track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation), dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)

class ASPP_block(nn.Module):
    def __init__(self, dim_in, dim_out,dilation_rates=[3,6,12,18,24]):
        super().__init__()
        
        self.conv0      = torch.nn.Sequential(nn.Conv2d(dim_in, dim_out*2, 3, 1, 1, bias=False),
                                              nn.ELU())
        
        self.daspp_3    = atrous_conv(dim_out*2, dim_out, dilation_rates[0])
        self.daspp_6    = atrous_conv(dim_in + dim_out, dim_out, dilation_rates[1])
        self.daspp_12   = atrous_conv(dim_in + dim_out*2, dim_out, dilation_rates[2])
        self.daspp_18   = atrous_conv(dim_in + dim_out*3, dim_out, dilation_rates[3])
        self.daspp_24   = atrous_conv(dim_in + dim_out*4, dim_out, dilation_rates[4])
        self.daspp_conv = torch.nn.Sequential(nn.Conv2d(dim_out*7, dim_out, 3, 1, 1, bias=False),
                                              nn.ELU())

    def change_dilation_rates(self,dilation_rates):
        for i,name in enumerate(["daspp_3","daspp_6","daspp_12","daspp_18","daspp_24"]):
            getattr(self,name).atrous_conv.aconv_sequence[4].padding=(dilation_rates[i],dilation_rates[i])
            getattr(self,name).atrous_conv.aconv_sequence[4].dilation=(dilation_rates[i],dilation_rates[i])
        #print(self.daspp_3.atrous_conv.aconv_sequence[4])

    def forward(self, x):
        iconv0 = self.conv0(x)

        daspp_0 = self.daspp_3(iconv0)
        concat0 = torch.cat([x, daspp_0], dim=1)
        daspp_1 = self.daspp_6(concat0)
        concat1 = torch.cat([concat0, daspp_1], dim=1)
        daspp_2 = self.daspp_12(concat1)
        concat2 = torch.cat([concat1, daspp_2], dim=1)
        daspp_3 = self.daspp_18(concat2)
        concat3 = torch.cat([concat2, daspp_3], dim=1)
        daspp_4 = self.daspp_24(concat3)
        concat_daspp = torch.cat([iconv0, daspp_0, daspp_1, daspp_2, daspp_3, daspp_4], dim=1)
        daspp_feat = self.daspp_conv(concat_daspp)
        return daspp_feat

# class ASPP_block(nn.Module):
#     def __init__(self, in_1,in_2, dim_out):
#         super().__init__()
#         dim_in = in_1 + in_2
#         self.dim_in = dim_in
#         self.dim_out = dim_out
        
#         self.elu = nn.ELU()
#         self.bn_1       = nn.BatchNorm2d(in_1, momentum=0.01, affine=True, eps=1.1e-5)
#         self.bn_2       = nn.BatchNorm2d(in_2, momentum=0.01, affine=True, eps=1.1e-5)
#         self.conv4      = torch.nn.Sequential(nn.Conv2d(dim_in, dim_out*2, 3, 1, 1, bias=False),
#                                               nn.ELU())
#         self.bn4_2      = nn.BatchNorm2d(dim_out*2, momentum=0.01, affine=True, eps=1.1e-5)
        
#         self.daspp_3    = atrous_conv(dim_out*2, dim_out, 3, apply_bn_first=False)
#         self.daspp_6    = atrous_conv(dim_in + dim_out, dim_out, 6)
#         self.daspp_12   = atrous_conv(dim_in + dim_out*2, dim_out, 12)
#         self.daspp_18   = atrous_conv(dim_in + dim_out*3, dim_out, 18)
#         self.daspp_24   = atrous_conv(dim_in + dim_out*4, dim_out, 24)
#         self.daspp_conv = torch.nn.Sequential(nn.Conv2d(dim_out*7, dim_out, 3, 1, 1, bias=False),
#                                               nn.ELU())

#     def forward(self, x ,x2):
#         x = self.bn_1(self.elu(x))
#         x2 = self.bn_2(self.elu(x2))
#         concat4 = torch.cat([x, x2], dim=1)
#         iconv4 = self.conv4(concat4)
#         iconv4 = self.bn4_2(iconv4)

#         daspp_3 = self.daspp_3(iconv4)
#         concat4_2 = torch.cat([concat4, daspp_3], dim=1)
#         daspp_6 = self.daspp_6(concat4_2)
#         concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
#         daspp_12 = self.daspp_12(concat4_3)
#         concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
#         daspp_18 = self.daspp_18(concat4_4)
#         concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
#         daspp_24 = self.daspp_24(concat4_5)
#         concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
#         daspp_feat = self.daspp_conv(concat4_daspp)
#         return daspp_feat