import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class DynFilter(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(DynFilter, self).__init__()

        self.padding = padding
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)

        filter_localexpand_np = np.reshape(np.eye(np.prod(kernel_size), np.prod(kernel_size)), (np.prod(kernel_size), 1, kernel_size[0], kernel_size[1]))
        self.filter_localexpand = torch.FloatTensor(filter_localexpand_np).cuda()

    def forward(self, x, filter):
        x_localexpand = []

        for c in range(x.size(1)):
            x_localexpand.append(F.conv2d(x[:, c:c + 1, :, :], self.filter_localexpand, padding=self.padding))  # 5x5 kernel --> padding = 2

        x_localexpand = torch.cat(x_localexpand, dim=1)
        x = torch.sum(torch.mul(x_localexpand, filter), dim=1).unsqueeze(1)

        return x

class Make_Dense(nn.Module):
    def __init__(self,intInput,intOutput):
        super(Make_Dense,self).__init__()

        self.dense_layer = nn.Sequential(
            # torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1), padding=dilation, dilation=dilation) #dilated convolution
            torch.nn.Conv2d(in_channels= intInput, out_channels= intOutput, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.ReLU()
        )

    def forward(self, input):
        Feature_d_output = self.dense_layer(input)
        return torch.cat((input, Feature_d_output), dim =1)


class RDB(nn.Module):
    def __init__(self, fusion_Feat, Conv_Num, Growth_rate):
        super(RDB, self).__init__()

        def LFF(intInput, intOutput):
            return torch.nn.Conv2d(in_channels= intInput, out_channels= intOutput, kernel_size=(1,1), stride = (1,1), padding= (0,0)) # Conv_1x1 layer

        self.Conv_Num = Conv_Num        # The number of Convolution layer before LFF

        Modules = [] # Sub_layer list #

        for i in range(Conv_Num):
            Modules.append(Make_Dense(fusion_Feat + i * Growth_rate, Growth_rate)) # Append dense layer

        self.Layer = nn.Sequential(*Modules) # Contiguous memory Mechanism

        self.local_feature_fusion = LFF(fusion_Feat + Conv_Num * Growth_rate, fusion_Feat) # G_0 : fusion_Feat, G : Growth_rate

    def forward(self, input):
        Feature_d_LF = self.local_feature_fusion(self.Layer(input))
        Feature_d = Feature_d_LF + input
        return Feature_d


class DFNet(nn.Module):
    def __init__(self,fusion_Feat, RDB_Num, Growth_rate, Conv_Num):
        """
        You are able to manipulate hyper parameters of RDN but,
        not able to load the ckpt model which has different hyper parameters.
        Arguments:
            fusion_Feat (int) : The number of channel of convolution layer.
            RDB_num     (int) : The number of Residual Dense Block.
            Growth_rate (int) : Growth rate of the number of convolution channel at dense layer.
            Conv_Num    (int) : The number of Convolution layers in Residual Dense Block.
        """
        super(DFNet, self).__init__()

        def SFE(intInput, intOutput):
            return torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1))

        def GFF(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(1, 1), stride=(1, 1),
                                padding=(0, 0)),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )



        self.fusion_Feat = fusion_Feat
        self.RDB_Num = RDB_Num
        self.Growth_rate = Growth_rate

        self.SFE_First = SFE(67 * 6 + 3 * 2, self.fusion_Feat)  # 6 Candidates(with context) + 2 input images

        self.SFE_Second = SFE(self.fusion_Feat, self.fusion_Feat)

        Modules = []

        for i in range(self.RDB_Num):
            Modules.append(RDB(fusion_Feat=self.fusion_Feat, Conv_Num=Conv_Num, Growth_rate=self.Growth_rate))

        self.GRL = nn.Sequential(*Modules)
        self.Global_feature_fusion = GFF(self.RDB_Num * self.fusion_Feat, self.fusion_Feat)

        self.Conv_Output = torch.nn.Conv2d(in_channels=self.fusion_Feat, out_channels=5 ** 2 * 6, kernel_size=(3, 3),
                                           stride=(1, 1), padding=(1, 1))

    def forward(self, input):
        F_B1 = self.SFE_First(input)  # F_B1 == F_(-1).
        F_d = self.SFE_Second(F_B1)  # F_0 (d==0).

        for i in range(self.RDB_Num):
            F_d = self.GRL[i](F_d)

            if i == 0:
                F_RDB_group = F_d

            else:
                F_RDB_group = torch.cat((F_RDB_group, F_d), dim=1)

        F_GF = self.Global_feature_fusion(F_RDB_group)

        F_DF = F_GF + F_B1

        output = self.Conv_Output(F_DF)

        return output