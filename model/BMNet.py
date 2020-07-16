import torch
import torch.nn as nn
from Bilateral_CostVolume import BilateralCostVolume
import numpy as np


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride=stride,
                  padding = padding, dilation = dilation, bias= True),
        nn.LeakyReLU(0.1))


def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)


class BMNet(nn.Module):
    def __init__(self):
        super(BMNet, self).__init__()

        self.conv1a  = conv(3,    16, kernel_size=3, stride=1)
        self.conv1aa = conv(16,   16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,   16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,   32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,   32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,   32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,   64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,   64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,   64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,   96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,   96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,   96, kernel_size=3, stride=1)
        self.conv5a  = conv(96,  128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b  = conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2)
        self.conv6a  = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b  = conv(196, 196, kernel_size=3, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1)

        # nd = (2*md + 1)**2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = (2*6+1) ** 2
        self.bilateral_corr6 = BilateralCostVolume(md=6)
        self.conv6_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.deconv6 = deconv(2,          2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2*4+1) ** 2 + 128*2 + 4
        self.bilateral_corr5 = BilateralCostVolume(md=4)
        self.conv5_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2,          2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2*4+1) ** 2  + 96*2 + 4
        self.bilateral_corr4 = BilateralCostVolume(md=4)
        self.conv4_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2,          2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2*2+1) ** 2  + 64*2 + 4
        self.bilateral_corr3 = BilateralCostVolume(md=2)
        self.conv3_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2,          2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2*2+1) ** 2  + 32*2 + 4
        self.bilateral_corr2 = BilateralCostVolume(md=2)
        self.conv2_0 = conv(od,         128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1],  96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2],  64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3],  32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128,        128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128,        128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128,         96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96,          64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,          32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask

    def forward(self, x, time=0.5):
        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]


        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        bicorr6 = self.bilateral_corr6(c16, c26, torch.zeros(c16.size(0), 2, c16.size(2), c16.size(3)).cuda(), time)
        bicorr6 = self.leakyRELU(bicorr6)

        x = torch.cat((self.conv6_0(bicorr6), bicorr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp1_5= self.warp(c15, up_flow6* (-1.25)*time*2)
        # warp1_5 = self.warp(c15, up_flow6* (-1.25)*time*2)
        warp2_5 = self.warp(c25, up_flow6* 1.25*(1-time)*2)
        bicorr5 = self.bilateral_corr5(c15, c25, up_flow6*1.25, time)
        bicorr5 = self.leakyRELU(bicorr5)
        x = torch.cat((bicorr5, warp1_5, warp2_5, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp1_4 = self.warp(c14, up_flow5* (-2.5)*time*2)
        warp2_4 = self.warp(c24, up_flow5* 2.5*(1-time)*2)
        bicorr4 = self.bilateral_corr4(c14,c24,up_flow5*2.5, time)
        bicorr4 = self.leakyRELU(bicorr4)
        x = torch.cat((bicorr4, warp1_4, warp2_4, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp1_3 = self.warp(c13, up_flow4* (-5.0)*time*2)
        warp2_3 = self.warp(c23, up_flow4* 5.0*(1-time)*2)
        bicorr3 = self.bilateral_corr3(c13,c23, up_flow4* 5.0, time)
        bicorr3 = self.leakyRELU(bicorr3)
        x = torch.cat((bicorr3, warp1_3, warp2_3, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp1_2 = self.warp(c12, up_flow3* (-10.0)*time*2)
        warp2_2 = self.warp(c22, up_flow3* 10.0*(1-time)*2)
        bicorr2 = self.bilateral_corr2(c12,c22, up_flow3* 10.0, time)
        bicorr2 = self.leakyRELU(bicorr2)
        x = torch.cat((bicorr2, warp1_2, warp2_2, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        flow = nn.functional.interpolate(flow2, (im1.size(2),im1.size(3)), mode='bilinear') * 20.0

        return flow
