import torch
import torch.nn as nn

class BilateralCostVolume(nn.Module):
    def __init__(self,md=4):
        super(BilateralCostVolume, self).__init__()
        self.md = md #displacement (default = 4pixels)
        self.grid = torch.ones(1).cuda()

        self.range = (md*2 + 1) ** 2 #(default = 9*9 = 81)
        d_u = torch.linspace(-self.md, self.md, 2 * self.md + 1).view(1, -1).repeat((2 * self.md + 1, 1)).view(self.range, 1)  # (25,1)
        d_v = torch.linspace(-self.md, self.md, 2 * self.md + 1).view(-1, 1).repeat((1, 2 * self.md + 1)).view(self.range, 1)  # (25,1)

        self.d = torch.cat((d_u, d_v), dim=1).cuda()  # (25,2)

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x/norm)

    def Make_UniformGrid(self, Input):
        '''
        Make uniform grid
        :param Input: tensor(N,C,H,W)
        :return grid: (N,2,H,W)
        '''
        # torchHorizontal = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(N, 1, H, W)
        # torchVertical = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(N, 1, H, W)
        # grid = torch.cat([torchHorizontal, torchVertical], 1).cuda()

        B, _, H, W = Input.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(self.range, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(self.range, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if Input.is_cuda:
            grid = grid.cuda()

        return grid

    def warp(self, x, BM_d):
        vgrid = self.grid + BM_d
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(x.size(3) - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(x.size(2) - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode = 'border') #800MB memory occupied (d=2,C=64,H=256,W=256)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid) #300MB memory occpied (d=2,C=64,H=256,W=256)

        mask = mask.masked_fill_(mask<0.999,0)
        mask = mask.masked_fill_(mask>0,1)

        return output * mask

    def forward(self,feature1, feature2, BM, time=0.5):
        '''
        Return bilateral cost volume(Set of bilateral correlation map)
        :param feature1: feature at time t-1(N,C,H,W)
        :param feature2: feature at time t+1(N,C,H,W)
        :param BM: (N,2,H,W)
        :param time(float): intermediate time step from 0 to 1 (default: 0.5 [half])
        :return BC: (N,(2d+1)^2,H,W)
        '''
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)

        if torch.equal(self.grid, torch.ones(1).cuda()):
            # Initialize first uniform grid
            self.grid = torch.autograd.Variable(self.Make_UniformGrid(BM))

        if BM.size(2) != self.grid.size(2) or BM.size(3) != self.grid.size(3):
            # Update uniform grid to fit on input tensor shape
            self.grid = torch.autograd.Variable(self.Make_UniformGrid(BM))

        # Displacement volume (N,(2d+1)^2,2,H,W) d = (i,j) , i in [-md,md] & j in [-md,md]
        D_vol = self.d.view(1, self.range, 2, 1, 1).expand(BM.size(0), -1, -1, BM.size(2), BM.size(3))  # (N,(2d+1)^2,2,H,W) \\ ex. D_vol(0,0,:,0,0) == (-2,-2), D_vol(0,13,:,0,0) == (1,0)

        # BM_d(N,(2*d+1)^2,2,H,W) : BM + d (displacement d from [-2,-2] to[2,2])
        BM_d = BM.view(BM.size(0), 1, BM.size(1), BM.size(2), BM.size(3)).expand(-1, self.range, -1, -1, -1) + D_vol

        # Bilateral Cost Volume(list)
        BC_list = []

        for i in range(BM.size(0)):
            if time == 0:
                bw_feature = feature1[i].view((1,) + feature1[i].size()).expand(self.range, -1, -1, -1)  # (D**2,C,H,W)
                fw_feature = self.warp(feature2[i].view((1,) + feature2[i].size()).expand(self.range, -1, -1, -1), 2*(1-time)*BM_d[i]) # (D**2,C,H,W)

            elif time > 0 and time<1:
                bw_feature = self.warp(feature1[i].view((1,) + feature1[i].size()).expand(self.range, -1, -1, -1),  2*(-time)*BM_d[i]) # (D**2,C,H,W)
                fw_feature = self.warp(feature2[i].view((1,) + feature2[i].size()).expand(self.range, -1, -1, -1), 2*(1-time)*BM_d[i]) # (D**2,C,H,W)

            elif time == 1:
                bw_feature = self.warp(feature1[i].view((1,) + feature1[i].size()).expand(self.range, -1, -1, -1),  2*(-time)*BM_d[i]) # (D**2,C,H,W)
                fw_feature = feature2[i].view((1,) + feature2[i].size()).expand(self.range, -1, -1, -1)  # (D**2,C,H,W)

            BC_list.append(torch.sum(torch.mul(fw_feature, bw_feature), dim=1).view(1,self.range,BM.size(2),BM.size(3)))

        # BC (N,(2d+1)^2,H,W)
        return torch.cat(BC_list)
