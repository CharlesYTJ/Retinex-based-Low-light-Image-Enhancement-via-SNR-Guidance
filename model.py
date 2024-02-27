import torch
import torch.nn as nn  
import kornia
import torch.nn.functional as F
from .SFBlock import *
import torch.nn.init as init
import functools
        
class FirstLayer(nn.Module):
    def __init__(self, num):
        super(FirstLayer,self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(3, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU()
            )
    def forward(self, x):
        return self.m(x)

class ConvP(nn.Module):
    def __init__(self, num,):
        super(ConvP,self).__init__()
        
        self.downd = nn.Sequential(
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, 3, 3, 1, 1),
            nn.Sigmoid()
            )
        self.upd = nn.Sequential(
            nn.Conv2d(3, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU()
            )
    def forward(self, x):
        p = self.downd(x)
        return self.upd(p)+x

class ConvI(nn.Module):
    def __init__(self, num,rate=0.5):
        super(ConvI,self).__init__()
        
        self.downd = nn.Sequential(
            nn.Conv2d(num, int(num*rate), 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(int(num*rate), 1, 3, 1, 1),
            nn.Sigmoid()
            )
        self.upd = nn.Sequential(
            nn.Conv2d(1, int(num*rate), 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(int(num*rate), num, 3, 1, 1),
            nn.ReLU()
            )
    def forward(self, x):
        i = self.downd(x)
        return self.upd(i)+x

        
class PILayer(nn.Module):
    def __init__(self, num):
        super(PILayer,self).__init__()
        self.p = ConvP(num)
        self.i = ConvI(num)
        
    def forward(self, x):
        u1 = self.p(x)
        u2 = self.i(x)

        return u1+u2

        
class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        self.r = nn.Sequential(
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, 3, 3, 1, 1),
            nn.Sigmoid()
            )
        self.l = nn.Sequential(
            nn.Conv2d(num, num, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num, 1, 3, 1, 1),
            nn.Sigmoid() 
            )
    def forward(self, x):
        return self.r(x), self.l(x)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
    
def initialize_weights(net_l, scale=1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)

class SNR(nn.Module):
    def __init__(self, num=64):  # num=64
        super(SNR,self).__init__()
        self.fl = FirstLayer(num)
        self.pi = PILayer(num)
        self.head = Head(num)
        
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=num)
        
        self.conv_first_1 = nn.Conv2d(3 * 2, num, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(num, num, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(num, num, 3, 2, 1, bias=True)
        
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, 1)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, 1)
        self.sfnet = SFNet(num)
        
        self.upconv1 = nn.Conv2d(num*2, num * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num*2, num * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(num*2, num, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num, 3, 3, 1, 1, bias=True)
        self.recon_trunk_light = make_layer(ResidualBlock_noBN_f, 6)
        
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)     
        self.lrelu = nn.ReLU()
    
    def get_mask(self,dark):

        light = kornia.filters.gaussian_blur2d(dark, (5, 5), (1.5, 1.5))
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        return mask.float()    
    
    def forward(self, x):
        _, _, H, W = x.shape
        x1 = self.fl(x)
        u = self.pi(x1)
        R,L = self.head(u)
        
        # x_center = X1
        x_center = x
        
        # rate = 2 ** 4  # default rate=2**3
        rate = 2 ** 4
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_center = F.pad(x_center, (0, pad_w, 0, pad_h), "reflect")
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")
        # print('x_center', x.size())    
        L1_fea_1 = self.lrelu(self.conv_first_1(torch.cat((x_center,x),dim=1)))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))
        
        fea = self.feature_extraction(L1_fea_3)
        fea_light = self.recon_trunk_light(fea)
        
        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = self.get_mask(x_center)
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')
        fea_unfold = self.sfnet(fea)

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask) + fea_light * mask

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x
        out_noise = out_noise[:, :, :H, :W]
        
        return L, R, out_noise
        
# if __name__ == '__main__':
    # model = SNR()
    # a=torch.randn(1,3,224,224)
    # L, R, X, out_noise = model(a)
    # print('out', out_noise.size())
