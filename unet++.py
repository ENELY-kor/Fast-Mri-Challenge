import torch
from torch import nn
from torch.nn import functional as F

# import keras
# import tensorflow as tf
# from keras.layers import concatenate
#unet -> mnet
#densenet, resnet, ...

#batch normalization
#dropout
#random.seed or fixed seed

#using unet++

class Unet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        base_n = 64

        # self.first_block = ConvBlock(in_chans, 2)
        # self.down1 = Down(2, 4)
        # self.up1 = Up(4, 2)#up samp. 
        # self.last_block = nn.Conv2d(2, out_chans, kernel_size=1)

        self.first_block = ConvBlock(in_chans, 64)#conv. channel 2 -> 64
        self.down1 = Down(64, 128)#down sampling
        #######################
        self.up1 = Up(128, 64)
        # self.down1_conv = ConvBlock(down1, 128)#?
        self.down2 = Down(128, 256)
        self.up2 = Up(256, 128)
        self.up2_1 = Up_cplus(128, 64, 64)
        # self.up1_conv = ConvBlock(up1, 256)#?
        self.down3 = Down(256, 512)       
        self.up3 = Up(512, 256)
        self.up3_1 = Up_cplus(256, 128, 128)
        self.up3_2 = Up_cplusplus(128, 64, 64)

        self.down4 = Down(512, 1024)       
        self.up4 = Up(1024, 512)
        self.up4_1 = Up_cplus(512, 256, 256)
        self.up4_2 = Up_cplusplus(256, 128, 128)
        self.up4_3 = Up_cplusplusplus(128, 64, 64)
        ####################### unet++
        
        
        # self.down2 = Down(128, 256)
        # self.up1 = Up(256, 128)#up samp.
        # self.up2 = Up(128, 64) 
        self.last_block = nn.Conv2d(64, out_chans, kernel_size=1)


    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        # input, mean, std = self.norm(input)#normalization
        # input = input.unsqueeze(1)
        # d1 = self.first_block(input)
        # m0 = self.down1(d1)
        # u1 = self.up1(m0, d1)
        # output = self.last_block(u1)
        # output = output.squeeze(1) dim matching
        # output = self.unnorm(output, mean, std)

        input, mean, std = self.norm(input)#normalization
        input = input.unsqueeze(1)

        d1 = self.first_block(input)#banila
        m0 = self.down1(d1)
        u1 = self.up1(m0 ,d1)#first output
        # u1 = self.up1(m1, m0)

        ########
        m1 = self.down2(m0) #down + convolution
        u2 = self.up2(m1, m0)#up + concatenate
        u2_1 = self.up2_1(u2, d1, u1)#second output
        
        m2 = self.down3(m1)
        u3 = self.up3(m2, m1)
        u3_1 = self.up3_1(u3, m0, u2)
        u3_2 = self.up3_2(u3_1, d1, u1, u2_1)#third output

        m3 = self.down4(m2)
        u4 = self.up4(m3, m2)
        u4_1 = self.up4_1(u4, m1, u3)
        u4_2 = self.up4_2(u4_1, m0, u2, u3_1)
        u4_3 = self.up4_3(u4_2, d1, u1, u2_1, u3_2)#last output
        ########

        # u2 = self.up2(u1, d1)

        output = self.last_block(u4_3)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(inplace=True),#ReLu->Leaky,

            #####dropout#######

            # nn.Dropout(0.5),

            ###################

            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(inplace=True)#,

            #####dropout#######

            # nn.Dropout(0.5)

            ###################
        )
    #convolution 부분
    def forward(self, x):
        return self.layers(x)


class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # self.do = nn.Dropout(0.5)#dropout!

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.activation(x)
        # x = self.do(x)#d

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        # output = self.do(x)#rop

        return output


class Down(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, out_chans)
        )
    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)

########################unet++

class Up_cplus(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = conv_block_nested(in_chans + mid_chans, mid_chans, out_chans)

    def forward(self, x, concat_input1, concat_input2):
        x = self.up(x)
        concat_output = torch.cat([concat_input2, concat_input1, x], dim=1)
        return self.conv(concat_output)


class Up_cplusplus(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = conv_block_nested(in_chans + mid_chans*2, mid_chans, out_chans)

    def forward(self, x, concat_input1, concat_input2, concat_input3):
        x = self.up(x)
        concat_output = torch.cat([concat_input3, concat_input2, concat_input1,x], dim=1)
        return self.conv(concat_output)

class Up_cplusplusplus(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = conv_block_nested(in_chans + mid_chans*3, mid_chans, out_chans)
        
    def forward(self, x, concat_input1, concat_input2, concat_input3, concat_input4):
        x = self.up(x)
        concat_output = torch.cat([concat_input4, concat_input3, concat_input2, concat_input1, x], dim=1)
        return self.conv(concat_output)
