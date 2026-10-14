

import numpy as np
import spconv.pytorch as spconv
import torch
from torch import nn
import math

from network.spconv_utils import conv3x3, conv1x3, conv3x1, conv3x3x1, conv1x1x3, conv1x3x1, conv3x1x1, conv1x1


### A module
class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)  # A

        return resA



#  DownSampling 520 
class ResBlock_NEW_520(nn.Module):    
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock_NEW_520, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1x3x3 = conv1x3(in_filters, out_filters, stride=1, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)
        self.conv3x1x3 = conv3x1(in_filters, out_filters, stride=1, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)
        self.conv3x3x1 = conv3x3x1(in_filters, out_filters, stride=1, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)

        self.conv1x3x3b = conv1x3(in_filters, out_filters, stride=2, indice_key=indice_key + "bef")
        self.act1b = nn.LeakyReLU()
        self.bn1b = nn.BatchNorm1d(out_filters)
        self.conv3x1x3b = conv3x1(in_filters, out_filters, stride=2, indice_key=indice_key + "bef")
        self.act2b = nn.LeakyReLU()
        self.bn2b = nn.BatchNorm1d(out_filters)
        self.conv3x3x1b = conv3x3x1(in_filters, out_filters, stride=2, indice_key=indice_key + "bef")
        self.act3b = nn.LeakyReLU()
        self.bn3b = nn.BatchNorm1d(out_filters)

        self.conv1x3x3c = conv1x3(in_filters, out_filters, stride=3, indice_key=indice_key + "bef")
        self.act1c = nn.LeakyReLU()
        self.bn1c = nn.BatchNorm1d(out_filters)
        self.conv3x1x3c = conv3x1(in_filters, out_filters, stride=3, indice_key=indice_key + "bef")
        self.act2c = nn.LeakyReLU()
        self.bn2c = nn.BatchNorm1d(out_filters)
        self.conv3x3x1c = conv3x3x1(in_filters, out_filters, stride=3, indice_key=indice_key + "bef")
        self.act3c = nn.LeakyReLU()
        self.bn3c = nn.BatchNorm1d(out_filters)

        

        # self.conv1x1a = conv1x1(in_filters, out_filters, indice_key=indice_key)
        self.conv_z1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(out_filters)

        self.conv1x1b = conv1x1(out_filters, out_filters, indice_key=indice_key)
        # self.conv_z2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        resA = self.conv1x3x3(x)
        resA = resA.replace_feature(self.act1(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))
        resB = self.conv3x1x3(x)
        resB = resB.replace_feature(self.act2(resB.features))
        resB = resB.replace_feature(self.bn2(resB.features))
        resC = self.conv3x3x1(x)
        resC = resC.replace_feature(self.act3(resC.features))
        resC = resC.replace_feature(self.bn3(resC.features))

        resA = resA.replace_feature(resA.features + resB.features + resC.features)


        resAA = self.conv3x3x1b(x)
        resAA = resAA.replace_feature(self.act1b(resAA.features))
        resAA = resAA.replace_feature(self.bn1b(resAA.features))
        resBB = self.conv3x1x3b(x)
        resBB = resBB.replace_feature(self.act2b(resBB.features))
        resBB = resBB.replace_feature(self.bn2b(resBB.features))
        resCC = self.conv1x3x3b(x)
        resCC = resCC.replace_feature(self.act3b(resCC.features))
        resCC = resCC.replace_feature(self.bn3b(resCC.features))

        resAA = resAA.replace_feature(resAA.features + resBB.features + resCC.features)

        resAAA = self.conv3x3x1c(x)
        resAAA = resAAA.replace_feature(self.act1c(resAAA.features))
        resAAA = resAAA.replace_feature(self.bn1c(resAAA.features))
        resBBB = self.conv3x1x3c(x)
        resBBB = resBBB.replace_feature(self.act2c(resBBB.features))
        resBBB = resBBB.replace_feature(self.bn2c(resBBB.features))
        resCCC = self.conv1x3x3c(x)
        resCCC = resCCC.replace_feature(self.act3c(resCCC.features))
        resCCC = resCCC.replace_feature(self.bn3c(resCCC.features))

        resAAA = resAAA.replace_feature(resAAA.features + resBBB.features  + resCCC.features)

        resA = resA.replace_feature(resA.features + resAA.features)
        resAA = resAA.replace_feature(resAA.features + resAAA.features)

        resA = resA.replace_feature(resA.features + resAA.features)

        x = self.conv_z1(x)
        x = x.replace_feature(self.act4(x.features))
        x = x.replace_feature(self.bn4(x.features))

        x = self.conv1x1b(x)

        resA = resA.replace_feature(resA.features + x.features)
        # print("resAllresAll.shape", resAA.dense().shape)

        if self.pooling:
            resBBBB = self.pool(resA)
            return resBBBB, resA
        else:
            return resA


# UpSampling   520 
class UpBlock_NEW_520(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock_NEW_520, self).__init__()
        #### self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)


        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.conv1x3x3 = conv1x3(out_filters, out_filters, stride=1, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)
        self.conv3x1x3 = conv3x1(out_filters, out_filters, stride=1, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)
        self.conv3x3x1 = conv3x3x1(out_filters, out_filters, stride=1, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)

        self.conv1x3x3b = conv1x3(out_filters, out_filters, stride=2, indice_key=indice_key + "bef")
        self.act1b = nn.LeakyReLU()
        self.bn1b = nn.BatchNorm1d(out_filters)
        self.conv3x1x3b = conv3x1(out_filters, out_filters, stride=2, indice_key=indice_key + "bef")
        self.act2b = nn.LeakyReLU()
        self.bn2b = nn.BatchNorm1d(out_filters)
        self.conv3x3x1b = conv3x3x1(out_filters, out_filters, stride=2, indice_key=indice_key + "bef")
        self.act3b = nn.LeakyReLU()
        self.bn3b = nn.BatchNorm1d(out_filters)

        self.conv1x3x3c = conv1x3(out_filters, out_filters, stride=3, indice_key=indice_key + "bef")
        self.act1c = nn.LeakyReLU()
        self.bn1c = nn.BatchNorm1d(out_filters)
        self.conv3x1x3c = conv3x1(out_filters, out_filters, stride=3, indice_key=indice_key + "bef")
        self.act2c = nn.LeakyReLU()
        self.bn2c = nn.BatchNorm1d(out_filters)
        self.conv3x3x1c = conv3x3x1(out_filters, out_filters, stride=3, indice_key=indice_key + "bef")
        self.act3c = nn.LeakyReLU()
        self.bn3c = nn.BatchNorm1d(out_filters)


        self.conv_z1 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(out_filters)

        self.conv1x1 = conv1x1(out_filters, out_filters, indice_key=indice_key)
        # self.conv_z2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))
       

        ## upsample
        upA = self.up_subm(upA)

        #print("After up_subm:", upA.features.shape)
        #print("Skip features:", skip.features.shape)

        upA = upA.replace_feature(upA.features + skip.features)

        #print("After skip addition:", upA.features.shape)
        #print("conv_z1 expects:", self.conv_z1.in_channels)

#####################################################
        resA = self.conv1x3x3(upA)
        resA = resA.replace_feature(self.act1(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))
        resB = self.conv3x1x3(upA)
        resB = resB.replace_feature(self.act2(resB.features))
        resB = resB.replace_feature(self.bn2(resB.features))
        resC = self.conv3x3x1(upA)
        resC = resC.replace_feature(self.act3(resC.features))
        resC = resC.replace_feature(self.bn3(resC.features))

        resA = resA.replace_feature(resA.features + resB.features + resC.features)


        resAA = self.conv3x3x1b(resA)
        resAA = resAA.replace_feature(self.act1b(resAA.features))
        resAA = resAA.replace_feature(self.bn1b(resAA.features))
        resBB = self.conv3x1x3b(resB)
        resBB = resBB.replace_feature(self.act2b(resBB.features))
        resBB = resBB.replace_feature(self.bn2b(resBB.features))
        resCC = self.conv1x3x3b(resC)
        resCC = resCC.replace_feature(self.act3b(resCC.features))
        resCC = resCC.replace_feature(self.bn3b(resCC.features))

        resAA = resAA.replace_feature(resAA.features + resBB.features + resCC.features)

        resAAA = self.conv3x3x1c(resA)
        resAAA = resAAA.replace_feature(self.act1c(resAAA.features))
        resAAA = resAAA.replace_feature(self.bn1c(resAAA.features))
        resBBB = self.conv3x1x3c(resB)
        resBBB = resBBB.replace_feature(self.act2c(resBBB.features))
        resBBB = resBBB.replace_feature(self.bn2c(resBBB.features))
        resCCC = self.conv1x3x3c(resC)
        resCCC = resCCC.replace_feature(self.act3c(resCCC.features))
        resCCC = resCCC.replace_feature(self.bn3c(resCCC.features))

        resAAA = resAAA.replace_feature(resAAA.features + resBBB.features + resCCC.features)

        resA = resA.replace_feature(resA.features + resAA.features)
        resAA = resAA.replace_feature(resAA.features + resAAA.features)
        
        resA = resA.replace_feature(resA.features + resAA.features)
        # print("resAll.shape", resAA.dense().shape)

        #print("upA features:", upA.features.shape)
        #print("conv_z1 expects:", self.conv_z1.in_channels)
        #print("conv_z1 outputs:", self.conv_z1.out_channels)
        x = self.conv_z1(upA)
        x = x.replace_feature(self.act4(x.features))
        x = x.replace_feature(self.bn4(x.features))

        x = self.conv1x1(x)

        resA = resA.replace_feature(resA.features + x.features)

        return resA

    
#DDCM 特征融合
class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))


        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut3.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)
        
        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


#DDCM 特征融合
class ReconBlock_NEW(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock_NEW, self).__init__()


        self.conv1x3x3 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)
        self.conv3x1x3 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)
        self.conv3x3x1 = conv3x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)

        self.conv1x3x3b = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1b = nn.LeakyReLU()
        self.bn1b = nn.BatchNorm1d(out_filters)
        self.conv3x1x3b = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act2b = nn.LeakyReLU()
        self.bn2b = nn.BatchNorm1d(out_filters)
        self.conv3x3x1b = conv3x3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3b = nn.LeakyReLU()
        self.bn3b = nn.BatchNorm1d(out_filters)

        self.conv1x1 = conv1x1(out_filters, out_filters, indice_key=indice_key)

    def forward(self, x):
        resA = self.conv1x3x3(x)
        resA = resA.replace_feature(self.act1(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))
        resB = self.conv3x1x3(x)
        resB = resB.replace_feature(self.act2(resB.features))
        resB = resB.replace_feature(self.bn2(resB.features))
        resC = self.conv3x3x1(x)
        resC = resC.replace_feature(self.act3(resC.features))
        resC = resC.replace_feature(self.bn3(resC.features))
        print("x.shape", x.dense().shape)


        resA = resA.replace_feature(resA.features + resB.features)
        resB = resB.replace_feature(resA.features + resC.features)
        resC = resC.replace_feature(resB.features + resC.features)


        resAA = self.conv3x3x1b(resA)
        resAA = resAA.replace_feature(self.act1b(resAA.features))
        resAA = resAA.replace_feature(self.bn1b(resAA.features))
        resBB = self.conv3x1x3b(resB)
        resBB = resBB.replace_feature(self.act2b(resBB.features))
        resBB = resBB.replace_feature(self.bn2b(resBB.features))
        resCC = self.conv1x3x3b(resC)
        resCC = resCC.replace_feature(self.act3b(resCC.features))
        resCC = resCC.replace_feature(self.bn3b(resCC.features))

        resAA = resAA.replace_feature(resAA.features + resBB.features + resCC.features)

        x = self.conv1x1(x)
        # print("xxxx.shape", x.dense().shape)
        shortcut = shortcut.replace_feature(resAA.features * x.features)

        return shortcut



# Network Building 
class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)

        self.sparse_shape = sparse_shape

###开始下采样
        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")  # A module

        self.resBlock2 = ResBlock_NEW_520(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock_NEW_520(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock_NEW_520(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock_NEW_520(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")
        self.resBlock6 = ResBlock_NEW_520(16 * init_size, 32 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down6")
        


###开始上采样
        self.upBlock00 = UpBlock_NEW_520(32 * init_size, 32 * init_size, indice_key="up00", up_key="down6")
        self.upBlock0 = UpBlock_NEW_520(32 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock_NEW_520(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock_NEW_520(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock_NEW_520(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")
        # self.upBlock4 = UpBlock_NEW(2 * init_size, 2 * init_size, indice_key="up4", up_key="down1")  ###  新加

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon") # DDCM

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, voxel_features, coors, batch_size):

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)   # A module

        down1c, down1b = self.resBlock2(ret)  #2 * init_size
        down2c, down2b = self.resBlock3(down1c)  #4 * init_size
        down3c, down3b = self.resBlock4(down2c)  #8 * init_size
        down4c, down4b = self.resBlock5(down3c)  #16 * init_size
        down5c, down5b = self.resBlock6(down4c)  #32 * init_size


        up5e = self.upBlock00(down5c, down5b) #32 * init_size
        up4e = self.upBlock0(up5e, down4b) #16 * init_size
        up3e = self.upBlock1(up4e, down3b)   #8 * init_size
        up2e = self.upBlock2(up3e, down2b)   #4 * init_size
        up1e = self.upBlock3(up2e, down1b)   #2 * init_size

        upe = self.ReconNet(up1e)  # DDCM

        upe = upe.replace_feature(torch.cat((upe.features, up1e.features), 1))

        logits = self.logits(upe)
        y = logits.dense()
        return y







