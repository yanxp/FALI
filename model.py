import torch
import torch.nn as nn
import torch.nn.functional as F


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class G12(nn.Module):
    """Generator for transfering from mnist to svhn"""
    def __init__(self, conv_dim=64):
        super(G12, self).__init__()
        # encoding blocks
        self.conv1 = conv(1, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        
        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        
        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)
        
        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )
        
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 3, 32, 32)
        return out
    
class G21(nn.Module):
    """Generator for transfering from svhn to mnist"""
    def __init__(self, conv_dim=64):
        super(G21, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        
        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        
        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1, 4, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)
        
        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )
        
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 1, 32, 32)
        return out
    
class G23(nn.Module):
    """Generator for transfering from svhn to mnist"""
    def __init__(self, conv_dim=64):
        super(G23, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        
        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        
        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)
        
        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )
        
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 1, 32, 32)
        return out
    
class D1(nn.Module):
    """Discriminator for mnist."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out

class D2(nn.Module):
    """Discriminator for svhn."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out

class IZ1(nn.Module):
    '''inferencer z for mnist:'''
    def __init__(self,conv_dim=64):
        super(IZ1,self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, 256, 4, 1, 0, False)
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)                                              
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = F.tanh(self.conv4(out))
        return out

class IZ2(nn.Module):
    """inferencer z for svhn."""
    def __init__(self, conv_dim=64):
        super(IZ2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.fc = conv(conv_dim*4, 256, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = F.tanh(self.fc(out))
        return out

class ZG1(nn.Module):
    '''Generator mnist from z '''
    def __init__(self,conv_dim=64):
        super(ZG1,self).__init__()
        self.deconv1 = deconv(256, conv_dim*8, 4,1,0)
        self.deconv2 = deconv(conv_dim*8,conv_dim*4,4)
        self.deconv3 = deconv(conv_dim*4,conv_dim*2,4)
        self.deconv4 = deconv(conv_dim*2,1,4,bn=False)
    def forward(self,x):
        x = F.leaky_relu(self.deconv1(x), 0.05) 
        x = F.leaky_relu(self.deconv2(x), 0.05) 
        x = F.leaky_relu(self.deconv3(x), 0.05) 
        x = F.tanh(self.deconv4(x))
        return x

class ZG2(nn.Module):
    '''Generator svhn from z '''
    def __init__(self,conv_dim=64):
        super(ZG2,self).__init__()
        self.deconv1 = deconv(256, conv_dim*8, 4,1,0)
        self.deconv2 = deconv(conv_dim*8,conv_dim*4,4)
        self.deconv3 = deconv(conv_dim*4,conv_dim*2,4)
        self.deconv4 = deconv(conv_dim*2,3,4,bn=False)
    def forward(self,x):
        x = F.leaky_relu(self.deconv1(x), 0.05) 
        x = F.leaky_relu(self.deconv2(x), 0.05) 
        x = F.leaky_relu(self.deconv3(x), 0.05) 
        x = F.tanh(self.deconv4(x))
        return x

class DZ1(nn.Module):
    """Discriminator for mnist and z."""
    def __init__(self,conv_dim=64, use_labels=False):
        super(DZ1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.deconv1 = deconv(256, conv_dim*4, 4,1,0)
        # self.deconv2 = deconv(conv_dim*,conv_dim*4,4)
        # self.deconv3 = deconv(conv_dim*4,conv_dim*2,4)
        # self.deconv4 = deconv(conv_dim*2,1,4,bn=False)

        self.fc = conv(conv_dim*8, 1, 4, 1, 0, False)
        
    def forward(self, x, z):

        z = self.deconv1(z)
        # z = F.sigmoid(z)
        # z = self.deconv2(z)
        # z = self.deconv3(z)
        # z = self.deconv4(z)
        # x = torch.cat((x,z),dim=1)
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        # out = out + z
        out = torch.cat((out,z),dim=1)

        out = self.fc(out).squeeze()
        return out

class DZ2(nn.Module):
    """Discriminator for svhn and z."""
    def __init__(self,conv_dim=64, use_labels=False):
        super(DZ2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        # self.conv4 = conv(conv_dim*4, conv_dim*2 , 4, 1, 0, False)
        self.convz1 = deconv(256, conv_dim*4, 4,1,0)
        # self.convz2 = conv(conv_dim*4,conv_dim*2,1,1,0,False)
        self.fc = conv(conv_dim*8,1,1,1,0,False)

    def forward(self, x, z):

        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        # x = F.leaky_relu(self.conv4(out),0.05)
        z = F.leaky_relu(self.convz1(z), 0.05)
        # z = F.leaky_relu(self.convz2(z), 0.05)
        # print(x.size(),z.size())
        out = torch.cat((out,z),dim=1)

        out = self.fc(out).squeeze()
        return out

#z = torch.randn((1,256,1,1))
#x = torch.randn((1,3,32,32))
#z = torch.autograd.Variable(z)
#x = torch.autograd.Variable(x)
#net = DZ2()
#print(net(x,z).size())
