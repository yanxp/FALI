import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np

from torch.autograd import Variable
from torch import optim
from model import IZ1,IZ2,ZG1,ZG2
from model import DZ1,DZ2


class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader,mnist_m_loader):
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.mnist_m_loader = mnist_m_loader

        self.IZ1 = None
        self.IZ2 = None
        self.IZ3 = None
        self.ZG1 = None
        self.ZG2 = None
        self.ZG3 = None
        self.DZ1 = None
        self.DZ2 = None
        self.DZ3 = None

        self.g_optimizer = None
        self.d_optimizer = None
        self.use_reconst_loss = config.use_reconst_loss
        self.use_labels = config.use_labels
        self.num_classes = config.num_classes
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.build_model()
        
    def build_model(self):
        """Builds a generator and a discriminator."""
        self.IZ1 = IZ1(conv_dim=self.g_conv_dim)
        self.IZ2 = IZ2(conv_dim=self.g_conv_dim)
        self.IZ3 = IZ2(conv_dim=self.g_conv_dim)
        self.ZG1 = ZG1(conv_dim=self.g_conv_dim)
        self.ZG2 = ZG2(conv_dim=self.g_conv_dim)
        self.ZG3 = ZG2(conv_dim=self.g_conv_dim)
        self.DZ1 = DZ1(conv_dim=self.d_conv_dim)
        self.DZ2 = DZ2(conv_dim=self.d_conv_dim)
        self.DZ3 = DZ2(conv_dim=self.d_conv_dim)

        # self.IZ1 = IZ1(conv_dim=self.g_conv_dim)
        # self.IZ2 = IZ2(conv_dim=self.g_conv_dim)
        # self.d1 = D1(conv_dim=self.d_conv_dim, use_labels=self.use_labels)
        # self.d2 = D2(conv_dim=self.d_conv_dim, use_labels=self.use_labels)
        
        g_params = list(self.IZ1.parameters()) + list(self.IZ2.parameters())+ list(self.ZG1.parameters())+ list(self.ZG2.parameters())+list(self.IZ3.parameters())+list(self.ZG3.parameters())
        d_params = list(self.DZ1.parameters()) + list(self.DZ2.parameters())+list(self.DZ3.parameters())
        
        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
        
        if torch.cuda.is_available():
            # self.IZ1.cuda()
            # self.IZ2.cuda()
            # self.d1.cuda()
            # self.d2.cuda()
            self.IZ1.cuda()
            self.IZ2.cuda()
            self.IZ3.cuda()
            self.ZG1.cuda()
            self.ZG2.cuda()
            self.ZG3.cuda()
            self.DZ1.cuda()
            self.DZ2.cuda()
            self.DZ3.cuda()
    
    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row*h, row*w*2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
        return merged.transpose(1, 2, 0)
    
    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()
    
    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        mnist_m_iter = iter(self.mnist_m_loader)
        iter_per_epoch = min(len(svhn_iter), len(mnist_iter),len(mnist_m_iter))
        
        # fixed mnist and svhn for sampling
        fixed_svhn = self.to_var(svhn_iter.next()[0])
        fixed_mnist = self.to_var(mnist_iter.next()[0])
        fixed_mnist_m = self.to_var(mnist_m_iter.next()[0])
         
        
        # loss if use_labels = True
        criterion = nn.CrossEntropyLoss()

        fixed_z_ = torch.randn((self.batch_size,256)).view(-1, 256, 1, 1)    # fixed noise
        fixed_z_ = Variable(fixed_z_.cuda())
        
        for step in range(self.train_iters+1):
            # reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0:
                mnist_iter = iter(self.mnist_loader)
                svhn_iter = iter(self.svhn_loader)
                mnist_m_iter = iter(self.mnist_m_loader)
            
            # load svhn and mnist dataset
            svhn, s_labels = svhn_iter.next() 
            svhn, s_labels = self.to_var(svhn), self.to_var(s_labels).long().squeeze()
            mnist, m_labels = mnist_iter.next() 
            mnist, m_labels = self.to_var(mnist), self.to_var(m_labels)

            mnist_m, mnist_m_labels = mnist_m_iter.next() 
            mnist_m, mnist_m_labels = self.to_var(mnist_m), self.to_var(mnist_m_labels)
            

            if self.use_labels:
                mnist_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes]*svhn.size(0)).long())
                svhn_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes]*mnist.size(0)).long())

            #z = torch.randn((self.batch_size, 256)).view(-1, 256, 1, 1)
            #z = Variable(z.cuda())
            #============ train D ============#
            
            # train with real images
            self.reset_grad()

            mb_size = min(mnist.size(0),svhn.size(0),mnist_m.size(0))
            mnist = mnist[:mb_size,:,:,:]
            m_labels = mnist[:mb_size]
            mnist_m = mnist_m[:mb_size,:,:,:]
            mnist_m_labels = mnist_m_labels[:mb_size]
            svhn = svhn[:mb_size,:,:,:]
            s_labels = s_labels[:mb_size]
            z = torch.randn((mb_size, 256)).view(-1, 256, 1, 1)
            z = Variable(z.cuda())

            z_hat_mnist = self.IZ3(mnist_m)
            out = self.DZ1(mnist,z_hat_mnist)

            if self.use_labels:
                d1_loss = criterion(out, m_labels)
            else:
                d1_loss = torch.mean((out-1)**2)
            
            # out = self.d2(svhn)
            z_hat_svhn =  self.IZ1(mnist)
            out = self.DZ2(svhn,z_hat_svhn)

            if self.use_labels:
                d2_loss = criterion(out, s_labels)
            else:
                d2_loss = torch.mean((out-1)**2)
            
            # out = self.d2(svhn)
            z_hat_mnist_m =  self.IZ2(svhn)
            out = self.DZ3(mnist_m,z_hat_mnist_m)

            if self.use_labels:
                d3_loss = criterion(out, s_labels)
            else:
                d3_loss = torch.mean((out-1)**2)
            
            d_mnist_loss = d1_loss
            d_svhn_loss = d2_loss
            d_mnist_m_loss = d3_loss
            d_real_loss = (d1_loss + d2_loss + d3_loss)
            d_real_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            for p in self.DZ1.parameters() :
                p.data.clamp_(-0.01, 0.01)

            for p in self.DZ2.parameters() :
                p.data.clamp_(-0.075, 0.075)
            
            for p in self.DZ3.parameters() :
                p.data.clamp_(-0.075, 0.075)

            # train with fake images
            self.reset_grad()
            
            z_hat_mnist_sample = z_hat_mnist
            z_hat_svhn_sample = z_hat_svhn
            z_hat_mnist_m_sample = z_hat_mnist_m
            if (step+1) % 2 == 0:
                z_hat_svhn_sample = z
                z_hat_mnist_sample = z
                z_hat_mnist_m_sample = z

            fake_svhn = self.ZG2(z_hat_svhn_sample)
            out = self.DZ2(fake_svhn,z)

            if self.use_labels:
                d2_loss = criterion(out, svhn_fake_labels)
            else:
                d2_loss = torch.mean(out**2)
            
            fake_mnist = self.ZG1(z_hat_mnist_sample)
            out = self.DZ1(fake_mnist,z)

            if self.use_labels:
                d1_loss = criterion(out, mnist_fake_labels)
            else:
                d1_loss = torch.mean(out**2)
            
            fake_mnist_m = self.ZG3(z_hat_mnist_m_sample)
            out = self.DZ3(fake_mnist_m,z)

            if self.use_labels:
                d3_loss = criterion(out, mnist_fake_labels)
            else:
                d3_loss = torch.mean(out**2)
            
            d_fake_loss = d1_loss + d2_loss + d3_loss
            d_fake_loss.backward(retain_graph=True)
            self.d_optimizer.step()
            
            for p in self.DZ1.parameters() :
                p.data.clamp_(-0.01, 0.01)

            for p in self.DZ2.parameters() :
                p.data.clamp_(-0.075, 0.075)

            for p in self.DZ3.parameters() :
                p.data.clamp_(-0.075, 0.075)
            #============ train G ============#
            
            # train mnist-svhn-mnist cycle
            self.reset_grad()

            out = self.DZ2(fake_svhn,z)
            reconst_svhn = self.ZG2(z_hat_mnist_m)
            reconst_z = self.IZ2(fake_svhn)
            
            if self.use_labels:
                g_loss = criterion(out, s_labels) 
            else:
                g_loss =  torch.mean((out-0.5)**2) 

            if self.use_reconst_loss:
                g_loss += 0.1* (torch.mean((svhn - reconst_svhn)**2) + torch.mean((z - reconst_z)**2))

            g_loss.backward(retain_graph=True)
            self.g_optimizer.step()

            # train svhn-mnist-svhn cycle
            self.reset_grad()
            
            out = self.DZ1(fake_mnist,z)
            reconst_mnist = self.ZG1(z_hat_svhn)
            reconst_z = self.IZ1(fake_mnist)
            if self.use_labels:
                g_loss = criterion(out, m_labels) 
            else:
                g_loss = torch.mean((out-0.5)**2) 

            if self.use_reconst_loss:
                g_loss += 0.1*(torch.mean((mnist - reconst_mnist)**2)+ torch.mean((z - reconst_z)**2))

            g_loss.backward(retain_graph=True)
            self.g_optimizer.step()
            
            out = self.DZ3(fake_mnist_m,z)
            reconst_mnist_m = self.ZG3(z_hat_mnist)
            reconst_z = self.IZ3(fake_mnist_m)
            if self.use_labels:
                g_loss = criterion(out, mnist_m_labels) 
            else:
                g_loss = torch.mean((out-0.5)**2) 

            if self.use_reconst_loss:
                g_loss += 0.1*(torch.mean((mnist_m - reconst_mnist_m)**2)+ torch.mean((z - reconst_z)**2))

            g_loss.backward()
            self.g_optimizer.step()
            
            # print the log info
            if (step+1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f,  '
                      'd_fake_loss: %.4f, g_loss: %.4f' 
                      %(step+1, self.train_iters, d_real_loss.data[0], 
                       d_fake_loss.data[0], g_loss.data[0]))

            # save the sampled images
            if (step+1) % self.sample_step == 0:
                # fake_svhn = self.IZ1(fixed_mnist)
                # fake_mnist = self.IZ2(fixed_svhn)
                z_hat_svhn = self.IZ1(fixed_mnist)
                fake_svhn = self.ZG2(z_hat_svhn)
                z_hat_mnist_m = self.IZ2(fixed_svhn)
                fake_mnist_m = self.ZG3(z_hat_mnist_m)
                z_hat_mnist = self.IZ3(fixed_mnist_m) 
                fake_mnist = self.ZG1(z_hat_mnist)
                
                mnist, fake_mnist = self.to_data(fixed_mnist), self.to_data(fake_mnist)
                svhn , fake_svhn = self.to_data(fixed_svhn), self.to_data(fake_svhn)
                mnist_m, fake_mnist_m = self.to_data(fixed_mnist_m), self.to_data(fake_mnist_m)
                merged = self.merge_images(mnist, fake_svhn)
                path = os.path.join(self.sample_path, 'sample-%d-m-s.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
                
                merged = self.merge_images(svhn, fake_mnist_m)
                path = os.path.join(self.sample_path, 'sample-%d-s-c.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
                
                merged = self.merge_images(mnist_m, fake_mnist)
                path = os.path.join(self.sample_path, 'sample-%d-c-m.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
            
            if (step+1) % 5000 == 0:
                # save the model parameters for each epoch
                IZ1_path = os.path.join(self.model_path, 'IZ1-%d.pkl' %(step+1))
                IZ2_path = os.path.join(self.model_path, 'IZ2-%d.pkl' %(step+1))
                IZ3_path = os.path.join(self.model_path, 'IZ2-%d.pkl' %(step+1))
                ZG1_path = os.path.join(self.model_path, 'ZG1-%d.pkl' %(step+1))
                ZG2_path = os.path.join(self.model_path, 'ZG2-%d.pkl' %(step+1))
                ZG3_path = os.path.join(self.model_path, 'ZG2-%d.pkl' %(step+1))
                DZ1_path = os.path.join(self.model_path, 'DZ1-%d.pkl' %(step+1))
                DZ2_path = os.path.join(self.model_path, 'DZ2-%d.pkl' %(step+1))
                DZ3_path = os.path.join(self.model_path, 'DZ2-%d.pkl' %(step+1))
                torch.save(self.IZ1.state_dict(), IZ1_path)
                torch.save(self.IZ2.state_dict(), IZ2_path)
                torch.save(self.IZ3.state_dict(), IZ2_path)
                torch.save(self.ZG1.state_dict(), ZG1_path)
                torch.save(self.ZG2.state_dict(), ZG2_path)
                torch.save(self.ZG3.state_dict(), ZG2_path)
                torch.save(self.DZ1.state_dict(), DZ1_path)
                torch.save(self.DZ1.state_dict(), DZ1_path)
                torch.save(self.DZ3.state_dict(), DZ2_path)
