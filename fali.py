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
    def __init__(self, config, svhn_loader, mnist_loader,mnist_color_loader):
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.mnist_color_loader = mnist_color_loader

        self.IZ1 = None
        self.IZ2 = None
        self.ZG1 = None
        self.ZG2 = None
        self.DZ1 = None
        self.DZ2 = None

        self.trans_domain = config.trans_domain
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
        self.ZG1 = ZG1(conv_dim=self.g_conv_dim)
        self.ZG2 = ZG2(conv_dim=self.g_conv_dim)
        self.DZ1 = DZ1(conv_dim=self.d_conv_dim)
        self.DZ2 = DZ2(conv_dim=self.d_conv_dim)

        g_params = list(self.IZ1.parameters()) + list(self.IZ2.parameters())+ list(self.ZG1.parameters())+ list(self.ZG2.parameters())
        d_params = list(self.DZ1.parameters()) + list(self.DZ2.parameters())
        
        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
        
        if torch.cuda.is_available():
         
            self.IZ1.cuda()
            self.IZ2.cuda()
            self.ZG1.cuda()
            self.ZG2.cuda()
            self.DZ1.cuda()
            self.DZ2.cuda()
    
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
        mnist_color_iter = iter(self.mnist_color_loader)
        iter_per_epoch = min(len(svhn_iter), len(mnist_iter),len(mnist_color_iter))
        
        # fixed mnist, mnist-color and svhn for sampling
        fixed_mnist = self.to_var(mnist_iter.next()[0])
        fixed_mnist_color = self.to_var(mnist_color_iter.next()[0])
        fixed_svhn = self.to_var(svhn_iter.next()[0])

        # if only two domain transforms, select two domain images 
        if self.trans_domain == 'mnist2color':
            source_iter = mnist_iter
            target_iter = mnist_color_iter
            fixed_source = fixed_mnist
            fixed_target = fixed_mnist_color
        elif self.trans_domain == 'mnist2svhn':
            source_iter = mnist_iter
            target_iter = svhn_iter
            fixed_source = fixed_mnist
            fixed_target = fixed_svhn
        elif self.trans_domain == 'color2svhn':
            source_iter = mnist_color_iter
            target_iter = svhn_iter
            fixed_source = fixed_mnist_color
            fixed_target = fixed_svhn

        # loss if use_labels = True
        criterion = nn.CrossEntropyLoss()

        fixed_z_ = torch.randn((self.batch_size,256)).view(-1, 256, 1, 1)    # fixed noise z
        fixed_z_ = Variable(fixed_z_.cuda())
        
        for step in range(self.train_iters+1):
            # reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0:
                if self.trans_domain == 'mnist2color':
            source_iter = mnist_iter
            target_iter = mnist_color_iter
        elif self.trans_domain == 'mnist2svhn':
            source_iter = mnist_iter
            target_iter = svhn_iter
        elif self.trans_domain == 'color2svhn':
            source_iter = mnist_color_iter
            target_iter = svhn_iter
            
            # load target and source dataset
            target, s_labels = target_iter.next() 
            target, s_labels = self.to_var(target), self.to_var(s_labels).long().squeeze()
            source, m_labels = source_iter.next() 
            source_color, source_color_labels = source_color_iter.next() 
            
            mb_size = min(source.size(0),target.size(0))
            
            source = source[:mb_size,:,:,:]
            target = target[:mb_size,:,:,:]

            # random sample noise z ever iter
            z = torch.randn((mb_size, 256)).view(-1, 256, 1, 1)
            z = Variable(z.cuda())

            # to generate two domain images pair in each batch size;only for mnist2color
            ## when the image paried ,so add the pair percentage pair_num/batch_size
            '''
            pair_num = 10
            idx = np.random.permutation(mb_size-pair_num)
            for i in range(len(idx)):
                source_color[pair_num+i] = source_color[idx[i]+pair_num]
            '''
            source = self.to_var(source)
            target = self.to_var(target)
            # for visulizer
            # source_save,target_save = self.to_data(source), self.to_data(target_save)  
            # merged = self.merge_images(source_save, target_save)
            # path = os.path.join(self.sample_path, 'pair-%d-m-c.png' %(step+1))
            # scipy.misc.imsave(path, merged)

            if self.use_labels:
                source_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes]*target.size(0)).long())
                target_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes]*source.size(0)).long())

            #============ train D ============#
            
            # train with real images
            self.reset_grad()

            z_hat_source = self.IZ2(target)
            out = self.DZ1(source,z_hat_source)

            if self.use_labels:
                d1_loss = criterion(out, m_labels)
            else:
                d1_loss = torch.mean((out-1)**2)
            
            z_hat_target =  self.IZ1(source)
            out = self.DZ2(target,z_hat_target)

            if self.use_labels:
                d2_loss = criterion(out, s_labels)
            else:
                d2_loss = torch.mean((out-1)**2)
            
            d_source_loss = d1_loss
            d_target_loss = d2_loss
            d_real_loss = (d1_loss + d2_loss)
            d_real_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            # the clamp hype need to reset for different domain
            for p in self.DZ1.parameters() :
                p.data.clamp_(-0.01, 0.01)

            for p in self.DZ2.parameters() :
                p.data.clamp_(-0.1, 0.1)

            # train with fake images
            self.reset_grad()
                   
            z_hat_target_sample = z_hat_target
            z_hat_source_sample = z_hat_source
           
            if (step+1) % 2 == 0:
                z_hat_target_sample = z
                z_hat_source_sample = z
            
            fake_target = self.ZG2(z_hat_target_sample)
            out = self.DZ2(fake_target,z)

            if self.use_labels:
                d2_loss = criterion(out, target_fake_labels)
            else:
                d2_loss = torch.mean(out**2)
            
            fake_source = self.ZG1(z_hat_source_sample)
            out = self.DZ1(fake_source,z)

            if self.use_labels:
                d1_loss = criterion(out, source_fake_labels)
            else:
                d1_loss = torch.mean(out**2)
            
            d_fake_loss = d1_loss + d2_loss
            d_fake_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            # the clamp hype need to reset for different domain
            for p in self.DZ1.parameters() :
                p.data.clamp_(-0.01, 0.01)

            for p in self.DZ2.parameters() :
                p.data.clamp_(-0.1, 0.1)

            #============ train G ============#
            
            # train source-target-source cycle
            self.reset_grad()
      
            out = self.DZ2(fake_target,z)
            reconst_source = self.ZG1(z_hat_target)
            reconst_z = self.IZ2(fake_target)

            '''
            # when the image paried ,so add the loss
            pair_loss_source = torch.mean((source[:pair_num] - fake_source[:pair_num])**2)
            '''

            if self.use_labels:
                g_loss = criterion(out, m_labels) 
            else:
                g_loss =  torch.mean((out-0.5)**2) 

            if self.use_reconst_loss:
                # only when paired
                #g_loss += pair_loss_source + 0.1* (torch.mean((source - reconst_source)**2) + torch.mean((z - reconst_z)**2))
                g_loss +=  0.1* (torch.mean((source - reconst_source)**2) + torch.mean((z - reconst_z)**2))
                
            g_loss.backward(retain_graph=True)
            self.g_optimizer.step()

            # train target-source-target cycle
            self.reset_grad()
     
            out = self.DZ1(fake_source,z)
            reconst_target = self.ZG2(z_hat_source)
            reconst_z = self.IZ1(fake_source)
            pair_loss_target = torch.mean((target[:pair_num] - fake_target[:pair_num])**2)

            if self.use_labels:
                g_loss = criterion(out, s_labels) 
            else:
                g_loss = torch.mean((out-0.5)**2) 

            if self.use_reconst_loss:
                # only when paired
                # g_loss += pair_loss_target + 0.1*(torch.mean((target - reconst_target)**2)+ torch.mean((z - reconst_z)**2))
                g_loss += 0.1*(torch.mean((target - reconst_target)**2)+ torch.mean((z - reconst_z)**2))
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # print the log info
            if (step+1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f, d_source_loss: %.4f, d_target_loss: %.4f, '
                      'd_fake_loss: %.4f, g_loss: %.4f' 
                      %(step+1, self.train_iters, d_real_loss.data[0], d_source_loss.data[0], 
                        d_target_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))

            # save the sampled images
            if (step+1) % self.sample_step == 0:
                # fixed sample
                # fake_target = self.IZ1(fixed_source)
                # fake_source = self.IZ2(fixed_target)
               
                # sample from image 
                z_hat_target = self.IZ1(source)
                fake_target = self.ZG2(z_hat_target)
                z_hat_source = self.IZ2(target)
                fake_source = self.ZG1(z_hat_source)
                
                '''
                # sample from z
                fake_target = self.ZG2(fixed_z_)
                fake_source = self.ZG1(fixed_z_)
                '''

                source, fake_source = self.to_data(source), self.to_data(fake_source)
                target , fake_target = self.to_data(target), self.to_data(fake_target)
                
                merged = self.merge_images(source, fake_target)
                path = os.path.join(self.sample_path, 'sample-%d-s-t.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
                
                merged = self.merge_images(target, fake_source)
                path = os.path.join(self.sample_path, 'sample-%d-t-s.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
            
            if (step+1) % 5000 == 0:
                # save the model parameters for each epoch
                IZ1_path = os.path.join(self.model_path, 'IZ1-%d.pkl' %(step+1))
                IZ2_path = os.path.join(self.model_path, 'IZ2-%d.pkl' %(step+1))
                ZG1_path = os.path.join(self.model_path, 'ZG1-%d.pkl' %(step+1))
                ZG2_path = os.path.join(self.model_path, 'ZG2-%d.pkl' %(step+1))
                DZ1_path = os.path.join(self.model_path, 'DZ1-%d.pkl' %(step+1))
                DZ2_path = os.path.join(self.model_path, 'DZ2-%d.pkl' %(step+1))
                torch.save(self.IZ1.state_dict(), IZ1_path)
                torch.save(self.IZ2.state_dict(), IZ2_path)
                torch.save(self.ZG1.state_dict(), ZG1_path)
                torch.save(self.ZG2.state_dict(), ZG2_path)
                torch.save(self.DZ1.state_dict(), DZ1_path)
                torch.save(self.DZ2.state_dict(), DZ2_path)

