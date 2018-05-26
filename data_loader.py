import torch
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import pickle as pkl
import torch.utils.data as data
from PIL import Image


class MNISTM(data.Dataset):
    def __init__(self,images,labels,transform=None,target_transform=None):
        super(MNISTM,self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.images = images
        self.labels = labels
    def __getitem__(self,index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform:
            img = img.transpose(1,2,0)
            img = Image.fromarray(img)
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(img)
        return img,label
    def __len__(self):
        return len(self.images)

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    fw = open('mnistm_data.pkl','rb')
    mnist_m = pkl.load(fw)
    images = mnist_m['train']['images']
    labels = mnist_m['train']['labels']
    
    svhn = datasets.SVHN(root='./svhn/', download = True,transform=transform)
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform)
    mnist_m = MNISTM(images,labels,transform = transform)
    
    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    

    mnist_m_loader = torch.utils.data.DataLoader(mnist_m,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers)

    return svhn_loader, mnist_loader,mnist_m_loader

