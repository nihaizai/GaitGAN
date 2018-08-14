#coding:utf-8
from my_dataSet import CASIABDataset,loadImage
import os
from test_myDataset import get_diff,check_sim,check_data,check_r
import time
import cv2
import numpy as np
import torch as th
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model import NetG,NetD,NetA
import torch.optim as optim
import visdom
from torchvision.utils import make_grid

vis = visdom.Visdom() #port = 8097
win = None
win1 = None
netg = NetG(nc=1)
netd = NetD(nc=1)
neta = NetA(nc=1)
device = th.device("cuda:0")

#weights init
all_mods = itertools.chain()
all_mods = itertools.chain(all_mods,[
    list(netg.children())[0].children(),
    list(netd.children())[0].children(),
    list(neta.children())[0].children()
])

for mod in all_mods:
    if isinstance(mod,nn.Conv2d) or isinstance(mod,nn.ConvTranspose2d):
        init.normal_(mod.weight,0.0,0.02)
    elif isinstance(mod,nn.BatchNorm2d):
        init.normal_(mod.weight,1.0,0.02)
        init.constant_(mod.bias,0.0)

data_dir = '/home/mg/code/data/GEI_CASIA_B/gei/'
netg = netg.to(device)
netd = netd.to(device)
neta = neta.to(device)
netg.train()
netd.train()
neta.train()
dataset = CASIABDataset(data_dir='/home/mg/code/data/GEI_CASIA_B/gei/')
#dataset.my_data()
#get_diff()
#check_r()
#check_data(data_dir='/home/mg/code/data/GEI_CASIA_B/gei/')

lr = 0.0002
real_label = 1
fake_label = 0
fineSize = 64

label = th.zeros((103,1),requires_grad = False).to(device)
optimG = optim.Adam(netg.parameters(),lr = lr/2)
optimD = optim.Adam(netd.parameters(),lr = lr/3)
optimA = optim.Adam(neta.parameters(),lr = lr/3)

print('Training Start epochs')
#------------------------------------------------epochs + iteration #-------------------------------------

epochs = 450
iterations = 198 * 2

time_start = time.time()
os.remove('./loss.txt')
for epoch in range(epochs):
    print('epoch:    '+str(epoch))
    start = 0
    batch_size = 103
    end = start + batch_size * 3
    dataset.my_data()
    get_diff()    #检测my_data2中的r1和r2中是否存在id相同的
    os.remove('./batch_data.txt')
    os.remove('./train_data.txt')
    print('iteration Start')
    for  iteration  in range(iterations):   #每次迭代处理103幅源图像
        #print('iteration:   ' + str(iteration))
        my_file1 = open('./my_data2.txt','r')
        batch_file1 = open('./batch_data.txt','a')
        content = my_file1.readlines()

        batch1 = []
        batch2 = []
        batch3 = []
        

        for j in range(start,end,3):
            batch_file1.write(content[j])
            batch_file1.write(content[j+1])
            batch_file1.write(content[j+2])
            #print(data_dir + content[j])
            

            img1 = loadImage(data_dir + content[j])
            img2 = loadImage(data_dir + content[j+1])
            img3 = loadImage(data_dir + content[j+2])
            


            batch1.append(img1)
            batch2.append(img2)
            batch3.append(img3)
        
        start = end
        end = start + batch_size * 3
        print('iteration End')
        batch_file1.close()
        my_file1.close()
   

        ass_label = th.stack(batch1)
        noass_label = th.stack(batch2)
        img = th.stack(batch3)

        ass_label = ass_label.to(device).to(th.float32)
        noass_label = noass_label.to(device).to(th.float32)
        img = img.to(device).to(th.float32)
        #updata D
        lossD = 0
        optimD.zero_grad()
        output = netd(ass_label)
        label.fill_(real_label)
        lossD_real1 = F.binary_cross_entropy(output,label)
        lossD += lossD_real1.item()
        lossD_real1.backward()

        label.fill_(real_label)
        output1 = netd(noass_label)
        lossD_real2 = F.binary_cross_entropy(output1,label)
        lossD += lossD_real2.item()
        lossD_real2.backward()

        fake = netg(img).detach()
        label.fill_(fake_label)
        output2 = netd(fake)
        lossD_fake = F.binary_cross_entropy(output2,label)
        lossD += lossD_fake.item()
        lossD_fake.backward()

        optimD.step()

        #update A
        lossA = 0
        optimA.zero_grad()
        assd = th.cat((img,ass_label),1)
        noassd = th.cat((img,noass_label),1)
        fake = netg(img).detach()
        faked = th.cat((img,fake),1)

        label.fill_(real_label)
        output1 = neta(assd)
        lossA_real1 = F.binary_cross_entropy(output1,label)
        lossA += lossA_real1.item()
        lossA_real1.backward()

        label.fill_(fake_label)
        output = neta(noassd)
        lossA_real2 = F.binary_cross_entropy(output,label)
        lossA += lossA_real2.item()
        lossA_real2.backward()

        label.fill_(fake_label)
        output = neta(faked)
        lossA_fake = F.binary_cross_entropy(output,label)
        lossA += lossA_fake.item()
        lossA_fake.backward()

        optimA.step()

        #update G
        lossG = 0
        optimG.zero_grad()
        fake = netg(img)
        output = netd(fake)

        label.fill_(real_label)
        lossGD = F.binary_cross_entropy(output,label)
        lossG += lossGD.item()
        lossGD.backward(retain_graph = True)

        faked = th.cat((img,fake),1)
        output = neta(faked)
        label.fill_(real_label)
        lossGA = F.binary_cross_entropy(output,label)
        lossG += lossGA.item()
        lossGA.backward()

        optimG.step()
 
    
    check_sim()  #查看一次epoch中my_data2中的内容与batch_data中的是否一致
   # with th.no_grad():
        #netg.eval()
        #fake = netg(img)
        #netg.train()
    
    #fake = (fake+1) /2 * 255
    #real = (ass_label + 1) / 2 * 255
    #ori = (img + 1) /2 * 255
    #a1 = th.cat((fake,real,ori),2)
    #display = make_grid(a1,20).cpu().numpy()
    #if win1 is None:
     #   win1 = vis.image(display,opts = dict(title='train',caption='train'))
    #else:
     #   vis.image(display,win= win1)

    state = {
        'netA' : neta.state_dict(),
        'netG' : netg.state_dict(),
        'netD' : netd.state_dict()
        }
    th.save(state,'./snapshots/snapshot_%d.t7' % epoch)
    print('epoch = {},ErrG = {},ErrA = {},ErrD = {}'.format(
        epoch,lossG/2,lossA/3,lossD/3))
    err_file = open('./loss.txt','a')
    err_file.write('epoch:   '+str(epoch)+'   lossG/2:   '+str(lossG/2)+\
                   '    lossA/3:    '+str(lossA/3) + '    lossD/3:    '+str(lossD/3) + '\n')
    err_file.close()
           
       
time_end = time.time()
count_time = time_end - time_start
print('count_time:    '+str(count_time))
