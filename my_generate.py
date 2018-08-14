import cv2
import torch as th
from model import NetG,NetD,NetA
from my_dataSet import CASIABDatasetGenerate


netg = NetG(nc=1)
netd = NetD(nc=1)
neta = NetA(nc=1)

device = th.device("cuda:0")
netg = netg.to(device)
netd = netd.to(device)
neta = neta.to(device)
fineSize = 64

checkpoint = '/home/mg/code/my_GAN_dataSet/snapshots/snapshot_449.t7'
checkpoint = th.load(checkpoint)
neta.load_state_dict(checkpoint['netA'])
netg.load_state_dict(checkpoint['netG'])
netd.load_state_dict(checkpoint['netD'])
neta.eval()
netg.eval()
netd.eval()

angles = ['000','018','036','054','072','090',
          '108','126','144','162','180']

for cond in ['nm-01','nm-02','nm-03','nm-04','cl-01',
             'cl-02']:
    dataset = CASIABDatasetGenerate(data_dir='/home/mg/code/data/GEI_CASIA_B/gei/',cond = cond)
    for i in range(1,125):
        ass_label,img = dataset.getbatch(i,11)
        img = img.to(device).to(th.float32)

        with th.no_grad():
            fake = netg(img)
            fake = (fake + 1) /2 * 255
            for j in range(11):
                fake_ = fake[j].squeeze().cpu().numpy()
                ang = angles[j]
                cv2.imwrite('/home/mg/code/my_GAN_dataSet/transformed/transformed_1/%03d-%s-%s.png' %(i,cond,ang),fake_)


