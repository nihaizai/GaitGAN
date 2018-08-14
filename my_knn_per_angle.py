#coding: utf-8
from sklearn.neighbors import KNeighborsClassifier
import cv2
import os
import numpy as np

result = np.zeros((11,11))
angles_gallery = ['000','018','036','054','072',
                  '090',
                  '108','126','144','162','180']
angles_probe = angles_gallery

test_path = open('./test_path.txt','w')

ix = 0
iy = 0
for g_ang in angles_gallery:
    iy = 0
    for p_ang in angles_probe:
        pid = 63 #测试数据集包括63 - 124
        X = []
        y = []
        for cond in ['nm-01','nm-02','nm-03','nm-04']:
            for p in range(pid,125):
                path = '/home/mg/code/data/GEI_CASIA_B/gei/%03d/%s/%03d-%s-%s.png' % (
                    p,cond,p,cond,g_ang)
                path1 = '/home/mg/code/my_GAN_dataSet/transformed/transformed_1/%03d-%s-%s.png' % (p,cond,g_ang)

                if not os.path.exists(path):
                    continue
                #if g_ang == '090':
                  #  img = cv2.imread(path,0)    #读取的是90度的原目标图
                   # test_path.write(path + '\n')
                #else:
                img = cv2.imread(path1,0)
                test_path.write(path1 + '\n')
                img = cv2.resize(img,(64,64))
                img = img.flatten().astype(np.float32)
                X.append(img)
                y.append(p - 63)

        nbrs = KNeighborsClassifier(n_neighbors = 1,p = 1,weights = 'distance')
        X = np.asarray(X)
        y = np.asarray(y).astype(np.int32)
        nbrs.fit(X,y)

        testX = []
        testy = []
        pid = 63
        for cond in ['cl-01','cl-02']:
            for p in range(pid,125):
                path = '/home/mg/code/data/GEI_CASIA_B/gei/%03d/%s/%03d-%s-%s.png' % (
                    p,cond,p,cond,p_ang)
                path1 = '/home/mg/code/my_GAN_dataSet/transformed/transformed_1/%03d-%s-%s.png' % (p,cond,p_ang)
                if not os.path.exists(path):
                    continue
                #if p_ang == '090':
                 #   img = cv2.imread(path,0)   #读取的是90度的原目标图，若为bg，cl情况，则不应这样做
                  #  test_path.write('test :     ' + path +'\n')
               # else:
                img = cv2.imread(path1,0)
                test_path.write('test :     ' + path1 +'\n')
                img = cv2.resize(img,(64,64))
                img = img.flatten().astype(np.float32)
                testX.append(img)
                testy.append(p-63)

        testX = np.asarray(testX).astype(np.float32)
        s = nbrs.score(testX,testy)
        #s = s *100
        #s = '%.2f' % s
        result[ix][iy] = s
        print(s)
        iy += 1
    ix += 1
print(result)
np.savetxt("view_analysis_cl_generate.csv",result)
test_path.close()
        


















        
            
