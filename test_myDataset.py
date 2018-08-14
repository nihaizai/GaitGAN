#coding: utf-8
from my_dataSet import CASIABDataset
import os


#检测一次迭代中my_data2中的r1和r2中是否存在id相同的
def get_diff():
    my_file = open('./my_data2.txt','r')
    content = my_file.readlines()
    len_my = len(content)
    sim_count = 0
    sim_index = []
    for j in range(0,len_my,3):
        r1 = content[j]
        r2 = content[j+1]
        r3 = content[j+2]

        #print("r1:    "+ r1)
        #print("r2:    " + r2)
        #print("r3:    "+ r3)
        
        

        str_r1 = r1.split('/')[0]
        str_r2 = r2.split('/')[0]
        if(str_r1 == str_r2):
            sim_count  += 1
            sim_index.append(j)

    print ('gei_diff:    sim_count:    ' + str(sim_count))
    if (len(sim_index) != 0):
        for i in len(sim_index):
            print sim_index[i]
    else:
        print('sim_index is empty')

#查看一次epoch中my_data2中的内容与batch_data中的是否一致
def check_sim():
    batch_file2 = open('./batch_data.txt','r')
    my_file2 = open('./my_data2.txt','r')
    content_batch = batch_file2.readlines()
    content_my  = my_file2.readlines()
    len_batch = len(content_batch)
    len_my = len(content_my)
    diff_count = 0
    diff_index = []

    if len_batch == len_my:
        for i in range(len_batch):
            if (content_batch[i] != content_my[i]):
                diff_count = 0
                diff_index.append(i)
        print diff_count
        if (len(diff_index) != 0):
            for i in range(len(diff_index)):
                print(diff_index[i])
        else:
            print('diff_index is empty')
    else:
        print('len_batch is not equal with len_my')

###查看数据集中是否均有nm 90度视角
def  check_data(data_dir):
    cond = ['nm-01','nm-02','nm-03',
                  'nm-04','nm-05','nm-06']
    n_id = 62
    n_cond = len(cond)

    loss_id = []
    loss_num = 0
    noLoss_num = 0
    for i in range(n_id):
        idx = i + 1
        idx = '%03d' % idx
        print(idx)
        for j in range(n_cond):
            condx = cond[j]
            r = idx + '/' + condx + '/' + idx + '-' + \
                condx + '-' + '090.png'
            if os.path.exists(data_dir + r):
                noLoss_num += 1
            else:
                loss_num += 1
                loss_id.append(r)

    print('loss_num:     ' + str(loss_num))
    if (len(loss_id) != 0):
        for k in range(len(loss_id)):
            print(loss_id[k])
    else:
        print('loss_id is empty')

    print('noLoss_num:      '+ str(noLoss_num))
    

#查看一次迭代中batch_size的r1，r2，r3是否正确
def check_r():
    start = 0
    batch_size = 206
    end = start + batch_size * 3
    os.remove('./batch_data.txt')
    for i in range(198): #198
        batch_file = open('./batch_data.txt','a')
        batch_file.write('iteration:   '+str(i) + '\n')
        batch_file.close()
        my_file1 = open('./my_data2.txt','r')
        batch_file1 = open('./batch_data.txt','a')
        content = my_file1.readlines()
        print("start:   " + str(start))
        print("end:   " + str(end))
    
    
        for j in range(start,end,3):
            batch_file1.write('batch iteration:   '+str(j) + '\n')
            batch_file1.write(content[j])
            batch_file1.write(content[j+1 ])
            batch_file1.write(content[j+2])

        start = end
        end = start + batch_size * 3
        
        
        batch_file1.close()
        my_file1.close()
        print('End')               
            
            
    
    
dataset = CASIABDataset(data_dir='D:/@mmg/GPU/code/data\GEI_CASIA_B/gei/')
##dataset.my_data()
##get_diff()
##check_r()
#check_data(data_dir='D:/@mmg/GPU/code/data/GEI_CASIA_B/gei/')


        
        


    



#------------------------------------------------epochs + iteration -------------------------------------
##epochs = 1
##iterations = 198
##
##for epoch in range(epochs):
##    start = 0
##    batch_size = 206
##    end = start + batch_size * 3
##    dataset.my_data()
##    os.remove('./batch_data.txt')
##    print('iteration Start')
##    for  iteration  in range(iterations):
##        print('iteration:   ' + str(iteration))
##        my_file1 = open('./my_data2.txt','r')
##        batch_file1 = open('./batch_data.txt','a')
##        content = my_file1.readlines()
##
##        for j in range(start,end,3):
##            batch_file1.write(content[j])
##            batch_file1.write(content[j+1])
##            batch_file1.write(content[j+2])
##            
##        start = end
##        end = start + batch_size * 3
##        print('iteration End')
##        batch_file1.close()
##        my_file1.close()
##        check_sim()



        

        
    
        
    




