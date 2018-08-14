my_data = [1,2,3,
           4,5,6,
           7,8,9,
           10,11,12,
           13,14,15,
           16,17,18,
           19,20,21,
           22,23,24,
           25,26,27,
           28,29,30]

start = 0
batch_size = 6
end = start + batch_size

for i in range(4):
    print("i:   "+str(i))
    print("start:   "+str(start))
    print("end:   "+str(end))
    for j in range(start,end,3):
        print("j:   "+str(j))
        print(my_data[j])
        print(my_data[j+1])
        print(my_data[j+2])

    start = end
    end = start +  batch_size
    
        
    

