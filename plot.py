import matplotlib
matplotlib.use('Agg')

import sys
from matplotlib import pyplot as plt


import numpy as np

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
file4 = sys.argv[4]

data = np.genfromtxt(file1, dtype=None,  delimiter="\n")

index = 0
arr = np.reshape(data,(3,5,7))
arr0 = [[],[],[],[],[],[],[]]
arr1 = [[],[],[],[],[],[],[]]
arr2 = [[],[],[],[],[],[],[]]
for i in range(0,5):
    for j in range(0,7):
        arr0[j].append(data[index])
        index += 1
        arr1[j].append(data[index])
        index += 1
        arr2[j].append(data[index])
        index += 1

plt.figure(figsize=(15,15))
plt.plot(range(1,8),[np.median(a) for a in arr0],"-b",label="Multiple Send Recieve")
plt.plot(range(1,8),[np.median(a) for a in arr1],"-r",label="Pack Unpack Send Receive")
plt.plot(range(1,8),[np.median(a) for a in arr2],"-g",label="Derived Data Send Recieve")
plt.legend(loc="upper left")


plt.boxplot(arr0,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.boxplot(arr1,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.boxplot(arr2,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.xlabel("Method")
plt.ylabel("Time(in seconds)")
plt.savefig("plot1.png")

data = np.genfromtxt(file2, dtype=None,  delimiter="\n")

index = 0
arr = np.reshape(data,(3,5,7))
arr0 = [[],[],[],[],[],[],[]]
arr1 = [[],[],[],[],[],[],[]]
arr2 = [[],[],[],[],[],[],[]]
for i in range(0,5):
    for j in range(0,7):
        arr0[j].append(data[index])
        index += 1
        arr1[j].append(data[index])
        index += 1
        arr2[j].append(data[index])
        index += 1

plt.figure(figsize=(15,15))
plt.plot(range(1,8),[np.median(a) for a in arr0],"-b",label="Multiple Send Recieve")
plt.plot(range(1,8),[np.median(a) for a in arr1],"-r",label="Pack Unpack Send Receive")
plt.plot(range(1,8),[np.median(a) for a in arr2],"-g",label="Derived Data Send Recieve")
plt.legend(loc="upper left")

plt.boxplot(arr0,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.boxplot(arr1,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.boxplot(arr2,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.xlabel("Method")
plt.ylabel("Time(in seconds)")
plt.savefig("plot2.png")

data = np.genfromtxt(file3, dtype=None,  delimiter="\n")

index = 0
arr = np.reshape(data,(3,5,7))
arr0 = [[],[],[],[],[],[],[]]
arr1 = [[],[],[],[],[],[],[]]
arr2 = [[],[],[],[],[],[],[]]
for i in range(0,5):
    for j in range(0,7):
        arr0[j].append(data[index])
        index += 1
        arr1[j].append(data[index])
        index += 1
        arr2[j].append(data[index])
        index += 1

plt.figure(figsize=(15,15))
plt.plot(range(1,8),[np.median(a) for a in arr0],"-b",label="Multiple Send Recieve")
plt.plot(range(1,8),[np.median(a) for a in arr1],"-r",label="Pack Unpack Send Receive")
plt.plot(range(1,8),[np.median(a) for a in arr2],"-g",label="Derived Data Send Recieve")
plt.legend(loc="upper left")

plt.boxplot(arr0,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.boxplot(arr1,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.boxplot(arr2,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.xlabel("Method")
plt.ylabel("Time(in seconds)")
plt.savefig("plot3.png")

data = np.genfromtxt(file4, dtype=None,  delimiter="\n")

index = 0
arr = np.reshape(data,(3,5,7))
arr0 = [[],[],[],[],[],[],[]]
arr1 = [[],[],[],[],[],[],[]]
arr2 = [[],[],[],[],[],[],[]]
for i in range(0,5):
    for j in range(0,7):
        arr0[j].append(data[index])
        index += 1
        arr1[j].append(data[index])
        index += 1
        arr2[j].append(data[index])
        index += 1

plt.figure(figsize=(15,15))
plt.plot(range(1,8),[np.median(a) for a in arr0],"-b",label="Multiple Send Recieve")
plt.plot(range(1,8),[np.median(a) for a in arr1],"-r",label="Pack Unpack Send Receive")
plt.plot(range(1,8),[np.median(a) for a in arr2],"-g",label="Derived Data Send Recieve")
plt.legend(loc="upper left")

plt.boxplot(arr0,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.boxplot(arr1,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.boxplot(arr2,labels=[16*16,32*32,64*64,128*128,256*256,512*512,1024*1024])
plt.xlabel("Method")
plt.ylabel("Time(in seconds)")
plt.savefig("plot4.png")
