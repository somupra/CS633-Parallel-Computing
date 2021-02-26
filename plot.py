import sys
from matplotlib import pyplot as plt
import numpy as np

P = sys.argv[1]
fileN = sys.argv[2]
data = np.genfromtxt(fileN, dtype=None,  delimiter="\n")

arr = np.reshape(data,(3,35))

plt.figure(figsize=(15,15))
plt.boxplot([arr[:,0],arr[:,1],arr[:,2]])
plt.xlabel("Method")
plt.ylabel("Time(in seconds)")
plt.show()
if(P == 16):
    plt.title("Plot for P = 16")
    plt.savefig("plot1.png")
if(P == 36):
    plt.title("Plot for P = 36")
    plt.savefig("plot2.png")
if(P == 49):
    plt.title("Plot for P = 49")
    plt.savefig("plot3.png")
if(P == 64):
    plt.title("Plot for P = 64")
    plt.savefig("plot4.png")
