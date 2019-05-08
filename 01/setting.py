import  numpy as np

import tkinter as tk  # 导入tkinter模块



def load_setting(filename="setting.txt"):
    if filename =="" :
        print("没有提供文件名")
    setting=np.loadtxt(filename)
    gamma = float(setting[0])
    #print("GAMMA=",gamma)
    batchsize = int(setting[1])
    #print("batchsize=",batchsize)
    istrain = int(setting[2])
    #print("istrain=",istrain)
    isrender = int(setting[3])
    #print("isrender=",isrender)
    timesperround = int(setting[4])
    #print("timesperround=",timesperround)
    roundpershow = int(setting[5])
    #print("roundpershow=",roundpershow)
    return istrain,gamma,batchsize,isrender,timesperround,roundpershow


def save_data(data,filename):
    intdata = [float(i) for i in data]
    np.savetxt(filename,intdata)
    #print(intdata)
    #print(type(intdata[0]))

def load_single_data(filename):
    data = np.loadtxt(filename)
    return  data
def save_signle_data(data,filename):
    data=[float(data)]
    np.savetxt(filename, data)

def load_data(filename):
    data = np.loadtxt(filename)
    print(data)
    outdata = []
    for i in range(len(data)):
        outdata.append(float(data[i]))
    #print(data)
    #print(outdata)
    #print(type(outdata[1]))
    return outdata #注意要返回list数组

if __name__ == "__main__":
    load_data(filename="data.txt")
    load_setting()