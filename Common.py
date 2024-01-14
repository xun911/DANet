
# 开发时间 2022/4/16 15:34
import os

def CheckFolder(dataPath):
    flag=os.path.exists(dataPath)
    if flag!=True:
        os.makedirs(dataPath)

if __name__ == '__main__':
    dataPath='model/'+'DBa/'
    CheckFolder(dataPath)
    print('1111')
