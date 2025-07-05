import os
import random

trainval_percent = 0.9  # 训练集和验证集占总数据集的比例
train_percent = 0.9  # 训练集占训练集和验证集之和的比例
xmlfilepath = "/media/zy/9361-7A75/img/img01/Annotations"
txtsavepath = "/media/zy/9361-7A75/img/img01/ImageSets"
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open("/media/zy/9361-7A75/img/img01/ImageSets/trainval.txt", "w")
ftest = open("/media/zy/9361-7A75/img/img01/ImageSets/test.txt", "w")
ftrain = open("/media/zy/9361-7A75/img/img01/ImageSets/train.txt", "w")
fval = open("/media/zy/9361-7A75/img/img01/ImageSets/val.txt", "w")

for i in list:
    name = total_xml[i][:-4] + "\n"
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
