"""
# ***************************
#  code author : Ren
#  papaer: unsupervised Domain Adaptation for face anti-spoofing
# ***************************
"""
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import os
from visualize import make_dot

#path
codeDirRoot = ''
#Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 20
learning_rate = 0.001
dacay_rate = 0.001
batch_size = 100

# data
trainset_path = 'C:\\Users\\ryyuan1\\Desktop\\1.jpg'
data_transform = transforms.Compose([
    transforms.Scale((227, 227), 2),                           #对图像大小统一
    transforms.RandomHorizontalFlip(),                        #图像翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[    #图像归一化
                             0.229, 0.224, 0.225])
         ])
trainloader = torch.utils.data.DataLoader(trainset_path, batch_size=1, shuffle=True, num_workers=8)
train_dataset = torchvision.datasets.ImageFolder(root='/path/data/train/', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#net

#调用模型
alexnet_model = models.alexnet(pretrained=True)
#修改类别为2
alexnet_model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
          )
# 卷积层参数进行训练,训练的时候全网络都BP
for param1 in alexnet_model.features.parameters():
    param1.requires_grad = True
# 将FC层进行初始化
for param2 in alexnet_model.classifier.parameters():
    torch.nn.init.normal(param2, mean=0, std=0.001)

# loss and optimizer
# optimizer = torch.optim.SGD(alexnet_model.parameters(), lr=0.02)
# loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

print(alexnet_model)
# for t in range(1):
#     out = alexnet_model(trainloader)                 # input x and predict based on x
#     loss = loss_func(out)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
#
#     optimizer.zero_grad()   # clear gradients for next train
#     loss.backward()         # backpropagation, compute gradients
#     optimizer.step()        # apply gradients
for epoch in range(num_epochs):
    batch_size_start = time.time()
    running_loss = 0.0
    for i, (inputs) in enumerate(train_loader):
        optimizer = torch.optim.SGD(alexnet_model.classifier.parameters(),
                                    lr=0.0001, momentum=0.9, weight_decay=0.0005,
                                   )
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1e-7) #每个epoch更新一次lr
        inputs = Variable(inputs)
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = alexnet_model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)  # 交叉熵
        loss.backward()
        optimizer.step()  # 更新权重
        running_loss += loss.data[0]

    print('Epoch [%d/%d], Loss: %.4f,need time %.4f'
          % (epoch + 1, num_epochs, running_loss / (4000 / batch_size), time.time() - batch_size_start))
#可视化
g = make_dot(alexnet_model)
g.view()
# 保存模型和特征
saveModelName = os.path.join(codeDirRoot, "model", "alexnet_model.pkl" + "_" + str(num_epochs))

torch.save(alexnet_model.state_dict(), saveModelName)


