import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import time

from models import ConvNet, nCrossEntropyLoss
from config import DefaultConfig
from data.dataset import data_loader, data, dataset_size
from utils.utils import equal


net = ConvNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = nCrossEntropyLoss()

best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0

since = time.time()
for epoch in range(DefaultConfig.EPOCH):

    running_loss = 0.0
    running_corrects = 0

    for step, (inputs, label) in enumerate(data_loader):
        # 用 0 填充 LongTensor
        pred = torch.LongTensor(DefaultConfig.BATCH_SIZE, 1).zero_()
        inputs = Variable(inputs)  # (bs, 3, 60, 160)
        label = Variable(label)  # (bs, 4)
        # 梯度清零
        optimizer.zero_grad()

        output = net(inputs)  # (bs, 40)
        loss = loss_func(output, label)

        for i in range(4):
            pre = F.log_softmax(output[:, 10 * i:10 * i + 10], dim=1)  # (bs, 10)
            # 按行取最大值并返回列的索引值
            pred = torch.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)  #

        loss.backward()
        optimizer.step()

        running_loss += loss.data * inputs.shape[0]
        running_corrects += equal(pred.numpy()[:, 1:], label.data.cpu().numpy().astype(int))

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())

    if epoch == DefaultConfig.EPOCH - 1:
        torch.save(best_model_wts, DefaultConfig.file_path + '../checkpoint/best_model_wts.pkl')

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Train Loss:{:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
