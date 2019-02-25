import torch
import torch.nn as nn
from torch.autograd import Variable


class nCrossEntropyLoss(torch.nn.Module):
    """
    自定义损失函数
    """

    def __init__(self, n=4):
        super(nCrossEntropyLoss, self).__init__()
        self.n = n
        self.total_loss = 0
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, label):
        output_t = output[:, 0:10]
        label = Variable(torch.LongTensor(label.data.cpu().numpy()))
        label_t = label[:, 0]

        for i in range(1, self.n):
            # 损失的思路是将一张图平均剪切为4张小图即4个多分类，然后再用多分类交叉熵方损失
            output_t = torch.cat((output_t, output[:, 10 * i:10 * i + 10]), 0)
            label_t = torch.cat((label_t, label[:, i]), 0)
            self.total_loss = self.loss(output_t, label_t)

        return self.total_loss
