# coding : utf-8

from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

from utils.visualizer import Visualizer
import models
from config import DefaultConfig
from data.dataset import DogCat
import torch as t

def train(opt):
    # 更新配置
    vis = Visualizer(opt.env)

    # step1: 加载模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # step2: 数据
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    # 验证集 data 不做变换
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)

    # step3: 目标函数和优化器
    # 交叉熵损失
    criterion = t.nn.CrossEntropyLoss()
    # 学习率
    lr = opt.lr
    # Adam 优化器
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    # 计算所有数的平均值和标准差，用来统计一个 epoch 中损失的平均值
    loss_meter = meter.AverageValueMeter()
    # 统计分类问题中的分类情况，错误矩阵
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 训练
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        # index，(data, label)
        for ii, (data, label) in enumerate(train_dataloader):

            # 训练模型
            input = data
            target = label
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            # 梯度清零
            optimizer.zero_grad()
            score = model(input)
            # 损失
            loss = criterion(score, target)
            loss.backward()
            # 优化步骤
            optimizer.step()

            # 更新统计指标以及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

        # checkpoint
        model.save()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}"
            .format(
            epoch=epoch,
            loss=loss_meter.value()[0],
            val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()),
            lr=lr))

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    # 使模型进入验证模式
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    # tqdm 可扩展的进度条
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))
    # 将模型重置为训练模式
    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

def test(opt):
    # 模型进入验证模式
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # 数据
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    results = []
    # index (data, id)
    for ii, (data, path) in enumerate(test_dataloader):
        # 包装 data 并记录用在它身上的 operations
        input = t.autograd.Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score)[:, 1].data.tolist()
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results
    write_csv(results, opt.result_file)
    return results

if __name__ == '__main__':
    opt = DefaultConfig()
    train(opt)
    # test(opt)