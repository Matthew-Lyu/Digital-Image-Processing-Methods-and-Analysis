import argparse

import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os.path as osp

from dataset.dataset import VeriDataset
from models import *

import os

# 重新定义print，用空格分隔
def log_string(out_str, end = '     '):
    print(out_str, end = end)

parser = argparse.ArgumentParser(description='PyTorch Relationship')

parser.add_argument('--data', default='./id_image',         # 车辆图片
                    help='path to dataset')
parser.add_argument('--trainlist', default='./list/train.txt', help='path to train list')   # 训练集
parser.add_argument('--testlist', default='./list/test.txt', help='path to test list')      #测试集
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',        # 用几个进程加载数据
                    help='number of data loading workers (defult: 4)')
parser.add_argument('--batch_size', '--batch-size', default=8, type=int, metavar='N',   # 批次，一次处理多少图片
                    help='mini-batch size (default: 1)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('-n', '--num_classes', default=57, type=int, metavar='N',       # 类别数
                    help='number of classes / categories')
parser.add_argument('--val_step', default=5, type=int,          # 每隔多少次迭代保存一次模型
                    help='val step')
parser.add_argument('--epochs', default=15, type=int,           # 迭代次数
                    help='epochs')
parser.add_argument('--save_dir', default='./checkpoints/att/', type=str,       # 模型保存路径
                    help='save_dir')


# gpu or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
def get_dataset(data_dir, list, type):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 图片归一化
                                     std=[0.229, 0.224, 0.225])

    if type == 'train':
        # 数据增强
        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),      # 随机翻转
            transforms.ToTensor(),
            normalize])
        data_set = VeriDataset(data_dir, list, train_data_transform)    # 用dataset中封装好的类定义数据集
    else:
        test_data_transform = transforms.Compose([
             transforms.ToTensor(),
             normalize])
        data_set = VeriDataset(data_dir, list, test_data_transform)

    data_loader = DataLoader(dataset=data_set, num_workers=args.workers,    # 导入数据
                              batch_size=args.batch_size, shuffle=True)

    return data_loader


def main():
    global args
    args = parser.parse_args()
    print(args)
    # Create dataloader
    log_string('====> Creating dataloader...' ,end='\n')

    data_dir = args.data
    train_list = args.trainlist
    test_list = args.testlist

    # 加载训练测试数据
    train_loader = get_dataset(data_dir, train_list, type = "train")
    test_loader = get_dataset(data_dir, test_list, type = "test")

    # 定义模型
    model = cnn2.DenseNet(num_classes=args.num_classes).to(device)
    mkdir_if_missing(args.save_dir)

    # 定义模型训练优化器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # 定义损失
    criterion = nn.CrossEntropyLoss().to(device)

    TRAIN_ACC = []
    TRAIN_LOSS = []
    TEST_ACC = []
    TEST_LOSS = []

    for epoch in range(args.start_epoch, args.epochs + 1):

        log_string('Epoch: %d' % epoch)

        Train_acc, train_loss = train(train_loader, model, criterion, optimizer)
        Test_acc, test_loss = test(test_loader, model, criterion)

        TRAIN_ACC.append(Train_acc)
        TRAIN_LOSS.append(train_loss)
        TEST_ACC.append(Test_acc)
        TEST_LOSS.append(test_loss)

        # 每隔5个epoch保存一次模型
        if epoch % args.val_step == 0:
            save_checkpoint(model, epoch, optimizer)

    # 画图，准确率、损失随epochs的变化曲线
    import matplotlib.pyplot as plt

    plt.figure(1)
    x = range(0, args.epochs + 1)
    plt.plot(x, TRAIN_ACC, color='r')
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.title("Train Accuracy")
    plt.savefig('./experiment/train_accuracy.png')
    plt.show()

    plt.figure(2)
    x = range(0, args.epochs + 1)
    plt.plot(x, TRAIN_LOSS, color='r')
    plt.xlabel('epoch')
    plt.ylabel('loss(%)')
    plt.title("Train Loss")
    plt.savefig('./experiment/train_loss.png')
    plt.show()

    plt.figure(3)
    x = range(0, args.epochs + 1)
    plt.plot(x, TEST_ACC, color='r')
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.title("Test Accuracy")
    plt.savefig('./experiment/test_accuracy.png')
    plt.show()

    plt.figure(4)
    x = range(0, args.epochs + 1)
    plt.plot(x, TEST_LOSS, color='r')
    plt.xlabel('epoch')
    plt.ylabel('loss(%)')
    plt.title("Test Loss")
    plt.savefig('./experiment/test_loss.png')
    plt.show()

    return

# 模型训练
def train(train_loader, model, criterion, optimizer):
    train_loss = 0
    model.train()   # 训练模式

    correct = 0
    total = 0

    for i, (image, target) in enumerate(train_loader):  # 小批量循环读取读取图片进行训练
        target = target.to(device)      # 标签
        image = image.to(device)        # 图片
        output = model(image)           # 模型输出，预测标签

        loss = criterion(output, target)    # 损失

        optimizer.zero_grad()       # 梯度归零
        loss.backward()             # 反向传播计算得到每个参数的梯度值
        optimizer.step()            # 梯度下降参数更新

        train_loss += loss.data
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

    Train_acc = 100 * float(correct) / total    # 准确率计算
    train_loss = train_loss / (len(train_loader))   # 损失计算
    log_string('train_loss: %.6f' % train_loss)
    log_string('Train_acc: %.6f' % Train_acc)
    return Train_acc, train_loss.cpu()

# 模型测试
def test(test_loader, model, criterion):
    test_loss = 0
    model.eval()  # 测试模式，无需计算梯度反向优化
    correct = 0
    total = 0

    for batch_idx, (image, target) in enumerate(test_loader):
        target = target.to(device)      # 标签
        image = image.to(device)        # 图片
        output = model(image)

        loss = criterion(output, target)
        test_loss += loss.data
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

    Test_acc = 100 * float(correct) / total
    test_loss = test_loss / (len(test_loader))
    log_string('test_loss: %.6f' % test_loss)
    log_string('Test_acc: %.6f' % Test_acc, end = '\n')
    return Test_acc, test_loss.cpu()

# 保存模型
def save_checkpoint(model, epoch, optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath = osp.join(args.save_dir, 'Car_epoch_' + str(epoch) + '.pth')
    torch.save(state, filepath)
    log_string(filepath + " model save!!!", end = '\n')

# 新建文件
def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == '__main__':
    main()
