# -*- coding: utf-8 -*-
import os
import argparse
import torch
from torch import nn,optim
from torch.backends import cudnn
from torch.cuda.random import manual_seed_all
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from config import Default_config
from display_train_statistics import display_statistics
import transforms
from models import resnest50

# 训练和测试代码
def train_val(args, device, net):
    os.makedirs(os.path.join(os.path.curdir, args.save_model_path),exist_ok=True)
    # 训练的 transform 可以理解为预处理
    train_transform = transforms.Compose([
                                          transforms.Resize((224,224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
                                         ])
    # 训练的数据集
    train_dataset = ImageFolder(os.path.join(args.datasets_name,'train'),train_transform)
    # 训练集数据 加载器
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True)
    # 测试集的 transform 可以理解为预处理
    val_transform = transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
    # 测试集的dataset
    val_dataset = ImageFolder(os.path.join(args.datasets_name,'test'), val_transform)
    # 测试集数据 加载器
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=True)
    # 优化器
    optimer = optim.SGD(net.parameters(),
                       args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                    nesterov=True)
    # 学习率衰减策略
    lr_schedualer = optim.lr_scheduler.StepLR(optimer,step_size=args.lr_decay_step,gamma=args.lr_decay_ratio)
    # 损失函数
    loss_function = nn.CrossEntropyLoss(reduction='sum')

    target_num = -1
    all_avg_loss = []
    all_avg_acc = []
    val_all_avg_loss = []
    val_all_avg_acc = []
    # 训练 遍历 epoch
    for epoch in range(target_num + 1, args.all_epoch):
        losses = 0
        counts = 0
        net.train(True)
        # 训练 遍历 所有数据
        for ith, (img, p_id) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
            # test code
            # if ith==2:
            #     break
            img, p_id = img.to(device), p_id.to(device)
            # 前项
            output = net(img)
            # 计算loss
            loss = loss_function(output, p_id)
            optimer.zero_grad()
            # 反传
            loss.backward()
            # 优化参数
            optimer.step()

            predict = output.argmax(dim=1)
            losses += loss.item()
            counts += predict.eq(p_id).sum().item()

            if ith % args.print_info_epoch == 0:
                data_count = args.batch_size * ith + len(p_id)
                print('               epoch%d-%dth,loss = %.4f, acc = %.4f' % (epoch, ith,
                                                                               losses / data_count,
                                                                               counts * 1.0 / data_count))

        lr_schedualer.step(epoch)
        avg_loss = losses / len(train_dataset)
        avg_acc = counts * 1.0 / len(train_dataset)
        all_avg_loss.append(avg_loss)
        all_avg_acc.append(avg_acc)

        print('Train: epoch = %d,loss = %.4f, acc = %.4f' %(epoch,avg_loss,avg_acc))

        net.eval()
        val_losses = 0
        val_counts = 0
        # 测试集 测试
        with torch.no_grad():
            for ith, (img, p_id) in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
                # test code
                # if ith == 2:
                #     break
                img, p_id = img.to(device), p_id.to(device)
                output = net(img)
                loss = loss_function(output,p_id)

                predict = output.argmax(dim=1)
                val_losses += loss.item()
                val_counts += predict.eq(p_id).sum().item()
            avg_loss = val_losses / len(val_dataset)
            avg_acc = val_counts * 1.0 / len(val_dataset)
            val_all_avg_loss.append(avg_loss)
            val_all_avg_acc.append(avg_acc)
            print('Val: epoch = %d,loss = %.4f, acc = %.4f' %(epoch,avg_loss,avg_acc))

        if (epoch + 1) % args.save_model_epoch == 0:
            # 保存权重文件
            save_path = os.path.join(os.path.curdir, args.save_model_path, 'epoch%d.pth' % epoch)
            torch.save(net.state_dict(), save_path)

    # 统计训练损失和准确度
    if args.display_train_statistics:
        display_statistics('train', args.all_epoch-target_num-1, all_avg_loss, all_avg_acc)
        display_statistics('test', args.all_epoch-target_num-1, val_all_avg_loss, val_all_avg_acc)

if __name__=='__main__':
    # 配置参数  具体参数在config.py中
    config = Default_config()
    parser = argparse.ArgumentParser(prog='Pytorch',description= 'Example of pytorch')
    parser.add_argument('--stage',type=str,default=config.stage,help='Train or val or test')
    parser.add_argument('--load_model_path',type=str,default=config.load_model_path,help='Load trained model path')
    parser.add_argument('--save_model_path',type=str,default=config.save_model_path,help='Save trained model path')
    parser.add_argument('--datasets_name',type=str,default=config.datasets_name,help='Train dataset path')
    parser.add_argument('--seed',type=int,default=config.seed,help='Set seed')
    parser.add_argument('--batch_size',type=int,default=config.batch_size,help='Batch size')
    parser.add_argument('--num_classes',type=int,default=config.num_classes,help='num_classes')
    parser.add_argument('--num_workers',type=int,default=config.num_workers,help='Number of workers')
    parser.add_argument('--print_info_epoch',type=int,default=config.print_info_epoch,help='How many epoch to print information')
    parser.add_argument('--all_epoch',type=int,default=config.all_epoch,help='All epoch')
    parser.add_argument('--save_model_epoch',type=int,default=config.save_model_epoch,help='Save model epoch')
    parser.add_argument('--lr',type=float,default=config.lr,help='Leaning rate')
    parser.add_argument('--lr_decay_ratio',type=float,default=config.lr_decay_ratio,help='Leaning rate decay  ratio')
    parser.add_argument('--lr_decay_step',type=int,default=config.lr_decay_step,help='Leaning rate decay step')
    parser.add_argument('--weight_decay',type=float,default=config.weight_decay,help='Weight decay ratio')
    parser.add_argument('--continue_train',type=bool,default=config.continue_train,help='Continue train or from zero')
    parser.add_argument('--display_train_statistics',type=bool,default=config.display_train_statistics,help='Display train statistics')

    args = parser.parse_args()

    cudnn.benchmark = True

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    if use_cuda:
        manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)
    # 搭建网络
    net = resnest50(num_classes=args.num_classes).to(device)
    # 训练或者测试
    train_val(args = args,device = device, net = net)

    print('ok')
