# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.backends import cudnn
from torch.cuda.random import manual_seed_all
from config import Default_config
from models import resnest50
import transforms

# 给一张图像可以得到他的分类结果
def classification(img_path, device, net):
    net.eval()
    transform_ = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_root_path = os.path.join(os.path.curdir, args.save_model_path)
    target_model_path = os.path.join(model_root_path, args.load_model_path)
    state = torch.load(target_model_path,map_location=device)
    net_state = net.state_dict()
    net_state.update(state)
    net.load_state_dict(net_state)
    # net.load_state_dict(torch.load(target_model_path, map_location=torch.device('cpu')))
    results = os.path.join('results')
    os.makedirs(results,exist_ok=True)
    cls_names = ['n10148035',
                 'n10565667',
                 'n11879895',
                 'n11939491',
                 'n13037406',
                 'n13040303',
                 'n13044778',
                 'n13052670',
                 'n13054560',
                 'n13133613',
                 'n15075141']

    with open(img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    img_np = np.array(img)
    img = transform_(img)

    with torch.no_grad():
        img = img.unsqueeze(0)
        img = img.to(device)
        output = net(img)
        predict = output.argmax(dim=1).numpy()
    img_path_cla = cls_names[predict[0]]
    print('predict class:%s' % (img_path_cla))
    plt.imshow(img_np)
    plt.title('predict class:%s' % (img_path_cla))
    img_p = os.path.join(results,'demo.png')
    plt.savefig(img_p)
    plt.show()
    # plt.close()

if __name__=='__main__':
    # 同train
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
    # 可换路径 测试
    parser.add_argument('--target', default=r'D:\Demo\FYP\cls_data\test\n15075141\ILSVRC2012_val_00002663.JPEG',help='Display train statistics')

    args = parser.parse_args()

    cudnn.benchmark = True

    use_cuda = torch.cuda.is_available()
    device = torch.device('cpu')
    if use_cuda:
        manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)

    # 模型加载
    net = resnest50(num_classes=args.num_classes).to(device)
    # 训练 测试
    classification(args.target ,device = device, net = net)
