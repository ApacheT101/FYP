# -*- coding: utf-8 -*-
import itertools
import os
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.backends import cudnn
from torch.cuda.random import manual_seed_all
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from config import Default_config
import transforms
from models import resnest50

# 只做评估 也就是测试 最后得到confusionmatrix.png
def evaluation(args, device, net):
    net.eval()
    # 同训练 transform
    test_transform = transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
    # 同训练 dataset
    test_dataset = ImageFolder(os.path.join(args.datasets_name,'test'), test_transform)
    # 同训练 dataloader
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  pin_memory=True)
    # 加载模型参数
    model_root_path = os.path.join(os.path.curdir,args.save_model_path)
    target_model_path = os.path.join(model_root_path,args.load_model_path)
    state = torch.load(target_model_path,map_location=device)
    net_state = net.state_dict()
    net_state.update(state)
    net.load_state_dict(net_state)

    counts = 0

    true_classes = []
    pre_classes = []
    # 测试结果 得到acc
    begin = time.time()
    with torch.no_grad():
        for i, (img, p_id) in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
            # test code
            # if i==2:
            #     break
            print(i,'/',len(test_dataloader))
            img, p_id = img.to(device), p_id.to(device)
            output = net(img)

            predict = output.argmax(dim=1)

            true_classes.extend(p_id.cpu().numpy().tolist())
            pre_classes.extend(predict.cpu().numpy().tolist())

            counts += predict.eq(p_id).sum().item()
    end = time.time()-begin
    avg_acc = counts * 1.0 / len(test_dataset)
    print('Test acc is %.4f'%(avg_acc))
    # 画confusionmatrix.png
    test_acc = np.array(true_classes)==np.array(pre_classes)
    test_acc = test_acc.astype(np.int).mean()
    cm = confusion_matrix(y_true=true_classes, y_pred=pre_classes)
    cm = cm/np.sum(cm,1)*100
    cm = cm.astype(np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.rainbow)
    plt.title('Confusion Matrix: accuracy={:0.4f}'.format(test_acc))
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.yticks(range(cm.shape[0]))
    # plt.xticks(range(cm.shape[1]))
    plt.savefig(os.path.join('confusionmatrix.png'))
    plt.show()

if __name__=='__main__':
    # 配置参数  同train
    config = Default_config()
    parser = argparse.ArgumentParser(prog='Pytorch',description= 'Example of pytorch')
    parser.add_argument('--stage',type=str,default='test',help='Train or val or test')
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
    net = resnest50(num_classes=args.num_classes).to(device)
    evaluation(args = args,device = device, net = net)
# 0.7789