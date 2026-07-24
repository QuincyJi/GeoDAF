# -*-coding:utf-8-*-
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import random
import numpy as np
import argparse
from torchsummary import summary
import pickle

import datasets
from model import MeshCNN_Stego_divide3_IR_ALL4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False  


def model_ini(args, device):

    model = MeshCNN_Stego_divide3_IR_ALL4(args).to(device)
    
    # # 加载模型参数
    if args.use_trained:
        model_path = args.model_path
        load_network(model, model_path, args.num_epoch, device=device)

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.2, patience=100, verbose=True)

    pos_weight = torch.tensor([1.7]).to(device)  # 示例值，根据数据不平衡情况调整
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return model, optimizer, criterion, scheduler

def dataset_ini(args):
    dataset_path = args.dataset_path
    num_workers = args.batch_size
    # num_workers = 0
    train_dataset = datasets.BaseDataset(args, path=dataset_path, train=True, target_length=args.target_length)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=num_workers, shuffle=True)
    test_dataset = datasets.BaseDataset(args, path=dataset_path, train=False, target_length=args.target_length)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  num_workers=num_workers, shuffle=True)

    return train_dataloader, test_dataloader


def load_network(model, save_dir, which_epoch, device):
    """load model from disk"""
    save_filename = '%s_net.pth' % which_epoch
    load_path = os.path.join(save_dir, save_filename)
    net = model
    net.load_state_dict(torch.load(load_path, map_location=device))  
    print('loading the model from %s' % load_path)


def save_network(model, save_dir, which_epoch):
    """save model to disk"""
    save_filename = '%s_net.pth' % (which_epoch)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(model.state_dict(), save_path)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: %.3fM, Trainable parameters: %.3fM' % (total_num / 1e6, trainable_num / 1e6)

def get_savepath(args):
    # 保存路径
    checkpoint_folder = args.checkpoint_folder
    out = args.dataset_path.split('/')
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = current_time + '_' + out[-2]
    result_folder = os.path.join(checkpoint_folder, out[-2], name)
    weights_folder = os.path.join(checkpoint_folder, out[-2], name + '_Weights')

    if os.path.exists(result_folder):
        print("File exists!")
    else:
        os.makedirs(result_folder, exist_ok=True)
        os.makedirs(weights_folder, exist_ok=True)
        print("File create!\t" + name)

    return result_folder, weights_folder, name

def train(model, device, train_dataloader, optimizer, criterion, scheduler):
    model.train()
    loss_total = 0
    step_total = 0
    train_ncorrect_total = 0
    train_n_total = 0
    train_TP_total = 0
    train_TN_total = 0
    train_FP_total = 0
    train_FN_total = 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Train")):

        ne = batch['neighbor'].to(device).float().requires_grad_(True)
        fe = batch['feature'].to(device).requires_grad_(True)
        label = batch['label'].to(device)
        slices = batch['edges_count'].to(device)

        optimizer.zero_grad()
        pred = model(ne, fe)

        pred = pred.squeeze()
        label = label.squeeze(-1).float()
        loss = criterion(pred, label)
        loss_total += loss.item()
        step_total += 1
        loss.backward()
        optimizer.step()

        pred_class = (pred >= 0.5).float()
        TP = ((pred_class == 1) & (label == 1)).sum()
        TN = ((pred_class == 0) & (label == 0)).sum()
        FP = ((pred_class == 1) & (label == 0)).sum()
        FN = ((pred_class == 0) & (label == 1)).sum()
        ncorrect = pred_class.eq(label).sum()

        train_TP_total += TP
        train_TN_total += TN
        train_FP_total += FP
        train_FN_total += FN
        train_ncorrect_total += ncorrect
        train_n_total += slices.shape[0]
       
    train_FPR = train_FP_total / (train_FP_total + train_TN_total) if (train_FP_total + train_TN_total) > 0 else 0.0
    train_FNR = train_FN_total / (train_TP_total + train_FN_total) if (train_TP_total + train_FN_total) > 0 else 0.0

    train_acc = train_ncorrect_total / train_n_total
    train_loss = loss_total / step_total
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Train Accuracy: {train_acc:.4f}')
    print(f"FPR: {train_FPR:.4f}, FNR: {train_FNR:.4f}\n")

    return train_loss, train_acc, train_FPR, train_FNR


def test(model, device, test_dataloader, optimizer, criterion, scheduler):

    test_loss_total = 0
    test_step_total = 0
    test_ncorrect_total = 0
    test_n_total = 0
    test_TP_total = 0
    test_TN_total = 0
    test_FP_total = 0
    test_FN_total = 0
    for step, batch in enumerate(tqdm(test_dataloader, desc="Test")):

        model.eval()
        ne = batch['neighbor'].to(device)
        fe = batch['feature'].to(device)
        label = batch['label'].to(device)
        slices = batch['edges_count'].to(device)

        with torch.no_grad():
            pred = model(ne, fe)
           
            pred = pred.squeeze()
            label = label.squeeze(-1).float()

            test_loss = criterion(pred, label)
            test_loss_total += test_loss.item()
            test_step_total += 1

            pred_class = (pred >= 0.5).float()
            TP = ((pred_class == 1) & (label == 1)).sum()
            TN = ((pred_class == 0) & (label == 0)).sum()
            FP = ((pred_class == 1) & (label == 0)).sum()
            FN = ((pred_class == 0) & (label == 1)).sum()
            ncorrect = pred_class.eq(label).sum()

            test_TP_total += TP
            test_TN_total += TN
            test_FP_total += FP
            test_FN_total += FN
            test_ncorrect_total += ncorrect
            test_n_total += label.shape[0]

    test_FPR = test_FP_total / (test_FP_total + test_TN_total) if (test_FP_total + test_TN_total) > 0 else 0.0
    test_FNR = test_FN_total / (test_TP_total + test_FN_total) if (test_TP_total + test_FN_total) > 0 else 0.0
    test_acc = test_ncorrect_total / test_n_total
    test_loss = test_loss_total / test_step_total

    scheduler.step(test_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f"FPR: {test_FPR:.4f}, FNR: {test_FNR:.4f}\n")
    print(f'Learning Rate: {current_lr:.6f}')

    return test_loss, test_acc, test_FPR, test_FNR

def main(args):

    device = torch.device('cuda:'+args.gpu_id)
    model, optimizer, criterion, scheduler = model_ini(args, device)
    train_dataloader, test_dataloader = dataset_ini(args)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    train_FNRs = []
    train_FPRs = []
    test_FNRs= []
    test_FPRs = []

    # 保存路径和保存文件
    result_folder, weights_folder, name = get_savepath(args)
    arg_info_file = os.path.join(result_folder, 'arg_info.txt')
    model_info_file = os.path.join(result_folder, 'model_info.txt')
    train_loss_file = os.path.join(result_folder, 'train_loss.npy')
    train_accs_file = os.path.join(result_folder, 'train_accs.npy')
    test_loss_file = os.path.join(result_folder, 'test_loss.npy')
    test_accs_file = os.path.join(result_folder, 'test_accs.npy')

    train_FPRs_file = os.path.join(result_folder, 'train_FPRs.npy')
    train_FNRs_file = os.path.join(result_folder, 'train_FNRs.npy')
    test_FPRs_file = os.path.join(result_folder, 'test_FPRs.npy')
    test_FNRs_file = os.path.join(result_folder, 'test_FNRs.npy')

    # 保存训练参数
    with open(arg_info_file, 'a') as f:
        for eachArg, value in args.__dict__.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
    # 获得模型参数
    with open(model_info_file, 'a') as f:
        model_info = summary(model, input_size=(1, 1, 65742, 1))
        f.write('{}\n'.format(model_info))

    print(name)
    best_loss = 1
    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        print("\n=====Epoch {}".format(epoch))
        print('Training...')

        train_loss, train_acc, train_FPR, train_FNR = train(model, device, train_dataloader, optimizer, criterion, scheduler)
        test_loss, test_acc, test_FPR, test_FNR = test(model, device, test_dataloader, optimizer, criterion, scheduler)

        # 输出文件
        train_losses.append(train_loss)
        train_accs.append(train_acc.cpu())
        test_losses.append(test_loss)
        test_accs.append(test_acc.cpu())

        train_FPRs.append(train_FPR.cpu())
        train_FNRs.append(train_FNR.cpu())
        test_FPRs.append(test_FPR.cpu())
        test_FNRs.append(test_FNR.cpu())


        if epoch == 1:
            best_loss = test_loss
            best_acc = test_acc
            best_loss_model = model
            best_acc_model = model
            loss_best_name = 'loss_best_1'
            acc_best_name = 'acc_best_1'

        # 保存最好的模型
        if test_loss < best_loss:
            best_loss = test_loss
            best_loss_model = model
            loss_best_name = 'loss_best_' + str(epoch)
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_model = model
            acc_best_name = 'acc_best_' + str(epoch)
        # 每隔5epoch输出以此
        if epoch % 5 == 0:
            train_losses_array = np.array(train_losses)
            np.save(train_loss_file, train_losses_array)
            train_accs_array = np.array(train_accs)
            np.save(train_accs_file, train_accs_array)
            test_losses_array = np.array(test_losses)
            np.save(test_loss_file, test_losses_array)
            test_accs_array = np.array(test_accs)
            np.save(test_accs_file, test_accs_array)

            train_FPRs_array = np.array(train_FPRs)
            np.save(train_FPRs_file, train_FPRs_array)
            train_FNRs_array = np.array(train_FNRs)
            np.save(train_FNRs_file, train_FNRs_array)
            test_FPRs_array = np.array(test_FPRs)
            np.save(test_FPRs_file, test_FPRs_array)
            test_FNRs_array = np.array(test_FNRs)
            np.save(test_FNRs_file, test_FNRs_array)

            save_network(model, weights_folder, epoch)
            save_network(best_loss_model, weights_folder, loss_best_name)
            save_network(best_acc_model, weights_folder, acc_best_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSCG_NET')

    parser.add_argument('--gpu_id', type=str, default='2',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU.')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Value of random seed.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Epoch of training the model.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size during training and testing.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Batch size during training and testing.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help=' ')
    parser.add_argument('--target_length', type=int, default=2000,
                        help=' ')
    parser.add_argument('--save_fold', type=str,
                        # default='SaveMesh8000_4',
                        # default='SaveMesh6000_5',
                        # default='SaveMesh4000_8',
                        default='SaveMesh2000_16',
                        # default='SaveMesh1000_32',
                        # default='SaveMesh500_64',
                        help='the name of features saved fold ')

    parser.add_argument('--dataset_path', type=str, default='/home/jiqingzhi/Dataset/PSB_C1P6/SaveMesh2000_16_AllInOne_obj', help='')

    parser.add_argument('--use_trained', type=bool, default=False,
                        help='Whether to load trained weights')
    parser.add_argument('--model_path', type=str,                       
                        # default='/home/jiqingzhi/checkpoints/PSB_C1P6/2024-06-24_23-25-57_PSB_C1P6_Weights',   
                        help='Path of the model weight, you do not have to set it when using our trained weights.')
    parser.add_argument('--num_epoch', type=str, default='80',
                        help='Num of the model weight epoch')

    parser.add_argument('--checkpoint_folder', type=str, default=r'/home/jiqingzhi/checkpoints',
                        help='Whether to save the model.')

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(args.seed)

    main(args)


