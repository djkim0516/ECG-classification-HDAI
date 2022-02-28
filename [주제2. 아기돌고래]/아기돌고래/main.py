import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from resnet import resnet34, resnet18
from utils import cal_f1, cal_aucs
from sklearn.preprocessing import scale
from data_preprocessing import data_lbl_to_array
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for data dir')
    parser.add_argument('--leads', type=int, default=8, help='ECG leads to use')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=113, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='test', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='models/epoch12_auc9938.pth', help='Path to saved model')
    return parser.parse_args()

def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for idx, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.type(torch.FloatTensor).to(device), labels.type(torch.FloatTensor).to(device)
        output = net(data).type(torch.FloatTensor).to(device)
        labels = torch.reshape(labels, (output.shape[0], output.shape[1]))
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())

    print('Train Loss: %.4f' % (running_loss))

def evaluate(dataloader, net, args, criterion, device):
    print('\n----------------Validating----------------')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.type(torch.FloatTensor).to(device), labels.type(torch.FloatTensor).to(device)
        output = net(data).type(torch.FloatTensor).to(device)
        labels = torch.reshape(labels, (output.shape[0], output.shape[1]))
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Validation Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1(y_trues, y_scores, True)
    print('Validation F1: %.4f' % f1s)
    aucs = cal_aucs(y_trues, y_scores)
    print('Validation AUC: %.4f' % aucs)
    if args.phase == 'train' and aucs > args.best_metric:       # f1이 아니라 auc를 기준으로 최상 모델 저장
        args.best_metric = aucs
        torch.save(net.state_dict(), args.model_path)

if __name__ == "__main__":
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)

    if not args.model_path:
        args.model_path = f'models/resnet34_{database}_{args.leads}.pth'

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'

    #train 코드
    if args.phase == 'train':
        train_org = np.concatenate((np.load('./data/train/arrhythmia.npy'),np.load('./data/train/normal.npy')), axis=0)
        valid_org = np.concatenate((np.load('./data/val/arrhythmia.npy'),np.load('./data/val/normal.npy')), axis=0)
        
        DATA_PATH = glob.glob(os.path.join(os.path.relpath(path='./electrocardiogram'), 'data', '*', '*'))
        train_arryh, train_nor, valid_arryh, valid_nor = data_lbl_to_array(DATA_PATH)
        
        train_org = np.concatenate((train_arryh, train_nor), axis=0)
        valid_org = np.concatenate((valid_arryh, valid_nor), axis=0)
        
        train_X, valid_X = train_org[:, :-1], valid_org[:, :-1]          # 라벨 떼고
        train_label, valid_label = train_org[:, -1], valid_org[:, -1]    # 라벨만

        train_label[np.where(train_label == 0)] = 0
        train_label[np.where(train_label != 0)] = 1
        valid_label[np.where(valid_label == 0)] = 0
        valid_label[np.where(valid_label != 0)] = 1


        Y_train = torch.from_numpy(train_label.astype(int))
        X_train = torch.from_numpy(scale(train_X.reshape(-1, 5000), axis = 1).reshape(-1, 8, 5000))
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        Y_valid = torch.from_numpy(valid_label.astype(int))
        X_valid = torch.from_numpy(scale(valid_X.reshape(-1,5000), axis = 1).reshape(-1, 8, 5000))
        valid_dataset = TensorDataset(X_valid, Y_valid)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    #test 코드
    else:
        DATA_PATH = glob.glob(os.path.join(os.path.relpath(path='./electrocardiogram'), 'data', 'test', '*'))
        test_arryh, test_nor = data_lbl_to_array(DATA_PATH)
        test_org = np.concatenate((test_arryh, test_nor), axis=0)      # 정상 / 부정맥 2개의 폴더로 구성되어있을 시 해당 코드로 실행
        
        # test_org = data_lbl_to_array(DATA_PATH)       # 폴더 한개에 들어있을 시 사용                         #

        test_X = test_org[:, :-1]          # 라벨 떼고
        test_label = test_org[:, -1]       # 라벨만
        
        test_label[np.where(test_label == 0)] = 0
        test_label[np.where(test_label != 0)] = 1
        
        Y_test = torch.from_numpy(test_label.astype(int))
        X_test = torch.from_numpy(scale(test_X.reshape(-1,5000), axis = 1).reshape(-1, 8, 5000))
        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    

    net = resnet34(input_channels=args.leads, num_classes=1)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()

    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(args.epochs):
            train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            evaluate(valid_loader, net, args, criterion, device)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(test_loader, net, args, criterion, device)