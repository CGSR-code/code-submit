import torch
import pickle
import os
import argparse
import datetime

from dataloader import MultiSessionsGraph
from torch_geometric.data import DataLoader
from model import CGSR
from metric import cal_hr, cal_mrr, cal_ndcg


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/gowalla/amazon')
parser.add_argument('--item_num', type=int, default=43098, help='gowalla:29510')
parser.add_argument('--batch_size', type=int, default=20, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=110, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_decay_epoch', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--reg', type=float, default=1e-6, help='l2 penalty')
parser.add_argument('--dropout', type=float, default=0.6, help='dropout')
parser.add_argument('--leaky_relu', type=float, default=0.2, help='leaky_relu')
parser.add_argument('--WGAT_heads', type=int, default=6, help='WGAT_heads')
parser.add_argument("--use_cuda",type=bool,default=True)
parser.add_argument('--metrics', nargs='?', default='[5, 10, 20]',help='topk')
parser.add_argument('--gpu',type=str, default='0', help='gpu card ID')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
train_dataset = MultiSessionsGraph(f"./datasets/{args.dataset}", phrase='train')
test_dataset = MultiSessionsGraph(f"./datasets/{args.dataset}", phrase='test')
args.metrics = eval(args.metrics)

if torch.cuda.is_available() and args.use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

training_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
testing_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = CGSR(args,device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay)
criterion = torch.nn.CrossEntropyLoss()

best_hr = 0
best_mrr = 0
best_ndcg = 0
total_loss = []
for epoch in range(args.epoch):
    print("epoch:(%d/%d)" % (epoch + 1, args.epoch))
    print("Start Training:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # train
    model.train()
    cur_loss = 0
    for i, data in enumerate(training_dataloader):
        optimizer.zero_grad()
        # forward & backward
        outputs = model(data.to(device))
        labels = data.y - 1
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        cur_loss += loss.item() * len(data)
        if i % 5000 == 0 and i != 0:
            print('(%d/%d) loss: %f' % (i, len(training_dataloader), loss.item()))
    scheduler.step()
    print(f"total loss: {cur_loss / len(training_dataloader.dataset)}")
    total_loss += [cur_loss]

    # test
    model.eval()
    test_loss = 0
    print("Start testing:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("effect_weight:",model.fuse_weight1.item(),"causal_weight:",model.fuse_weight2.item(),"relation_weight:",model.fuse_weight3.item())
    print("edge_weight1:",model.edge_weight1.item(),"edge_weight2:",model.edge_weight2.item(),"edge_weight3:",model.edge_weight3.item())  
    with torch.no_grad():
        hrs = [0 for _ in range(len(args.metrics))]
        mrrs = [0 for _ in range(len(args.metrics))]
        ndcgs = [0 for _ in range(len(args.metrics))]
        for data in testing_dataloader:
            # forward & backward
            outputs = model(data.to(device))
            labels = data.y - 1
            loss = criterion(outputs, labels.to(device))

            test_loss += loss.item() * len(data)

            # metric
            result = torch.topk(outputs, k=args.metrics[-1], dim=1)[1]
            for i, k in enumerate(args.metrics):
                hrs[i] += cal_hr(result[:, :k].cpu().numpy(), labels.cpu().numpy())
                mrrs[i] += cal_mrr(result[:, :k].cpu().numpy(), labels.cpu().numpy())
                ndcgs[i] += cal_ndcg(result[:, :k].cpu().numpy(), labels.cpu().numpy())

        test_loss = test_loss / len(testing_dataloader.dataset)

        for i, k in enumerate(args.metrics):
            hrs[i] = hrs[i] / len(testing_dataloader.dataset)
            mrrs[i] = mrrs[i] / len(testing_dataloader.dataset)
            ndcgs[i] = ndcgs[i] / len(testing_dataloader.dataset)
            print(f'HR@{k}: {hrs[i]:.4f} MRR@{k}: {mrrs[i]:.4f} NDCG@{k}: {ndcgs[i]:.4f}')

        if hrs[-1] > best_hr:
            best_hr = hrs[-1]
            best_mrr = mrrs[-1]
            best_ndcg = ndcgs[-1]
            for i, k in enumerate(args.metrics):
                print(f'best ever HR@{k}: {hrs[i]:.4f} MRR@{k}: {mrrs[i]:.4f} NDCG@{k}: {ndcgs[i]:.4f}')
        print('================================')

