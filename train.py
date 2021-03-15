from Data_union import PatchDataTrain, PatchDataTest
from Net import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import numpy as np

source = ('160_normal_4', '32_sobel_4', '32_canny_4')
# ['牙冠', '牙根', '牙髓', '非牙齿']

lr = 5e-5
batch_size = 100
epochs = 20


def train(data_loader, net, loss, optimizer):
    data_loader = tqdm(data_loader,
                       bar_format='{l_bar}{bar}| Batch: {n_fmt}/{total_fmt} [已运行:{elapsed}，剩余:{remaining}]')

    for batch, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = net.forward(x[0], x[1], x[2])
        loss_batch = loss(output, y)
        data_loader.set_description('Loss: {loss:.3f}'.format(loss=loss_batch))
        loss_batch.backward()
        optimizer.step()

    data_loader.close()  # 清理进度条，防止异常


def test(data_loader, net):
    truth_count = 0
    total_count = 0
    confusion_matrix = np.zeros((4, 4), dtype=int).tolist()
    for batch, (x, y) in enumerate(data_loader):
        predict = torch.argmax(net.forward(x[0], x[1], x[2])).cpu().numpy()
        truth = y[0].cpu().numpy()
        truth_count += (predict == truth)
        total_count += 1
        confusion_matrix[predict][truth] += 1
    acc = int(truth_count) / total_count
    print("Accuracy: {acc:.2f}%".format(acc=acc*100))
    print("Confusion matrix:", confusion_matrix)

    TP = np.zeros(4, dtype=int).tolist()
    FP = np.zeros(4, dtype=int).tolist()
    TN = np.zeros(4, dtype=int).tolist()
    FN = np.zeros(4, dtype=int).tolist()
    for i in range(4):
        for j in range(4):
            if i == j:
                TP[i] += confusion_matrix[i][j]
                for n in range(4):
                    if n != i:
                        TN[n] += confusion_matrix[i][j]
            else:
                FP[i] += confusion_matrix[i][j]
                FN[j] += confusion_matrix[i][j]
                for n in range(4):
                    if n != i and n != j:
                        TN[n] += confusion_matrix[i][j]
    print("牙冠: Precision: {precision:.2f}%, Recall: {recall:.2f}%".format(precision=TP[0]/(TP[0]+FP[0])*100, recall=TP[0]/(TP[0]+FN[0])*100))
    print("牙根: Precision: {precision:.2f}%, Recall: {recall:.2f}%".format(precision=TP[1]/(TP[1]+FP[1])*100, recall=TP[1]/(TP[1]+FN[1])*100))
    print("牙髓: Precision: {precision:.2f}%, Recall: {recall:.2f}%".format(precision=TP[2]/(TP[2]+FP[2])*100, recall=TP[2]/(TP[2]+FN[2])*100))
    print("非牙齿: Precision: {precision:.2f}%, Recall: {recall:.2f}%".format(precision=TP[3]/(TP[3]+FP[3])*100, recall=TP[3]/(TP[3]+FN[3])*100))

    return acc


def main():
    train_data = PatchDataTrain(source)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = PatchDataTest(source)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    net = Union(int(source[0][-1]))
    net = net.cuda()
    loss = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    total_acc = []

    for epoch in range(epochs):
        print('\nEpoch', epoch+1)
        print('lr = {lr:.2e}'.format(lr=optimizer.param_groups[0]['lr']))
        train(train_loader, net, loss, optimizer)
        scheduler.step()
        acc = test(test_loader, net)
        total_acc.append(acc)
        torch.save(net, 'model/{eph}.pth'.format(eph=epoch+1))  # 保存pth

    # 绘图
    fig = plt.figure(figsize=(11, 9))
    plt.title('Accuracy of ' + str(source))
    plt.plot(range(1, epochs + 1), total_acc, color='red', marker='o', linestyle='--', linewidth=2.0)
    plt.text(epochs, total_acc[-1], total_acc[-1]*1000//1/10)

    fig.savefig('result/' + str(source) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.jpg', dpi=800)
    torch.save(net, 'result/' + str(source) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.pth')
    plt.show()


if __name__ == '__main__':
    main()
