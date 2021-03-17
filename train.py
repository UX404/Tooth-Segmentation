from Data import PatchDataTrain, PatchDataTest
from Net import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import numpy as np

source = '160_sobel_4'
lr = 5e-5
batch_size = 100
epochs = 30

# ['其他', '牙冠', '牙齿', '牙龈']
epoch_time = -1
log = ''


def train(data_loader, net, loss, optimizer):
    timer = time.time()
    data_loader = tqdm(data_loader,
                       bar_format='{l_bar}{bar}| Batch: {n_fmt}/{total_fmt} [已运行:{elapsed}，剩余:{remaining}]')

    for batch, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = net.forward(x)
        loss_batch = loss(output, y)
        data_loader.set_description('Loss: {loss:.3f}'.format(loss=loss_batch))
        loss_batch.backward()
        optimizer.step()

    data_loader.close()  # 清理进度条，防止异常
    global epoch_time
    epoch_time = time.time() - timer


def test(data_loader, net):
    truth_count = 0
    total_count = 0
    confusion_matrix = np.zeros((4, 4), dtype=int).tolist()
    for batch, (x, y) in enumerate(data_loader):
        predict = torch.argmax(net.forward(x)).cpu().numpy()
        truth = y[0].cpu().numpy()
        truth_count += (predict == truth)
        total_count += 1
        confusion_matrix[predict][truth] += 1
    acc = int(truth_count) / total_count
    print("Accuracy: {acc:.2f}%".format(acc=acc*100))
    print("Confusion matrix:", confusion_matrix)

    # 计算灵敏度和特异度
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
    print("其他: Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%".format(sensitivity=TP[0]/(TP[0]+FN[0]+0.0001)*100, specificity=TN[0]/(FP[0]+TN[0]+0.0001)*100))
    print("牙冠: Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%".format(sensitivity=TP[1]/(TP[1]+FN[1]+0.0001)*100, specificity=TN[1]/(FP[1]+TN[1]+0.0001)*100))
    print("牙齿: Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%".format(sensitivity=TP[2]/(TP[2]+FN[2]+0.0001)*100, specificity=TN[2]/(FP[2]+TN[2]+0.0001)*100))
    print("牙龈: Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%".format(sensitivity=TP[3]/(TP[3]+FN[3]+0.0001)*100, specificity=TN[3]/(FP[3]+TN[3]+0.0001)*100))

    # log
    global log
    log += ('Epoch_time: {time:.2f}s\n'.format(time=epoch_time))
    log += ("Accuracy: {acc:.2f}%\n".format(acc=acc*100))
    log += ("Confusion matrix: " + str(confusion_matrix) + '\n')
    log += ("其他: Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%\n".format(sensitivity=TP[0]/(TP[0]+FN[0]+0.0001)*100, specificity=TN[0]/(FP[0]+TN[0]+0.0001)*100))
    log += ("牙冠: Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%\n".format(sensitivity=TP[1]/(TP[1]+FN[1]+0.0001)*100, specificity=TN[1]/(FP[1]+TN[1]+0.0001)*100))
    log += ("牙齿: Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%\n".format(sensitivity=TP[2]/(TP[2]+FN[2]+0.0001)*100, specificity=TN[2]/(FP[2]+TN[2]+0.0001)*100))
    log += ("牙龈: Sensitivity: {sensitivity:.2f}%, Specificity: {specificity:.2f}%\n\n".format(sensitivity=TP[3]/(TP[3]+FN[3]+0.0001)*100, specificity=TN[3]/(FP[3]+TN[3]+0.0001)*100))

    return acc


def main():
    train_data = PatchDataTrain(source)
    # print(train_data.classification)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = PatchDataTest(source)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    net = VGG(int(source[-1]))
    net = net.cuda()
    loss = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    total_acc = []

    for epoch in range(epochs):
        # log
        global log
        log += ('Epoch {epoch:d}\n'.format(epoch=epoch+1))
        log += ('lr = {lr:.2e}\n'.format(lr=optimizer.param_groups[0]['lr']))
        
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
    with open('result/' + str(source) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.txt', mode='w') as log_txt:
        log_txt.write(log)

    plt.show()


if __name__ == '__main__':
    main()
