import torch
from torch import optim,nn
import visdom
import torchvision
from torch.utils.data import DataLoader

from butterfly import Butterfly
from resnet18 import ResNet18
from resnet34 import ResNet34

batchsz = 32
lr = 1e-3
epochs = 80

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = Butterfly('butterfly', 224, mode='train')
val_db = Butterfly('butterfly', 224, mode='val')
test_db = Butterfly('butterfly', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)


viz = visdom.Visdom()

def evalute(model,loader):
    correct = 0                               #测试正确数初始为0
    total = len(loader.dataset)               #计算loader.datas中共有多少数据

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim = 1)
        correct += torch.eq(pred, y).sum().float().item() #用预测值与真实值进行比较，每对一次correct+1
    return correct / total

def main():

    model = ResNet34(20).to(device)
    optimizer = optim.SGD(model.parameters(),lr=lr)
    # optimizer = optim.Adam(model.parameters(),lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [0], win='loss', opts=dict(title='loss'))
    viz.line([0], [0], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            #x:[b, 3, 224, 224],y:[b]
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:
            val_acc= evalute(model, val_loader)
            if val_acc> best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best_model_weight.pth')
                torch.save(model, 'best_model.pth')

                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best_acc:', best_acc, 'best_epoch:', best_epoch)

    # model.load_state_dict(torch.load('best_model_weight.pth'))
    # print('loaded form checkpoint!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


if __name__ == '__main__':
    main()