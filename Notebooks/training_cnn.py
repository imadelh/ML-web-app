import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np

# Define transformers

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize((mean,), (std,)),
    ]
)

transform_valid = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((mean,), (std,)),
    ]
)


# Fixed Params
root = ""

batch_size = 128
test_batch_size = 128
lr = 1e-3
EPOCHS = 200


# Prepare loaders

train = datasets.EMNIST(
    root, split="balanced", train=True, download=True, transform=transform_train
)
test = datasets.EMNIST(
    root, split="balanced", train=False, download=True, transform=transform_valid
)


train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
)


test_loader = torch.utils.data.DataLoader(
    test, batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
)

# Define model


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(28, 64, (5, 5), padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 2, padding=2)

        self.fc1 = nn.Linear(2048, 1024)

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)

        self.bn = nn.BatchNorm1d(1)

        self.fc3 = nn.Linear(512, 128)

        self.fc4 = nn.Linear(128, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv1_bn(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        x = x.view(-1, 1, 512)
        x = self.bn(x)

        x = x.view(-1, 512)
        x = self.fc3(x)
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)
        #
        return x


net = Net()
net.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(count_parameters(net))


logging.basicConfig(filename="cnn2.log", level=logging.DEBUG)
logging.info(net)
logging.info("Number of parameters: {}".format(count_parameters(net)))


# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.view(-1, 28, 28, 1)
        x = torch.transpose(x, 1, 2)

        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 47)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


data_gen = inf_generator(train_loader)
batches_per_epoch = len(train_loader)


optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss().to(device)

best_acc = 0
net.to(device)

for itr in range(EPOCHS * batches_per_epoch):

    optimizer.zero_grad()
    x, y = data_gen.__next__()
    x = x.view(-1, 28, 28, 1)
    x = torch.transpose(x, 1, 2)

    x = x.to(device)
    y = y.to(device)
    logits = net(x)
    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

    if itr % batches_per_epoch == 0:
        with torch.no_grad():
            train_acc = accuracy(net, train_loader)
            val_acc = accuracy(net, test_loader)
            if val_acc > best_acc:
                torch.save({"state_dict": net.state_dict()}, "cnn2.pth")
                best_acc = val_acc
            logging.info(
                "Epoch {:04d}"
                "Train Acc {:.4f} | Test Acc {:.4f}".format(
                    itr // batches_per_epoch, train_acc, val_acc
                )
            )

            print(
                "Epoch {:04d}"
                "Train Acc {:.4f} | Test Acc {:.4f}".format(
                    itr // batches_per_epoch, train_acc, val_acc
                )
            )
