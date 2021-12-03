import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, n_classes=100):
        super(Model, self).__init__()
        # Convolutional layers
        # N x 3 x 64 x 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(4, 4)
        # N x 64 x 16 x 16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(4, 4)
        # N x 128 x 4 x 4
        # Linear layer
        # N x 2048
        self.fc1 = nn.Linear(2048, 1024)
        # N x 1024
        self.fc2 = nn.Linear(1024, n_classes)
        # N x n_classes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.ImageFolder('datasets/TinyImageNet/train',
                                         transform=train_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=512,
                                                   shuffle=True)
    model = Model().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in tqdm(range(100), desc='Epoch'):
        losses = []
        for i, (x, label) in enumerate(tqdm(train_dataloader)):
            y = model(x.cuda())
            loss = criterion(y, label.cuda())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                tqdm.write(f'{i + 1}/{len(train_dataloader)}: loss={sum(losses) / len(losses)}')
                losses = []


if __name__ == '__main__':
    main()
