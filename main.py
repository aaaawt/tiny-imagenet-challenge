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
        self.conv1 = nn.Conv2d(3, 16, (5, 5), (2, 2), 2)
        # N x 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, (5, 5), (2, 2), 2)
        # N x 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, (5, 5), (2, 2), 2)
        # N x 64 x 8 x 8
        self.conv4 = nn.Conv2d(64, 128, (5, 5), (2, 2), 2)
        # N x 128 x 4 x 4
        self.conv5 = nn.Conv2d(128, 256, (5, 5), (2, 2), 2)
        # N x 256 x 2 x 2
        # Linear layer
        # N x 1024
        self.fc1 = nn.Linear(1024, 512)
        # N x 512
        self.fc2 = nn.Linear(512, n_classes)
        # N x n_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 1024)
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
                                                   batch_size=32,
                                                   shuffle=True)
    model = Model()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.)
    for x, label in tqdm(train_dataloader):
        y = model(x)
        l = loss(y, label)
        tqdm.write(f'{l.item()}')
        optimizer.zero_grad()
        l.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
