import torch
from torchvision import datasets, transforms


def main():
    train_dataset = datasets.ImageFolder('datasets/TinyImageNet/train',
                                         transform=transforms.Compose([transforms.ToTensor()]))
    for i in range(len(train_dataset)):
        if train_dataset[i][0].size() != torch.Size([3, 64, 64]):
            print(i)
            break
    else:
        print('hhhhh')


if __name__ == '__main__':
    main()
