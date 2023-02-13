import argparse
import numpy as np

import torch
import torchvision

from model import Net
from train import Trainer
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='mnist classification')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # model
    model = Net()

    # datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )

    # trainer
    trainer = Trainer(model=model)

    # model training
    trainer.train(train_loader=train_loader, test_loader=test_loader,epochs=args.epochs, lr=args.lr, save_dir="./save/")

    # model evaluation
    trainer.load_model('./save/mnist.pth')
    trainer.eval(test_loader=test_loader)

    # model inference  
    # image = Image.open(input("Enter Image Path: ")) # Load the input image
    image = Image.open("./images/six.png") # Load the input image
    image = image.resize((28, 28)) # Resize the image to 28x28
    image = image.convert("L") # Convert the image to grayscale
    image_tensor = transform(image)
    sample = image_tensor.unsqueeze(0) # inserting a new dimension of size 1 at position 0
    trainer.infer(sample=sample)

    return


if __name__ == "__main__":
    main()
