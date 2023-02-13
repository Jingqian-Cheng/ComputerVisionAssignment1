import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from model import Net

from utils import AverageMeter


class Trainer:
    """ Trainer for MNIST classification """

    def __init__(self, model: nn.Module):
        self._model = model

    def train(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        """ Model training, TODO: consider adding model evaluation into the training loop """

        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        loss_track = AverageMeter()
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            loss_track.reset()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self._model(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=data.size(0))

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg))
            self.eval(test_loader) # evaluate the model after each loop

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))

        return

    def eval(self, test_loader: DataLoader) -> float:
        """ Model evaluation, return the model accuracy over test set """
        self._model.eval()
        correct = 0.0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self._model(data)
                _, predicted = torch.max(output.data, 1) # predicted class indexes of this batch
                total += target.size(0) # batch size
                correct += (predicted == target).sum().item() # extract the single number from tensor
        percentage = (correct / total) * 100
        percentage = round(percentage, 2)
        print("Accuracy: "+str(percentage)+"%")
        return correct / total # accuracy rate

    def infer(self, sample: Tensor) -> int:
        """ Model inference: input an image, return its class index """
        self.load_model('./save/mnist.pth')
        self._model.eval()
        with torch.no_grad():
            output = self._model(sample)
            _, predicted = torch.max(output.data, 1)
        print("Infer result: "+str(predicted.item()))
        return predicted.item()

    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        model = Net()
        model.load_state_dict(torch.load(path))
        self._model = model
        return

