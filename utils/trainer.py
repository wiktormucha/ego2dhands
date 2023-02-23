import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from config import *


class Trainer:
    """
    Class for training the model
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, config: dict, scheduler: torch.optim = None, grad_clip: int = None) -> None:
        """
        Initialisation

        Args:
            model (torch.nn.Module): Input modle used for training
            criterion (torch.nn.Module): Loss function
            optimizer (torch.optim): Optimiser
            config (dict): Config dictionary (needed max epochs and device)
            scheduler (torch.optim, optional): Learning rate scheduler. Defaults to None.
            grad_clip (int, optional): Gradient clipping. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.epochs = config["epochs"]
        self.device = config["device"]
        self.scheduler = scheduler
        self.early_stopping_epochs = config["early_stopping"]
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.best_val_loss = 100000
        self.grad_clip = grad_clip

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
        """
        Training loop

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        Returns:
            torch.nn.Module: Trained model
        """
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                )
            )

            # Save loss every epoch
            df = pd.DataFrame()
            df['train_los'] = self.loss["train"]
            df['val_los'] = self.loss["val"]
            df.to_csv(f'{EXPERIMENT_NAME}.csv')

            # reducing LR if no improvement in training
            if self.scheduler is not None:
                self.scheduler.step(self.loss["train"][-1])

            self.__save_best_model(val_loss=np.round(
                self.loss["val"][-1], 10), epoch=epoch)

            # early stopping if no progress
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(
                    np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]),
                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(), f'{EXPERIMENT_NAME}_final')
        return self.model

    def _epoch_train(self, dataloader: torch.utils.data.DataLoader):
        """
        Training step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.train()
        running_loss = []

        for i, data in enumerate(tqdm(dataloader, 0)):

            inputs = data["image"].to(self.device)
            labels = data["heatmaps"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _epoch_eval(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluation step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):

                inputs = data["image"].to(self.device)
                labels = data["heatmaps"].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())

            epoch_loss = np.mean(running_loss)
            self.loss["val"].append(epoch_loss)

    def __save_best_model(self, val_loss: float, epoch: int):
        """
        Saves best model

        Args:
            val_loss (float): Current validation loss
            epoch (int): Current epoch
        """

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print("Saving best model..")
            torch.save(self.model.state_dict(), f'{EXPERIMENT_NAME}_{epoch}')
