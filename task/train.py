import torch
from torch import nn, optim
import os
from tqdm.auto import tqdm
from model import RNN, LSTM, GRU
from data_utils import load_data
from evaluate import evaluate

class Train_Task:
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.best_metric = config["best_metric"]
        self.save_path = config["save_path"]
        self.patience = config["patience"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config["model"]["model_name"]
        if self.model_name == "RNN":
            self.model = RNN.RNN(config).to(self.device)
        elif self.model_name == "LSTM":
            self.model = LSTM.LSTM(config).to(self.device)
        elif self.model_name == "GRU":
            self.model = GRU.GRU(config).to(self.device)
        self.dataloader = load_data.Load_Data(config)
        self.loss = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), lr= self.learning_rate)

    def train(self):
        train, dev = self.dataloader.load_train_dev()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        last_model = f"{self.model_name}_last_model.pth"
        best_model = f"{self.model_name}_best_model.pth"

        if os.path.exists(os.path.join(self.save_path, last_model)):
            checkpoint = torch.load(os.path.join(self.save_path, last_model))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optim_state_dict"])
            print("Load the last model")
            initial_epoch = checkpoint["epoch"] + 1
            print(f"Continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("First time training!!!")

        if os.path.exists(os.path.join(self.save_path, best_model)):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        threshold = 0

        self.model.train()
        for epoch in range(initial_epoch, initial_epoch + self.num_epochs):
            train_loss = 0.

            for _, item in enumerate(tqdm(train)):
                self.optim.zero_grad()
                X, y = item["sentence"], item["label"].to(self.device)

                # Forward
                y_logits = self.model(X)
                loss = self.loss(y_logits, y)
                train_loss += loss

                # Backward
                loss.backward()
                self.optim.step()

            valid_preds = []
            valid_trues = []
            with torch.inference_mode():
                for _, item in enumerate(tqdm(dev)):
                    X, y = item["sentence"], item["label"]
                    valid_trues += y.tolist()
                    y_logits = self.model(X)
                    y_preds = torch.softmax(y_logits, dim = 1).argmax(dim= 1)
                    valid_preds += y_preds.tolist()

                valid_acc, valid_precision, valid_recall, valid_f1 = evaluate.compute_score(valid_trues, valid_preds)
                train_loss /= len(train)

                print(f"Epoch {epoch + 1}/{initial_epoch + self.num_epochs}")
                print(f"Train loss: {train_loss:.5f}")
                print(f"valid acc: {valid_acc:.4f} | valid f1: {valid_f1:.4f} | valid precision: {valid_precision:.4f} | valid recall: {valid_recall:.4f}")

                if self.best_metric == 'accuracy':
                    score= valid_acc
                if self.best_metric == 'f1':
                    score= valid_f1
                if self.best_metric == 'precision':
                    score= valid_precision
                if self.best_metric == 'recall':
                    score= valid_recall

                # save last model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'score': score
                }, os.path.join(self.save_path, last_model))

                # save the best model
                if epoch > 0 and score < best_score:
                    threshold += 1
                else:
                    threshold = 0

                if score >= best_score:
                    best_score = score
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'score':score
                    }, os.path.join(self.save_path, best_model))
                    print(f"Saved the best model with {self.best_metric} of {score:.4f}")

            # early stopping
            if threshold >= self.patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break