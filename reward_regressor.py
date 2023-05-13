import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from agents.models import MLP
from agents.utils import MetricMonitor
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from math import sqrt
from comet_ml import Experiment
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser


class RewardDataset(Dataset):
    def __init__(self, json_files:list, json_base_dir:str) -> None:
        self.json_files = json_files
        self.json_base_dir = json_base_dir

        assert(os.path.exists(json_base_dir)), f"{json_base_dir} does not exist!"
    
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, index):
        json_file = self.json_files[index]
        assert(os.path.exists(os.path.join(self.json_base_dir, json_file))), f"{os.path.join(self.json_base_dir, json_file)} does not exist!"
        with open(os.path.join(self.json_base_dir, json_file), "r+") as f:
            d = json.load(f)

        return torch.tensor(d["state_actions"]), torch.tensor(d["global_rewards"]), torch.tensor(d["agent_rewards"])

class TransformerRewardPredictor(nn.Module):
    def __init__(self, e_dim, d_k, mlp_hidden_layers) -> None:
        super(TransformerRewardPredictor, self).__init__()
        self.key_net = MLP(e_dim, [d_k])
        self.query_net = MLP(e_dim, [d_k])
        self.value_net = MLP(e_dim, [d_k])
        self.mlp = MLP(d_k, mlp_hidden_layers)
        self.attention_weights = None

    def forward(self, state_actions):
        # obtaining keys queries and values
        self.query = self.query_net(torch.sum(state_actions, dim=1, keepdim=True))
        self.key = self.key_net(state_actions)
        self.value = self.value_net(state_actions)

        # self attention layer
        self.attention_weights = F.softmax((self.query @ self.key.permute(0, 2, 1) / sqrt(self.d_k)), dim=-1)
        self.attention_values =  (self.attention_weights @ self.value).squeeze(1)

        # MLP for predicting reward
        y_hat = self.mlp(self.attention_values)
        return y_hat
    
class MLP_RewardPredictor(nn.Module):
    def __init__(self, input_dim:int, hidden_layers:list) -> None:
        super(MLP_RewardPredictor, self).__init__()
        self.mlp = MLP(input_dim, hidden_layers)
    
    def forward(self, x):
        return self.mlp(x)

class Trainer:
    def __init__(self, model, train_dataset:RewardDataset, params, run_name, plot_dir) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.params = params
        self.run_name = run_name
        self.metric_monitor = MetricMonitor()
        self.plot_dir = plot_dir
        
        if not os.path.exists(self.plot_dir):
            os.makedirs(plot_dir)

    def _setup_comet(self):
        self.experiment = Experiment(
            api_key="8U8V63x4zSaEk4vDrtwppe8Vg",
            project_name="credit-assignment",
            parse_args=False
        )
        self.experiment.set_name(self.run_name)
        # logging hparams to comet_ml
        self.experiment.log_parameters(self.params)

    def collate_fn(self, batch):
        state_actions, global_rewards, agent_rewards = list(zip(*batch))
        return torch.cat(state_actions, dim=0), torch.cat(global_rewards, dim=0), torch.cat(agent_rewards, dim=0)
    
    def plot(self, epoch, fig=None):
        if fig is not None:
            self.experiment.log_figure(figure=fig)

        for metric in self.metric_monitor.metrics.keys():
            self.experiment.log_metric(name=metric, value=self.metric_monitor.metrics[metric]["avg"], epoch=epoch)

    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, params):
        model.train(True)
        self.epoch_loss = 0.0
        self.weight_values = None

        fig, ax = plt.subplots(1, 1)
        y_hats = []
        ys = []
        
        for i, (X, y, _) in enumerate(train_loader, start=1):
            X = X.float().to(params["device"])
            y = y.float().to(params["device"])

            # forward pass
            y_hat = model(X.reshape(X.shape[0], -1)).squeeze(-1)

            # computing loss
            loss = criterion(y_hat, y)
            
            # back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging metrics
            self.metric_monitor.update("loss", loss.item(), self.plot_dir)
            y_hats.extend(y_hat.cpu().detach().numpy().tolist())
            ys.extend(y.cpu().detach().numpy().tolist())

        # plotting metrics
        ax.plot(range(len(ys)), ys, 'g', label="True Reward")
        ax.plot(range(len(y_hats)), y_hats, 'r', label="Predicted Reward")
        ax.legend()
        ax.set_ylabel("Reward")
        self.plot(epoch, fig)

        plt.clf()
        

    def train(self):
        self._setup_comet()

        train_loader = DataLoader(
            self.train_dataset,
            self.params["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn
        )

        criterion = nn.HuberLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["lr"])
        
        for epoch in tqdm(range(self.params["num_epochs"])):
            self.train_one_epoch(train_loader, self.model, criterion, optimizer, epoch+1, self.params)
        

if __name__ == "__main__":
    train_json_list = [file.split("/")[-1] for file in glob("jsons/*.json")]
    train_dataset = RewardDataset(train_json_list, "jsons")

    params = {
        "batch_size": 4,
        "num_epochs": 5,
        "lr": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    model = MLP_RewardPredictor(60, [64, 64, 1])
    trainer = Trainer(model, train_dataset, params, "temp", "plots")
    trainer.train()
