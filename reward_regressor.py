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
        self.metric_monitor = None
        self.plot_dir = plot_dir
        self.total_steps = 0
        
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
    
    def plot(self, epoch=None, step=None, **kwargs):
        if epoch is not None:
            self.experiment.log_metrics(kwargs, epoch=epoch)
        
        elif step is not None:
            self.experiment.log_metrics(kwargs, step=step)
        
        else:
            raise NotImplementedError


    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, params):
        model.train(True)
        self.epoch_loss = 0.0
        self.weight_values = None
        self.metric_monitor = MetricMonitor()
        stream = tqdm(train_loader)
        
        for i, (X, y, _) in enumerate(stream, start=1):
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

            # logging and plotting metrics
            self.metric_monitor.update("loss", loss.item(), self.plot_dir)
            self.plot(y_hat=torch.mean(y_hat.cpu().detach()).item(), step=(i + (epoch-1) * len(train_loader)))
            self.plot(y=torch.mean(y.cpu().detach()).item(), step=(i + (epoch-1) * len(train_loader)))
            stream.set_description(
                "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=self.metric_monitor)
            )

        # plotting metrics
        self.plot(avg_loss=self.metric_monitor.metrics["loss"]["avg"], epoch=epoch)

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
    ap = ArgumentParser()
    ap.add_argument("-n", "--num_epochs", required=True, type=int)
    ap.add_argument("-d", "--data_dir", required=True, type=str)
    ap.add_argument("-r", "--run_name", required=True, default=1e-3)
    ap.add_argument("-b", "--batch_size", required=False, default=32, type=int)
    ap.add_argument("-l", "--lr", required=False, default=1e-3)
    ap.add_argument("-t", "--network_type", required=False, default="mlp")
    args = vars(ap.parse_args())
    data_dir = args["data_dir"]
    train_json_list = [file.split("/")[-1] for file in glob(f"{data_dir}/*.json")]
    train_dataset = RewardDataset(train_json_list, data_dir)

    params = {
        "batch_size": args["batch_size"],
        "num_epochs": args["num_epochs"],
        "lr": args["lr"],
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    if args["network_type"] == "mlp":
        model = MLP_RewardPredictor(60, [64, 64, 1])
    else:
        raise NotImplementedError
    trainer = Trainer(model, train_dataset, params, args["run_name"], "plots")
    trainer.train()
