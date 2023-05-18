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
import matplotlib.pyplot as plt


class RewardDataset(Dataset):
    def __init__(self, json_files:list, json_base_dir:str, episode_length=25) -> None:
        self.json_files = json_files
        self.json_base_dir = json_base_dir
        self.episode_length = episode_length
        assert(os.path.exists(json_base_dir)), f"{json_base_dir} does not exist!"
    
    def __len__(self):
        return len(self.json_files) * self.episode_length
    
    def __getitem__(self, index):
        json_file_index = index // self.episode_length
        data_index = index % self.episode_length
        json_file = self.json_files[json_file_index]
        assert(os.path.exists(os.path.join(self.json_base_dir, json_file))), f"{os.path.join(self.json_base_dir, json_file)} does not exist!"
        with open(os.path.join(self.json_base_dir, json_file), "r+") as f:
            d = json.load(f)
        return torch.tensor(d["state_actions"][data_index]), torch.tensor(d["global_rewards"][data_index]), torch.tensor(d["agent_rewards"][data_index])

class TransformerRewardPredictor(nn.Module):
    def __init__(self, e_dim, d_k, mlp_hidden_layers) -> None:
        super(TransformerRewardPredictor, self).__init__()
        self.key_net = MLP(e_dim, [d_k])
        self.query_net = MLP(e_dim, [d_k])
        self.value_net = MLP(e_dim, [d_k])
        self.mlp = MLP(d_k, mlp_hidden_layers)
        self.attention_weights = None
        self.d_k = d_k

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
        return y_hat, self.attention_weights
    
class MLP_RewardPredictor(nn.Module):
    def __init__(self, input_dim:int, hidden_layers:list) -> None:
        super(MLP_RewardPredictor, self).__init__()
        self.mlp = MLP(input_dim, hidden_layers)
    
    def forward(self, x):
        return self.mlp(x)

class Trainer:
    def __init__(self, model, train_dataset:RewardDataset, params, run_name, plot_dir, disable_comet=False) -> None:
        self.model = model.to(params["device"])
        self.train_dataset = train_dataset
        self.params = params
        self.run_name = run_name
        self.metric_monitor = MetricMonitor()
        self.plot_dir = plot_dir
        self.disable_comet = disable_comet
        
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
    
    def plot(self, metrics:dict, epoch=None, step=None):
        if self.disable_comet: return

        if epoch is not None:
            self.experiment.log_metrics(metrics, epoch=epoch)
        
        elif step is not None:
            self.experiment.log_metrics(metrics, step=step)
        
        else:
            raise NotImplementedError


    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, params):
        model.train(True)
        self.epoch_loss = 0.0
        self.weight_values = None
        stream = tqdm(train_loader)
        epoch_loss = 0
        for i, (X, y, agent_rewards) in enumerate(stream, start=1):
            X = X.float().to(params["device"])
            y = y.float().to(params["device"])

            # forward pass
            if isinstance(model, MLP_RewardPredictor):
                y_hat = model(X.reshape(X.shape[0], -1).to(params["device"])).squeeze(-1)
            elif isinstance(model, TransformerRewardPredictor):
                y_hat, attention_weights = model(X)
                y_hat = y_hat.squeeze(-1)

            # computing loss
            loss = criterion(y_hat, y)
            
            # back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging and plotting metrics
            self.metric_monitor.update("loss", loss.item(), self.plot_dir)
            self.metric_monitor.update("y_hat", torch.mean(y_hat.cpu().detach()).item(), self.plot_dir)
            self.metric_monitor.update("y", torch.mean(y.cpu().detach()).item(), self.plot_dir)
            metrics = {}
            metrics["y"] = torch.mean(y.cpu().detach()).item()
            metrics["y_hat"] = torch.mean(y_hat.cpu().detach()).item()
            self.plot(metrics, step=(i + (epoch-1) * len(train_loader)))

            if isinstance(model, TransformerRewardPredictor):
                other_metrics = {}
                weight_entropy = -torch.mean(torch.sum(attention_weights * torch.log(torch.clamp(attention_weights, 1e-10,1.0)), dim=-1))
                other_metrics["weight_entropy"] = weight_entropy.item()
                batch_global_reward = torch.mean(y.cpu().detach())
                batch_agent_rewards = torch.mean(agent_rewards, dim=0)
                mean_attention_weights = torch.mean(attention_weights.cpu().detach().squeeze(1), dim=0)
                other_metrics["batch_global_reward"] = batch_global_reward.item()
                assert(batch_agent_rewards.shape == mean_attention_weights.shape)
                for agent_index in range(batch_agent_rewards.shape[0]):
                    other_metrics[f"batch_agent_reward_{agent_index}"] = batch_agent_rewards[agent_index].item()
                    other_metrics[f"batch_agent_weight_{agent_index}"] = mean_attention_weights[agent_index].item()
                    other_metrics[f"batch_agent_reward_{agent_index}/batch_global_reward"] = batch_agent_rewards[agent_index].item() / (batch_global_reward.item() + 1e-7)

                self.plot(other_metrics, step=(i + (epoch-1) * len(train_loader)))

            epoch_loss += loss.item()
            stream.set_description(
                "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=self.metric_monitor)
            )

        # plotting metrics
        self.plot({"loss": epoch_loss}, epoch=epoch)

    def train(self, save_path):
        if not self.disable_comet:
            self._setup_comet()
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_loader = DataLoader(
            self.train_dataset,
            self.params["batch_size"],
            shuffle=True,
        )

        criterion = nn.HuberLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["lr"])
        
        for epoch in range(self.params["num_epochs"]):
            self.train_one_epoch(train_loader, self.model, criterion, optimizer, epoch+1, self.params)
            if (epoch+1) % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path, "epoch_" + str(epoch).zfill(5) + ".pth"))

        
if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-n", "--num_epochs", required=True, type=int)
    ap.add_argument("-d", "--data_dir", required=True, type=str)
    ap.add_argument("-r", "--run_name", required=True, type=str)
    ap.add_argument("-s", "--save_path", required=True, type=str)
    ap.add_argument("-b", "--batch_size", required=False, default=32, type=int)
    ap.add_argument("-l", "--lr", required=False, default=1e-3)
    ap.add_argument("-t", "--network_type", required=False, default="mlp")
    ap.add_argument("-x", "--disable_comet", required=False, default=False)
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
        model = MLP_RewardPredictor(60, [128, 64, 4, 1])
    elif args["network_type"] == "transformer":
        model = TransformerRewardPredictor(15, 15, [128, 64, 4, 1])
    trainer = Trainer(model, train_dataset, params, args["run_name"], "plots")
    trainer.train(args["save_path"])
