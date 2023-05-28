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
from agents.models import MLP_RewardPredictor, TransformerRewardPredictor, RewardDataset
from torch.utils.tensorboard import SummaryWriter

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

    def _setup_tensorboard(self):
        self.writer = SummaryWriter("runs/" + self.run_name)
        # layout = {
        #     "Loss Curve": {
        #         "loss": ["Multiline", ["loss"]]
        #     },
        #     "Metrics": {
        #         "Predicted vs Global Reward": ["Multiline", ["y", "y_hat"]],
        #         "Weight Entropy": ["Multiline", ["weight_entropy"]]
        #     }

        # }
        # self.writer.add_custom_scalars(layout)
    
    def plot(self, metrics:dict, epoch=None, step=None):
        if self.disable_comet: return

        if epoch is not None:
            # self.experiment.log_metrics(metrics, epoch=epoch)
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, epoch)
        
        elif step is not None:
            # self.experiment.log_metrics(metrics, step=step)
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
        
        else:
            raise NotImplementedError


    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, params):
        model.train(True)
        self.epoch_loss = 0.0
        self.weight_values = None
        # stream = tqdm(train_loader)
        epoch_loss = 0
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(40, 40))
        agent_reward_by_global_reward = {}
        agent_weight = {}
        for i in range(4):
            agent_reward_by_global_reward[i] = []
            agent_weight[i] = []

        for i in range(4):
            ax[i].set_xlabel("batch_agent_weight")
            ax[i].set_ylabel("batch_agent_reward / global_reward ")
        

        for i, (X, y, agent_rewards) in enumerate(train_loader, start=1):
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
            # metrics = {}
            # metrics["y"] = torch.mean(y.cpu().detach()).item()
            # metrics["y_hat"] = torch.mean(y_hat.cpu().detach()).item()
            # self.plot(metrics, step=(i + (epoch-1) * len(train_loader)))
            self.writer.add_scalars("Global_reward_prediction", {"y": torch.mean(y.cpu().detach()).item(),
                                                                 "y_hat": torch.mean(y_hat.cpu().detach()).item()}, (i + (epoch-1) * len(train_loader)))

            if isinstance(model, TransformerRewardPredictor):
                # other_metrics = {}
                weight_entropy = -torch.mean(torch.sum(attention_weights * torch.log(torch.clamp(attention_weights, 1e-10,1.0)), dim=-1))
                # other_metrics["weight_entropy"] = weight_entropy.item()
                batch_global_reward = torch.mean(y.cpu().detach())
                batch_agent_rewards = torch.mean(agent_rewards, dim=0)
                mean_attention_weights = torch.mean(attention_weights.cpu().detach().squeeze(1), dim=0)
                # other_metrics["batch_global_reward"] = batch_global_reward.item()
                assert(batch_agent_rewards.shape == mean_attention_weights.shape)
                for agent_index in range(batch_agent_rewards.shape[0]):
                    # other_metrics[f"batch_agent_reward_{agent_index}"] = batch_agent_rewards[agent_index].item()
                    # other_metrics[f"batch_agent_weight_{agent_index}"] = mean_attention_weights[agent_index].item()
                    # other_metrics[f"batch_agent_reward_{agent_index}/batch_global_reward"] = batch_agent_rewards[agent_index].item() / (batch_global_reward.item() + 1e-7)
                    agent_weight[agent_index].append(mean_attention_weights[agent_index].item())
                    agent_reward_by_global_reward[agent_index].append(batch_agent_rewards[agent_index].item() / (batch_global_reward.item() + 1e-7))

                # self.plot(other_metrics, step=(i + (epoch-1) * len(train_loader)))
                self.writer.add_scalar("weight_entropy", weight_entropy.item(), (i + (epoch-1) * len(train_loader)))

            epoch_loss += loss.item()
            # stream.set_description(
            #     "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=self.metric_monitor)
            # )

        # plotting metrics
        self.plot({"loss": epoch_loss}, epoch=epoch)
        for i in range(4):
            ax[i].scatter(agent_weight[i], agent_reward_by_global_reward[i])
        # self.writer.add_figure("(Agent_reward / global reward) vs weight", fig, epoch)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "fig_" + str(epoch).zfill(3) + ".png"))
        self.writer.flush()
        plt.clf()
        plt.close()

    def train(self, save_path):
        if not self.disable_comet:
            # self._setup_comet()
            self._setup_tensorboard()
        
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
    ap.add_argument("-x", "--disable_comet", required=False, default=False, type=bool)
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
    trainer = Trainer(model, train_dataset, params, args["run_name"], "plots", args["disable_comet"])
    trainer.train(args["save_path"])
