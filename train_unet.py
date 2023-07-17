import argparse
from dataset.dataset_VOC import get_VOC_dataset
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from model.UNet import UNet
# from model.unet_x import UNet
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import train_loop


def main(config):
    # Outdir
    outdir = Path(config["outdir"]) / 'train'
    os.makedirs(outdir, exist_ok=True)
    
    model_outdir = outdir / "ckpt"
    os.makedirs(model_outdir, exist_ok=True)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Loss
    loss = nn.CrossEntropyLoss()
    
    # Dataloader
    train_dataset, eval_dataset = get_VOC_dataset(data_dir="./data", height=256, width=256)
    
    
    torch.manual_seed(config["random_seed"])
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True)
    eval_loader = DataLoader(
        eval_dataset, batch_size=config["batch_size"],
        shuffle=False)
    print("train size: ", len(train_loader))
    print("eval size: ", len(eval_loader))
    
    data_loader = {"Training": train_loader, "Evaluation": eval_loader}
    
    # Model
    print("in_ch: ", train_dataset[0][0].shape[0])
    print("out_ch: ", train_dataset[0][1].shape[0])
    model = UNet(in_ch=train_dataset[0][0].shape[0], out_ch=train_dataset[0][1].shape[0])

    if "pretrained" in config:
        if os.path.exists(config["pretrained"]):
            model.load_state_dict(torch.load(config["pretrained"]))
        else:
            raise FileNotFoundError

    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    losses = {"Training": [], "Evaluation": []}
    epochs = []
    
    # Train
    for epoch in range(1, config["num_epoch"]+1):
        epochs.append(epoch)
        
        epoch_losses = train_loop(
            device,
            epoch,
            data_loader,
            model,
            optimizer,
            loss
        )
        
        losses["Training"].append(epoch_losses["Training"])
        losses["Evaluation"].append(epoch_losses["Evaluation"])
        
        # plot
        fig, ax = plt.subplots()
        ax.plot(epochs, losses["Training"], color=(0,0,1), label="Training")
        ax.plot(epochs, losses["Evaluation"], color=(1,0,0), label="Evaluation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(bottom=0.0)
        plt.legend(loc="best")
        fig.savefig(outdir / "loss.png")
        plt.cla()
        plt.close()

        # save model
        if epoch % config["save_epoch_freq"] == 0:
            model_path = model_outdir / "model_{}.pth".format(epoch)
            torch.save(model.state_dict(), model_path)    


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, mode="r") as f:
        config = json.load(f)

    main(config)
