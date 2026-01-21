import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader
from dataset import BindingDataset
from torch.utils.data import random_split
from model_pl import BindingAffinityModelPL
import pandas as pd

def main():
    lr = 0.0005
    # Load dataset
    dataframe = pd.read_csv('pdbbind_refined_dataset.csv')
    dataframe.dropna(inplace=True)
    print("Dataset loaded with {} samples".format(len(dataframe)))
    dataset = BindingDataset(dataframe)
    print("Dataset transformed with {} samples".format(len(dataset)))

    if len(dataset) == 0:
        print("Dataset is empty")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_features = train_dataset[0].x.shape[1]
    print("Number of node features:", num_features)

    model = BindingAffinityModelPL(num_node_features=84, hidden_channels_gnn=128, lr=lr)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=3,
        mode='min'
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto", # Use GPU if available
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()