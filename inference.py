import random

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from dataset import BindingDataset
from model import BindingAffinityModel
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import random_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "best_model_gat.pth"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)

def predict_and_plot():
    gen = set_seed(42)
    print("Loading data...")

    dataframe = pd.read_csv('pdbbind_refined_dataset.csv')
    dataframe.dropna(inplace=True)
    dataset = BindingDataset(dataframe)
    if len(dataset) == 0:
        print("Dataset is empty")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size], generator=gen)

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_features = test_dataset[0].x.shape[1]

    print("Loading model...")
    model = BindingAffinityModel(num_node_features=num_features, hidden_channels_gnn=128).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    y_true = []
    y_pred = []
    print("Predicting...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch, batch.protein_seq)

            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(out.squeeze().cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    pearson_corr, _ = pearsonr(y_true, y_pred)  # Pearson correlation

    print("Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")

    plt.figure(figsize=(9, 9))
    plt.scatter(y_true, y_pred, alpha=0.4, s=15, c='blue', label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=2,
             label='Ideal')

    plt.xlabel('Experimental Affinity (pK)')
    plt.ylabel('Predicted Affinity (pK)')
    plt.title(f'Binding affinity Results\nRMSE={rmse:.3f}, Pearson R={pearson_corr:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_file = 'final_results_gat.png'
    plt.savefig(plot_file)
    print(f"График сохранен в {plot_file}")
    plt.show()

if __name__ == "__main__":
    predict_and_plot()