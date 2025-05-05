import torch
from segger.data.parquet.sample import STSampleParquet
from segger.training.segger_data_module import SeggerDataModule
from segger.training.train import LitSegger
from segger.models.segger_model import Segger
from segger.prediction.predict import load_model
from torch_geometric.nn import to_hetero
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pytorch_lightning import Trainer



# Function to visualize attention weights as a heatmap
def visualize_attention_weights(attention_weights, edge_index, layer_idx, head_idx, edge_type, num_nodes):
    attention_weights = attention_weights.cpu().detach().numpy()
    edge_index = edge_index.cpu().detach().numpy()
    # Create adjacency matrix for visualization
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i, (src, dst) in enumerate(edge_index.T):
        adj_matrix[src, dst] = attention_weights[i]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_matrix, cmap='viridis', annot=False, cbar=True)
    plt.title(f'Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}, Edge Type: {edge_type}')
    plt.xlabel('Target Node (Transcript)')
    plt.ylabel('Source Node (Transcript)')
    plt.savefig(Path('figures') / f'attention_layer_{layer_idx + 1}_head_{head_idx + 1}_{edge_type.replace(" ", "_")}.png')
    plt.close()

# Main function to load model and visualize attention weights
def main():
    # Paths to data and models
    model_version = 1
    model_path = Path('models') / "lightning_logs" / f"version_{model_version}"
    ls = load_model(model_path / "checkpoints")
    
    ls.eval()
    
    # # Initialize trainer for prediction
    # trainer = Trainer(accelerator='cuda', devices=1, precision='16-mixed')
    
    xenium_data_dir = Path('data_xenium')
    segger_data_dir = Path('data_segger')

    sample = STSampleParquet(
        base_dir=xenium_data_dir,
        n_workers=4,
        sample_type='xenium', # this could be 'xenium_v2' in case one uses the cell boundaries from the segmentation kit.
        # weights=gene_celltype_abundance_embedding, # uncomment if gene-celltype embeddings are available
    )


    # Base directory to store Pytorch Lightning models
    models_dir = Path('models')

    # Initialize the Lightning data module
    dm = SeggerDataModule(
        data_dir=segger_data_dir,
        batch_size=2,
        num_workers=2,
    )

    dm.setup()
    
    # Get a sample batch from the data module
    batch = dm.train[0]
    
    # # Move batch to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = batch.to(device)
    ls = ls.to(device)
    
    # Run forward pass to get attention weights for tx-neighbors-tx edges
    with torch.no_grad():
        # Access the heterogeneous model
        hetero_model = ls.model
        # Get node features and edge indices for tx-neighbors-tx
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_type = ('tx', 'neighbors', 'tx')
        x_tx = x_dict['tx']
        edge_index = edge_index_dict[edge_type]
        
        # Run forward pass through the tx-neighbors-tx module
        x_out, attention_weights = hetero_model(x_dict, edge_index_dict)
    # print("Attention weights structure:", attention_weights)
    # Visualize attention weights for each layer and head
    num_nodes = batch.x_dict['tx'].shape[0]  # Number of transcript nodes
    
    # Visualize attention weights for each layer and head
    for layer_idx, (edge_index, alpha) in enumerate(attention_weights):
        num_nodes = batch.x_dict['tx'].shape[0]  # Number of transcript nodes
        print(f"Layer {layer_idx + 1}: edge_index type = {type(edge_index)}, alpha type = {type(alpha)}")
        # Handle alpha if it's a dictionary
        if isinstance(alpha, dict):
            # print(f"Alpha keys: {list(alpha.keys())}") # "tx", "bd"
            # print(f"Edge keys: {list(edge_index.keys())}") # "tx", "bd"
            # Assume the attention weights are under a key like 'attention' or the edge type
            alpha_tensor = alpha.get('bd')
            edge_index_tensor = edge_index.get('bd')
        else:
            alpha_tensor = alpha
        
        # Verify alpha_tensor is a tensor and has the expected shape
        if not isinstance(alpha_tensor, torch.Tensor):
            raise ValueError(f"Expected alpha to be a tensor, got {type(alpha_tensor)}")
        print(f"Alpha tensor shape: {alpha_tensor.shape}")
           
        for head_idx in range(alpha_tensor.shape[1]):  # Iterate over attention heads
            visualize_attention_weights(
                attention_weights=alpha_tensor[:, head_idx],
                edge_index=edge_index_tensor,
                layer_idx=layer_idx,
                head_idx=head_idx,
                edge_type=str(edge_type),
                num_nodes=num_nodes
            )


if __name__ == '__main__':
    main()