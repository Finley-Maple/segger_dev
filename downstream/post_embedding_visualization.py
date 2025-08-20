"""
Example Script for Embedding Visualization

This script demonstrates how to use the embedding visualization tools
both for post-training analysis and during training with Lightning callbacks.
"""

import sys
from pathlib import Path
import argparse
import torch
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from segger.training.segger_data_module import SeggerDataModule
from segger.models.segger_model import Segger
from torch_geometric.nn import to_hetero
from segger.training.train import LitSegger
from visualization.embedding_visualization import (
    visualize_embeddings_from_model,
    visualize_spatial_from_dataloader,
    EmbeddingVisualizationConfig,
    EmbeddingExtractor,
    EmbeddingVisualizer
)
from visualization.embedding_callback import create_embedding_callbacks

# Configure paths (adjust these to your setup)
DATA_DIR = Path('/dkfz/cluster/gpu/data/OE0606/fengyun')


def post_training_visualization():
    """
    Example of how to visualize embeddings from a trained model.
    """
    print("=== Post-Training Embedding Visualization Example ===")
    
    # Configuration
    dataset = 'pancreas'  # or 'colon'
    model_type = 'no_seq'
    model_version = 1
    
    # Set up paths
    model_dir_path = DATA_DIR / 'segger_model' / f'segger_{dataset}_{model_type}'
    model_path = Path(model_dir_path) / "lightning_logs" / f"version_{model_version}"
    XENIUM_DATA_DIR = DATA_DIR / 'xenium_data' / f'xenium_{dataset}'
    SEGGER_DATA_DIR = DATA_DIR / 'segger_data' / f'segger_{dataset}_{model_type}'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data module
    dm = SeggerDataModule(
        data_dir=SEGGER_DATA_DIR,
        batch_size=3,
        num_workers=2,
    )
    dm.setup()
    
    # Set up model
    if model_type == 'no_seq':
        num_tx_tokens = 500
    else:
        num_tx_tokens = dm.train[0].x_dict["tx"].shape[1]
    
    model = Segger(
        num_tx_tokens=num_tx_tokens,
        init_emb=8,
        hidden_channels=64,
        out_channels=16,
        heads=4,
        num_mid_layers=3,
    )
    model = to_hetero(model, (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]), aggr="sum")
    
    # Load trained model
    ls = LitSegger(model=model).to(device)
    if dataset == 'pancreas':
        ckpt_path = model_path / "checkpoints" / "epoch=99-step=48300.ckpt"
    else:  # colon
        ckpt_path = model_path / "checkpoints" / "epoch=79-step=70160.ckpt"
    
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        ls.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        print(f"Checkpoint not found at {ckpt_path}")
        return
    
    ls.eval()
    
    # Load transcripts and metadata
    transcripts = pd.read_parquet(XENIUM_DATA_DIR / 'transcripts.parquet')
    
    gene_types_dict = None
    cell_types_dict = None
    if dataset == 'pancreas':
        # Load gene and cell type information
        gene_types = pd.read_excel(XENIUM_DATA_DIR / 'gene_groups_modified.xlsx')
        gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
        
        cell_types = pd.read_csv(XENIUM_DATA_DIR / 'cell_groups.csv')
        cell_types_dict = dict(zip(cell_types['cell_id'], cell_types['group']))
        
        # Merge Endocrine 1 and Endocrine 2 into Endocrine
        # Merge Tumor Cells and CFTR- Tumor Cells into Tumor Cells
        for k, v in cell_types_dict.items():
            if v in ["Endocrine 1", "Endocrine 2"]:
                cell_types_dict[k] = "Endocrine"
            elif v in ["Tumor Cells", "CFTR- Tumor Cells"]:
                cell_types_dict[k] = "Tumor Cells"
    
    # Create visualization config
    config = EmbeddingVisualizationConfig(
        method='umap',  # Try 'tsne' or 'pca' as well
        n_components=2,
        figsize=(12, 8),
        point_size=3.0,
        alpha=0.7,
        max_points_per_type=1000,  # Subsample for better visualization
        subsample_method='balanced',
        umap_n_neighbors=15,
        umap_min_dist=0.1
    )
    
    # Set up save directory
    save_dir = Path('./embedding_visualization_results') / dataset / model_type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating embeddings visualizations...")
    print(f"Dataset: {dataset}")
    print(f"Model type: {model_type}")
    print(f"Visualization method: {config.method}")
    print(f"Save directory: {save_dir}")
    
    # Generate visualizations
    try:
        plots = visualize_embeddings_from_model(
            model=ls.model,
            dataloader=dm.train[:40],  # Use first 10 batches
            save_dir=save_dir,
            transcripts_df=transcripts,
            gene_types_dict=gene_types_dict,
            cell_types_dict=cell_types_dict,
            max_batches=40,
            config=config
        )
        
        print(f"\nVisualization complete! Generated plots:")
        for plot_name, plot_path in plots.items():
            print(f"  - {plot_name}: {plot_path}")
            
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()


def spatial_visualization_by_batch():
    """
    Example of how to visualize spatial distribution of transcripts and boundaries.
    Creates two combined plots: one for all transcripts (tx) and one for all boundaries (bd),
    with points colored by batch index.
    """
    print("=== Spatial Visualization by Batch Example ===")
    
    # Configuration
    dataset = 'pancreas'  # or 'colon'
    model_type = 'no_seq'
    
    # Set up paths
    XENIUM_DATA_DIR = DATA_DIR / 'xenium_data' / f'xenium_{dataset}'
    SEGGER_DATA_DIR = DATA_DIR / 'segger_data' / f'segger_{dataset}_{model_type}'
    
    # Load data module
    dm = SeggerDataModule(
        data_dir=SEGGER_DATA_DIR,
        batch_size=3,
        num_workers=2,
    )
    dm.setup()
    
    # Load transcripts and metadata
    transcripts = pd.read_parquet(XENIUM_DATA_DIR / 'transcripts.parquet')
    
    gene_types_dict = None
    cell_types_dict = None
    if dataset == 'pancreas':
        # Load gene and cell type information
        gene_types = pd.read_excel(XENIUM_DATA_DIR / 'gene_groups_modified.xlsx')
        gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
        
        cell_types = pd.read_csv(XENIUM_DATA_DIR / 'cell_groups.csv')
        cell_types_dict = dict(zip(cell_types['cell_id'], cell_types['group']))
        
        # Merge Endocrine 1 and Endocrine 2 into Endocrine
        # Merge Tumor Cells and CFTR- Tumor Cells into Tumor Cells
        for k, v in cell_types_dict.items():
            if v in ["Endocrine 1", "Endocrine 2"]:
                cell_types_dict[k] = "Endocrine"
            elif v in ["Tumor Cells", "CFTR- Tumor Cells"]:
                cell_types_dict[k] = "Tumor Cells"
    
    # Create visualization config
    config = EmbeddingVisualizationConfig(
        figsize=(10, 8),
        spatial_alpha=0.7,
        spatial_tx_size=8.0,
        spatial_bd_size=15.0,
        spatial_max_points_per_gene_type=500,  # Subsample for better visualization
        save_format='png',
        dpi=300
    )
    
    # Set up save directory
    save_dir = Path('./spatial_visualization_results') / dataset / model_type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating spatial visualizations...")
    print(f"Dataset: {dataset}")
    print(f"Model type: {model_type}")
    print(f"Save directory: {save_dir}")
    
    # Generate visualizations
    try:
        plots = visualize_spatial_from_dataloader(
            dataloader=dm.train,
            save_dir=save_dir,
            transcripts_df=transcripts,
            gene_types_dict=gene_types_dict,
            cell_types_dict=cell_types_dict,
            max_batches=80,
            max_batches_to_plot=80,
            config=config,
            combined_plot=True  # Create combined plots colored by batch index
        )
        
        print(f"\nSpatial visualization complete! Generated plots:")
        for plot_name, plot_path in plots.items():
            print(f"  - {plot_name}: {plot_path}")
            
    except Exception as e:
        print(f"Error during spatial visualization: {str(e)}")
        import traceback
        traceback.print_exc()


def spatial_visualization_separate_batches():
    """
    Example of how to visualize spatial distribution with separate plots for each batch.
    Creates individual plots for each batch showing transcripts and boundaries.
    """
    print("=== Spatial Visualization Separate Batches Example ===")
    
    # Configuration
    dataset = 'pancreas'  # or 'colon'
    model_type = 'no_seq'
    
    # Set up paths
    XENIUM_DATA_DIR = DATA_DIR / 'xenium_data' / f'xenium_{dataset}'
    SEGGER_DATA_DIR = DATA_DIR / 'segger_data' / f'segger_{dataset}_{model_type}'
    
    # Load data module
    dm = SeggerDataModule(
        data_dir=SEGGER_DATA_DIR,
        batch_size=3,
        num_workers=2,
    )
    dm.setup()
    
    # Load transcripts and metadata
    transcripts = pd.read_parquet(XENIUM_DATA_DIR / 'transcripts.parquet')
    
    gene_types_dict = None
    cell_types_dict = None
    if dataset == 'pancreas':
        # Load gene and cell type information
        gene_types = pd.read_excel(XENIUM_DATA_DIR / 'gene_groups_modified.xlsx')
        gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
        
        cell_types = pd.read_csv(XENIUM_DATA_DIR / 'cell_groups.csv')
        cell_types_dict = dict(zip(cell_types['cell_id'], cell_types['group']))
        
        # Merge Endocrine 1 and Endocrine 2 into Endocrine
        # Merge Tumor Cells and CFTR- Tumor Cells into Tumor Cells
        for k, v in cell_types_dict.items():
            if v in ["Endocrine 1", "Endocrine 2"]:
                cell_types_dict[k] = "Endocrine"
            elif v in ["Tumor Cells", "CFTR- Tumor Cells"]:
                cell_types_dict[k] = "Tumor Cells"
    
    # Create visualization config
    config = EmbeddingVisualizationConfig(
        figsize=(10, 8),
        spatial_alpha=0.7,
        spatial_tx_size=8.0,
        spatial_bd_size=15.0,
        spatial_max_points_per_gene_type=500,
        save_format='png',
        dpi=300
    )
    
    # Set up save directory
    save_dir = Path('./spatial_visualization_results') / dataset / model_type / 'separate_batches'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating separate spatial visualizations...")
    print(f"Dataset: {dataset}")
    print(f"Model type: {model_type}")
    print(f"Save directory: {save_dir}")
    
    # Generate separate visualizations
    try:
        plots = visualize_spatial_from_dataloader(
            dataloader=dm.train,
            save_dir=save_dir,
            transcripts_df=transcripts,
            gene_types_dict=gene_types_dict,
            cell_types_dict=cell_types_dict,
            max_batches=5,  # Process fewer batches for separate plots
            max_batches_to_plot=5,
            config=config,
            combined_plot=False  # Create separate plots for each batch
        )
        
        print(f"\nSeparate spatial visualization complete! Generated plots:")
        for plot_name, plot_path in plots.items():
            print(f"  - {plot_name}: {plot_path}")
            
    except Exception as e:
        print(f"Error during separate spatial visualization: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    # Run embedding visualization
    post_training_visualization()
    
    # Run combined spatial visualization (all batches in 2 plots colored by batch)
    # spatial_visualization_by_batch()


if __name__ == '__main__':
    main()
