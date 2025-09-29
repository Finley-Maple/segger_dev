import sys
from pathlib import Path
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import pandas as pd
import pickle
from segger.training.segger_data_module import SeggerDataModule
from segger.prediction.predict import load_model
from visualization.batch_visualization import extract_attention_df
from visualization.gene_visualization import AttentionSummarizer
from visualization.utils import safe_divide_sparse_numpy
from visualization.embedding_visualization import (
    visualize_embeddings_from_model, 
    EmbeddingVisualizationConfig,
    EmbeddingExtractor,
    EmbeddingVisualizer
)
from segger.models.segger_model import Segger
from torch_geometric.nn import to_hetero
from segger.training.train import LitSegger
from scipy.sparse import lil_matrix
import numpy as np
from segger.data.parquet.sample import STSampleParquet
import os

DATA_DIR = Path('/dkfz/cluster/gpu/data/OE0606/fengyun')


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process dataset with specified edge type and model type')
    parser.add_argument('--dataset', type=str, choices=['colon', 'CRC', 'pancreas', 'breast'], required=True, default='colon',
                       help='Dataset: "colon", "CRC", or "pancreas"')
    parser.add_argument('--edge_type', type=str, choices=['tx-bd', 'tx-tx'], required=True,
                       help='Edge type: "tx-bd" or "tx-tx"')
    parser.add_argument('--model_type', type=str, choices=['seq', 'no_seq'], required=True, default='no_seq',
                       help='Model type: "seq" or "no_seq"')
    parser.add_argument('--test_mode', action='store_true',
                       help='Test mode: if True, only process the first 10 batches')
    parser.add_argument('--visualize_embeddings', action='store_true',
                       help='Generate embedding visualizations using UMAP/t-SNE')
    parser.add_argument('--embedding_method', type=str, choices=['umap', 'tsne', 'pca'], default='umap',
                       help='Dimensionality reduction method for embedding visualization')
    parser.add_argument('--max_embedding_batches', type=int, default=20,
                       help='Maximum number of batches to use for embedding visualization')
    args = parser.parse_args()
    
    # Configuration
    edge_type = args.edge_type
    model_type = args.model_type
    max_cells = None  # Maximum number of cells to process
    test_mode = args.test_mode
    # Paths to data and models
    model_version = 1
    model_dir_path = DATA_DIR / 'segger_model' / f'segger_{args.dataset}_{model_type}'
    model_path = Path(model_dir_path) / "lightning_logs" / f"version_{model_version}"
    
    XENIUM_DATA_DIR=DATA_DIR / 'xenium_data' / f'xenium_{args.dataset}'
    SEGGER_DATA_DIR=DATA_DIR / 'segger_data' / f'segger_{args.dataset}_{model_type}'
    
    results_dir = DATA_DIR / 'attention_results' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sample = STSampleParquet(
        base_dir=XENIUM_DATA_DIR,
        n_workers=4,
        sample_type="xenium",
        # scale_factor=0.8,
        # weights=gene_celltype_abundance_embedding
    )

    # only run once if you want to create the dataset
    if not os.path.exists(SEGGER_DATA_DIR):
        sample.save(
            data_dir=SEGGER_DATA_DIR,
            k_bd=3,
            dist_bd=15.0,
            k_tx=3,
            dist_tx=5.0,
            tile_width=120,
            tile_height=120,
            neg_sampling_ratio=5.0,
            frac=1.0,
            val_prob=0.1,
            test_prob=0.2,
        )
    
    # Initialize the Lightning data module
    dm = SeggerDataModule(
        data_dir=SEGGER_DATA_DIR,
        batch_size=3,
        num_workers=2,
    )
    dm.setup()
    
    if model_type == 'no_seq':
        num_tx_tokens = 500
        print(f"No RNA seq data is used. Using {num_tx_tokens} tokens for tx nodes")
    elif model_type == 'seq':
        num_tx_tokens = dm.train[0].x_dict["tx"].shape[1]
        print(f"RNA seq data is used. Using {num_tx_tokens} tokens for tx nodes")
    else:
        raise ValueError(f"Model type {model_type} is not supported")
    
    model = Segger(
        num_tx_tokens=num_tx_tokens,
        init_emb=8,
        hidden_channels=64,
        out_channels=16,
        heads=4,
        num_mid_layers=3,
    )
    model = to_hetero(model, (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]), aggr="sum")
    
    # ls = load_model(Path(model_path) / "checkpoints")
    # Wrap the model in LitSegger
    ls = LitSegger(model=model).to(device)
    if model_type == 'no_seq':
        if args.dataset == 'colon':
            ckpt = torch.load(model_path / "checkpoints" / "epoch=79-step=70160.ckpt", map_location=torch.device(device))
        elif args.dataset == 'CRC':
            print(f"Not implemented for CRC")
            exit()
        elif args.dataset == 'pancreas':
            ckpt = torch.load(model_path / "checkpoints" / "epoch=99-step=48300.ckpt", map_location=torch.device(device))
        elif args.dataset == 'breast':
            ckpt = torch.load(model_path / "checkpoints" / "epoch=79-step=70160.ckpt", map_location=torch.device(device))
        ls.load_state_dict(ckpt["state_dict"], strict=True)
    elif model_type == 'seq':
        ckpt = torch.load(model_path / "checkpoints" / "epoch=79-step=71440.ckpt", map_location=torch.device(device))
        ls.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        raise ValueError(f"Model type {model_type} is not supported")
    ls.eval()

    # Load transcripts
    transcripts = pd.read_parquet(Path(XENIUM_DATA_DIR) / 'transcripts.parquet')

    # Get cell types and gene types
    if args.dataset == 'pancreas':
        # Get classified genes (it contains the gene names and groups)
        gene_types = pd.read_csv(XENIUM_DATA_DIR / 'gene_groups.csv')
        gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
            
        # Get cell types
        cell_types = pd.read_csv(XENIUM_DATA_DIR / 'cell_groups.csv')
        cell_types_dict = dict(zip(cell_types['cell_id'], cell_types['group']))
    else:
        gene_types_dict = None
        cell_types_dict = None

    if edge_type == 'tx-bd':
        cell_num = 0
        cell_ids = []
        for batch in dm.train:
            cell_id_batch = batch['bd'].id
            cell_num += len(cell_id_batch)
            cell_ids.extend(cell_id_batch)
        if max_cells is not None and cell_num > max_cells:
            selected_cell_ids = []
            # Use the first max_cells cells
            for batch_idx, batch in enumerate(dm.train):
                cell_ids_batch = batch['bd'].id
                # add the cell ids to the selected_cell_ids
                selected_cell_ids = np.concatenate([selected_cell_ids, cell_ids_batch])
                # if the number of selected cells is greater than max_cells, break
                if len(selected_cell_ids) >= max_cells:
                    print(f"The first {batch_idx} batches are used to select {max_cells} cells")
                    max_batch_idx = batch_idx
                    break
        else:
            selected_cell_ids = np.unique(np.array(cell_ids))
            print(f"Using all {len(selected_cell_ids)} cells")
            max_batch_idx = len(dm.train) - 1
    
        # Create cell ID to index mapping for selected cells
        cell_to_idx = {cell: idx for idx, cell in enumerate(selected_cell_ids)}
    else:
        cell_to_idx = None
        max_batch_idx = len(dm.train) - 1
    
    # Get a sample batch ------------------------------------------------------------
    batch = dm.train[0].to(device)
    
    # Get gene names
    transcript_ids = batch['tx'].id.cpu().numpy()
    id_to_gene = dict(zip(transcripts['transcript_id'], transcripts['feature_name']))
    gene_names_batch = [id_to_gene[id] for id in transcript_ids]
    cell_ids_batch = batch['bd'].id

    # Run forward pass to get attention weights
    with torch.no_grad():
        hetero_model = ls.model
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        _, attention_weights = hetero_model(x_dict, edge_index_dict)

    # Extract attention weights
    attention_df = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_ids_batch, edge_type = edge_type, cell_types_dict=cell_types_dict)

    # Gene-level visualization
    print("Computing gene-level attention patterns...")
    num_genes = len(transcripts['feature_name'].unique())
    gene_names = transcripts['feature_name'].unique().tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    layers = 5
    heads = 4
    
    # Initialize attention matrix dictionary for both cases
    if edge_type == 'tx-tx':
        num_genes = len(gene_names)
        attention_gene_matrix_dict = {
            "adj_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "count_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "adj_avg_matrix": [[lil_matrix((num_genes, num_genes), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "gene_names": gene_names,
            "gene_types_dict": gene_types_dict
        }
    elif edge_type == 'tx-bd':
        num_cells = len(selected_cell_ids)
        attention_gene_matrix_dict = {
            "adj_matrix": [[lil_matrix((num_genes, num_cells), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "count_matrix": [[lil_matrix((num_genes, num_cells), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "adj_avg_matrix": [[lil_matrix((num_genes, num_cells), dtype=np.float32) for _ in range(heads)] for _ in range(layers)],
            "cell_ids": selected_cell_ids.tolist(),
            "cell_types_dict": cell_types_dict,
            "gene_names": gene_names,
            "gene_types_dict": gene_types_dict
        }
    else:
        raise ValueError(f"Only tx-bd and tx-tx edge types are supported in this version")
    
    if test_mode:
        max_batch_idx = 80 # for testing

    # Process each batch ------------------------------------------------------------
    # If not file exists, process the batches and save the results
    if not (results_dir / f'attention_gene_matrix_dict_{model_type}_{edge_type}_{test_mode}.pkl').exists():
        # Initialize attention summarizer
        attention_summarizer = AttentionSummarizer(
            edge_type=edge_type,
            gene_to_idx=gene_to_idx,
            cell_to_idx=cell_to_idx
        )
        # Process each batch
        for batch_idx, batch in enumerate(dm.train[:max_batch_idx]):
            with torch.no_grad():
                print(f"Processing batch {batch_idx} of {max_batch_idx}")
                batch = batch.to(device)
                x_dict = batch.x_dict
                edge_index_dict = batch.edge_index_dict
                _, attention_weights = hetero_model(x_dict, edge_index_dict)
                
                transcript_ids = batch['tx'].id.cpu().numpy()
                gene_names_batch = [id_to_gene[id] for id in transcript_ids]
                
                cell_ids_batch = batch['bd'].id
                
                attention_df = extract_attention_df(attention_weights = attention_weights, gene_names = gene_names_batch, cell_ids = cell_ids_batch, edge_type = edge_type, cell_types_dict = cell_types_dict)
                
                #print(f"shape of attention_df: {attention_gene_matrix_dict['adj_matrix'][0][0].shape}")
                for layer_idx in range(layers):
                    for head_idx in range(heads):
                        adj_matrix, count_matrix = attention_summarizer.summarize_attention_by_gene_df(
                            attention_df, 
                            layer_idx=layer_idx, 
                            head_idx=head_idx
                        )
                        #print(f"Layer {layer_idx}, head {head_idx} shape: {adj_matrix.shape}")
                        attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx] += adj_matrix
                        attention_gene_matrix_dict["count_matrix"][layer_idx][head_idx] += count_matrix
        
        # Compute the average attention matrix
        for layer_idx in range(layers):
            for head_idx in range(heads):
                attention_gene_matrix_dict["adj_avg_matrix"][layer_idx][head_idx] = safe_divide_sparse_numpy(attention_gene_matrix_dict["adj_matrix"][layer_idx][head_idx], attention_gene_matrix_dict["count_matrix"][layer_idx][head_idx])
        
        # Save results
        if test_mode:
            with open(results_dir / f'attention_gene_matrix_dict_{model_type}_{edge_type}_{test_mode}.pkl', 'wb') as f:
                pickle.dump(attention_gene_matrix_dict, f)
        else: # split the results into multiple files (adj_matrix, count_matrix, adj_avg_matrix) to avoid memory issues
            with open(results_dir / f'attention_gene_matrix_dict_{model_type}_{edge_type}_{test_mode}_adj_matrix.pkl', 'wb') as f:
                pickle.dump(attention_gene_matrix_dict["adj_matrix"], f)
            with open(results_dir / f'attention_gene_matrix_dict_{model_type}_{edge_type}_{test_mode}_count_matrix.pkl', 'wb') as f:
                pickle.dump(attention_gene_matrix_dict["count_matrix"], f)
            with open(results_dir / f'attention_gene_matrix_dict_{model_type}_{edge_type}_{test_mode}_adj_avg_matrix.pkl', 'wb') as f:
                pickle.dump(attention_gene_matrix_dict["adj_avg_matrix"], f)
    else:
        print(f"Loading results from {results_dir / f'attention_gene_matrix_dict_{model_type}_{edge_type}_{test_mode}.pkl'}")
        # Load results
        with open(results_dir / f'attention_gene_matrix_dict_{model_type}_{edge_type}_{test_mode}.pkl', 'rb') as f:
            attention_gene_matrix_dict = pickle.load(f)
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Model type: {model_type}")
    print(f"Edge type: {edge_type}")
    print(f"Number of genes: {num_genes}")
    print(f"Number of layers: {layers}")
    print(f"Number of heads: {heads}")
    if edge_type == 'tx-bd':
        print(f"Number of cells: {len(selected_cell_ids)}")
        print(f"Matrix shape: {num_genes} x {len(selected_cell_ids)}")
    else:
        print(f"Matrix shape: {num_genes} x {num_genes}")
    
    # Generate embedding visualizations if requested
    if args.visualize_embeddings:
        print(f"\nGenerating embedding visualizations using {args.embedding_method}...")
        
        # Create embedding visualization config
        embedding_config = EmbeddingVisualizationConfig(
            method=args.embedding_method,
            n_components=2,
            figsize=(12, 8),
            point_size=2.0,
            alpha=0.7,
            max_points_per_type=5000,  # Limit points for better visualization
            subsample_method='balanced'
        )
        
        # Create embedding visualization directory
        embedding_results_dir = results_dir / 'embeddings'
        embedding_results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate embeddings visualization
            plots = visualize_embeddings_from_model(
                model=hetero_model,
                dataloader=dm.train[:args.max_embedding_batches],
                save_dir=embedding_results_dir,
                transcripts_df=transcripts,
                gene_types_dict=gene_types_dict,
                cell_types_dict=cell_types_dict,
                max_batches=args.max_embedding_batches,
                config=embedding_config
            )
            
            print(f"Embedding visualizations saved to: {embedding_results_dir}")
            for plot_name, plot_path in plots.items():
                print(f"  - {plot_name}: {plot_path}")
                
        except Exception as e:
            print(f"Error generating embedding visualizations: {str(e)}")
            print("Continuing without embedding visualizations...")

if __name__ == '__main__':
    main() 