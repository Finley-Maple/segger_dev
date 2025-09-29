#!/usr/bin/env python3
"""
Model training script for Segger with proper Python path configuration.
This script ensures the segger module can be found when running in batch job environments.
"""

import sys
import os
from pathlib import Path

# Add the segger source directory to Python path
# This ensures the segger module can be imported when running in batch jobs
script_dir = Path(__file__).parent.absolute()
segger_src_dir = script_dir.parent / "src"
if segger_src_dir.exists():
    sys.path.insert(0, str(segger_src_dir))
else:
    # Fallback: try to find segger in the current working directory structure
    cwd = Path.cwd()
    potential_paths = [
        cwd / "src",
        cwd / "segger" / "src", 
        cwd.parent / "src",
        Path.home() / "segger" / "src"
    ]
    for path in potential_paths:
        if path.exists() and (path / "segger").exists():
            sys.path.insert(0, str(path))
            break
    else:
        print("Warning: Could not find segger source directory. Module imports may fail.")

# Now import segger modules
from segger.data.parquet.sample import STSampleParquet
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
from segger.data.parquet._utils import find_markers
from segger.training.segger_data_module import SeggerDataModule
from segger.training.train import LitSegger
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch import Trainer
import pandas as pd
from segger.models.segger_model import Segger
from torch_geometric.nn import to_hetero
import argparse
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple
import anndata as ad
from lightning.pytorch.loggers import WandbLogger
# Import embedding visualization modules conditionally
create_embedding_callbacks = None
EmbeddingVisualizationConfig = None

def setup_wandb_key():
    """Simple wandb setup with embedded API key."""
    try:
        import wandb
        
        # Set the API key directly
        wandb_api_key = 'a086b720bef11264e58d29b2779369ad2582c326'
        
        # Login with the key
        wandb.login(key=wandb_api_key)
        print("✓ Wandb configured successfully")
        return True
        
    except ImportError:
        print("❌ Wandb not installed. Please install with: pip install wandb")
        return False
    except Exception as e:
        print(f"❌ Error setting up wandb: {e}")
        return False

DATA_DIR = Path('/dkfz/cluster/gpu/data/OE0606/fengyun')



def find_mutually_exclusive_genes(
    adata: ad.AnnData, markers: Dict[str, Dict[str, List[str]]], cell_type_column: str
) -> List[Tuple[str, str]]:
    """Patched version of find_mutually_exclusive_genes that fixes boolean indexing issues.
    
    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - markers: dict
        Dictionary where keys are cell types and values are dictionaries containing:
            'positive': list of top x% highly expressed genes
            'negative': list of top x% lowly expressed genes.
    - cell_type_column: str
        Column name in `adata.obs` that specifies cell types.

    Returns:
    - exclusive_pairs: list
        List of mutually exclusive gene pairs.
    """
    exclusive_genes = {}
    all_exclusive = []
    
    for cell_type, marker_sets in markers.items():
        positive_markers = marker_sets["positive"]
        exclusive_genes[cell_type] = []
        
        for gene in positive_markers:
            if gene not in adata.var_names:
                continue
                
            gene_expr = adata[:, gene].X
            
            # Convert to dense array if sparse to avoid indexing issues
            if hasattr(gene_expr, 'toarray'):
                gene_expr = gene_expr.toarray().flatten()
            else:
                gene_expr = gene_expr.flatten()
            
            # Create boolean masks as numpy arrays
            cell_type_mask = (adata.obs[cell_type_column] == cell_type).values
            non_cell_type_mask = ~cell_type_mask
            
            # Calculate expression fractions
            cell_type_expr_frac = (gene_expr[cell_type_mask] > 0).mean()
            non_cell_type_expr_frac = (gene_expr[non_cell_type_mask] > 0).mean()
            
            if cell_type_expr_frac > 0.2 and non_cell_type_expr_frac < 0.05:
                exclusive_genes[cell_type].append(gene)
                all_exclusive.append(gene)
    
    unique_genes = list(
        {
            gene
            for i in exclusive_genes.keys()
            for gene in exclusive_genes[i]
            if gene in all_exclusive
        }
    )
    
    filtered_exclusive_genes = {
        i: [gene for gene in exclusive_genes[i] if gene in unique_genes]
        for i in exclusive_genes.keys()
    }
    
    mutually_exclusive_gene_pairs = [
        tuple(sorted((gene1, gene2)))
        for key1, key2 in combinations(filtered_exclusive_genes.keys(), 2)
        if key1 != key2
        for gene1 in filtered_exclusive_genes[key1]
        for gene2 in filtered_exclusive_genes[key2]
    ]
    
    return set(mutually_exclusive_gene_pairs)



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Model training with specified dataset and scRNA-seq file')
    parser.add_argument('--dataset', type=str, choices=['colon', 'CRC', 'pancreas', 'breast'], required=True,
                       help='Dataset: "colon" or "CRC" or "pancreas" or "breast"')
    parser.add_argument('--use_scRNAseq', action='store_true', default=False,
                       help='Whether to use the scRNA-seq file')
    parser.add_argument('--enable_embedding_viz', action='store_true',
                       help='Enable embedding visualization during training')
    parser.add_argument('--embedding_log_freq', type=int, default=10,
                       help='Log embeddings every N epochs')
    parser.add_argument('--enable_align_loss', action='store_true',
                       help='Enable alignment loss during training (automatically enables mutually exclusive genes)')
    parser.add_argument('--align_loss_weight', type=float, default=0.5,
                       help='Weight for alignment loss (default: 0.5)')
    parser.add_argument('--align_cycle_length', type=int, default=10000,
                       help='Cycle length for alignment loss (default: 10000)')
    parser.add_argument('--use_mutually_exclusive_genes', action='store_true',
                       help='Generate and use mutually exclusive genes for training (requires --use_scRNAseq)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate for training (default: 1e-3)')
    parser.add_argument('--wandb', action='store_true',
                       help='Use wandb for logging')
    
    args = parser.parse_args()
    
    # Setup wandb if requested
    if args.wandb:
        wandb_available = setup_wandb_key()
        if not wandb_available:
            print("⚠️  Wandb setup failed. Falling back to CSV logging.")
            args.wandb = False
    
    # Validate argument combinations
    if args.use_mutually_exclusive_genes and not args.use_scRNAseq:
        raise ValueError("--use_mutually_exclusive_genes requires --use_scRNAseq to be True")
    
    # Align loss requires mutually exclusive genes to function properly
    if args.enable_align_loss and not args.use_mutually_exclusive_genes:
        print("WARNING: --enable_align_loss requires mutually exclusive genes. Automatically enabling --use_mutually_exclusive_genes")
        args.use_mutually_exclusive_genes = True
        if not args.use_scRNAseq:
            raise ValueError("--enable_align_loss requires --use_scRNAseq to be True (for mutually exclusive genes generation)")
    
    if args.use_scRNAseq:
        print("Using scRNA-seq file")
        if args.dataset == 'colon' or args.dataset == 'CRC':
            SCRNASEQ_FILE = DATA_DIR / 'xenium_data' / 'xenium_colon' / 'scRNAseq.h5ad'
            CELLTYPE_COLUMN = "Level1"
        elif args.dataset == 'breast':
            SCRNASEQ_FILE = DATA_DIR / 'xenium_data' / 'xenium_breast' / 'scRNAseq.h5ad'
            CELLTYPE_COLUMN = "celltype_major"
        elif args.dataset == 'pancreas':
            print("Not implemented for pancreas. Please do not use --use_scRNAseq for pancreas")
            exit()
        scrnaseq = sc.read(SCRNASEQ_FILE)
        sc.pp.subsample(scrnaseq, 0.1)
        scrnaseq.var_names_make_unique()
        
        # Preprocess scRNA-seq data for marker finding (log-transform as recommended)
        sc.pp.normalize_total(scrnaseq)
        sc.pp.log1p(scrnaseq)
        
        # Calculate gene-celltype embeddings from reference data
        gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(
            scrnaseq,
            CELLTYPE_COLUMN
        )
        num_tx_tokens = gene_celltype_abundance_embedding.shape[1]
        model_suffix = "seq"
    else:
        print("Not using scRNA-seq file")
        gene_celltype_abundance_embedding = None
        num_tx_tokens = 500
        model_suffix = "no_seq"
    
    print(f"Number of tokens: {num_tx_tokens}")
    print(f"The dataset is {args.dataset}")
    if args.dataset == 'colon':
        XENIUM_DATA_DIR = DATA_DIR / 'xenium_data' / 'xenium_colon'
    elif args.dataset == 'CRC':
        XENIUM_DATA_DIR = DATA_DIR / 'xenium_data' / 'xenium_CRC'
    elif args.dataset == 'pancreas':
        XENIUM_DATA_DIR = DATA_DIR / 'xenium_data' / 'xenium_pancreas'
    elif args.dataset == 'breast':
        XENIUM_DATA_DIR = DATA_DIR / 'xenium_data' / 'xenium_breast'
    
    # Base directory to store segger data
    if args.enable_align_loss:
        segger_data_dir = DATA_DIR / 'segger_data_align' / f'segger_{args.dataset}_{model_suffix}'
    else:
        segger_data_dir = DATA_DIR / 'segger_data' / f'segger_{args.dataset}_{model_suffix}'

    sample = STSampleParquet(
        base_dir=XENIUM_DATA_DIR,
        n_workers=4,
        sample_type="xenium",
        # scale_factor=0.8,
        weights=gene_celltype_abundance_embedding
    )

    # Generate mutually exclusive genes if requested and scRNA-seq data is available
    mutually_exclusive_gene_pairs = None
    if args.use_mutually_exclusive_genes and args.use_scRNAseq:
        print("Generating mutually exclusive gene pairs...")
        # Find common genes between scRNA-seq and spatial data
        common_genes = list(set(scrnaseq.var_names) & set(sample.transcripts_metadata['feature_names']))
        print(f"Found {len(common_genes)} common genes between scRNA-seq and spatial data")
        
        # Find marker genes using the same parameters as create_data_fast_sample.py
        markers = find_markers(
            scrnaseq[:, common_genes], 
            cell_type_column=CELLTYPE_COLUMN, 
            pos_percentile=90, 
            neg_percentile=20, 
            percentage=20
        )
        
        # Find mutually exclusive gene pairs
        # Create a copy to avoid modifying the original data
        scrnaseq_copy = scrnaseq.copy()
        mutually_exclusive_gene_pairs = find_mutually_exclusive_genes(
            adata=scrnaseq_copy,
            markers=markers,
            cell_type_column=CELLTYPE_COLUMN
        )
        print(f"Generated {len(mutually_exclusive_gene_pairs)} mutually exclusive gene pairs")

    # only run once if you want to create the dataset
    if not os.path.exists(segger_data_dir):
        sample.save(
            data_dir=segger_data_dir,
            k_bd=3,
            dist_bd=15.0,
            k_tx=3,
            dist_tx=5.0,
            k_tx_ex=20,  # Additional transcript neighbors for exclusive genes
            dist_tx_ex=20,  # Additional search radius for exclusive genes
            tile_width=120,
            tile_height=120,
            neg_sampling_ratio=5.0,
            frac=1.0,
            val_prob=0.1,
            test_prob=0.2,
            mutually_exclusive_genes=mutually_exclusive_gene_pairs,
        )

    # Base directory to store Pytorch Lightning models
    if args.enable_align_loss:
        models_dir = DATA_DIR / 'segger_model_align' / f'segger_{args.dataset}_{model_suffix}'
    else:
        models_dir = DATA_DIR / 'segger_model' / f'segger_{args.dataset}_{model_suffix}'

    # Initialize the Lightning data module
    dm = SeggerDataModule(
        data_dir=segger_data_dir,
        batch_size=6,
        num_workers=2,
    )

    dm.setup()

    model = Segger(
        num_tx_tokens=num_tx_tokens,
        init_emb=8,
        hidden_channels=64,
        out_channels=16,
        heads=4,
        num_mid_layers=3,
    )
    model = to_hetero(model, (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")]), aggr="sum")

    batch = dm.train[0]
    model.forward(batch.x_dict, batch.edge_index_dict)
    
    # Wrap the model in LitSegger with optional align loss parameters
    if args.enable_align_loss:
        print(f"Enabling align loss with weight={args.align_loss_weight} and cycle_length={args.align_cycle_length}")
        ls = LitSegger(
            model=model,
            learning_rate=args.learning_rate,
            align_loss=True,
            align_lambda=args.align_loss_weight,
            cycle_length=args.align_cycle_length
        )
    else:
        print("Using standard LitSegger without align loss")
        ls = LitSegger(model=model, learning_rate=args.learning_rate)

    # Set up callbacks
    callbacks = []
    
    # Add embedding visualization callbacks if enabled
    if args.enable_embedding_viz:
        try:
            # Import embedding visualization modules
            import sys
            sys.path.append(str(Path(__file__).parent))
            from visualization.embedding_callback import create_embedding_callbacks
            from visualization.embedding_visualization import EmbeddingVisualizationConfig
            
            # Use the already defined XENIUM_DATA_DIR for transcripts
            transcripts = pd.read_parquet(XENIUM_DATA_DIR / 'transcripts.parquet')
            
            # Load metadata if available
            gene_types_dict = None
            cell_types_dict = None
            spatial_region = None
            
            # Extract gene-to-cell-type mapping from scRNAseq (your simple approach!)
            if args.use_scRNAseq and 'scrnaseq' in locals():
                try:
                    from utils.simple_gene_celltype_mapping import extract_gene_celltype_dict
                    print("Extracting gene-to-cell-type mapping from scRNAseq...")
                    gene_types_dict = extract_gene_celltype_dict(
                        scrnaseq, 
                        celltype_column=CELLTYPE_COLUMN
                    )
                    print(f"Mapped {len(gene_types_dict)} genes to cell types")
                except Exception as e:
                    print(f"Warning: Could not extract gene-cell type mapping: {e}")
            
            # Load dataset-specific metadata if available (only for pancreas and only if scRNA gene types not already loaded)
            if args.dataset == 'pancreas' and gene_types_dict is None:
                try:
                    gene_types = pd.read_excel(XENIUM_DATA_DIR / 'gene_groups_modified.xlsx')
                    gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
                except FileNotFoundError:
                    print("Gene type files not found, proceeding without gene type metadata")
            
            # Load cell type metadata for pancreas
            if args.dataset == 'pancreas':
                try:
                    cell_types = pd.read_csv(XENIUM_DATA_DIR / 'cell_groups.csv')
                    cell_types_dict = dict(zip(cell_types['cell_id'], cell_types['group']))
                    
                    # Merge Endocrine 1 and Endocrine 2 into Endocrine
                    # Merge Tumor Cells and CFTR- Tumor Cells into Tumor Cells
                    for k, v in cell_types_dict.items():
                        if v in ["Endocrine 1", "Endocrine 2"]:
                            cell_types_dict[k] = "Endocrine"
                        elif v in ["Tumor Cells", "CFTR- Tumor Cells"]:
                            cell_types_dict[k] = "Tumor Cells"
                except FileNotFoundError:
                    print("Cell type files not found, proceeding without cell type metadata")
            
            # Create embedding visualization config with robust settings
            embedding_config = EmbeddingVisualizationConfig(
                method='umap',
                n_components=2,
                figsize=(10, 8),
                point_size=2.0,
                alpha=0.7,
                max_points_per_type=1000, 
                subsample_method='balanced'
            )
            if args.dataset == 'colon' or args.dataset == 'CRC':
                spatial_region = [2300, 2500, 2100, 2300]
                spatial_region_gene = 'all' # [2000, 4000, 3000, 5000]
            elif args.dataset == 'pancreas':
                spatial_region = [0, 0, 1000, 1000] # placeholder
                spatial_region_gene = [0, 1000, 0, 1000]
            elif args.dataset == 'breast':
                spatial_region = [5000, 5200, 5800, 6000]
                spatial_region_gene = 'all' # [4000, 6000, 5000, 7000]
            else:
                raise ValueError("Dataset not supported for embedding visualization. Please add the spatial region for gene embeddings and tx interactive plots.")
            
            # Build two dataloaders: full combined for gene embeddings, region-filtered for tx
            try:
                from visualization.utils.spatial_batch_utils import get_spatial_combined_dataloader
            except ImportError:
                from utils.spatial_batch_utils import get_spatial_combined_dataloader

            # 1) Full combined dataloader (train+val+test) for gene embeddings
            spatial_cache_dir = segger_data_dir / "spatial_cache"
            if spatial_region_gene == 'all':
                gene_combined_dataloader = get_spatial_combined_dataloader(
                    dm, all_regions=True, save_dir = spatial_cache_dir
                )
            else:
                gene_combined_dataloader = get_spatial_combined_dataloader(
                    dm, 
                    x_range=[spatial_region_gene[0], spatial_region_gene[1]], 
                    y_range=[spatial_region_gene[2], spatial_region_gene[3]], all_regions=False, save_dir = spatial_cache_dir
            )

            # 2) Region-filtered combined dataloader for tx interactive plots
            if spatial_region is None:
                raise ValueError("spatial_region must be set for transcript visualization")
            spatial_cache_dir = segger_data_dir / "spatial_cache"
            spatial_cache_dir.mkdir(parents=True, exist_ok=True)
            tx_region_dataloader = get_spatial_combined_dataloader(
                dm,
                x_range=[spatial_region[0], spatial_region[1]],
                y_range=[spatial_region[2], spatial_region[3]],
                all_regions=False,
                save_dir=spatial_cache_dir
            )

            # Two separate configs (can be tuned independently)
            gene_cfg = EmbeddingVisualizationConfig(
                method='umap', n_components=2, figsize=(10, 8),
                point_size=2.0, alpha=0.7, max_points_per_type=1000,
                subsample_method='balanced'
            )
            tx_cfg = EmbeddingVisualizationConfig(
                method='umap', n_components=2, figsize=(10, 8),
                point_size=2.0, alpha=0.7, max_points_per_type=1000,
                subsample_method='balanced'
            )

            # 1) Gene-level visualization on FULL dataset (Unknown excluded downstream)
            gene_callbacks = create_embedding_callbacks(
                dataloader=gene_combined_dataloader,
                transcripts_df=transcripts,
                gene_types_dict=gene_types_dict,
                cell_types_dict=cell_types_dict,
                log_every_n_epochs=max(args.embedding_log_freq, 20),
                comparison_epochs=[0, 20, 40, 60, 80],
                config=gene_cfg,
                use_fixed_coordinates=False,
                reference_epoch=None,
                create_interactive_plots=False,
                create_gene_level_plot=True
            )

            # 2) Transcript interactive visualization on REGION-FILTERED dataset
            tx_callbacks = create_embedding_callbacks(
                dataloader=tx_region_dataloader,
                transcripts_df=transcripts,
                spatial_region=spatial_region,
                gene_types_dict=gene_types_dict,
                cell_types_dict=cell_types_dict,
                log_every_n_epochs=max(args.embedding_log_freq, 20),
                comparison_epochs=[0, 20, 40, 60, 80],
                config=tx_cfg,
                use_fixed_coordinates=False,
                reference_epoch=None,
                create_interactive_plots=True,
                create_gene_level_plot=False
            )

            # Update callbacks to use wandb if requested
            if args.wandb:
                for callback in gene_callbacks + tx_callbacks:
                    if hasattr(callback, 'log_to_wandb'):
                        callback.log_to_wandb = True
                        if hasattr(callback, 'log_to_tensorboard'):
                            callback.log_to_tensorboard = False
                        print("Configured embedding callbacks to use Weights & Biases")
                    else:
                        print("Warning: Embedding callback doesn't support wandb logging")

            callbacks.extend(gene_callbacks + tx_callbacks)
            print(f"✅ Added {len(gene_callbacks)} gene-viz and {len(tx_callbacks)} tx-viz callbacks")
            print(f"📊 Will log embeddings every {max(args.embedding_log_freq, 20)} epochs")
            
            # Debug: print callback details
            for i, callback in enumerate(gene_callbacks + tx_callbacks):
                if hasattr(callback, 'save_plots'):
                    print(f"   Callback {i}: save_plots={callback.save_plots}, log_to_wandb={getattr(callback, 'log_to_wandb', 'N/A')}")
                else:
                    print(f"   Callback {i}: {type(callback).__name__}")
            
        except ImportError as e:
            print(f"Could not import embedding visualization modules: {e}")
            print("Proceeding without embedding visualization...")
        except Exception as e:
            print(f"Error setting up embedding visualization: {e}")
            print("Proceeding without embedding visualization...")
            # Disable embedding visualization if there's an error
            args.enable_embedding_viz = False

    # Set up logger
    if args.wandb:
        project_name = "segger_with_embeddings" if args.enable_embedding_viz else "segger_training"
        run_name = f"segger_{args.dataset}_{model_suffix}"
        if args.enable_align_loss:
            run_name += f"_align_{args.align_loss_weight}"
        logger = WandbLogger(project=project_name, name=run_name, save_dir=models_dir)
        print(f"Using Weights & Biases logger: project={project_name}, run={run_name}")
    else:
        logger = CSVLogger(models_dir)
        print(f"Using CSV logger: {models_dir}")

    # Initialize the Lightning trainer
    trainer = Trainer(
        accelerator='auto', # 'gpu' or 'cpu'
        #accelerator='cuda',
        strategy='auto',
        precision='32',
        devices=1, # set higher number if more gpus are available
        max_epochs=150 if args.enable_align_loss else 100, # extend the training time for align loss
        default_root_dir=models_dir,
        logger=logger,
        callbacks=callbacks,
    )

    # Fit model
    trainer.fit(
        model=ls,
        datamodule=dm
    )
    
if __name__ == "__main__":
    main()