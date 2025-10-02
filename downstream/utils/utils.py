#!/usr/bin/env python3
"""
Utility functions for spatial batch combination from downstream tasks.
"""

import sys
from pathlib import Path
import scanpy as sc
import anndata as ad
from typing import Dict, Optional, List, Tuple
import os
from itertools import combinations

# Ensure project and src paths are available for imports (src/ layout)
project_root_path = Path(__file__).resolve().parents[2]
src_path = project_root_path / "src"
if str(project_root_path) not in sys.path:
    sys.path.append(str(project_root_path))
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Optional, List
import time
import torch
from torch_geometric.nn import to_hetero
from segger.training.segger_data_module import SeggerDataModule
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from typing import Dict, Optional, List
from dataclasses import dataclass
try:
    # Optional: load environment variables from .env if available
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

DATA_DIR = Path('/dkfz/cluster/gpu/data/OE0606/fengyun')

@dataclass
class VisualizationConfig:
    """Unified configuration for visualization parameters."""
    dataset: str  # 'colon', 'CRC', 'pancreas', 'breast'
    model_type: str  # 'seq', 'no_seq'
    align_loss: bool = False  # Whether to use align loss model directories
    model_version: Optional[int] = None  # Auto-detect if None
    load_scrna_gene_types: bool = True  # Whether to load gene-to-cell-type mapping from scRNAseq
    
    # Embedding visualization settings
    embedding_method: str = 'umap'
    n_components: int = 2
    figsize: tuple = (12, 8)
    point_size: float = 3.0
    alpha: float = 0.7
    max_points_per_type: int = 1000
    subsample_method: str = 'balanced'
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    
    # Spatial visualization settings
    spatial_alpha: float = 0.7
    spatial_tx_size: float = 8.0
    spatial_bd_size: float = 15.0
    spatial_max_points_per_gene_type: int = 500
    save_format: str = 'png'
    dpi: int = 300
    
    def get_data_paths(self):
        """Get data directory paths based on configuration."""
        xenium_data_dir = DATA_DIR / 'xenium_data' / f'xenium_{self.dataset}'
        
        if self.align_loss:
            segger_data_dir = DATA_DIR / 'segger_data_align' / f'segger_{self.dataset}_{self.model_type}'
            model_dir_path = DATA_DIR / 'segger_model_align' / f'segger_{self.dataset}_{self.model_type}'
        else:
            segger_data_dir = DATA_DIR / 'segger_data' / f'segger_{self.dataset}_{self.model_type}'
            model_dir_path = DATA_DIR / 'segger_model' / f'segger_{self.dataset}_{self.model_type}'
            
        return xenium_data_dir, segger_data_dir, model_dir_path
    
    def auto_detect_model_version(self, model_dir_path: Path) -> int:
        """Automatically detect the latest model version."""
        if self.model_version is not None:
            return self.model_version
            
        # Look for version directories
        version_dirs = list(model_dir_path.glob("segger_with_embeddings/version_*"))
        if not version_dirs:
            # Fallback: look for any version directories
            version_dirs = list(model_dir_path.glob("*/version_*"))
        
        if not version_dirs:
            raise FileNotFoundError(f"No model versions found in {model_dir_path}")
        
        # Extract version numbers and get the latest
        versions = []
        for version_dir in version_dirs:
            try:
                version_num = int(version_dir.name.split('_')[-1])
                versions.append(version_num)
            except ValueError:
                continue
        
        if not versions:
            raise FileNotFoundError(f"No valid model versions found in {model_dir_path}")
        
        latest_version = max(versions)
        print(f"Auto-detected latest model version: {latest_version}")
        return latest_version
    
    def find_checkpoint(self, model_path: Path) -> Path:
        """Automatically find the best checkpoint in the model directory."""
        checkpoint_dir = model_path / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Look for checkpoint files
        checkpoint_patterns = [
            "*.ckpt",
            "epoch=*-step=*.ckpt",
            "last.ckpt",
            "best.ckpt"
        ]
        
        checkpoint_files = []
        for pattern in checkpoint_patterns:
            checkpoint_files.extend(checkpoint_dir.glob(pattern))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Prefer specific epoch checkpoints over generic ones
        specific_checkpoints = [f for f in checkpoint_files if "epoch=" in f.name and "step=" in f.name]
        if specific_checkpoints:
            # Sort by modification time, get the latest
            latest_checkpoint = max(specific_checkpoints, key=lambda x: x.stat().st_mtime)
        else:
            # Fallback to any checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        print(f"Auto-detected checkpoint: {latest_checkpoint}")
        return latest_checkpoint


def get_metadata_cache_path(config: VisualizationConfig, load_scrna_gene_types: bool = True) -> Path:
    """Get the path for cached metadata file."""
    cache_dir = DATA_DIR / 'metadata_cache'
    cache_dir.mkdir(exist_ok=True)
    
    # Create unique cache filename based on dataset and settings
    cache_name = f"{config.dataset}_metadata"
    if load_scrna_gene_types:
        cache_name += "_scrna"
    cache_name += ".pkl"
    
    return cache_dir / cache_name


def check_metadata_freshness(cache_path: Path, source_files: list) -> bool:
    """Check if cached metadata is newer than source files."""
    if not cache_path.exists():
        return False
    
    cache_mtime = cache_path.stat().st_mtime
    
    # Check if any source file is newer than the cache
    for source_file in source_files:
        if source_file.exists() and source_file.stat().st_mtime > cache_mtime:
            return False
    
    return True


def save_metadata_cache(cache_path: Path, transcripts: pd.DataFrame, 
                       gene_types_dict: Optional[Dict], cell_types_dict: Optional[Dict]):
    """Save metadata to cache file."""
    metadata_cache = {
        'transcripts': transcripts,
        'gene_types_dict': gene_types_dict,
        'cell_types_dict': cell_types_dict
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(metadata_cache, f)
    print(f"Saved metadata cache to: {cache_path}")


def load_metadata_cache(cache_path: Path) -> tuple:
    """Load metadata from cache file."""
    with open(cache_path, 'rb') as f:
        metadata_cache = pickle.load(f)
    
    return (
        metadata_cache['transcripts'],
        metadata_cache['gene_types_dict'],
        metadata_cache['cell_types_dict']
    )


def load_metadata(config: VisualizationConfig, load_scrna_gene_types: bool = True, force_reload: bool = True):
    """Load transcripts and metadata based on configuration with caching support.
    
    Args:
        config: Visualization configuration
        load_scrna_gene_types: Whether to load scRNAseq gene types
        force_reload: If True, ignore cache and reload from source files
    """
    xenium_data_dir, _, _ = config.get_data_paths()
    cache_path = get_metadata_cache_path(config, load_scrna_gene_types)
    
    # Determine source files for freshness check
    source_files = [xenium_data_dir / 'transcripts.parquet']
    
    if load_scrna_gene_types:
        scrnaseq_file = DATA_DIR / 'xenium_data' / f'xenium_{config.dataset}' / 'scRNAseq.h5ad'
        source_files.append(scrnaseq_file)
    
    if config.dataset == 'pancreas':
        source_files.extend([
            xenium_data_dir / 'gene_groups_modified.xlsx',
            xenium_data_dir / 'cell_groups.csv'
        ])
    
    # Try to load from cache if it's fresh and not forced to reload
    if not force_reload and check_metadata_freshness(cache_path, source_files):
        try:
            start_time = time.time()
            print(f"Loading metadata from cache: {cache_path}")
            transcripts, gene_types_dict, cell_types_dict = load_metadata_cache(cache_path)
            load_time = time.time() - start_time
            print(f"✓ Loaded cached metadata in {load_time:.2f} seconds:")
            print(f"  - Transcripts: {len(transcripts):,} rows")
            if gene_types_dict:
                print(f"  - Gene types: {len(gene_types_dict):,} mappings")
            if cell_types_dict:
                print(f"  - Cell types: {len(cell_types_dict):,} mappings")
            return transcripts, gene_types_dict, cell_types_dict
        except Exception as e:
            print(f"Warning: Failed to load metadata cache: {e}")
            print("Falling back to loading from source files...")
    
    # Load from source files
    start_time = time.time()
    print(f"Loading metadata from source files (cache {'disabled' if force_reload else 'outdated/missing'})...")
    
    # Load transcripts
    print(f"Loading transcripts from: {xenium_data_dir / 'transcripts.parquet'}")
    transcripts = pd.read_parquet(xenium_data_dir / 'transcripts.parquet')
    
    # Load gene and cell type metadata
    gene_types_dict = None
    cell_types_dict = None
    
    # Try to load gene-to-cell-type mapping from scRNAseq data first
    if load_scrna_gene_types:
        scrnaseq_file = DATA_DIR / 'xenium_data' / f'xenium_{config.dataset}' / 'scRNAseq.h5ad'
        if scrnaseq_file.exists():
            try:
                from utils.simple_gene_celltype_mapping import create_gene_celltype_dict
                print(f"Loading gene-to-cell-type mapping from scRNAseq: {scrnaseq_file}")
                if config.dataset == 'colon':
                    gene_types_dict = create_gene_celltype_dict(
                        str(scrnaseq_file), 
                        celltype_column="Level1"
                    )
                elif config.dataset == 'breast':
                    gene_types_dict = create_gene_celltype_dict(
                        str(scrnaseq_file), 
                        celltype_column="celltype_major"
                    )
                else:
                    raise ValueError(f"The cell type column needs to be specified for dataset {config.dataset}")
                print(f"Loaded {len(gene_types_dict)} gene-to-cell-type mappings from scRNAseq")
            except Exception as e:
                print(f"Warning: Could not load scRNAseq gene types: {e}")
                gene_types_dict = None
        else:
            print(f"scRNAseq file not found: {scrnaseq_file}")
    
    # Load dataset-specific metadata (fallback or additional)
    if config.dataset == 'pancreas':
        try:
            # Load pancreas-specific gene and cell type information
            if gene_types_dict is None:  # Only load if scRNAseq types not available
                gene_types = pd.read_excel(xenium_data_dir / 'gene_groups_modified.xlsx')
                gene_types_dict = dict(zip(gene_types['gene'], gene_types['group']))
                print(f"Loaded {len(gene_types_dict)} gene types from pancreas-specific file")
            
            cell_types = pd.read_csv(xenium_data_dir / 'cell_groups.csv')
            cell_types_dict = dict(zip(cell_types['cell_id'], cell_types['group']))
            
            # Merge Endocrine 1 and Endocrine 2 into Endocrine
            # Merge Tumor Cells and CFTR- Tumor Cells into Tumor Cells
            for k, v in cell_types_dict.items():
                if v in ["Endocrine 1", "Endocrine 2"]:
                    cell_types_dict[k] = "Endocrine"
                elif v in ["Tumor Cells", "CFTR- Tumor Cells"]:
                    cell_types_dict[k] = "Tumor Cells"
        except FileNotFoundError:
            print(f"Metadata files not found for {config.dataset}, proceeding without metadata")
    
    # Save to cache for next time
    try:
        save_metadata_cache(cache_path, transcripts, gene_types_dict, cell_types_dict)
    except Exception as e:
        print(f"Warning: Failed to save metadata cache: {e}")
    
    load_time = time.time() - start_time
    print(f"✓ Loaded metadata from source files in {load_time:.2f} seconds")
    print(f"  - Transcripts: {len(transcripts):,} rows")
    if gene_types_dict:
        print(f"  - Gene types: {len(gene_types_dict):,} mappings")
    if cell_types_dict:
        print(f"  - Cell types: {len(cell_types_dict):,} mappings")
    
    return transcripts, gene_types_dict, cell_types_dict


def clear_metadata_cache(dataset: str = None):
    """Clear metadata cache files.
    
    Args:
        dataset: If specified, only clear cache for this dataset. If None, clear all.
    """
    cache_dir = DATA_DIR / 'metadata_cache'
    if not cache_dir.exists():
        print("No metadata cache directory found.")
        return
    
    if dataset:
        # Clear cache for specific dataset
        cache_files = list(cache_dir.glob(f"{dataset}_metadata*.pkl"))
    else:
        # Clear all cache files
        cache_files = list(cache_dir.glob("*_metadata*.pkl"))
    
    if not cache_files:
        print(f"No metadata cache files found{f' for dataset {dataset}' if dataset else ''}.")
        return
    
    for cache_file in cache_files:
        cache_file.unlink()
        print(f"Removed cache file: {cache_file}")
    
    print(f"Cleared {len(cache_files)} metadata cache file(s).")


def setup_model_and_data(config: VisualizationConfig):
    """Set up model and data module based on configuration."""
    xenium_data_dir, segger_data_dir, model_dir_path = config.get_data_paths()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data module
    dm = SeggerDataModule(
        data_dir=segger_data_dir,
        batch_size=3,
        num_workers=2,
    )
    dm.setup()
    
    # Set up model
    if config.model_type == 'no_seq':
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
    
    # Auto-detect model version and checkpoint
    model_version = config.auto_detect_model_version(model_dir_path)
    model_path = model_dir_path / "segger_with_embeddings" / f"version_{model_version}"
    ckpt_path = config.find_checkpoint(model_path)
    
    # Load trained model
    ls = LitSegger(model=model).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        ls.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"Successfully loaded model from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    ls.eval()
    return ls, dm


def create_combined_dataloader(dm, batch_indices: Dict[str, List[int]]) -> List:
    """Create a combined dataloader from train/test/val using specified batch indices.
    
    Args:
        dm: SeggerDataModule instance
        batch_indices: Dictionary mapping dataset names to lists of batch indices
    
    Returns:
        List of combined batches from all datasets
    """
    combined_batches = []
    
    for dataset, indices in batch_indices.items():
        if not indices:
            continue
            
        dataloader = getattr(dm, dataset)
        print(f"Adding {len(indices)} batches from {dataset} dataset")
        
        for batch_idx in indices:
            if batch_idx < len(dataloader):
                combined_batches.append(dataloader[batch_idx])
            else:
                print(f"Warning: batch_idx {batch_idx} >= {dataset} dataset size {len(dataloader)}")
    
    print(f"Created combined dataloader with {len(combined_batches)} total batches")
    return combined_batches


def setup_wandb_key():
    """Configure Weights & Biases using the WANDB_API_KEY from environment.

    Expects the API key to be provided via the environment variable
    WANDB_API_KEY (e.g., export WANDB_API_KEY=... or set in a .env file).
    """
    try:
        import wandb
        # Load .env if present (current working dir and project root)
        if load_dotenv is not None:
            try:
                # Load from CWD first
                load_dotenv()
                # Also try the repository/project root
                load_dotenv(dotenv_path=project_root_path / ".env")
            except Exception:
                pass

        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            print("⚠️  WANDB_API_KEY not set. Skipping wandb login; falling back to CSV logging.")
            return False

        wandb.login(key=wandb_api_key)
        print("✓ Wandb configured successfully via environment variable")
        return True

    except ImportError:
        print("❌ Wandb not installed. Please install with: pip install wandb")
        return False
    except Exception as e:
        print(f"❌ Error setting up wandb: {e}")
        return False

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