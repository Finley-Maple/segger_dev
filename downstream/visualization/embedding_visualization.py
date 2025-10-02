"""
Embedding Visualization Module for Segger Model

This module provides tools for visualizing the final transcript ('tx') node embeddings from the Segger model
using dimensionality reduction techniques like UMAP, along with interactive Plotly dashboards
for synchronized selection across multiple visualization types.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pickle
import warnings
from tqdm import tqdm
from collections import defaultdict
try:
    from .utils import load_cell_type_color_palette
except ImportError:
    try:
        from utils import load_cell_type_color_palette
    except ImportError:
        warnings.warn("Could not import color palette utilities. Using default colors.")
        def load_cell_type_color_palette():
            return {}

try:
    import umap
except ImportError:
    warnings.warn("UMAP not installed. Please install with: pip install umap-learn")
    umap = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    warnings.warn("scikit-learn not installed. Please install with: pip install scikit-learn")
    TSNE = None
    PCA = None
    NearestNeighbors = None

try:
    import leidenalg
    import igraph as ig
except ImportError:
    warnings.warn("leidenalg or igraph not installed. Please install with: pip install leidenalg python-igraph")
    leidenalg = None
    ig = None

try:
    import scanpy as sc
    import scipy.sparse as sp
except ImportError:
    warnings.warn("scanpy not installed. Please install with: pip install scanpy")
    sc = None
    sp = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    warnings.warn("Plotly not installed. Please install with: pip install plotly")
    go = None
    make_subplots = None
    px = None

@dataclass
class EmbeddingVisualizationConfig:
    """Configuration for embedding visualization."""
    method: str = 'umap'  # 'umap', 'tsne', 'pca'
    n_components: int = 2
    random_state: int = 42
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    cmap: str = 'tab10'
    point_size: float = 3.0
    alpha: float = 0.7
    save_format: str = 'png'
    
    # UMAP-specific parameters
    umap_n_neighbors: int = 50
    umap_min_dist: float = 0.5
    metric: str = 'euclidean'
    
    # t-SNE-specific parameters
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    
    # Sampling for large datasets
    max_points_per_type: int = 10000
    subsample_method: str = 'random'  # 'random', 'balanced'
    max_points_interactive: int = 100000  # Max total points for interactive dashboard
    
    # Spatial visualization parameters
    spatial_max_points_per_gene_type: int = 1000  # Max points per gene type for spatial plots
    spatial_alpha: float = 0.6
    spatial_tx_size: float = 2.0
    
    # Jitter settings
    jitter_amount: float = 0.1
    


class EmbeddingExtractor:
    """Extracts and processes node embeddings from Segger model forward passes."""
    
    def __init__(self, device: torch.device = None):
        """
        Initialize the embedding extractor.
        
        Args:
            device: PyTorch device to use for computations
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def extract_batch_embeddings(self, 
                                model: torch.nn.Module, 
                                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from a single batch.
        
        Args:
            model: The trained Segger model (hetero)
            batch: Batch data containing x_dict and edge_index_dict
            
        Returns:
            Dictionary containing extracted embeddings for 'tx' nodes
        """
        model.eval()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            # Sanity check: verify edge_index dimensions
            for edge_type, edge_index in batch.edge_index_dict.items():
                if edge_index.dim() != 2:
                    print(f"Warning: Skipping batch due to invalid edge_index shape for {edge_type}: "
                          f"expected 2D, got {edge_index.dim()}D with shape {edge_index.shape}")
                    # print("edge_index", edge_index)
                    # print("batch", batch)
                    return {}
            
            # Forward pass to get final embeddings
            try:
                embeddings_dict, _ = model(batch.x_dict, batch.edge_index_dict)
            except Exception as e:
                print(f"Warning: Skipping batch due to model forward pass error: {e}")
                return {}
            
            # Extract embeddings for tx nodes only
            extracted_embeddings = {}
            if 'tx' in embeddings_dict:
                # Keep on device for now, move to CPU only when needed for final storage
                extracted_embeddings['tx'] = embeddings_dict['tx']
            # Add bd (nuclei) embedding extraction
            if 'bd' in embeddings_dict:
                # Keep on device for now, move to CPU only when needed for final storage
                extracted_embeddings['bd'] = embeddings_dict['bd']
                    
        return extracted_embeddings
    
    def extract_embeddings_from_batches(self,
                                      model: torch.nn.Module,
                                      dataloader,
                                      max_batches: Optional[int] = None,
                                      gene_names_dict: Optional[Dict] = None,
                                      cell_types_dict: Optional[Dict] = None,
                                      transcripts_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
        """
        Extract embeddings from multiple batches and aggregate metadata.
        
        Args:
            model: The trained Segger model
            dataloader: DataLoader containing batches
            max_batches: Maximum number of batches to process
            gene_names_dict: Mapping from transcript ID to gene name
            cell_types_dict: Mapping from cell ID to cell type
            transcripts_df: DataFrame containing transcript information
            
        Returns:
            Dictionary containing embeddings and metadata for visualization
        """
        model.eval()
        
        all_tx_embeddings = []
        all_tx_metadata = []
        all_bd_embeddings = []
        all_bd_metadata = []
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            # Extract embeddings for this batch
            batch_embeddings = self.extract_batch_embeddings(model, batch)
            
            # Skip batch if no valid embeddings extracted
            if not batch_embeddings:
                print(f"Skipping batch {batch_idx + 1}: no valid embeddings extracted")
                continue
            
            # Process tx (transcript) embeddings
            if 'tx' in batch_embeddings:
                tx_emb = batch_embeddings['tx']
                # Ensure embeddings are on the same device as indices/model for safe advanced indexing
                if tx_emb.device != self.device:
                    tx_emb = tx_emb.to(self.device)
                tx_ids = batch['tx'].id.cpu().numpy()
                tx_pos = batch['tx'].pos.cpu()  # Get spatial positions
                
                # Move to CPU for final storage to avoid memory issues
                all_tx_embeddings.append(tx_emb.cpu())
                
                # Add metadata for tx nodes
                for i, tx_id in enumerate(tx_ids):
                    metadata = {
                        'node_id': tx_id,
                        'node_type': 'tx',
                        'batch_idx': batch_idx,
                        'within_batch_idx': i,
                        'x': tx_pos[i, 0].item(),
                        'y': tx_pos[i, 1].item()
                    }
                    
                    # Add gene name if available
                    if gene_names_dict and tx_id in gene_names_dict:
                        metadata['gene_name'] = gene_names_dict[tx_id]
                    elif transcripts_df is not None:
                        gene_name = transcripts_df[transcripts_df['transcript_id'] == tx_id]['feature_name'].iloc[0] if len(transcripts_df[transcripts_df['transcript_id'] == tx_id]) > 0 else 'Unknown'
                        metadata['gene_name'] = gene_name
                    else:
                        metadata['gene_name'] = f'tx_{tx_id}'
                    
                    all_tx_metadata.append(metadata)
            
            # Add bd (nuclei) embeddings processing
            if 'bd' in batch_embeddings:
                bd_emb_full = batch_embeddings['bd']
                bd_ids = batch['bd'].id
                bd_pos = batch['bd'].pos.cpu()
                if len(bd_ids) > len(bd_emb_full):
                    print(f"Warning: Skipping batch {batch_idx + 1}: more bd ids than embeddings")
                    num_seed_bd = len(bd_ids)
                    bd_emb = bd_emb_full[:num_seed_bd]
                    print(f"Sliced bd embeddings from {len(bd_emb_full)} to {len(bd_emb)} and {bd_pos.shape} to match seed nodes")
                    # Move to CPU for final storage
                    all_bd_embeddings.append(bd_emb.cpu())
                else:
                    print(f"Batch {batch_idx + 1}: No need to slice bd embeddings")
                    # Move to CPU for final storage
                    all_bd_embeddings.append(bd_emb_full.cpu())
                for i, bd_id in enumerate(bd_ids):
                    metadata = {
                        'node_id': bd_id,
                        'node_type': 'bd',
                        'batch_idx': batch_idx,
                        'within_batch_idx': i,
                        'x': bd_pos[i, 0].item(),
                        'y': bd_pos[i, 1].item()
                    }
                    all_bd_metadata.append(metadata)
        
        # Concatenate all embeddings
        result = {}
        if all_tx_embeddings:
            result['tx'] = {
                'embeddings': torch.cat(all_tx_embeddings, dim=0),
                'metadata': pd.DataFrame(all_tx_metadata)
            }
        # bd (nuclei) embeddings in result
        if all_bd_embeddings:
            result['bd'] = {
                'embeddings': torch.cat(all_bd_embeddings, dim=0),
                'metadata': pd.DataFrame(all_bd_metadata)
            }
            
        return result
    
    def extract_gene_embeddings_from_batches(self,
                                           model: torch.nn.Module,
                                           dataloader,
                                           max_batches: Optional[int] = None,
                                           gene_names_dict: Optional[Dict] = None,
                                           transcripts_df: Optional[pd.DataFrame] = None) -> Dict[str, torch.Tensor]:
        """
        Memory-efficient extraction of gene embeddings by processing batches individually.
        
        Args:
            model: The trained Segger model
            dataloader: DataLoader containing batches
            max_batches: Maximum number of batches to process
            gene_names_dict: Mapping from transcript ID to gene name
            transcripts_df: DataFrame containing transcript information
            
        Returns:
            Dictionary containing aggregated gene embeddings and metadata
        """
        model.eval()
        
        # OPTIMIZATION: Pre-compute gene name lookup dictionary from DataFrame
        # This replaces the expensive pandas lookups with O(1) dictionary access
        if transcripts_df is not None and gene_names_dict is None:
            print("Creating gene name lookup dictionary from transcripts DataFrame...")
            gene_names_dict = dict(zip(transcripts_df['transcript_id'], transcripts_df['feature_name']))
            print(f"Created lookup dictionary with {len(gene_names_dict)} entries")
        
        # GPU OPTIMIZATION: Detect device and ensure consistent device usage
        device = next(model.parameters()).device
        print(f"Using device: {device} for gene embedding extraction")
        
        # GPU OPTIMIZATION: Print initial GPU memory stats
        if device.type == 'cuda':
            print(f"Initial GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved")
        
        # Dictionaries to accumulate gene embeddings across batches
        gene_embedding_sums = {}  # gene_name -> sum of embeddings
        gene_transcript_counts = {}  # gene_name -> total transcript count
        gene_metadata_accumulator = {}  # gene_name -> metadata dict
        
        print("Starting memory-efficient gene embedding extraction...")
        
        # GPU OPTIMIZATION: Use torch.no_grad() and CUDA memory management
        with torch.no_grad():  # Disable gradient computation for inference
            # replace with tqdm
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Extract transcript embeddings for this batch
                batch_embeddings = self.extract_batch_embeddings(model, batch)
                
                # Process tx (transcript) embeddings only
                if not batch_embeddings or 'tx' not in batch_embeddings:
                    print(f"Skipping batch {batch_idx + 1}: no valid embeddings extracted")
                    continue
                
                tx_emb = batch_embeddings['tx']
                # Ensure tx_emb is on the same device as the model for consistent processing
                if tx_emb.device != device:
                    tx_emb = tx_emb.to(device)
                # GPU OPTIMIZATION: Keep spatial positions on GPU, only transfer IDs once
                tx_ids = batch['tx'].id.cpu().numpy()
                tx_pos_gpu = batch['tx'].pos.to(device)  # Ensure on GPU
            
                # GPU OPTIMIZATION: Batch create gene names to minimize CPU operations
                batch_gene_names = []
                for tx_id in tx_ids:
                    if gene_names_dict and tx_id in gene_names_dict:
                        batch_gene_names.append(gene_names_dict[tx_id])
                    else:
                        batch_gene_names.append(f'tx_{tx_id}')
            
                # Create metadata for this batch (minimized CPU operations)
                batch_tx_metadata = []
                for i, (tx_id, gene_name) in enumerate(zip(tx_ids, batch_gene_names)):
                    metadata = {
                        'node_id': tx_id,
                        'gene_name': gene_name,
                        'batch_idx': i  # Store index for GPU tensor operations
                    }
                    batch_tx_metadata.append(metadata)
            
                # OPTIMIZATION: Use more efficient grouping with defaultdict
                batch_gene_groups = defaultdict(list)
                
                # Group transcripts by gene within this batch
                for i, meta in enumerate(batch_tx_metadata):
                    batch_gene_groups[meta['gene_name']].append((i, meta))
            
                # GPU OPTIMIZATION: Batch process all genes at once to reduce GPU kernel launches
                for gene_name, transcript_list in batch_gene_groups.items():
                    # GPU OPTIMIZATION: Create indices tensor directly on GPU device
                    indices = torch.tensor([i for i, _ in transcript_list], dtype=torch.long, device=device)
                    gene_embedding = tx_emb[indices].mean(dim=0)  # Average within batch
                    transcript_count = indices.numel()  # Use tensor operation instead of len()
                
                    # Accumulate across batches
                    if gene_name in gene_embedding_sums:
                        # Weighted sum: sum += embedding * count
                        gene_embedding_sums[gene_name] += gene_embedding * transcript_count
                        gene_transcript_counts[gene_name] += transcript_count
                    else:
                        # First time seeing this gene
                        gene_embedding_sums[gene_name] = gene_embedding * transcript_count
                        gene_transcript_counts[gene_name] = transcript_count
                        
                        # Initialize metadata (will be updated with averages)
                        gene_metadata_accumulator[gene_name] = {
                            'x_sum': 0.0,
                            'y_sum': 0.0,
                            'total_transcripts': 0
                        }
                
                    # GPU OPTIMIZATION: Use GPU tensor operations for spatial coordinates
                    # Extract spatial coordinates directly from GPU tensor using indices
                    gene_positions = tx_pos_gpu[indices]  # Shape: [num_transcripts_in_gene, 2]
                    x_sum = gene_positions[:, 0].sum().item()  # GPU sum, then single CPU transfer
                    y_sum = gene_positions[:, 1].sum().item()  # GPU sum, then single CPU transfer
                    
                    gene_metadata_accumulator[gene_name]['x_sum'] += x_sum
                    gene_metadata_accumulator[gene_name]['y_sum'] += y_sum
                    gene_metadata_accumulator[gene_name]['total_transcripts'] += transcript_count
                    
                    # GPU OPTIMIZATION: Clean up GPU tensors to free VRAM
                    del gene_positions
            
                # OPTIMIZATION: Clean up batch-level variables to free memory
                del batch_gene_groups, batch_tx_metadata, batch_gene_names
                
                # GPU OPTIMIZATION: Periodic GPU memory cleanup
                if device.type == 'cuda' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Final aggregation: compute averages
        # print(f"Finalizing gene embeddings for {len(gene_embedding_sums)} genes...")
        
        if len(gene_embedding_sums) == 0:
            print("Warning: No gene embeddings were successfully extracted from any batch")
            return {
                'embeddings': torch.empty((0, 0)),
                'metadata': pd.DataFrame()
            }
        
        gene_embeddings_list = []
        gene_metadata_list = []
        
        for gene_name in gene_embedding_sums.keys():
            # Average embedding across all batches
            final_embedding = gene_embedding_sums[gene_name] / gene_transcript_counts[gene_name]
            gene_embeddings_list.append(final_embedding)
            
            # Create final metadata
            meta_acc = gene_metadata_accumulator[gene_name]
            gene_meta = {
                'gene_name': gene_name,
                'transcript_count': gene_transcript_counts[gene_name],
                'x': meta_acc['x_sum'] / meta_acc['total_transcripts'],
                'y': meta_acc['y_sum'] / meta_acc['total_transcripts']
            }
            gene_metadata_list.append(gene_meta)
        
        # GPU OPTIMIZATION: Stack tensors on GPU then move to CPU for final storage
        final_gene_embeddings = torch.stack(gene_embeddings_list, dim=0)
        
        # Move final embeddings to CPU for storage and visualization
        if final_gene_embeddings.device != torch.device('cpu'):
            final_gene_embeddings = final_gene_embeddings.cpu()
        
        # GPU OPTIMIZATION: Clean up intermediate embedding list to free GPU memory
        del gene_embeddings_list
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        final_gene_metadata = pd.DataFrame(gene_metadata_list)
        
        # print(f"✅ Extracted gene embeddings: {final_gene_embeddings.shape}")
        print(f"   Total genes: {len(final_gene_metadata)}")
        print(f"   Average transcripts per gene: {final_gene_metadata['transcript_count'].mean():.1f}")
        print(f"   Transcript count range: {final_gene_metadata['transcript_count'].min()}-{final_gene_metadata['transcript_count'].max()}")
        
        # GPU OPTIMIZATION: Print final GPU memory stats
        if device.type == 'cuda':
            print(f"Final GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved")
        
        return {
            'embeddings': final_gene_embeddings,
            'metadata': final_gene_metadata
        }


class EmbeddingVisualizer:
    """Visualizes node embeddings using dimensionality reduction techniques."""
    
    def __init__(self, config: EmbeddingVisualizationConfig = None):
        """
        Initialize the embedding visualizer.
        
        Args:
            config: Configuration for visualization parameters
        """
        self.config = config or EmbeddingVisualizationConfig()
        # Store fitted reducers for consistent coordinates across epochs
        self.fitted_reducers = {}
        
        
    def _apply_dimensionality_reduction(self, 
                                      embeddings: torch.Tensor,
                                      method: str = None,
                                      node_type: str = None,
                                      fit_reducer: bool = False) -> np.ndarray:
        """
        Apply dimensionality reduction to embeddings.
        
        Args:
            embeddings: Input embeddings tensor
            method: Dimensionality reduction method ('umap', 'tsne', 'pca')
            node_type: Type of node ('tx' or 'bd') for storing fitted reducers
            fit_reducer: Whether to fit a new reducer (for reference epoch)
            
        Returns:
            Reduced embeddings as numpy array
        """
        method = method or self.config.method
        embeddings_np = embeddings.numpy()
        
        # If dimension of embedding is too large, use PCA first as preprocessing
        if embeddings_np.shape[1] > 50:
            print(f"Embedding dimension is too large ({embeddings_np.shape[1]}), using PCA as preprocessing...")
            reducer = PCA(
                n_components= 50,
                random_state=self.config.random_state
            )
            embeddings_np = reducer.fit_transform(embeddings_np)
        
        # Apply jitter for improved UMAP performance if enabled
        embeddings_np = self._apply_jitter(embeddings_np)
        
        # Create reducer key for storing fitted reducers
        reducer_key = f"{method}_{node_type}" if node_type else method
        
        # If we already have a fitted reducer and not explicitly fitting, use it
        if reducer_key in self.fitted_reducers and not fit_reducer:
            try:
                return self.fitted_reducers[reducer_key].transform(embeddings_np)
            except Exception as e:
                print(f"Warning: Failed to transform using fitted reducer: {e}")
                print("Falling back to fitting new reducer...")
        
        # Create new reducer
        if method == 'umap':
            if umap is None:
                raise ImportError("UMAP not available. Please install with: pip install umap-learn")
            reducer = umap.UMAP(
                n_components=self.config.n_components,
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                metric=self.config.metric,
                random_state=self.config.random_state
            )
        elif method == 'tsne':
            if TSNE is None:
                raise ImportError("t-SNE not available. Please install scikit-learn")
            reducer = TSNE(
                n_components=self.config.n_components,
                perplexity=self.config.tsne_perplexity,
                # n_iter=self.config.tsne_n_iter,
                random_state=self.config.random_state,
                metric=self.config.metric
            )
        elif method == 'pca':
            if PCA is None:
                raise ImportError("PCA not available. Please install scikit-learn")
            reducer = PCA(
                n_components=self.config.n_components,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Fit and transform with error handling for UMAP spectral initialization
        try:
            reduced_embeddings = reducer.fit_transform(embeddings_np)
        except Exception as e:
            if method == 'umap' and 'spectral' in str(e).lower():
                print(f"Warning: UMAP spectral initialization failed: {e}")
                print("Retrying with random initialization...")
                reducer = umap.UMAP(
                    n_components=self.config.n_components,
                    n_neighbors=self.config.umap_n_neighbors,
                    min_dist=self.config.umap_min_dist,
                    metric=self.config.metric,
                    random_state=self.config.random_state,
                    init='random'
                )
                reduced_embeddings = reducer.fit_transform(embeddings_np)
            else:
                raise e
        
        # Store fitted reducer if requested
        if fit_reducer and reducer_key:
            self.fitted_reducers[reducer_key] = reducer
            
        return reduced_embeddings
    
    def _apply_jitter(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply jitter to embeddings for improved UMAP performance.
        
        Args:
            embeddings: Input embeddings numpy array
    
        Returns:
            Jittered embeddings numpy array
        """
        if self.config.jitter_amount > 0:
            return embeddings + np.random.normal(0, self.config.jitter_amount, embeddings.shape)
        return embeddings
    
    def _apply_clustering(self, raw_embeddings: np.ndarray, n_neighbors: int = 15, n_clusters: int = [12]) -> np.ndarray:
        """
        Apply clustering to raw embeddings using cosine metric via graph-based spectral clustering.
        
        Args:
            raw_embeddings: Raw embedding vectors (n_samples, n_features)
            n_neighbors: Number of neighbors for KNN graph construction
            n_clusters: Number of clusters to find
            
        Returns:
            Array of cluster labels
        """
        # Convert to numpy array if it's a torch tensor
        if hasattr(raw_embeddings, 'detach'):
            raw_embeddings = raw_embeddings.detach().cpu().numpy()
        elif hasattr(raw_embeddings, 'numpy'):
            raw_embeddings = raw_embeddings.numpy()
        
        n_samples, n_features = raw_embeddings.shape
        print(f"Processing {n_samples} points ({n_features} dims) for clustering...")
        
        try:
            from sklearn.cluster import SpectralClustering, MiniBatchKMeans
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import normalize
            from sklearn.metrics import silhouette_score
            
            # L2-normalize for cosine similarity equivalence
            normalized_embeddings = normalize(raw_embeddings, norm='l2', axis=1)
            
            # Optional PCA if features high
            if n_features > 50:
                n_components = min(50, n_features)
                pca = PCA(n_components=n_components, random_state=42)
                reduced_embeddings = pca.fit_transform(normalized_embeddings)
                print(f"Reduced from {n_features} to {n_components} dims via PCA.")
            else:
                reduced_embeddings = normalized_embeddings
                print("Skipping PCA as dims are low.")
            
            # Handle single vs multiple n_clusters
            if not isinstance(n_clusters, list):
                n_clusters_list = [n_clusters]
            else:
                n_clusters_list = n_clusters
            results = {}
            for n in n_clusters_list:
                kmeans = MiniBatchKMeans(
                    n_clusters=n,
                    random_state=42,
                    batch_size=max(1000, n_samples // 100),  # Adaptive for large data
                    max_iter=100,  # Increased for better convergence
                    init='k-means++'
                )
                labels = kmeans.fit_predict(reduced_embeddings)
                actual_n = len(set(labels))
                # Silhouette on subsample for speed (cosine metric)
                sil = silhouette_score(reduced_embeddings, labels, metric='cosine', sample_size=min(10000, n_samples), random_state=42) if actual_n > 1 else 0.0
                results[n] = {'labels': labels, 'n_clusters': actual_n, 'silhouette': sil}
                print(f"MiniBatchKMeans (n={n}): {n_samples} points -> {actual_n} clusters, silhouette={sil:.4f}")
            
            return results[n_clusters_list[0]]['labels']
        
        except Exception as e:
            print(f"Critical error in clustering: {e}. Returning dummy labels.")
            return np.full(n_samples, -1, dtype=int)  # Use -1 for unassigned instead of zeros
    
    def _aggregate_embeddings_by_gene(self, 
                                     embeddings: torch.Tensor, 
                                     metadata: pd.DataFrame) -> Tuple[torch.Tensor, pd.DataFrame]:
        """
        Aggregate transcript embeddings by gene name to create gene-level embeddings.
        
        Args:
            embeddings: Transcript embeddings tensor (n_transcripts, embedding_dim)
            metadata: Transcript metadata containing gene_name and gene_type columns
            
        Returns:
            Tuple of (gene_embeddings, gene_metadata)
        """
        if 'gene_name' not in metadata.columns:
            raise ValueError("gene_name column not found in metadata")
        
        # Group by gene name and calculate mean embeddings
        gene_embeddings_list = []
        gene_metadata_list = []
        
        for gene_name, group in metadata.groupby('gene_name'):
            # Get indices for this gene
            gene_indices = group.index.tolist()
            
            # Average embeddings for this gene
            gene_embedding = embeddings[gene_indices].mean(dim=0)
            gene_embeddings_list.append(gene_embedding)
            
            # Create metadata for this gene
            gene_meta = {
                'gene_name': gene_name,
                'transcript_count': len(gene_indices),
            }
            
            # Get gene type (should be the same for all transcripts of the same gene)
            if 'gene_type' in group.columns:
                gene_types = group['gene_type'].dropna().unique()
                if len(gene_types) > 0:
                    gene_meta['gene_type'] = gene_types[0]  # Take first non-NaN gene type
                else:
                    gene_meta['gene_type'] = 'Unknown'
            else:
                gene_meta['gene_type'] = 'Unknown'
            
            # Add spatial information if available (average coordinates)
            if 'x' in group.columns and 'y' in group.columns:
                gene_meta['x'] = group['x'].mean()
                gene_meta['y'] = group['y'].mean()
            
            gene_metadata_list.append(gene_meta)
        
        # Convert to tensors and DataFrame
        gene_embeddings = torch.stack(gene_embeddings_list, dim=0)
        gene_metadata = pd.DataFrame(gene_metadata_list)
        
        print(f"Aggregated {len(metadata)} transcripts into {len(gene_metadata)} genes")
        print(f"Average transcripts per gene: {gene_metadata['transcript_count'].mean():.1f}")
        
        return gene_embeddings, gene_metadata
    
    def fit_reference_reducers(self, reference_embeddings_data: Dict[str, Dict]) -> None:
        """
        Fit dimensionality reducers on reference embeddings for consistent coordinates.
        
        Args:
            reference_embeddings_data: Reference embeddings data (typically from final epoch)
        """
        print("Fitting reference reducers for consistent coordinates across epochs...")
        
        for node_type, data in reference_embeddings_data.items():
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            print(f"Fitting reducer for {node_type} nodes ({len(embeddings)} nodes)...")
            
            # For gene embeddings, we might need to filter or aggregate first
            if node_type == 'gene':
                # Gene embeddings are already aggregated, use them directly
                self._apply_dimensionality_reduction(
                    embeddings, 
                    node_type=node_type, 
                    fit_reducer=True
                )
            elif node_type == 'tx' and 'gene_name' in metadata.columns:
                # For transcript embeddings, also fit a gene reducer based on aggregated data
                try:
                    gene_embeddings, gene_metadata = self._aggregate_embeddings_by_gene(
                        embeddings, metadata
                    )
                    print(f"Fitting gene reducer from {len(gene_embeddings)} aggregated genes...")
                    self._apply_dimensionality_reduction(
                        gene_embeddings, 
                        node_type='gene', 
                        fit_reducer=True
                    )
                except Exception as e:
                    print(f"Warning: Could not fit gene reducer from transcript data: {e}")
                
                # Also fit the transcript reducer
                self._apply_dimensionality_reduction(
                    embeddings, 
                    node_type=node_type, 
                    fit_reducer=True
                )
            else:
                # Standard fitting for other node types
                self._apply_dimensionality_reduction(
                    embeddings, 
                    node_type=node_type, 
                    fit_reducer=True
                )
    
    def visualize_embeddings(self,
                           embeddings_data: Dict[str, Dict],
                           save_dir: Path,
                           title_prefix: str = "",
                           gene_types_dict: Optional[Dict] = None,
                           create_interactive_plots: bool = True) -> Dict[str, str]:
        """
        Create visualization plots for node embeddings.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata
            save_dir: Directory to save plots
            title_prefix: Prefix for plot titles
            gene_types_dict: Mapping from gene name to gene type
            create_interactive_plots: Whether to create interactive Plotly dashboards
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        # If a Weights & Biases run is active, redirect saves to the run directory
        try:
            import wandb  # type: ignore
            if getattr(wandb, "run", None) is not None and getattr(wandb.run, "dir", None):
                save_dir = Path(wandb.run.dir) / 'embeddings_plots'
        except Exception:
            pass

        # If a Weights & Biases run is active, redirect saves to the run directory
        try:
            import wandb  # type: ignore
            if getattr(wandb, "run", None) is not None and getattr(wandb.run, "dir", None):
                save_dir = Path(wandb.run.dir) / 'embeddings_plots'
        except Exception:
            pass

        save_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = {}
        
        # Create interactive dashboard if tx data is available and requested
        if create_interactive_plots and 'tx' in embeddings_data:
            # For interactive dashboard, we need spatial data, so extract it if not provided
            if 'x' in embeddings_data['tx']['metadata'].columns:
                # Use embeddings_data as spatial_data since it contains spatial coordinates
                spatial_data_for_dashboard = {
                    'tx': {
                        'positions': torch.column_stack([
                            torch.tensor(embeddings_data['tx']['metadata']['x'].values),
                            torch.tensor(embeddings_data['tx']['metadata']['y'].values)
                        ]),
                        'metadata': embeddings_data['tx']['metadata']
                    }
                }
                
                # Adding bd nodes (nuclei) to spatial data
                if 'bd' in embeddings_data and 'x' in embeddings_data['bd']['metadata'].columns:
                    spatial_data_for_dashboard['bd'] = {
                        'positions': torch.column_stack([
                            torch.tensor(embeddings_data['bd']['metadata']['x'].values),
                            torch.tensor(embeddings_data['bd']['metadata']['y'].values)
                        ]),
                        'metadata': embeddings_data['bd']['metadata']
                }
                
                # Pre-compute reduced embeddings once to ensure consistency between dashboards
                print("Computing reduced embeddings for consistent visualization...")
                tx_embeddings = embeddings_data['tx']['embeddings']
                precomputed = None
                if tx_embeddings.sum() != 0:  # Check if embeddings are not all zeros (dummy)
                    # If bd embeddings exist and share dimensionality, compute combined reduction
                    if 'bd' in embeddings_data and 'embeddings' in embeddings_data['bd'] and len(embeddings_data['bd']['embeddings']) > 0:
                        print("Computing combined reduced embeddings for tx and bd...")
                        bd_embeddings = embeddings_data['bd']['embeddings']
                        # Sanity check: same dimensionality
                        if tx_embeddings.shape[1] == bd_embeddings.shape[1]:
                            combined = torch.cat([tx_embeddings, bd_embeddings], dim=0)
                            combined_reduced = self._apply_dimensionality_reduction(combined, node_type='tx')
                            n_tx = tx_embeddings.shape[0]
                            precomputed = {
                                'tx': combined_reduced[:n_tx, :],
                                'bd': combined_reduced[n_tx:, :]
                            }
                        else:
                            print(f"Warning: tx and bd embedding dims differ (tx={tx_embeddings.shape[1]}, bd={bd_embeddings.shape[1]}). Skipping combined reduction.")
                            precomputed = self._apply_dimensionality_reduction(tx_embeddings, node_type='tx')
                    else:
                        precomputed = self._apply_dimensionality_reduction(tx_embeddings, node_type='tx')
                else:
                    # For spatial-only data, use spatial coordinates as "embeddings"
                    x_coords = embeddings_data['tx']['metadata']['x'].values
                    y_coords = embeddings_data['tx']['metadata']['y'].values
                    precomputed = np.column_stack([x_coords, y_coords])
                
                # Create gene type dashboard
                gene_type_dashboard_path = self.create_interactive_dashboard(
                    embeddings_data=embeddings_data,
                    spatial_data=spatial_data_for_dashboard,
                    save_dir=save_dir,
                    title_prefix=title_prefix,
                    gene_types_dict=gene_types_dict,
                    dashboard_type='gene_type',
                    precomputed_reduced_embeddings=precomputed
                )
                if gene_type_dashboard_path:
                    saved_plots['gene_type_dashboard'] = gene_type_dashboard_path
                
                # Create cluster dashboard
                cluster_dashboard_path = self.create_interactive_dashboard(
                    embeddings_data=embeddings_data,
                    spatial_data=spatial_data_for_dashboard,
                    save_dir=save_dir,
                    title_prefix=title_prefix,
                    gene_types_dict=gene_types_dict,
                    dashboard_type='cluster',
                    precomputed_reduced_embeddings=precomputed
                )
                if cluster_dashboard_path:
                    saved_plots['cluster_dashboard'] = cluster_dashboard_path
            
            
        return saved_plots

    def _create_gene_level_plots(self,
                                gene_embeddings: torch.Tensor,
                                gene_metadata: pd.DataFrame,
                                save_dir: Path,
                                title_prefix: str,
                                min_transcript_count: int = 1,
                                exclude_unknown: bool = False) -> Dict[str, str]:
        """Create plots for gene-level embeddings with gene names as labels."""
        plots = {}
        
        # Apply gene filtering
        print(f"Original gene count: {len(gene_metadata)}")
        
        # Filter by minimum transcript count
        transcript_count_mask = gene_metadata['transcript_count'] >= min_transcript_count
        
        # Filter by excluding unknown genes if requested
        if exclude_unknown:
            unknown_mask = ~gene_metadata['gene_type'].str.contains('Unknown', case=False, na=False)
        else:
            unknown_mask = pd.Series([True] * len(gene_metadata), index=gene_metadata.index)
        
        # Combine filters
        final_mask = transcript_count_mask & unknown_mask
        
        if final_mask.sum() == 0:
            print("Warning: No genes passed filtering criteria")
            return plots
        
        # Apply filtering
        filtered_gene_metadata = gene_metadata[final_mask].reset_index(drop=True)
        
        print(f"Filtered gene count: {len(filtered_gene_metadata)} "
              f"(min_transcript_count={min_transcript_count}, exclude_unknown={exclude_unknown})")
        
        # Apply dimensionality reduction to the whole gene embeddings with reference coordinates and then filter the embeddings
        reduced_embeddings = self._apply_dimensionality_reduction(
            gene_embeddings, 
            node_type='gene',
            fit_reducer=False  # Use existing reducer if available for consistency
        )
        
        reduced_embeddings = reduced_embeddings[final_mask]
        
        # Create plot colored by gene type with gene name labels
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Get unique gene types from filtered data
        gene_types = sorted(filtered_gene_metadata['gene_type'].unique())
        
        # Load cell type color palette from Excel file
        cell_type_color_palette = load_cell_type_color_palette()
        
        # Fallback colors for additional gene types not in the palette
        fallback_colors = [
            '#999999',  # Gray
            '#1f77b4',  # Blue2
            '#ff7f0e',  # Orange2
            '#2ca02c',  # Green2
            '#d62728',  # Red2
            '#e377c2',  # Pink2
            '#7f7f7f',  # Gray2
            '#bcbd22',  # Olive
            '#aec7e8'   # Light Blue
        ]
        
        # Assign colors: use specific palette first, then fallback colors
        colors = []
        fallback_index = 0
        for gene_type in gene_types:
            if gene_type in cell_type_color_palette:
                colors.append(cell_type_color_palette[gene_type])
            else:
                colors.append(fallback_colors[fallback_index % len(fallback_colors)])
                fallback_index += 1
        
        # Add gene name labels with flexible positioning to avoid overlap
        from adjustText import adjust_text
        texts = []
        
        # Plot each gene type
        for gene_type, color in zip(gene_types, colors):
            mask = filtered_gene_metadata['gene_type'] == gene_type
            gene_subset = filtered_gene_metadata[mask]
            embeddings_subset = reduced_embeddings[mask]
            
            # Scatter plot
            ax.scatter(embeddings_subset[:, 0], embeddings_subset[:, 1],
                      c=color, label=gene_type, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add gene name labels
            for i, (_, gene_row) in enumerate(gene_subset.iterrows()):
                text = ax.annotate(gene_row['gene_name'], 
                                 (embeddings_subset[i, 0], embeddings_subset[i, 1]),
                                 fontsize=8, alpha=0.8,
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
                texts.append(text)
        
        # Adjust text positions to avoid overlap
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5))
        
        ax.set_xlabel(f'{self.config.method.upper()} 1')
        ax.set_ylabel(f'{self.config.method.upper()} 2')
        ax.set_title(f'{title_prefix}Gene Embeddings by Gene Type ({len(filtered_gene_metadata)} genes)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, markerscale=1.5)
        # Remove grid lines as requested
        ax.grid(False)
        
        plt.tight_layout()
        plot_path = save_dir / f'gene_embeddings_by_type.{self.config.save_format}'
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        plots['gene_embeddings_by_type'] = str(plot_path)
        
        print(f"Created gene-level plot with {len(filtered_gene_metadata)} genes")
        
        return plots
    
    def create_gene_level_plots(self,
                               embeddings_data: Dict[str, Dict],
                               save_dir: Path,
                               title_prefix: str = "",
                               gene_types_dict: Optional[Dict] = None,
                               min_transcript_count: int = 1,
                               exclude_unknown: bool = True) -> Dict[str, str]:
        """
        Create gene-level plots from embeddings data that may include pre-computed gene embeddings.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata (may include 'gene' key)
            save_dir: Directory to save plots
            title_prefix: Prefix for plot titles
            gene_types_dict: Mapping from gene name to gene type
            min_transcript_count: Minimum transcript count for genes to include
            exclude_unknown: Whether to exclude unknown genes
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        plots = {}
        
        # Check if we have pre-computed gene embeddings
        if 'gene' in embeddings_data:
            print("Using pre-computed gene embeddings for visualization...")
            gene_data = embeddings_data['gene']
            gene_embeddings = gene_data['embeddings']
            gene_metadata = gene_data['metadata'].copy()
            
            # Add gene types if available
            if gene_types_dict and 'gene_name' in gene_metadata.columns:
                gene_metadata['gene_type'] = gene_metadata['gene_name'].map(gene_types_dict)
                
                # Fill NA values
                is_negative_control = gene_metadata['gene_name'].str.contains('BLANK|Neg', case=False, na=False)
                gene_metadata['gene_type'] = np.where(
                    gene_metadata['gene_type'].isna() & ~is_negative_control,
                    'Unknown',
                    gene_metadata['gene_type']
                )
            
            # Create gene-level plots
            if ('gene_name' in gene_metadata.columns and 
                'gene_type' in gene_metadata.columns):
                
                gene_plots = self._create_gene_level_plots(
                    gene_embeddings=gene_embeddings,
                    gene_metadata=gene_metadata,
                    save_dir=save_dir,
                    title_prefix=title_prefix,
                    min_transcript_count=min_transcript_count,
                    exclude_unknown=exclude_unknown
                )
                plots.update(gene_plots)
            else:
                print("Warning: Missing gene information for gene-level plots")
        
        # Fallback: aggregate from transcript embeddings if available
        elif 'tx' in embeddings_data:
            print("Aggregating gene embeddings from transcript data...")
            tx_data = embeddings_data['tx']
            tx_embeddings = tx_data['embeddings']
            tx_metadata = tx_data['metadata'].copy()
            
            # Add gene types if available
            if gene_types_dict and 'gene_name' in tx_metadata.columns:
                tx_metadata['gene_type'] = tx_metadata['gene_name'].map(gene_types_dict)
                
                # Fill NA values
                is_negative_control = tx_metadata['gene_name'].str.contains('BLANK|Neg', case=False, na=False)
                tx_metadata['gene_type'] = np.where(
                    tx_metadata['gene_type'].isna() & ~is_negative_control,
                    'Unknown',
                    tx_metadata['gene_type']
                )
            
            # Only create gene-level plots if we have gene information and non-dummy embeddings
            if ('gene_name' in tx_metadata.columns and 
                'gene_type' in tx_metadata.columns and 
                tx_embeddings.sum() != 0):  # Check if embeddings are not all zeros (dummy)
                
                # Aggregate transcript embeddings by gene
                gene_embeddings, gene_metadata = self._aggregate_embeddings_by_gene(
                    tx_embeddings, tx_metadata
                )
                
                # Create gene-level plots
                gene_plots = self._create_gene_level_plots(
                    gene_embeddings=gene_embeddings,
                    gene_metadata=gene_metadata,
                    save_dir=save_dir,
                    title_prefix=title_prefix,
                    min_transcript_count=min_transcript_count,
                    exclude_unknown=exclude_unknown
                )
                plots.update(gene_plots)
            else:
                print("Warning: Missing gene information or dummy embeddings for gene-level plots")
        else:
            print("Warning: No suitable data found for gene-level plots")
        
        return plots
    
    def create_interactive_dashboard(self,
                                   embeddings_data: Dict[str, Dict],
                                   spatial_data: Dict[str, Dict],
                                   save_dir: Path,
                                   title_prefix: str = "",
                                   gene_types_dict: Optional[Dict] = None,
                                   epoch_data: Optional[Dict[str, Dict[str, Dict]]] = None,
                                   dashboard_type: str = 'gene_type',
                                   precomputed_reduced_embeddings: Optional[np.ndarray] = None) -> str:
        """
        Create an interactive Plotly dashboard with two synchronized plots.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata
            spatial_data: Dictionary containing spatial coordinates and metadata
            save_dir: Directory to save the interactive plot
            title_prefix: Prefix for plot titles
            gene_types_dict: Mapping from gene name to gene type
            epoch_data: Optional dict with format {epoch_name: {embeddings_data: {...}, spatial_data: {...}}}
                       for multi-epoch visualization
            dashboard_type: Type of dashboard ('gene_type' or 'cluster')
            
        Returns:
            Path to the saved interactive HTML file
        """
        if go is None or make_subplots is None:
            raise ImportError("Plotly not available. Please install with: pip install plotly")
        
        # Check if we have multi-epoch data or single epoch data
        if epoch_data is not None:
            # Multi-epoch mode not supported - fallback to single epoch
            print("Warning: Multi-epoch mode not supported. Using first epoch only.")
            if epoch_data:
                first_epoch_key = next(iter(epoch_data.keys()))
                embeddings_data = epoch_data[first_epoch_key]['embeddings_data']
                spatial_data = epoch_data[first_epoch_key]['spatial_data']
        
        # Single epoch mode (existing functionality)
        # Only process tx nodes
        if 'tx' not in embeddings_data:
            print("Warning: No tx embeddings data available for interactive dashboard")
            return ""
            
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get embeddings data
        embeddings = embeddings_data['tx']['embeddings']
        metadata = embeddings_data['tx']['metadata'].copy()
        has_bd = 'bd' in embeddings_data and 'embeddings' in embeddings_data['bd'] and len(embeddings_data['bd']['embeddings']) > 0
        has_bd = False
        bd_embeddings = None
        bd_metadata = None
        
        # Get spatial coordinates
        x_coords = metadata['x'].values
        y_coords = metadata['y'].values
        
        # Use precomputed reduced embeddings if provided, otherwise compute them
        reduced_embeddings = None
        reduced_embeddings_bd = None
        if precomputed_reduced_embeddings is not None:
            # precomputed can be either ndarray (tx only)
            if isinstance(precomputed_reduced_embeddings, dict):
                reduced_embeddings = precomputed_reduced_embeddings.get('tx')
            else:
                reduced_embeddings = precomputed_reduced_embeddings
                print(f"Using precomputed reduced embeddings: {reduced_embeddings.shape}")
        else:
            # Apply dimensionality reduction for embedding plots (only if embeddings are not dummy)
            if embeddings.sum() != 0:  # Check if embeddings are not all zeros (dummy)
                reduced_embeddings = self._apply_dimensionality_reduction(embeddings, node_type='tx')
            else:
                # For spatial-only data, use spatial coordinates as "embeddings"
                reduced_embeddings = np.column_stack([x_coords, y_coords])

        # Disable any bd reduced embeddings computation
        
        # Apply scanpy clustering only for cluster dashboard using raw embeddings
        if dashboard_type == 'cluster':
            # apply clustering to the embeddings of the known tx nodes (without 'Unknown' gene types)
            known_embeddings = embeddings[metadata['gene_type'] != 'Unknown']            
            cluster_labels = self._apply_clustering(known_embeddings)
            metadata.loc[metadata['gene_type'] != 'Unknown', 'cluster'] = cluster_labels
            metadata.loc[metadata['gene_type'] == 'Unknown', 'cluster'] = 'Unknown'
        else:
            cluster_labels = None
        
        # Calculate counts for subplot titles
        total_transcripts = len(metadata)
        gene_type_transcripts = metadata['gene_type'].notna().sum() if 'gene_type' in metadata.columns else 0
        nuclei_count = len(spatial_data['bd']['metadata']) if 'bd' in spatial_data else 0 
        n_clusters = len(set(cluster_labels)) if cluster_labels is not None else 0
        
        # Create subplot figure with 1 row, 2 columns (16:9 optimized)
        if dashboard_type == 'gene_type':
            subplot_titles = [
                f'{title_prefix}Embeddings by Gene Type ({gene_type_transcripts:,} transcripts)',
                f'{title_prefix}Spatial by Gene Type ({total_transcripts:,} transcripts' + (f', {nuclei_count:,} nuclei' if nuclei_count > 0 else '') + ')'
            ]
        else:  # cluster
            subplot_titles = [
                f'{title_prefix}Embeddings by Cluster ({n_clusters} clusters)',
                f'{title_prefix}Spatial by Cluster ({n_clusters} clusters' + (f', {nuclei_count:,} nuclei' if nuclei_count > 0 else '') + ')'
            ]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08
        )
        
        print("gene types: ", metadata['gene_type'].unique())
        # Prepare colors for gene types using the same palette as static plots
        # Filter gene types to only include valid ones (not NA) for consistent coloring
        if 'gene_type' in metadata.columns:
            valid_gene_type_mask = metadata['gene_type'].notna()
            gene_types = sorted(metadata[valid_gene_type_mask]['gene_type'].unique()) if valid_gene_type_mask.sum() > 0 else []
        else:
            gene_types = []
        
        # Load cell type color palette from Excel file
        cell_type_color_palette = load_cell_type_color_palette()
        
        # Fallback colors for additional gene types not in the palette
        fallback_colors = [
            '#999999', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#e377c2', '#7f7f7f', '#bcbd22', '#aec7e8'
        ]
        
        # Assign colors: use specific palette first, then fallback colors
        gene_type_colors = {}
        fallback_index = 0
        for gene_type in gene_types:
            if gene_type in cell_type_color_palette:
                gene_type_colors[gene_type] = cell_type_color_palette[gene_type]
            else:
                gene_type_colors[gene_type] = fallback_colors[fallback_index % len(fallback_colors)]
                print(f"Fallback color for gene type {gene_type}: {gene_type_colors[gene_type]}")
                fallback_index += 1
        
        # Prepare cluster colors using discrete palette
        cluster_colors = {}
        unique_clusters = []
        if cluster_labels is not None:
            unique_clusters = sorted(set(cluster_labels))
            if px is not None:
                # Use plotly's discrete color sequence
                color_sequence = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Set1
            else:
                # Fallback colors
                color_sequence = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5']
            
            for i, cluster in enumerate(unique_clusters):
                cluster_colors[cluster] = color_sequence[i % len(color_sequence)]
        
        # Create unique IDs for each transcript for selection synchronization
        transcript_ids = list(range(len(metadata)))
        # Prepare nuclei ids right after transcript ids for consistent selection space
        bd_ids_for_embedding = []
        if has_bd and bd_metadata is not None:
            bd_ids_for_embedding = list(range(len(transcript_ids), len(transcript_ids) + len(bd_metadata)))
        
        # Create plots based on dashboard type
        if dashboard_type == 'gene_type':
            # Plot 1: Embeddings by Gene Type
            if gene_types and 'gene_type' in metadata.columns:
                # Apply gene type filtering as the final step to match static plots
                valid_gene_type_mask = metadata['gene_type'].notna()
                
                for gene_type in gene_types:
                    # Only include transcripts with valid gene types (not NA)
                    mask = (metadata['gene_type'] == gene_type) & valid_gene_type_mask
                    if mask.any():
                        indices = np.where(mask)[0]
                        # Validate indices to prevent out-of-bounds errors
                        valid_indices = [i for i in indices if i < len(metadata) and i < len(transcript_ids) and i < len(x_coords) and i < len(y_coords)]
                        if len(valid_indices) == 0:
                            continue
                            
                        # Create valid mask for the valid indices
                        valid_mask = np.zeros(len(metadata), dtype=bool)
                        valid_mask[valid_indices] = True
                        
                        fig.add_trace(
                            go.Scatter(
                                x=reduced_embeddings[valid_mask, 0],
                                y=reduced_embeddings[valid_mask, 1],
                                mode='markers',
                                marker=dict(
                                    color=gene_type_colors[gene_type],
                                    size=5,
                                    opacity=0.7
                                ),
                                name=gene_type,
                                text=[f"ID: {transcript_ids[i]}<br>Gene: {metadata.iloc[i]['gene_name']}<br>Type: {gene_type}<br>X: {x_coords[i]:.1f}<br>Y: {y_coords[i]:.1f}" 
                                      for i in valid_indices],
                                hovertemplate='%{text}<extra></extra>',
                                customdata=valid_indices,  # Store indices for selection sync
                                legendgroup=gene_type,
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                # Nuclei in embedding space
                if has_bd and reduced_embeddings_bd is not None and len(bd_ids_for_embedding) == len(reduced_embeddings_bd):
                    fig.add_trace(
                        go.Scatter(
                            x=reduced_embeddings_bd[:, 0],
                            y=reduced_embeddings_bd[:, 1],
                            mode='markers',
                            marker=dict(
                                color='black',
                                size=8,
                                opacity=0.8,
                                symbol='circle'
                            ),
                            name='Nuclei',
                            text=[f"Nucleus ID: {bd_metadata.iloc[i]['node_id']}" for i in range(len(bd_metadata))],
                            hovertemplate='%{text}<extra></extra>',
                            customdata=bd_ids_for_embedding,
                            showlegend=True,
                            legendgroup='nuclei'
                        ),
                        row=1, col=1
                    )
        
        else:  # cluster dashboard
            # Plot 1: Embeddings by Cluster
            for cluster in unique_clusters:
                cluster_mask = metadata['cluster'] == cluster
                if cluster_mask.any():
                    indices = np.where(cluster_mask)[0]
                    valid_indices = [i for i in indices if i < len(metadata)]
                    if len(valid_indices) == 0:
                        continue
                        
                    fig.add_trace(
                        go.Scatter(
                            x=reduced_embeddings[cluster_mask, 0],
                            y=reduced_embeddings[cluster_mask, 1],
                            mode='markers',
                            marker=dict(
                                color=cluster_colors[cluster],
                                size=5,
                                opacity=0.7
                            ),
                            name=f'Cluster {cluster}',
                            text=[f"ID: {transcript_ids[i]}<br>Cluster: {cluster}<br>Gene: {metadata.iloc[i]['gene_name']}<br>X: {x_coords[i]:.1f}<br>Y: {y_coords[i]:.1f}" 
                                  for i in valid_indices],
                            hovertemplate='%{text}<extra></extra>',
                            customdata=valid_indices,
                            legendgroup=f'cluster_{cluster}',
                            showlegend=True
                        ),
                        row=1, col=1
                    )
            # Nuclei in embedding space for cluster dashboard
            if has_bd and reduced_embeddings_bd is not None and len(bd_ids_for_embedding) == len(reduced_embeddings_bd):
                fig.add_trace(
                    go.Scatter(
                        x=reduced_embeddings_bd[:, 0],
                        y=reduced_embeddings_bd[:, 1],
                        mode='markers',
                        marker=dict(
                            color='black',
                            size=8,
                            opacity=0.8,
                            symbol='circle'
                        ),
                        name='Nuclei',
                        text=[f"Nucleus ID: {bd_metadata.iloc[i]['node_id']}" for i in range(len(bd_metadata))],
                        hovertemplate='%{text}<extra></extra>',
                        customdata=bd_ids_for_embedding,
                        showlegend=True,
                        legendgroup='nuclei'
                    ),
                    row=1, col=1
                )
        # Plot 2: Spatial Distribution (conditional based on dashboard type)
        if dashboard_type == 'gene_type':
            # Spatial Distribution colored by Gene Type
            if gene_types and 'gene_type' in metadata.columns:
                valid_gene_type_mask = metadata['gene_type'].notna()
                for gene_type in gene_types:
                    mask = (metadata['gene_type'] == gene_type) & valid_gene_type_mask
                    if mask.any():
                        indices = np.where(mask)[0]
                        valid_indices = [i for i in indices if i < len(metadata)]
                        if len(valid_indices) == 0:
                            continue
                        fig.add_trace(
                            go.Scatter(
                                x=x_coords[valid_indices],
                                y=y_coords[valid_indices],
                                mode='markers',
                                marker=dict(
                                    color=gene_type_colors[gene_type],
                                    size=5,
                                    opacity=0.7
                                ),
                                name=gene_type,
                                text=[f"ID: {transcript_ids[i]}<br>Gene: {metadata.iloc[i]['gene_name']}<br>Type: {gene_type}<br>X: {x_coords[i]:.1f}<br>Y: {y_coords[i]:.1f}" for i in valid_indices],
                                hovertemplate='%{text}<extra></extra>',
                                customdata=valid_indices,
                                legendgroup=gene_type,
                                showlegend=False
                            ),
                            row=1, col=2
                        )
            else:
                # Fallback: single trace when gene types are unavailable
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        marker=dict(
                            color='#999999',
                            size=5,
                            opacity=0.7
                        ),
                        name='Spatial Distribution',
                        text=[f"ID: {transcript_ids[i]}<br>Gene: {metadata.iloc[i]['gene_name']}<br>X: {x_coords[i]:.1f}<br>Y: {y_coords[i]:.1f}" for i in range(len(metadata))],
                        hovertemplate='%{text}<extra></extra>',
                        customdata=transcript_ids,
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        else:  # cluster dashboard
            # Spatial Distribution colored by Cluster
            for cluster in unique_clusters:
                cluster_mask = metadata['cluster'] == cluster
                if cluster_mask.any():
                    indices = np.where(cluster_mask)[0]
                    valid_indices = [i for i in indices if i < len(metadata)]
                    if len(valid_indices) == 0:
                        continue
                        
                    fig.add_trace(
                        go.Scatter(
                            x=x_coords[valid_indices],
                            y=y_coords[valid_indices],
                            mode='markers',
                            marker=dict(
                                color=cluster_colors[cluster],
                                size=5,
                                opacity=0.7
                            ),
                            name=f'Cluster {cluster}',
                            text=[f"ID: {transcript_ids[i]}<br>Cluster: {cluster}<br>Gene: {metadata.iloc[i]['gene_name']}<br>X: {x_coords[i]:.1f}<br>Y: {y_coords[i]:.1f}" 
                                  for i in valid_indices],
                            hovertemplate='%{text}<extra></extra>',
                            customdata=valid_indices,
                            legendgroup=f'cluster_{cluster}',
                            showlegend=False
                        ),
                        row=1, col=2
                    )
        
        # Adding nuclei (bd nodes) to spatial plot
        if 'bd' in spatial_data:
            bd_data = spatial_data['bd']
            bd_positions = bd_data['positions']
            bd_metadata = bd_data['metadata']
            bd_ids = list(range(len(transcript_ids), len(transcript_ids) + len(bd_metadata)))
            fig.add_trace(
                go.Scatter(
                    x=bd_positions[:, 0].numpy(),
                    y=bd_positions[:, 1].numpy(),
                    mode='markers',
                    marker=dict(
                        color='black',
                        size=8,
                        opacity=0.8,
                        symbol='circle'
                    ),
                    name='Nuclei',
                    text=[f"Nucleus ID: {bd_metadata.iloc[i]['node_id']}<br>X: {bd_positions[i, 0]:.1f}<br>Y: {bd_positions[i, 1]:.1f}" for i in range(len(bd_metadata))],
                    hovertemplate='%{text}<extra></extra>',
                    customdata=bd_ids,
                    showlegend=True,
                    legendgroup='nuclei'
                ),
                row=1, col=2
            )
        
        # Update layout with 16:9 optimized size for 1x2
        dashboard_title = f'{title_prefix}Interactive {dashboard_type.replace("_", " ").title()} Dashboard ({len(metadata):,} transcripts'
        if dashboard_type == 'cluster':
            dashboard_title += f', {n_clusters} clusters'
        if nuclei_count > 0:
            dashboard_title += f', {nuclei_count:,} nuclei'
        dashboard_title += ')'
        
        fig.update_layout(
            title=dashboard_title,
            height=900,   # Reduced height for 1x2
            width=1800,   # Reduced width for 1x2
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=12),
                itemsizing="constant",
                itemwidth=30  
            ),
            dragmode='lasso'
        )
        
        # Update axes labels for 1x2 layout
        # Col 1: Embeddings
        fig.update_xaxes(title_text=f'{self.config.method.upper()} 1', row=1, col=1)
        fig.update_yaxes(title_text=f'{self.config.method.upper()} 2', row=1, col=1)
        
        # Col 2: Spatial
        fig.update_xaxes(title_text='X Coordinate (µm)', row=1, col=2)
        fig.update_yaxes(title_text='Y Coordinate (µm)', row=1, col=2)
        
        # Save the base figure with appropriate filename
        filename = f'interactive_{"gene_types" if dashboard_type == "gene_type" else "clusters"}.html'
        plot_path = save_dir / filename
        fig.write_html(str(plot_path))
        
        # Read the HTML and add improved JavaScript for cross-plot synchronization
        with open(plot_path, 'r') as f:
            html_content = f.read()
        
        # Add enhanced JavaScript for selection synchronization and legend interaction
        dashboard_name = dashboard_type.replace('_', ' ').title()
        item_type = 'gene type' if dashboard_type == 'gene_type' else 'cluster'
        item_types = 'gene types' if dashboard_type == 'gene_type' else 'clusters'
        
        selection_js = f"""
        <script>
        // Enhanced interactive dashboard with legend controls and selection synchronization
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        var isUpdating = false;
        var selectedPoints = new Set();
        var hiddenGeneTypes = new Set();
        var exclusiveGeneType = null;
        var clickTimer = null;
        var DOUBLE_CLICK_DELAY = 300; // milliseconds
        
        // Track original visibility state for all traces
        var originalVisibility = {{}};
        
        function initializeTraceVisibility() {{
            for (var i = 0; i < plotDiv.data.length; i++) {{
                originalVisibility[i] = plotDiv.data[i].visible !== false;
            }}
        }}
        
        function getGeneTypeFromTraceName(traceName) {{
            // Extract gene type from trace name (assumes trace name is the gene type)
            return traceName;
        }}
        
        function getTracesByGeneType(geneType) {{
            var traces = [];
            for (var i = 0; i < plotDiv.data.length; i++) {{
                if (plotDiv.data[i].name === geneType || 
                    (plotDiv.data[i].legendgroup && plotDiv.data[i].legendgroup === geneType)) {{
                    traces.push(i);
                }}
            }}
            return traces;
        }}
        
        function updateTraceVisibility() {{
            var updates = {{visible: []}};
            var traceIndices = [];
            
            for (var i = 0; i < plotDiv.data.length; i++) {{
                var trace = plotDiv.data[i];
                var geneType = trace.name || trace.legendgroup;
                var shouldBeVisible = true;
                
                // Always keep nuclei visible
                if (geneType === 'Nuclei' || trace.legendgroup === 'nuclei') {{
                    shouldBeVisible = true;
                }} else if (exclusiveGeneType !== null) {{
                    // Exclusive mode: only show the selected gene type (but always show nuclei)
                    shouldBeVisible = (geneType === exclusiveGeneType);
                }} else {{
                    // Normal mode: hide explicitly hidden gene types (but always show nuclei)
                    shouldBeVisible = !hiddenGeneTypes.has(geneType);
                }}
                
                updates.visible.push(shouldBeVisible);
                traceIndices.push(i);
            }}
            
            Plotly.restyle(plotDiv, updates, traceIndices);
        }}
        
        function updateTraceOpacity(traceIndex, selectedIds) {{
            var trace = plotDiv.data[traceIndex];
            if (!trace.customdata || trace.visible === false) return;
            
            var opacities = [];
            var hasSelectedPoints = false;
            var hasSelection = selectedIds.size > 0;
            
            var isNucleiTrace = (trace.name === 'Nuclei') || (trace.legendgroup === 'nuclei');
            if (Array.isArray(trace.customdata)) {{
                for (var i = 0; i < trace.customdata.length; i++) {{
                    if (selectedIds.has(trace.customdata[i])) {{
                        opacities.push(1.0);  // Full opacity for selected points
                        hasSelectedPoints = true;
                    }} else {{
                        // Keep nuclei visible even when not selected
                        if (isNucleiTrace) {{
                            opacities.push(0.7);
                        }} else {{
                            // Much lower opacity for unselected when there's a selection
                            opacities.push(hasSelection ? 0.00 : 0.7);
                        }}
                    }}
                }}
            }} else {{
                // Single value customdata
                if (selectedIds.has(trace.customdata)) {{
                    opacities = 1.0;
                    hasSelectedPoints = true;
                }} else {{
                    // Keep nuclei visible even when not selected
                    if (isNucleiTrace) {{
                        opacities = 0.7;
                    }} else {{
                        opacities = hasSelection ? 0.00 : 0.7;
                    }}
                }}
            }}
            
            // Always update opacity when there's a selection or when clearing selection
            if (hasSelection || (!hasSelection && trace.marker && trace.marker.opacity !== 0.7)) {{
                Plotly.restyle(plotDiv, {{'marker.opacity': [opacities]}}, [traceIndex]);
            }}
        }}
        
        function toggleGeneTypeVisibility(geneType) {{
            // Don't allow hiding nuclei
            if (geneType === 'Nuclei') {{
                console.log('Nuclei cannot be hidden - always visible for spatial reference');
                return;
            }}
            
            if (exclusiveGeneType !== null) {{
                // Exit exclusive mode first
                exclusiveGeneType = null;
            }}
            
            if (hiddenGeneTypes.has(geneType)) {{
                hiddenGeneTypes.delete(geneType);
                console.log('Showing {item_type}:', geneType);
            }} else {{
                hiddenGeneTypes.add(geneType);
                console.log('Hiding {item_type}:', geneType);
            }}
            
            updateTraceVisibility();
        }}
        
        function showOnlyGeneType(geneType) {{
            exclusiveGeneType = geneType;
            hiddenGeneTypes.clear();
            console.log('Exclusive view for {item_type}:', geneType);
            updateTraceVisibility();
        }}
        
        function showAllGeneTypes() {{
            exclusiveGeneType = null;
            hiddenGeneTypes.clear();
            console.log('Showing all {item_types}');
            updateTraceVisibility();
        }}
        
        // Initialize when plot is ready
        plotDiv.on('plotly_afterplot', function() {{
            initializeTraceVisibility();
        }});
        
        // Handle legend clicks with single/double-click detection
        plotDiv.on('plotly_legendclick', function(eventData) {{
            var geneType = eventData.data[eventData.curveNumber].name || 
                          eventData.data[eventData.curveNumber].legendgroup;
            
            if (clickTimer) {{
                // Double click detected
                clearTimeout(clickTimer);
                clickTimer = null;
                showOnlyGeneType(geneType);
            }} else {{
                // Single click - wait to see if double click follows
                clickTimer = setTimeout(function() {{
                    clickTimer = null;
                    toggleGeneTypeVisibility(geneType);
                }}, DOUBLE_CLICK_DELAY);
            }}
            
            return false; // Prevent default legend click behavior
        }});
        
        // Handle plot area clicks to restore all gene types
        plotDiv.on('plotly_click', function(eventData) {{
            if (!eventData.points || eventData.points.length === 0) {{
                // Clicked on empty area
                showAllGeneTypes();
            }}
        }});
        
        // Selection synchronization with enhanced visual contrast
        plotDiv.on('plotly_selected', function(eventData) {{
            if (!eventData || !eventData.points || isUpdating) return;
            
            isUpdating = true;
            selectedPoints.clear();
            
            // Collect all selected point IDs
            eventData.points.forEach(function(pt) {{
                if (pt.customdata !== undefined) {{
                    if (Array.isArray(pt.customdata)) {{
                        selectedPoints.add(pt.customdata[pt.pointIndex]);
                    }} else {{
                        selectedPoints.add(pt.customdata);
                    }}
                }}
            }});
            
            console.log('🎯 Selected', selectedPoints.size, 'transcripts across all plots');
            
            // Update all traces with enhanced contrast
            for (var i = 0; i < plotDiv.data.length; i++) {{
                updateTraceOpacity(i, selectedPoints);
            }}
            
            // Provide user feedback about the selection
            if (selectedPoints.size > 0) {{
                console.log('💡 Selected transcripts are highlighted with full opacity (1.0)');
                console.log('💡 Unselected transcripts are dimmed with low opacity (0.00)');
            }}
            
            isUpdating = false;
        }});
        
        plotDiv.on('plotly_deselect', function() {{
            if (isUpdating) return;
            
            isUpdating = true;
            selectedPoints.clear();
            
            // Reset all visible traces to normal opacity using the updateTraceOpacity function
            // This ensures consistent opacity handling
            for (var i = 0; i < plotDiv.data.length; i++) {{
                if (plotDiv.data[i].visible !== false) {{
                    updateTraceOpacity(i, selectedPoints);  // Empty set will restore normal opacity
                }}
            }}
            
            console.log('🔄 Selection cleared - all transcripts restored to normal opacity (0.7)');
            isUpdating = false;
        }});
        
        // Add instructions
        console.log('📊 Enhanced Interactive {dashboard_name} Dashboard Loaded!');
        console.log('🔍 Use lasso or box select to highlight transcripts with high contrast');
        console.log('✨ Selected transcripts: Full opacity (1.0), Unselected: Dimmed (0.00)');
        console.log('👆 Single click legend: Toggle {item_type} visibility');
        console.log('👆👆 Double click legend: Show only that {item_type} (exclusive view)');
        console.log('🎯 Click plot background: Restore all {item_types}');
        console.log('💡 All selection interactions work across both plots simultaneously');
        </script>
        """
        
        # Insert the JavaScript before the closing body tag
        html_content = html_content.replace('</body>', f'{selection_js}</body>')
        
        # Write back the modified HTML
        with open(plot_path, 'w') as f:
            f.write(html_content)
        
        print(f"Saved interactive dashboard with cross-plot synchronization to {plot_path}")
        print(f"Dashboard contains {len(metadata):,} transcripts after filtering and subsampling")
        return str(plot_path)
    
    def save_embeddings(self, 
                       embeddings_data: Dict[str, Dict], 
                       save_path: Path) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata
            save_path: Path to save the data
        """
        save_data = {}
        for node_type, data in embeddings_data.items():
            save_data[node_type] = {
                'embeddings': data['embeddings'].numpy(),
                'metadata': data['metadata']
            }
            
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Embeddings saved to {save_path}")
    
    def load_embeddings(self, load_path: Path) -> Dict[str, Dict]:
        """
        Load embeddings and metadata from disk.
        
        Args:
            load_path: Path to load the data from
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
            
        embeddings_data = {}
        for node_type, data in save_data.items():
            embeddings_data[node_type] = {
                'embeddings': torch.tensor(data['embeddings']),
                'metadata': data['metadata']
            }
            
        return embeddings_data


def _apply_spatial_filtering(embeddings_data: Dict[str, Dict], 
                           spatial_region: List[float]) -> Dict[str, Dict]:
    """
    Apply spatial filtering to embeddings data.
    
    Args:
        embeddings_data: Dictionary containing embeddings and metadata
        spatial_region: List [x_min, x_max, y_min, y_max] for spatial filtering
        
    Returns:
        Filtered embeddings data
    """
    if len(spatial_region) != 4:
        raise ValueError("spatial_region must be a list of 4 values: [x_min, x_max, y_min, y_max]")
    
    x_min, x_max, y_min, y_max = spatial_region
    filtered_data = {}
    
    for node_type, data in embeddings_data.items():
        embeddings = data['embeddings']
        metadata = data['metadata']
        
        # Check if spatial coordinates are available
        if 'x' not in metadata.columns or 'y' not in metadata.columns:
            print(f"Warning: No spatial coordinates found for {node_type} nodes, skipping spatial filtering")
            filtered_data[node_type] = data
            continue
            
        # Apply spatial filtering
        x_coords = metadata['x'].values
        y_coords = metadata['y'].values
        
        spatial_mask = (
            (x_coords >= x_min) & (x_coords <= x_max) &
            (y_coords >= y_min) & (y_coords <= y_max)
        )
        
        n_original = len(metadata)
        n_filtered = spatial_mask.sum()
        
        print(f"Spatial filtering {node_type}: {n_original:,} -> {n_filtered:,} nodes "
              f"(region: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}])")
        
        if n_filtered == 0:
            print(f"Warning: No {node_type} nodes found in spatial region")
            filtered_data[node_type] = {
                'embeddings': torch.empty((0, embeddings.shape[1])),
                'metadata': metadata.iloc[:0].copy()
            }
        else:
            # Slice embeddings for both transcripts and nuclei so we can place nuclei in embedding space
            if hasattr(embeddings, 'shape') and embeddings.shape[0] == len(metadata):
                filtered_embeddings = embeddings[spatial_mask, :]
            else:
                # Safety: if embeddings are missing or mismatched, keep as empty tensor
                filtered_embeddings = torch.empty((0, embeddings.shape[1] if hasattr(embeddings, 'shape') and len(embeddings.shape) == 2 else 0))
                print(f"Warning: {node_type} embeddings shape mismatch with metadata, keeping as empty tensor")
            filtered_data[node_type] = {
                'embeddings': filtered_embeddings,
                'metadata': metadata[spatial_mask].reset_index(drop=True)
            }
    
    return filtered_data


def visualize_embeddings_from_model(model: torch.nn.Module,
                                   dataloader,
                                   save_dir: Path,
                                   transcripts_df: pd.DataFrame,
                                   gene_types_dict: Optional[Dict] = None,
                                   cell_types_dict: Optional[Dict] = None,
                                   max_batches: Optional[int] = None,
                                   config: EmbeddingVisualizationConfig = None,
                                   spatial_region: Optional[List[float]] = None,
                                   create_interactive_plots: bool = True) -> Dict[str, str]:
    """
    Convenience function to extract and visualize embeddings from a trained model.
    
    Args:
        model: Trained Segger model
        dataloader: DataLoader containing batches
        save_dir: Directory to save visualizations
        transcripts_df: DataFrame containing transcript information
        gene_types_dict: Mapping from gene name to gene type
        cell_types_dict: Mapping from cell ID to cell type
        max_batches: Maximum number of batches to process
        config: Visualization configuration
        spatial_region: Optional list [x_min, x_max, y_min, y_max] to filter transcripts by spatial coordinates
        create_interactive_plots: Whether to create interactive Plotly dashboards
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create gene names dictionary from transcripts
    gene_names_dict = dict(zip(transcripts_df['transcript_id'], transcripts_df['feature_name']))
    
    # Prepare visualizer
    visualizer = EmbeddingVisualizer(config)

    # If the embeddings are already saved, load them; otherwise extract them
    if spatial_region is not None and (save_dir / f'embeddings_data_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl').exists():
        embeddings_data = visualizer.load_embeddings(save_dir / f'embeddings_data_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl')
        print(f"Embeddings loaded from {save_dir / f'embeddings_data_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl'}")
    elif spatial_region is None and (save_dir / f'embeddings_data.pkl').exists():
        embeddings_data = visualizer.load_embeddings(save_dir / f'embeddings_data.pkl')
        print(f"Embeddings loaded from {save_dir / f'embeddings_data.pkl'}")
    else:
        extractor = EmbeddingExtractor()
        embeddings_data = extractor.extract_embeddings_from_batches(
            model=model,
            dataloader=dataloader,
            max_batches=max_batches,
            gene_names_dict=gene_names_dict,
            cell_types_dict=cell_types_dict,
            transcripts_df=transcripts_df
        )
        if spatial_region is not None:
            print(f"Embeddings extracted and saved to {save_dir / f'embeddings_data_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl'}")
        else:
            print(f"Embeddings extracted and saved to {save_dir / f'embeddings_data.pkl'}")
    # Apply spatial filtering if specified
    if spatial_region is not None:
        embeddings_data = _apply_spatial_filtering(embeddings_data, spatial_region)
    
    plots = visualizer.visualize_embeddings(
        embeddings_data=embeddings_data,
        save_dir=save_dir,
        gene_types_dict=gene_types_dict,
        create_interactive_plots=create_interactive_plots
    )
    
    # Save embeddings
    if spatial_region is not None:
        visualizer.save_embeddings(embeddings_data, save_dir / f'embeddings_data_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl')
    else:
        visualizer.save_embeddings(embeddings_data, save_dir / f'embeddings_data.pkl')
    
    return plots


def visualize_gene_embeddings_from_model(model: torch.nn.Module,
                                        dataloader,
                                        save_dir: Path,
                                        transcripts_df: pd.DataFrame,
                                        gene_types_dict: Optional[Dict] = None,
                                        max_batches: Optional[int] = None,
                                        config: EmbeddingVisualizationConfig = None,
                                        spatial_region: Optional[List[float]] = None,
                                        min_transcript_count: int = 1,
                                        exclude_unknown: bool = False) -> Dict[str, str]:
    """
    Memory-efficient gene-only visualization workflow.
    
    Args:
        model: Trained Segger model
        dataloader: DataLoader containing batches
        save_dir: Directory to save visualizations
        transcripts_df: DataFrame containing transcript information
        gene_types_dict: Mapping from gene name to gene type
        max_batches: Maximum number of batches to process
        config: Visualization configuration
        spatial_region: Optional list [x_min, x_max, y_min, y_max] to filter genes by spatial coordinates
        min_transcript_count: Minimum transcript count for genes to include
        exclude_unknown: Whether to exclude unknown genes
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create gene names dictionary from transcripts
    gene_names_dict = dict(zip(transcripts_df['transcript_id'], transcripts_df['feature_name']))
    
    # Prepare visualizer and extractor
    visualizer = EmbeddingVisualizer(config)
    extractor = EmbeddingExtractor()
    
    # Check if gene embeddings are already saved
    if spatial_region is not None:
        embeddings_file = save_dir / f'gene_embeddings_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl'
    else:
        embeddings_file = save_dir / 'gene_embeddings.pkl'
    
    if embeddings_file.exists():
        gene_data = visualizer.load_embeddings(embeddings_file)['gene']  # Extract gene data from dict
        print(f"Gene embeddings loaded from {embeddings_file}")
    else:
        # Extract gene embeddings using memory-efficient method
        gene_data = extractor.extract_gene_embeddings_from_batches(
            model=model,
            dataloader=dataloader,
            max_batches=max_batches,
            gene_names_dict=gene_names_dict,
            transcripts_df=transcripts_df
        )
        
        # Check if any gene embeddings were successfully extracted
        if len(gene_data['embeddings']) == 0:
            print("Error: No gene embeddings were successfully extracted. Cannot create visualizations.")
            return {}
        
        print(f"Gene embeddings extracted")
    
    # Add gene types if available
    if gene_types_dict and 'gene_name' in gene_data['metadata'].columns:
        gene_data['metadata']['gene_type'] = gene_data['metadata']['gene_name'].map(gene_types_dict)
        
        # Fill NA values
        is_negative_control = gene_data['metadata']['gene_name'].str.contains('BLANK|Neg', case=False, na=False)
        gene_data['metadata']['gene_type'] = np.where(
            gene_data['metadata']['gene_type'].isna() & ~is_negative_control,
            'Unknown',
            gene_data['metadata']['gene_type']
        )
    
    # Apply spatial filtering if specified
    if spatial_region is not None:
        gene_data = _apply_spatial_filtering({'gene': gene_data}, spatial_region)['gene']
    
    # Create gene-level plots only
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if ('gene_name' in gene_data['metadata'].columns and 
        'gene_type' in gene_data['metadata'].columns):
        
        plots = visualizer._create_gene_level_plots(
            gene_embeddings=gene_data['embeddings'],
            gene_metadata=gene_data['metadata'],
            save_dir=save_dir,
            title_prefix="",
            min_transcript_count=min_transcript_count,
            exclude_unknown=exclude_unknown
        )
        
        # Save gene embeddings
        visualizer.save_embeddings({'gene': gene_data}, embeddings_file)
        print(f"Gene embeddings saved to {embeddings_file}")
        
        return plots
    else:
        print("Warning: Missing gene information for visualization")
        return {}