"""
Embedding Visualization Module for Segger Model

This module provides tools for visualizing the final node embeddings from the Segger model,
including both transcript ('tx') and boundary ('bd') node embeddings using dimensionality
reduction techniques like UMAP.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pickle
import warnings

try:
    import umap
except ImportError:
    warnings.warn("UMAP not installed. Please install with: pip install umap-learn")
    umap = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except ImportError:
    warnings.warn("scikit-learn not installed. Please install with: pip install scikit-learn")
    TSNE = None
    PCA = None


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
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = 'euclidean'
    
    # t-SNE-specific parameters
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    
    # Sampling for large datasets
    max_points_per_type: int = 10000
    subsample_method: str = 'random'  # 'random', 'balanced'
    
    # Spatial visualization parameters
    spatial_max_points_per_gene_type: int = 1000  # Max points per gene type for spatial plots
    spatial_alpha: float = 0.6
    spatial_tx_size: float = 2.0
    spatial_bd_size: float = 5.0


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
            Dictionary containing extracted embeddings for 'tx' and 'bd' nodes
        """
        model.eval()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            # Forward pass to get final embeddings
            embeddings_dict, _ = model(batch.x_dict, batch.edge_index_dict)
            
            # Extract embeddings for each node type
            extracted_embeddings = {}
            for node_type in ['tx', 'bd']:
                if node_type in embeddings_dict:
                    extracted_embeddings[node_type] = embeddings_dict[node_type].cpu()
                    
        return extracted_embeddings
    
    def extract_spatial_data_from_batches(self,
                                        dataloader,
                                        max_batches: Optional[int] = None,
                                        gene_names_dict: Optional[Dict] = None,
                                        cell_types_dict: Optional[Dict] = None,
                                        transcripts_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
        """
        Extract spatial coordinates and metadata from multiple batches.
        
        Args:
            dataloader: DataLoader containing batches
            max_batches: Maximum number of batches to process
            gene_names_dict: Mapping from transcript ID to gene name
            cell_types_dict: Mapping from cell ID to cell type
            transcripts_df: DataFrame containing transcript information
            
        Returns:
            Dictionary containing spatial coordinates and metadata for visualization
        """
        all_tx_positions = []
        all_bd_positions = []
        all_tx_metadata = []
        all_bd_metadata = []
        
        # Handle both dataset indexing and dataloader iteration
        if hasattr(dataloader, '__getitem__'):
            # It's a dataset, access items by index
            num_items = min(len(dataloader), max_batches) if max_batches else len(dataloader)
            for batch_idx in range(num_items):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                batch = dataloader[batch_idx]
                print(f"Processing batch {batch_idx + 1} for spatial data...")
                
                # Check if this is a heterogeneous graph
                if hasattr(batch, 'x_dict') and 'tx' in batch.x_dict:
                    # Extract spatial coordinates for tx (transcript) nodes
                    if hasattr(batch, 'x_dict') and 'tx' in batch.x_dict:
                        tx_pos = batch['tx'].pos.cpu()
                        tx_ids = batch['tx'].id.cpu().numpy()
                        
                        all_tx_positions.append(tx_pos)
                        
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
                    
                    # Extract spatial coordinates for bd (boundary/cell) nodes
                    if 'bd' in batch.x_dict:
                        bd_pos = batch['bd'].pos.cpu()
                        bd_ids = batch['bd'].id
                        
                        all_bd_positions.append(bd_pos)
                        
                        # Add metadata for bd nodes
                        for i, bd_id in enumerate(bd_ids):
                            metadata = {
                                'node_id': bd_id,
                                'node_type': 'bd',
                                'batch_idx': batch_idx,
                                'within_batch_idx': i,
                                'x': bd_pos[i, 0].item(),
                                'y': bd_pos[i, 1].item()
                            }
                            
                            # Add cell type if available
                            if cell_types_dict and bd_id in cell_types_dict:
                                metadata['cell_type'] = cell_types_dict[bd_id]
                            else:
                                metadata['cell_type'] = 'Unknown'
                                
                            all_bd_metadata.append(metadata)
                else:
                    print(f"  Warning: Unexpected batch structure: {type(batch)}")
        else:
            # It's a regular dataloader, iterate over it
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                print(f"Processing batch {batch_idx + 1} for spatial data...")
                print(f"  Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else 'No keys available'}")
                
                # Extract spatial coordinates for tx (transcript) nodes
                if 'tx' in batch:
                    print(f"  TX data found, has pos: {hasattr(batch['tx'], 'pos')}")
                    if hasattr(batch['tx'], 'pos'):
                        tx_pos = batch['tx'].pos.cpu()
                        tx_ids = batch['tx'].id.cpu().numpy()
                        print(f"  TX: {len(tx_ids)} transcripts")
                        
                        all_tx_positions.append(tx_pos)
                        
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
                else:
                    print(f"  No TX data in batch")
                
                # Extract spatial coordinates for bd (boundary/cell) nodes
                if 'bd' in batch:
                    print(f"  BD data found, has pos: {hasattr(batch['bd'], 'pos')}")
                    if hasattr(batch['bd'], 'pos'):
                        bd_pos = batch['bd'].pos.cpu()
                        bd_ids = batch['bd'].id
                        print(f"  BD: {len(bd_ids)} boundaries")
                        
                        all_bd_positions.append(bd_pos)
                        
                        # Add metadata for bd nodes
                        for i, bd_id in enumerate(bd_ids):
                            metadata = {
                                'node_id': bd_id,
                                'node_type': 'bd',
                                'batch_idx': batch_idx,
                                'within_batch_idx': i,
                                'x': bd_pos[i, 0].item(),
                                'y': bd_pos[i, 1].item()
                            }
                            
                            # Add cell type if available
                            if cell_types_dict and bd_id in cell_types_dict:
                                metadata['cell_type'] = cell_types_dict[bd_id]
                            else:
                                metadata['cell_type'] = 'Unknown'
                                
                            all_bd_metadata.append(metadata)
                else:
                    print(f"  No BD data in batch")
        
        # Combine all spatial data
        result = {}
        if all_tx_positions:
            result['tx'] = {
                'positions': torch.cat(all_tx_positions, dim=0),
                'metadata': pd.DataFrame(all_tx_metadata)
            }
        
        if all_bd_positions:
            result['bd'] = {
                'positions': torch.cat(all_bd_positions, dim=0),
                'metadata': pd.DataFrame(all_bd_metadata)
            }
            
        return result

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
        all_bd_embeddings = []
        all_tx_metadata = []
        all_bd_metadata = []
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            print(f"Processing batch {batch_idx + 1}...")
            
            # Extract embeddings for this batch
            batch_embeddings = self.extract_batch_embeddings(model, batch)
            
            # Process tx (transcript) embeddings
            if 'tx' in batch_embeddings:
                tx_emb = batch_embeddings['tx']
                tx_ids = batch['tx'].id.cpu().numpy()
                tx_pos = batch['tx'].pos.cpu()  # Get spatial positions
                
                all_tx_embeddings.append(tx_emb)
                
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
            
            # Process bd (boundary/cell) embeddings
            if 'bd' in batch_embeddings:
                bd_emb = batch_embeddings['bd']
                bd_ids = batch['bd'].id
                bd_pos = batch['bd'].pos.cpu()  # Get spatial positions
                
                all_bd_embeddings.append(bd_emb)
                
                # Add metadata for bd nodes
                for i, bd_id in enumerate(bd_ids):
                    metadata = {
                        'node_id': bd_id,
                        'node_type': 'bd',
                        'batch_idx': batch_idx,
                        'within_batch_idx': i,
                        'x': bd_pos[i, 0].item(),
                        'y': bd_pos[i, 1].item()
                    }
                    
                    # Add cell type if available
                    if cell_types_dict and bd_id in cell_types_dict:
                        metadata['cell_type'] = cell_types_dict[bd_id]
                    else:
                        metadata['cell_type'] = 'Unknown'
                        
                    all_bd_metadata.append(metadata)
        
        # Concatenate all embeddings
        result = {}
        if all_tx_embeddings:
            result['tx'] = {
                'embeddings': torch.cat(all_tx_embeddings, dim=0),
                'metadata': pd.DataFrame(all_tx_metadata)
            }
        
        if all_bd_embeddings:
            result['bd'] = {
                'embeddings': torch.cat(all_bd_embeddings, dim=0),
                'metadata': pd.DataFrame(all_bd_metadata)
            }
            
        return result


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
                metric=self.config.umap_metric,
                random_state=self.config.random_state
            )
        elif method == 'tsne':
            if TSNE is None:
                raise ImportError("t-SNE not available. Please install scikit-learn")
            reducer = TSNE(
                n_components=self.config.n_components,
                perplexity=self.config.tsne_perplexity,
                n_iter=self.config.tsne_n_iter,
                random_state=self.config.random_state
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
                    metric=self.config.umap_metric,
                    random_state=self.config.random_state,
                    init='random'  # Use random instead of spectral
                )
                reduced_embeddings = reducer.fit_transform(embeddings_np)
            else:
                raise e
        
        # Store fitted reducer if requested
        if fit_reducer and reducer_key:
            self.fitted_reducers[reducer_key] = reducer
            
        return reduced_embeddings
    
    def _subsample_data(self, 
                       embeddings: torch.Tensor, 
                       metadata: pd.DataFrame,
                       color_column: str) -> Tuple[torch.Tensor, pd.DataFrame]:
        """
        Subsample data for visualization if it's too large.
        
        Args:
            embeddings: Input embeddings
            metadata: Metadata DataFrame
            color_column: Column to use for balanced sampling
            
        Returns:
            Subsampled embeddings and metadata
        """
        if len(embeddings) <= self.config.max_points_per_type * len(metadata[color_column].unique()):
            return embeddings, metadata
            
        if self.config.subsample_method == 'random':
            indices = np.random.choice(len(embeddings), 
                                     min(len(embeddings), self.config.max_points_per_type * len(metadata[color_column].unique())), 
                                     replace=False)
        elif self.config.subsample_method == 'balanced':
            indices = []
            for group in metadata[color_column].unique():
                group_indices = metadata[metadata[color_column] == group].index
                n_sample = min(len(group_indices), self.config.max_points_per_type)
                sampled_indices = np.random.choice(group_indices, n_sample, replace=False)
                indices.extend(sampled_indices)
            indices = np.array(indices)
        else:
            raise ValueError(f"Unknown subsample method: {self.config.subsample_method}")
            
        return embeddings[indices], metadata.iloc[indices].reset_index(drop=True)
    
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
            
            # Determine color column for subsampling
            color_column = 'gene_name' if node_type == 'tx' else 'cell_type'
            
            # Subsample if necessary
            embeddings, metadata = self._subsample_data(embeddings, metadata, color_column)
            
            # Fit reducer
            self._apply_dimensionality_reduction(
                embeddings, 
                node_type=node_type, 
                fit_reducer=True
            )
    
    def visualize_embeddings(self,
                           embeddings_data: Dict[str, Dict],
                           save_dir: Path,
                           title_prefix: str = "",
                           gene_types_dict: Optional[Dict] = None) -> Dict[str, str]:
        """
        Create visualization plots for node embeddings.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata
            save_dir: Directory to save plots
            title_prefix: Prefix for plot titles
            gene_types_dict: Mapping from gene name to gene type
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = {}
        
        for node_type, data in embeddings_data.items():
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            print(f"Visualizing {node_type} embeddings ({len(embeddings)} nodes)...")
            
            # Determine color column and add gene types if available
            if node_type == 'tx':
                color_column = 'gene_name'
                if gene_types_dict and 'gene_name' in metadata.columns:
                    metadata['gene_type'] = metadata['gene_name'].map(gene_types_dict)
                    # Don't fill NA values - they will be filtered out in plotting
            else:  # bd
                color_column = 'cell_type'
                
            # Subsample if necessary
            embeddings, metadata = self._subsample_data(embeddings, metadata, color_column)
            
            # Apply dimensionality reduction with consistent coordinates
            reduced_embeddings = self._apply_dimensionality_reduction(
                embeddings, 
                node_type=node_type
            )
            
            # Create plots
            if node_type == 'tx':
                plots = self._create_tx_plots(reduced_embeddings, metadata, save_dir, title_prefix, gene_types_dict)
            else:
                plots = self._create_bd_plots(reduced_embeddings, metadata, save_dir, title_prefix)
                
            saved_plots.update(plots)
        
        # Add spatial plots if spatial coordinates are available
        spatial_plots = self._create_spatial_plots(embeddings_data, save_dir, title_prefix)
        saved_plots.update(spatial_plots)
            
        return saved_plots
    


    def _create_tx_plots(self,
                        reduced_embeddings: np.ndarray,
                        metadata: pd.DataFrame,
                        save_dir: Path,
                        title_prefix: str,
                        gene_types_dict: Optional[Dict] = None) -> Dict[str, str]:
        """Create plots for transcript (tx) embeddings."""
        plots = {}
        
        # Plot 1: Color by gene name (if not too many unique genes)
        unique_genes = metadata['gene_name'].nunique()
        if unique_genes <= 50:  # Only plot if manageable number of genes
            fig, ax = plt.subplots(figsize=self.config.figsize)
            
            for i, gene in enumerate(metadata['gene_name'].unique()):
                mask = metadata['gene_name'] == gene
                ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                          label=gene, s=self.config.point_size, alpha=self.config.alpha)
            
            ax.set_xlabel(f'{self.config.method.upper()} 1')
            ax.set_ylabel(f'{self.config.method.upper()} 2')
            ax.set_title(f'{title_prefix}Transcript Embeddings by Gene Name')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plot_path = save_dir / f'tx_embeddings_by_gene.{self.config.save_format}'
            plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            plots['tx_by_gene'] = str(plot_path)
        
        # Plot 2: Color by gene type (if available)
        if gene_types_dict and 'gene_type' in metadata.columns:
            # Filter out genes with unknown/NA gene types
            valid_gene_type_mask = metadata['gene_type'].notna()
            if valid_gene_type_mask.sum() > 0:  # Only create plot if there are valid gene types
                filtered_metadata = metadata[valid_gene_type_mask]
                filtered_embeddings = reduced_embeddings[valid_gene_type_mask]
                
                fig, ax = plt.subplots(figsize=self.config.figsize)
                
                gene_types = filtered_metadata['gene_type'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(gene_types)))
                
                for gene_type, color in zip(gene_types, colors):
                    mask = filtered_metadata['gene_type'] == gene_type
                    ax.scatter(filtered_embeddings[mask, 0], filtered_embeddings[mask, 1],
                              c=[color], label=gene_type, s=self.config.point_size, alpha=self.config.alpha)
                
                ax.set_xlabel(f'{self.config.method.upper()} 1')
                ax.set_ylabel(f'{self.config.method.upper()} 2')
                ax.set_title(f'{title_prefix}Transcript Embeddings by Gene Type')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plot_path = save_dir / f'tx_embeddings_by_gene_type.{self.config.save_format}'
                plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots['tx_by_gene_type'] = str(plot_path)
        

        
        return plots
    
    def _create_bd_plots(self,
                        reduced_embeddings: np.ndarray,
                        metadata: pd.DataFrame,
                        save_dir: Path,
                        title_prefix: str) -> Dict[str, str]:
        """Create plots for boundary (bd) embeddings."""
        plots = {}
        
        # Plot 1: Color by cell type
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        cell_types = metadata['cell_type'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(cell_types)))
        
        for cell_type, color in zip(cell_types, colors):
            mask = metadata['cell_type'] == cell_type
            ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                      c=[color], label=cell_type, s=self.config.point_size, alpha=self.config.alpha)
        
        ax.set_xlabel(f'{self.config.method.upper()} 1')
        ax.set_ylabel(f'{self.config.method.upper()} 2')
        ax.set_title(f'{title_prefix}Cell Embeddings by Cell Type')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plot_path = save_dir / f'bd_embeddings_by_cell_type.{self.config.save_format}'
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        plots['bd_by_cell_type'] = str(plot_path)
        

        
        return plots
    
    def _create_spatial_plots(self,
                            embeddings_data: Dict[str, Dict],
                            save_dir: Path,
                            title_prefix: str) -> Dict[str, str]:
        """
        Create plots for embeddings colored by spatial location metrics.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata with spatial coordinates
            save_dir: Directory to save plots
            title_prefix: Prefix for plot titles
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plots = {}
        
        for node_type, data in embeddings_data.items():
            if 'x' not in data['metadata'].columns or 'y' not in data['metadata'].columns:
                print(f"Warning: No spatial coordinates found for {node_type} nodes")
                continue
                
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            # Calculate spatial metrics
            x_coords = metadata['x'].values
            y_coords = metadata['y'].values
            
            # Metric 1: Distance from origin
            distance_from_origin = np.sqrt(x_coords**2 + y_coords**2)
            
            # Metric 2: Spatial quadrants
            x_median = np.median(x_coords)
            y_median = np.median(y_coords)
            quadrants = []
            for x, y in zip(x_coords, y_coords):
                if x >= x_median and y >= y_median:
                    quadrants.append('Q1 (Top-Right)')
                elif x < x_median and y >= y_median:
                    quadrants.append('Q2 (Top-Left)')
                elif x < x_median and y < y_median:
                    quadrants.append('Q3 (Bottom-Left)')
                else:
                    quadrants.append('Q4 (Bottom-Right)')
            
            # Apply dimensionality reduction
            reduced_embeddings = self._apply_dimensionality_reduction(
                embeddings, 
                node_type=node_type
            )
            
            # Plot 1: Colored by distance from origin
            fig, ax = plt.subplots(figsize=self.config.figsize)
            
            scatter = ax.scatter(
                reduced_embeddings[:, 0], 
                reduced_embeddings[:, 1],
                c=distance_from_origin,
                s=self.config.point_size,
                alpha=self.config.alpha,
                cmap='viridis'
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Distance from Origin (µm)', rotation=270, labelpad=20)
            
            ax.set_xlabel(f'{self.config.method.upper()} 1')
            ax.set_ylabel(f'{self.config.method.upper()} 2')
            ax.set_title(f'{title_prefix}{node_type.upper()} Embeddings by Distance from Origin')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = save_dir / f'{node_type}_embeddings_by_spatial_distance.{self.config.save_format}'
            plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            plots[f'{node_type}_by_spatial_distance'] = str(plot_path)
            
            # Plot 2: Colored by spatial quadrants
            fig, ax = plt.subplots(figsize=self.config.figsize)
            
            unique_quadrants = sorted(set(quadrants))
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_quadrants)))
            
            for quadrant, color in zip(unique_quadrants, colors):
                mask = np.array(quadrants) == quadrant
                ax.scatter(
                    reduced_embeddings[mask, 0], 
                    reduced_embeddings[mask, 1],
                    c=[color],
                    s=self.config.point_size,
                    alpha=self.config.alpha,
                    label=quadrant
                )
            
            ax.set_xlabel(f'{self.config.method.upper()} 1')
            ax.set_ylabel(f'{self.config.method.upper()} 2')
            ax.set_title(f'{title_prefix}{node_type.upper()} Embeddings by Spatial Quadrants')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = save_dir / f'{node_type}_embeddings_by_spatial_quadrants.{self.config.save_format}'
            plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            plots[f'{node_type}_by_spatial_quadrants'] = str(plot_path)
        
        return plots
    
    def _subsample_spatial_data_by_gene_type(self, 
                                           spatial_data: Dict[str, Dict],
                                           gene_types_dict: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Subsample spatial data for transcript nodes by gene type.
        
        Args:
            spatial_data: Dictionary containing spatial coordinates and metadata
            gene_types_dict: Mapping from gene name to gene type
            
        Returns:
            Subsampled spatial data
        """
        result = {}
        
        for node_type, data in spatial_data.items():
            positions = data['positions']
            metadata = data['metadata']
            
            if node_type == 'tx' and gene_types_dict and 'gene_name' in metadata.columns:
                # Add gene type information
                metadata['gene_type'] = metadata['gene_name'].map(gene_types_dict)
                
                # Subsample by gene type
                sampled_indices = []
                for gene_type in metadata['gene_type'].dropna().unique():
                    gene_type_mask = metadata['gene_type'] == gene_type
                    gene_type_indices = metadata[gene_type_mask].index
                    n_sample = min(len(gene_type_indices), self.config.spatial_max_points_per_gene_type)
                    sampled_indices.extend(np.random.choice(gene_type_indices, n_sample, replace=False))
                
                # Include transcripts without gene type information (up to limit)
                no_gene_type_mask = metadata['gene_type'].isna()
                if no_gene_type_mask.any():
                    no_gene_type_indices = metadata[no_gene_type_mask].index
                    n_sample = min(len(no_gene_type_indices), self.config.spatial_max_points_per_gene_type)
                    sampled_indices.extend(np.random.choice(no_gene_type_indices, n_sample, replace=False))
                
                sampled_indices = np.array(sampled_indices)
                
                result[node_type] = {
                    'positions': positions[sampled_indices],
                    'metadata': metadata.iloc[sampled_indices].copy().reset_index(drop=True)
                }
            else:
                # No subsampling for bd nodes or when gene_types_dict is not provided
                result[node_type] = data
                
        return result
    
    def visualize_spatial_all_batches(self,
                                    spatial_data: Dict[str, Dict],
                                    save_dir: Path,
                                    gene_types_dict: Optional[Dict] = None,
                                    max_batches_to_plot: Optional[int] = None) -> Dict[str, str]:
        """
        Create spatial visualization plots with all batches combined, colored by batch index.
        Creates separate plots for tx and bd nodes.
        
        Args:
            spatial_data: Dictionary containing spatial coordinates and metadata
            save_dir: Directory to save plots
            gene_types_dict: Mapping from gene name to gene type (used for subsampling only)
            max_batches_to_plot: Maximum number of batches to include
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = {}
        
        # Subsample data if gene types are provided
        if gene_types_dict:
            spatial_data = self._subsample_spatial_data_by_gene_type(spatial_data, gene_types_dict)
        
        # Get all unique batch indices
        all_batch_indices = set()
        for node_type, data in spatial_data.items():
            all_batch_indices.update(data['metadata']['batch_idx'].unique())
        
        all_batch_indices = sorted(all_batch_indices)
        
        if max_batches_to_plot:
            all_batch_indices = all_batch_indices[:max_batches_to_plot]
        
        print(f"Creating combined spatial plots for {len(all_batch_indices)} batches...")
        
        # Create colormap for batches - if more than 10 batches, group them into consecutive ranges
        if len(all_batch_indices) > 10:
            # Group batches into consecutive ranges of 10
            batches_per_group = 10
            n_groups = (len(all_batch_indices) + batches_per_group - 1) // batches_per_group  # Ceiling division
            
            batch_groups = {}
            batch_ranges = {}
            
            for i, batch_idx in enumerate(all_batch_indices):
                group_idx = i // batches_per_group  # Integer division for consecutive grouping
                batch_groups[batch_idx] = group_idx
                
                # Track the range for each group
                if group_idx not in batch_ranges:
                    batch_ranges[group_idx] = []
                batch_ranges[group_idx].append(batch_idx)
            
            # Use distinct colors for each group
            group_colors = plt.cm.tab10(np.linspace(0, 1, min(n_groups, 10)))
            batch_to_color = {batch_idx: group_colors[group_idx % 10] for batch_idx, group_idx in batch_groups.items()}
            
            print(f"  Grouping {len(all_batch_indices)} batches into {n_groups} consecutive groups of {batches_per_group}")
            for group_idx, batch_list in batch_ranges.items():
                print(f"    Group {group_idx}: Batches {min(batch_list)}-{max(batch_list)}")
        else:
            # Use individual colors for each batch
            batch_colors = plt.cm.tab10(np.linspace(0, 1, len(all_batch_indices)))
            batch_to_color = dict(zip(all_batch_indices, batch_colors))
            print(f"  Using individual colors for {len(all_batch_indices)} batches")
        
        # Plot 1: All transcripts colored by batch
        if 'tx' in spatial_data:
            print("Creating transcript plot colored by batch...")
            fig, ax = plt.subplots(figsize=self.config.figsize)
            
            tx_data = spatial_data['tx']
            
            for batch_idx in all_batch_indices:
                batch_mask = tx_data['metadata']['batch_idx'] == batch_idx
                if batch_mask.any():
                    batch_positions = tx_data['positions'][batch_mask]
                    
                    # Create label based on grouping
                    if len(all_batch_indices) > 10:
                        group_idx = batch_groups[batch_idx]
                        # Get the batch range for this group
                        batches_in_group = batch_ranges[group_idx]
                        # Only label the first batch in each group
                        if batch_idx == batches_in_group[0]:
                            if len(batches_in_group) == 1:
                                label = f'Batch {batch_idx}'
                            else:
                                label = f'Batches {min(batches_in_group)}-{max(batches_in_group)}'
                        else:
                            label = None  # Don't repeat label for same color group
                    else:
                        label = f'Batch {batch_idx}'
                    
                    ax.scatter(batch_positions[:, 0], batch_positions[:, 1],
                             c=[batch_to_color[batch_idx]], s=self.config.spatial_tx_size,
                             alpha=self.config.spatial_alpha, label=label)
            
            ax.set_xlabel('X Coordinate (µm)')
            ax.set_ylabel('Y Coordinate (µm)')
            ax.set_title('All Transcripts - Colored by Batch Index')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = save_dir / f'tx_all_batches.{self.config.save_format}'
            plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            saved_plots['tx_all_batches'] = str(plot_path)
        
        # Plot 2: All boundaries colored by batch
        if 'bd' in spatial_data:
            print("Creating boundary plot colored by batch...")
            fig, ax = plt.subplots(figsize=self.config.figsize)
            
            bd_data = spatial_data['bd']
            
            for batch_idx in all_batch_indices:
                batch_mask = bd_data['metadata']['batch_idx'] == batch_idx
                if batch_mask.any():
                    batch_positions = bd_data['positions'][batch_mask]
                    
                    # Create label based on grouping
                    if len(all_batch_indices) > 10:
                        group_idx = batch_groups[batch_idx]
                        # Get the batch range for this group
                        batches_in_group = batch_ranges[group_idx]
                        # Only label the first batch in each group
                        if batch_idx == batches_in_group[0]:
                            if len(batches_in_group) == 1:
                                label = f'Batch {batch_idx}'
                            else:
                                label = f'Batches {min(batches_in_group)}-{max(batches_in_group)}'
                        else:
                            label = None  # Don't repeat label for same color group
                    else:
                        label = f'Batch {batch_idx}'
                    
                    ax.scatter(batch_positions[:, 0], batch_positions[:, 1],
                             c=[batch_to_color[batch_idx]], s=self.config.spatial_bd_size,
                             alpha=self.config.spatial_alpha, marker='s', label=label)
            
            ax.set_xlabel('X Coordinate (µm)')
            ax.set_ylabel('Y Coordinate (µm)')
            ax.set_title('All Boundaries - Colored by Batch Index')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = save_dir / f'bd_all_batches.{self.config.save_format}'
            plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            saved_plots['bd_all_batches'] = str(plot_path)
        
        return saved_plots
    
    def visualize_spatial_by_batch(self,
                                 spatial_data: Dict[str, Dict],
                                 save_dir: Path,
                                 gene_types_dict: Optional[Dict] = None,
                                 max_batches_to_plot: Optional[int] = None) -> Dict[str, str]:
        """
        Create spatial visualization plots for different batches separately.
        (Legacy function - kept for compatibility)
        
        Args:
            spatial_data: Dictionary containing spatial coordinates and metadata
            save_dir: Directory to save plots
            gene_types_dict: Mapping from gene name to gene type
            max_batches_to_plot: Maximum number of batches to plot
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        # This is now just a wrapper that calls the combined plot function
        # with combined_plot=False for backwards compatibility
        return self.visualize_spatial_all_batches(
            spatial_data=spatial_data,
            save_dir=save_dir, 
            gene_types_dict=gene_types_dict,
            max_batches_to_plot=max_batches_to_plot
        )
    
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


def visualize_embeddings_from_model(model: torch.nn.Module,
                                   dataloader,
                                   save_dir: Path,
                                   transcripts_df: pd.DataFrame,
                                   gene_types_dict: Optional[Dict] = None,
                                   cell_types_dict: Optional[Dict] = None,
                                   max_batches: Optional[int] = None,
                                   config: EmbeddingVisualizationConfig = None) -> Dict[str, str]:
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
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create gene names dictionary from transcripts
    gene_names_dict = dict(zip(transcripts_df['transcript_id'], transcripts_df['feature_name']))
    
    # Extract embeddings
    extractor = EmbeddingExtractor()
    embeddings_data = extractor.extract_embeddings_from_batches(
        model=model,
        dataloader=dataloader,
        max_batches=max_batches,
        gene_names_dict=gene_names_dict,
        cell_types_dict=cell_types_dict,
        transcripts_df=transcripts_df
    )
    
    # Visualize embeddings
    visualizer = EmbeddingVisualizer(config)
    plots = visualizer.visualize_embeddings(
        embeddings_data=embeddings_data,
        save_dir=save_dir,
        gene_types_dict=gene_types_dict
    )
    
    # Save embeddings
    visualizer.save_embeddings(embeddings_data, save_dir / 'embeddings_data.pkl')
    
    return plots


def visualize_spatial_from_dataloader(dataloader,
                                    save_dir: Path,
                                    transcripts_df: pd.DataFrame,
                                    gene_types_dict: Optional[Dict] = None,
                                    cell_types_dict: Optional[Dict] = None,
                                    max_batches: Optional[int] = None,
                                    max_batches_to_plot: Optional[int] = None,
                                    config: EmbeddingVisualizationConfig = None,
                                    combined_plot: bool = True) -> Dict[str, str]:
    """
    Convenience function to extract and visualize spatial data from a dataloader.
    
    Args:
        dataloader: DataLoader containing batches
        save_dir: Directory to save visualizations
        transcripts_df: DataFrame containing transcript information
        gene_types_dict: Mapping from gene name to gene type
        cell_types_dict: Mapping from cell ID to cell type
        max_batches: Maximum number of batches to process
        max_batches_to_plot: Maximum number of batches to plot
        config: Visualization configuration
        combined_plot: If True, create combined plots with all batches colored by batch index.
                      If False, create separate plots for each batch.
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create gene names dictionary from transcripts
    gene_names_dict = dict(zip(transcripts_df['transcript_id'], transcripts_df['feature_name']))
    
    # Extract spatial data
    extractor = EmbeddingExtractor()
    spatial_data = extractor.extract_spatial_data_from_batches(
        dataloader=dataloader,
        max_batches=max_batches,
        gene_names_dict=gene_names_dict,
        cell_types_dict=cell_types_dict,
        transcripts_df=transcripts_df
    )
    
    # Visualize spatial data
    visualizer = EmbeddingVisualizer(config)
    if combined_plot:
        plots = visualizer.visualize_spatial_all_batches(
            spatial_data=spatial_data,
            save_dir=save_dir,
            gene_types_dict=gene_types_dict,
            max_batches_to_plot=max_batches_to_plot
        )
    else:
        plots = visualizer.visualize_spatial_by_batch(
            spatial_data=spatial_data,
            save_dir=save_dir,
            gene_types_dict=gene_types_dict,
            max_batches_to_plot=max_batches_to_plot
        )
    
    # Save spatial data
    save_data = {}
    for node_type, data in spatial_data.items():
        save_data[node_type] = {
            'positions': data['positions'].numpy(),
            'metadata': data['metadata']
        }
    
    with open(save_dir / 'spatial_data.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Spatial data saved to {save_dir / 'spatial_data.pkl'}")
    
    return plots
