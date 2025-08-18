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
                
                all_tx_embeddings.append(tx_emb)
                
                # Add metadata for tx nodes
                for i, tx_id in enumerate(tx_ids):
                    metadata = {
                        'node_id': tx_id,
                        'node_type': 'tx',
                        'batch_idx': batch_idx,
                        'within_batch_idx': i
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
                
                all_bd_embeddings.append(bd_emb)
                
                # Add metadata for bd nodes
                for i, bd_id in enumerate(bd_ids):
                    metadata = {
                        'node_id': bd_id,
                        'node_type': 'bd',
                        'batch_idx': batch_idx,
                        'within_batch_idx': i
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
