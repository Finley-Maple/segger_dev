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
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = 'euclidean'
    
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
            # Forward pass to get final embeddings
            embeddings_dict, _ = model(batch.x_dict, batch.edge_index_dict)
            
            # Extract embeddings for tx nodes only
            extracted_embeddings = {}
            if 'tx' in embeddings_dict:
                extracted_embeddings['tx'] = embeddings_dict['tx'].cpu()
            if 'bd' in embeddings_dict:
                extracted_embeddings['bd'] = embeddings_dict['bd'].cpu()
                    
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
                
            # print(f"Processing batch {batch_idx + 1}...")
            
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
            
            # Process bd (nuclei) embeddings
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
    
    def _apply_leiden_clustering(self, coordinates: np.ndarray, n_neighbors: int = 15, resolution: float = 0.001) -> np.ndarray:
        """
        Apply Leiden clustering to coordinates.
        
        Args:
            coordinates: 2D coordinates (e.g., UMAP, t-SNE, PCA results)
            n_neighbors: Number of neighbors for graph construction
            resolution: Resolution parameter for clustering
            
        Returns:
            Array of cluster labels
        """
        if leidenalg is None or ig is None or NearestNeighbors is None:
            print("Warning: Leiden clustering dependencies not available. Using dummy clusters.")
            # Return dummy clusters (just assign all points to cluster 0)
            return np.zeros(len(coordinates), dtype=int)
        
        try:
            # Build k-nearest neighbors graph
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(coordinates)
            distances, indices = nbrs.kneighbors(coordinates)
            
            # Create igraph from adjacency
            edges = []
            weights = []
            for i in range(len(indices)):
                for j in range(1, len(indices[i])):  # Skip self (index 0)
                    neighbor = indices[i][j]
                    weight = 1.0 / (1.0 + distances[i][j])  # Convert distance to weight
                    edges.append((i, neighbor))
                    weights.append(weight)
            
            # Create igraph
            g = ig.Graph(n=len(coordinates), edges=edges)
            g.es['weight'] = weights
            
            # Apply Leiden clustering
            partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, 
                                               resolution_parameter=resolution, weights='weight')
            
            # Get cluster labels
            cluster_labels = np.array(partition.membership)
            
            print(f"Leiden clustering: {len(coordinates)} points -> {len(set(cluster_labels))} clusters")
            return cluster_labels
            
        except Exception as e:
            print(f"Warning: Leiden clustering failed: {e}. Using dummy clusters.")
            return np.zeros(len(coordinates), dtype=int)
    
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
        
        # Check transcript count before processing
        if 'tx' in embeddings_data:
            tx_count = len(embeddings_data['tx']['metadata'])
            max_points_interactive = getattr(self.config, 'max_points_interactive', 100000)
            if tx_count > max_points_interactive:
                raise ValueError(f"Number of transcripts ({tx_count:,}) exceeds max_points_interactive limit ({max_points_interactive:,}). "
                               f"Please use spatial filtering to reduce the number of transcripts or increase max_points_interactive.")
        
        for node_type, data in embeddings_data.items():
            metadata = data['metadata']
            
            print(f"Visualizing {node_type} embeddings ({len(metadata):,} nodes)...")
            
            # Add gene types if available for tx nodes
            if node_type == 'tx' and gene_types_dict and 'gene_name' in metadata.columns:
                metadata['gene_type'] = metadata['gene_name'].map(gene_types_dict)
                
                # Fill NA values based on gene name
                is_negative_control = metadata['gene_name'].str.contains('BLANK|Neg', case=False, na=False)
                # calculate the number of negative controls
                negative_control_count = is_negative_control.sum()
                print(f"Number of negative controls: {negative_control_count}")
                metadata['gene_type'] = np.where(
                    metadata['gene_type'].isna() & ~is_negative_control,
                    'Unknown',
                    metadata['gene_type']  # Keep original (including NaN for negative controls)
                )
                # calculate the number of unknown gene types
                unknown_gene_type_count = metadata['gene_type'].str.contains('Unknown').sum()
                print(f"Number of unknown gene types: {unknown_gene_type_count}")
        
        # Create interactive dashboard if tx data is available
        if 'tx' in embeddings_data:
            #try:
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
                
                # Add bd nodes (nuclei) to spatial data if available
                if 'bd' in embeddings_data and 'x' in embeddings_data['bd']['metadata'].columns:
                    spatial_data_for_dashboard['bd'] = {
                        'positions': torch.column_stack([
                            torch.tensor(embeddings_data['bd']['metadata']['x'].values),
                            torch.tensor(embeddings_data['bd']['metadata']['y'].values)
                        ]),
                        'metadata': embeddings_data['bd']['metadata']
                }
                
                # Create gene type dashboard
                gene_type_dashboard_path = self.create_interactive_dashboard(
                    embeddings_data=embeddings_data,
                    spatial_data=spatial_data_for_dashboard,
                    save_dir=save_dir,
                    title_prefix=title_prefix,
                    gene_types_dict=gene_types_dict,
                    dashboard_type='gene_type'
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
                    dashboard_type='cluster'
                )
                if cluster_dashboard_path:
                    saved_plots['cluster_dashboard'] = cluster_dashboard_path
            # except Exception as e:
            #     print(f"Warning: Could not create interactive dashboard: {e}")
            
        return saved_plots
    


    def _create_tx_plots(self,
                        reduced_embeddings: np.ndarray,
                        metadata: pd.DataFrame,
                        save_dir: Path,
                        title_prefix: str,
                        gene_types_dict: Optional[Dict] = None) -> Dict[str, str]:
        """Create plots for transcript (tx) embeddings."""
        plots = {}
        
        # Plot: Color by gene type (if available)
        if gene_types_dict and 'gene_type' in metadata.columns:
            # Apply gene type filtering as the LAST step to ensure consistency with spatial plot
            # This ensures both plots use the same dimensionality reduction and base dataset
            valid_gene_type_mask = metadata['gene_type'].notna()
            if valid_gene_type_mask.sum() > 0:  # Only create plot if there are valid gene types
                # Filter data as the final step (after dimensionality reduction)
                filtered_metadata = metadata[valid_gene_type_mask]
                filtered_embeddings = reduced_embeddings[valid_gene_type_mask]
                
                fig, ax = plt.subplots(figsize=self.config.figsize)
                
                # Sort gene types for meaningful ordering
                gene_types = sorted(filtered_metadata['gene_type'].unique())
                
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
                
                for gene_type, color in zip(gene_types, colors):
                    mask = filtered_metadata['gene_type'] == gene_type
                    ax.scatter(filtered_embeddings[mask, 0], filtered_embeddings[mask, 1],
                              c=color, label=gene_type, s=self.config.point_size, alpha=self.config.alpha)
                
                ax.set_xlabel(f'{self.config.method.upper()} 1')
                ax.set_ylabel(f'{self.config.method.upper()} 2')
                ax.set_title(f'{title_prefix}Transcript Embeddings by Gene Type ({len(filtered_metadata):,} transcripts)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, markerscale=6.0)
                
                plt.tight_layout()
                plot_path = save_dir / f'tx_embeddings_by_gene_type.{self.config.save_format}'
                plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                plots['tx_by_gene_type'] = str(plot_path)
        
        return plots
    
    def create_interactive_dashboard(self,
                                   embeddings_data: Dict[str, Dict],
                                   spatial_data: Dict[str, Dict],
                                   save_dir: Path,
                                   title_prefix: str = "",
                                   gene_types_dict: Optional[Dict] = None,
                                   epoch_data: Optional[Dict[str, Dict[str, Dict]]] = None,
                                   dashboard_type: str = 'gene_type') -> str:
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
        
        # Get spatial coordinates
        x_coords = metadata['x'].values
        y_coords = metadata['y'].values
        
        # Apply dimensionality reduction for embedding plots (only if embeddings are not dummy)
        if embeddings.sum() != 0:  # Check if embeddings are not all zeros (dummy)
            reduced_embeddings = self._apply_dimensionality_reduction(embeddings, node_type='tx')
        else:
            # For spatial-only data, use spatial coordinates as "embeddings"
            reduced_embeddings = np.column_stack([x_coords, y_coords])
        
        # Apply Leiden clustering only for cluster dashboard
        if dashboard_type == 'cluster':
            cluster_labels = self._apply_leiden_clustering(reduced_embeddings)
            metadata['cluster'] = cluster_labels
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
        
        # # Convert spatial coordinates to RGB colors
        # rgb_colors_hex = [f'rgb({int(x_normalized[i]*255)},{int(y_normalized[i]*255)},{int(0.5*255)})' 
        #                  for i in range(len(x_normalized))]
        
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
        
        # Add nuclei (bd nodes) to spatial plot if available
        if 'bd' in spatial_data:
            bd_data = spatial_data['bd']
            bd_positions = bd_data['positions']
            bd_metadata = bd_data['metadata']
            
            # Create nuclei IDs for selection synchronization
            bd_ids = list(range(len(transcript_ids), len(transcript_ids) + len(bd_metadata)))
            
            fig.add_trace(
                go.Scatter(
                    x=bd_positions[:, 0].numpy(),
                    y=bd_positions[:, 1].numpy(),
                    mode='markers',
                    marker=dict(
                        color='black',
                        size=8,  # Slightly larger than transcripts
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
            # Only process tx nodes
            if node_type != 'tx':
                continue
                
            positions = data['positions']
            metadata = data['metadata']
            
            if gene_types_dict and 'gene_name' in metadata.columns:
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
                # No subsampling when gene_types_dict is not provided
                result[node_type] = data
                
        return result
    
    def visualize_spatial_all_batches(self,
                                    spatial_data: Dict[str, Dict],
                                    save_dir: Path,
                                    gene_types_dict: Optional[Dict] = None,
                                    max_batches_to_plot: Optional[int] = None) -> Dict[str, str]:
        """
        Create spatial visualization plots with all batches combined, colored by batch index.
        Creates plots for tx nodes only.
        
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
        
        if max_batches_to_plot and len(all_batch_indices) > max_batches_to_plot:
            # Randomly select batches instead of taking the first N
            original_count = len(all_batch_indices)
            selected_batch_indices = np.random.choice(all_batch_indices, max_batches_to_plot, replace=False)
            all_batch_indices = sorted(selected_batch_indices)
            print(f"Randomly selected {max_batches_to_plot} batches from {original_count} available: {all_batch_indices}")
        
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
            total_transcripts = len(tx_data['metadata'])
            ax.set_title(f'All Transcripts - Colored by Batch Index ({total_transcripts:,} transcripts)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, markerscale=6.0)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = save_dir / f'tx_all_batches.{self.config.save_format}'
            plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            saved_plots['tx_all_batches'] = str(plot_path)
        

        
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
            if node_type == 'tx':
                # Only slice embeddings for transcripts
                filtered_data[node_type] = {
                    'embeddings': embeddings[spatial_mask, :],
                    'metadata': metadata[spatial_mask].reset_index(drop=True)
                }
            else:
                # For nuclei ('bd'), we only need spatial metadata; keep embeddings empty to avoid mismatches
                filtered_data[node_type] = {
                    'embeddings': torch.empty((0, embeddings.shape[1])),
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
                                   spatial_region: Optional[List[float]] = None) -> Dict[str, str]:
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
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create gene names dictionary from transcripts
    gene_names_dict = dict(zip(transcripts_df['transcript_id'], transcripts_df['feature_name']))
    
    # Prepare visualizer
    visualizer = EmbeddingVisualizer(config)

    # If the embeddings are already saved, load them; otherwise extract them
    if (save_dir / f'embeddings_data_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl').exists():
        embeddings_data = visualizer.load_embeddings(save_dir / f'embeddings_data_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl')
        print(f"Embeddings loaded from {save_dir / 'embeddings_data.pkl'}")
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
        print(f"Embeddings extracted and saved to {save_dir / f'embeddings_data_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl'}")
    # Apply spatial filtering if specified
    if spatial_region is not None:
        embeddings_data = _apply_spatial_filtering(embeddings_data, spatial_region)
    
    plots = visualizer.visualize_embeddings(
        embeddings_data=embeddings_data,
        save_dir=save_dir,
        gene_types_dict=gene_types_dict
    )
    
    # Save embeddings
    visualizer.save_embeddings(embeddings_data, save_dir / f'embeddings_data_{spatial_region[0]}_{spatial_region[1]}_{spatial_region[2]}_{spatial_region[3]}.pkl')
    
    return plots
