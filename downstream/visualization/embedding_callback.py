"""
PyTorch Lightning Callback for Embedding Visualization

This module provides Lightning callbacks for logging and visualizing embeddings
during training, including TensorBoard integration and periodic plot generation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import pickle
import io
from PIL import Image
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import warnings

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    warnings.warn("TensorBoard not available. Please install with: pip install tensorboard")
    SummaryWriter = None

try:
    import umap
except ImportError:
    warnings.warn("UMAP not installed. Please install with: pip install umap-learn")
    umap = None

from .embedding_visualization import EmbeddingExtractor, EmbeddingVisualizer, EmbeddingVisualizationConfig


class EmbeddingVisualizationCallback(Callback):
    """
    Lightning callback for visualizing embeddings during training.
    
    This callback extracts and visualizes node embeddings at specified intervals
    during training, supporting both TensorBoard logging and file-based plot generation.
    """
    
    def __init__(self,
                 dataloader,
                 transcripts_df: pd.DataFrame,
                 log_every_n_epochs: int = 10,
                 max_batches_per_log: int = 80,
                 save_plots: bool = True,
                 log_to_tensorboard: bool = True,
                 save_embeddings: bool = False,
                 gene_types_dict: Optional[Dict] = None,
                 cell_types_dict: Optional[Dict] = None,
                 config: Optional[EmbeddingVisualizationConfig] = None,
                 plots_dir: Optional[Path] = None,
                 use_fixed_coordinates: bool = True,
                 reference_epoch: Optional[int] = None):
        """
        Initialize the embedding visualization callback.
        
        Args:
            dataloader: DataLoader to extract embeddings from
            transcripts_df: DataFrame containing transcript information
            log_every_n_epochs: Frequency of logging (in epochs)
            max_batches_per_log: Maximum number of batches to process per logging event
            save_plots: Whether to save plots to disk
            log_to_tensorboard: Whether to log embeddings to TensorBoard
            save_embeddings: Whether to save raw embeddings to disk
            gene_types_dict: Mapping from gene name to gene type
            cell_types_dict: Mapping from cell ID to cell type
            config: Visualization configuration
            plots_dir: Directory to save plots (defaults to trainer.log_dir/embedding_plots)
            use_fixed_coordinates: Whether to use consistent coordinates across epochs
            reference_epoch: Epoch to use as reference for coordinates (defaults to final epoch)
        """
        super().__init__()
        
        self.dataloader = dataloader
        self.transcripts_df = transcripts_df
        self.log_every_n_epochs = log_every_n_epochs
        self.max_batches_per_log = max_batches_per_log
        self.save_plots = save_plots
        self.log_to_tensorboard = log_to_tensorboard
        self.save_embeddings = save_embeddings
        self.gene_types_dict = gene_types_dict
        self.cell_types_dict = cell_types_dict
        self.config = config or EmbeddingVisualizationConfig()
        self.plots_dir = plots_dir
        self.use_fixed_coordinates = use_fixed_coordinates
        self.reference_epoch = reference_epoch
        
        # Initialize components
        self.extractor = EmbeddingExtractor()
        self.visualizer = EmbeddingVisualizer(self.config)
        
        # Create gene names dictionary
        self.gene_names_dict = dict(zip(transcripts_df['transcript_id'], transcripts_df['feature_name']))
        
        # Track logged epochs to avoid duplicates
        self.logged_epochs = set()
        
        # Store embeddings for reference fitting
        self.stored_embeddings = {} if use_fixed_coordinates else None
        self.reference_fitted = False
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""
        current_epoch = trainer.current_epoch
        
        # Store embeddings if using fixed coordinates
        if self.use_fixed_coordinates and self.stored_embeddings is not None:
            self._store_embeddings_for_reference(trainer, pl_module, current_epoch)
        
        # Check if we should log this epoch
        if (current_epoch % self.log_every_n_epochs == 0 and 
            current_epoch not in self.logged_epochs):
            
            self.logged_epochs.add(current_epoch)
            self._log_embeddings(trainer, pl_module, current_epoch)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        current_epoch = trainer.current_epoch
        
        # Store embeddings if using fixed coordinates
        if self.use_fixed_coordinates and self.stored_embeddings is not None:
            self._store_embeddings_for_reference(trainer, pl_module, current_epoch)
        
        # Log embeddings on validation end if it's a logging epoch
        if (current_epoch % self.log_every_n_epochs == 0 and 
            current_epoch not in self.logged_epochs):
            
            self.logged_epochs.add(current_epoch)
            self._log_embeddings(trainer, pl_module, current_epoch, prefix="val_")
    
    def _log_embeddings(self, trainer, pl_module, epoch: int, prefix: str = ""):
        """Extract and log embeddings."""
        try:
            print(f"Extracting embeddings at epoch {epoch}...")
            
            # Extract embeddings from model
            embeddings_data = self.extractor.extract_embeddings_from_batches(
                model=pl_module.model,
                dataloader=self.dataloader,
                max_batches=self.max_batches_per_log,
                gene_names_dict=self.gene_names_dict,
                cell_types_dict=self.cell_types_dict,
                transcripts_df=self.transcripts_df
            )
            
            # Set up save directory
            if self.plots_dir is None:
                log_dir = Path(trainer.log_dir) if trainer.log_dir else Path("./logs")
                save_dir = log_dir / "embedding_plots" / f"epoch_{epoch}"
            else:
                save_dir = self.plots_dir / f"epoch_{epoch}"
            
            # Save plots if requested
            if self.save_plots:
                title_prefix = f"{prefix}Epoch {epoch} - "
                plots = self.visualizer.visualize_embeddings(
                    embeddings_data=embeddings_data,
                    save_dir=save_dir,
                    title_prefix=title_prefix,
                    gene_types_dict=self.gene_types_dict
                )
                print(f"Saved embedding plots to {save_dir}")
            
            # Log to TensorBoard if requested
            if self.log_to_tensorboard and hasattr(trainer, 'logger'):
                self._log_to_tensorboard(trainer.logger, embeddings_data, epoch, prefix)
            
            # Save raw embeddings if requested
            if self.save_embeddings:
                embeddings_path = save_dir / f"{prefix}embeddings_epoch_{epoch}.pkl"
                self.visualizer.save_embeddings(embeddings_data, embeddings_path)
                
        except Exception as e:
            print(f"Error during embedding logging at epoch {epoch}: {str(e)}")
            # Don't raise the exception to avoid stopping training
    
    def _store_embeddings_for_reference(self, trainer, pl_module, epoch: int):
        """Store embeddings for reference epoch fitting."""
        try:
            embeddings_data = self.extractor.extract_embeddings_from_batches(
                model=pl_module.model,
                dataloader=self.dataloader,
                max_batches=self.max_batches_per_log,
                gene_names_dict=self.gene_names_dict,
                cell_types_dict=self.cell_types_dict,
                transcripts_df=self.transcripts_df
            )
            self.stored_embeddings[epoch] = embeddings_data
        except Exception as e:
            print(f"Error storing embeddings for reference at epoch {epoch}: {str(e)}")
    
    def on_train_end(self, trainer, pl_module):
        """Called when training ends - fit reference reducers for consistent coordinates."""
        if not self.use_fixed_coordinates or not self.stored_embeddings:
            return
            
        # Determine reference epoch (default to final epoch)
        available_epochs = list(self.stored_embeddings.keys())
        if not available_epochs:
            print("No stored embeddings available for reference fitting.")
            return
            
        ref_epoch = self.reference_epoch
        if ref_epoch is None or ref_epoch not in available_epochs:
            ref_epoch = max(available_epochs)
            
        print(f"Fitting reference reducers using epoch {ref_epoch}...")
        
        try:
            # Fit reducers on reference epoch
            self.visualizer.fit_reference_reducers(self.stored_embeddings[ref_epoch])
            self.reference_fitted = True
            
            # Re-generate all plots with consistent coordinates
            self._regenerate_all_plots_with_fixed_coordinates(trainer)
            
        except Exception as e:
            print(f"Error fitting reference reducers: {str(e)}")
    
    def _regenerate_all_plots_with_fixed_coordinates(self, trainer):
        """Regenerate all plots using consistent coordinates."""
        print("Regenerating all plots with consistent coordinates...")
        
        for epoch, embeddings_data in self.stored_embeddings.items():
            if epoch in self.logged_epochs:
                try:
                    # Set up save directory
                    if self.plots_dir is None:
                        log_dir = Path(trainer.log_dir) if trainer.log_dir else Path("./logs")
                        save_dir = log_dir / "embedding_plots" / f"epoch_{epoch}"
                    else:
                        save_dir = self.plots_dir / f"epoch_{epoch}"
                    
                    # Create plots with fixed coordinates
                    title_prefix = f"Epoch {epoch} - "
                    self.visualizer.visualize_embeddings(
                        embeddings_data=embeddings_data,
                        save_dir=save_dir,
                        title_prefix=title_prefix,
                        gene_types_dict=self.gene_types_dict
                    )
                    print(f"Regenerated plots for epoch {epoch} with fixed coordinates")
                    
                except Exception as e:
                    print(f"Error regenerating plots for epoch {epoch}: {str(e)}")
    
    def _log_to_tensorboard(self, logger, embeddings_data: Dict, epoch: int, prefix: str = ""):
        """Log embeddings and visualization plots to TensorBoard."""
        if not isinstance(logger, TensorBoardLogger) or SummaryWriter is None:
            print("TensorBoard logging not available")
            return
            
        writer = logger.experiment
        
        for node_type, data in embeddings_data.items():
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            # Subsample for TensorBoard (TensorBoard can be slow with large datasets)
            max_points = 1000
            if len(embeddings) > max_points:
                indices = np.random.choice(len(embeddings), max_points, replace=False)
                embeddings_subset = embeddings[indices]
                metadata_subset = metadata.iloc[indices].reset_index(drop=True)
            else:
                embeddings_subset = embeddings
                metadata_subset = metadata
            
            # Apply dimensionality reduction for visualization  
            reduced_embeddings = self.visualizer._apply_dimensionality_reduction(
                embeddings_subset, 
                node_type=node_type
            )
            
            # Log high-dimensional embeddings to TensorBoard projector
            self._log_embedding_projector(writer, embeddings_subset, metadata_subset, node_type, epoch, prefix)
            
            # Log 2D visualization plots as images
            self._log_embedding_plots(writer, reduced_embeddings, metadata_subset, node_type, epoch, prefix)
    
    def _log_embedding_projector(self, writer, embeddings: torch.Tensor, metadata: pd.DataFrame, 
                                node_type: str, epoch: int, prefix: str = ""):
        """Log high-dimensional embeddings to TensorBoard projector."""
        # Prepare labels for TensorBoard projector
        if node_type == 'tx':
            if self.gene_types_dict and 'gene_name' in metadata.columns:
                metadata['gene_type'] = metadata['gene_name'].map(self.gene_types_dict)
                # Filter out NA gene types
                valid_mask = metadata['gene_type'].notna()
                if valid_mask.sum() > 0:
                    labels = metadata[valid_mask]['gene_type'].tolist()
                    embeddings = embeddings[valid_mask]
                    tag_suffix = "by_gene_type"
                else:
                    labels = metadata['gene_name'].tolist()
                    tag_suffix = "by_gene_name"
            else:
                labels = metadata['gene_name'].tolist()
                tag_suffix = "by_gene_name"
        else:  # bd
            labels = metadata['cell_type'].tolist()
            tag_suffix = "by_cell_type"
        
        # Log to TensorBoard projector
        tag = f"{prefix}projector/{node_type}_{tag_suffix}"
        writer.add_embedding(
            mat=embeddings,
            metadata=labels,
            global_step=epoch,
            tag=tag
        )
    
    def _log_embedding_plots(self, writer, reduced_embeddings: np.ndarray, metadata: pd.DataFrame,
                           node_type: str, epoch: int, prefix: str = ""):
        """Log 2D embedding visualization plots as images to TensorBoard."""
        
        # Create plots for different metadata categories
        plots_to_create = []
        
        if node_type == 'tx':
            # Add gene type plot if available
            if self.gene_types_dict and 'gene_name' in metadata.columns:
                metadata['gene_type'] = metadata['gene_name'].map(self.gene_types_dict)
                valid_mask = metadata['gene_type'].notna()
                if valid_mask.sum() > 0:
                    plots_to_create.append({
                        'data': (reduced_embeddings[valid_mask], metadata[valid_mask]),
                        'color_column': 'gene_type',
                        'title': f'{prefix}Transcript Embeddings by Gene Type - Epoch {epoch}',
                        'tag': f'{prefix}plots/{node_type}_by_gene_type'
                    })
            
            # Add gene name plot if not too many genes
            if metadata['gene_name'].nunique() <= 20:
                plots_to_create.append({
                    'data': (reduced_embeddings, metadata),
                    'color_column': 'gene_name', 
                    'title': f'{prefix}Transcript Embeddings by Gene Name - Epoch {epoch}',
                    'tag': f'{prefix}plots/{node_type}_by_gene_name'
                })
        else:  # bd
            plots_to_create.append({
                'data': (reduced_embeddings, metadata),
                'color_column': 'cell_type',
                'title': f'{prefix}Cell Embeddings by Cell Type - Epoch {epoch}',
                'tag': f'{prefix}plots/{node_type}_by_cell_type'
            })
        

        
        # Create and log each plot
        for plot_info in plots_to_create:
            try:
                plot_image = self._create_plot_image(plot_info)
                writer.add_image(plot_info['tag'], plot_image, global_step=epoch, dataformats='HWC')
            except Exception as e:
                print(f"Error creating plot {plot_info['tag']}: {str(e)}")
        
        # Add spatial plots if spatial coordinates are available
        self._log_spatial_plots(writer, reduced_embeddings, metadata, node_type, epoch, prefix)
    
    def _create_plot_image(self, plot_info: Dict) -> np.ndarray:
        """Create a plot image as numpy array for TensorBoard logging."""
        reduced_embeddings, metadata = plot_info['data']
        color_column = plot_info['color_column']
        title = plot_info['title']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique values and colors
        unique_values = metadata[color_column].unique()
        if len(unique_values) > 20:
            # Too many categories, use a continuous color map
            ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                      c=pd.Categorical(metadata[color_column]).codes, 
                      s=self.config.point_size, alpha=self.config.alpha, cmap='tab20')
        else:
            # Use discrete colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
            for value, color in zip(unique_values, colors):
                mask = metadata[color_column] == value
                ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                          c=[color], label=str(value), s=self.config.point_size, alpha=self.config.alpha)
            
            # Add legend if not too many items
            if len(unique_values) <= 15:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        ax.set_xlabel(f'{self.config.method.upper()} 1')
        ax.set_ylabel(f'{self.config.method.upper()} 2')
        ax.set_title(title, fontsize=12)
        
        plt.tight_layout()
        
        # Convert plot to image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to numpy array
        img = Image.open(buf)
        img_array = np.array(img)
        
        plt.close(fig)
        buf.close()
        
        return img_array
    
    def _log_spatial_plots(self, writer, reduced_embeddings: np.ndarray, metadata: pd.DataFrame,
                          node_type: str, epoch: int, prefix: str = ""):
        """Log spatial-based visualization plots to TensorBoard."""
        if 'x' not in metadata.columns or 'y' not in metadata.columns:
            return  # No spatial coordinates available
            
        try:
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
            
            # Create spatial distance plot
            spatial_distance_plot = self._create_spatial_distance_plot(
                reduced_embeddings, distance_from_origin, 
                f'{prefix}{node_type.upper()} Embeddings by Spatial Distance - Epoch {epoch}'
            )
            writer.add_image(
                f'{prefix}plots/{node_type}_by_spatial_distance', 
                spatial_distance_plot, 
                global_step=epoch, 
                dataformats='HWC'
            )
            
            # Create spatial quadrants plot
            spatial_quadrants_plot = self._create_spatial_quadrants_plot(
                reduced_embeddings, quadrants,
                f'{prefix}{node_type.upper()} Embeddings by Spatial Quadrants - Epoch {epoch}'
            )
            writer.add_image(
                f'{prefix}plots/{node_type}_by_spatial_quadrants', 
                spatial_quadrants_plot, 
                global_step=epoch, 
                dataformats='HWC'
            )
            
        except Exception as e:
            print(f"Error creating spatial plots for {node_type}: {str(e)}")
    
    def _create_spatial_distance_plot(self, reduced_embeddings: np.ndarray, 
                                    distance_values: np.ndarray, title: str) -> np.ndarray:
        """Create spatial distance plot for TensorBoard."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1],
            c=distance_values,
            s=self.config.point_size,
            alpha=self.config.alpha,
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Distance from Origin (µm)', rotation=270, labelpad=20)
        
        ax.set_xlabel(f'{self.config.method.upper()} 1')
        ax.set_ylabel(f'{self.config.method.upper()} 2')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close(fig)
        buf.close()
        
        return img_array
    
    def _create_spatial_quadrants_plot(self, reduced_embeddings: np.ndarray, 
                                     quadrants: List[str], title: str) -> np.ndarray:
        """Create spatial quadrants plot for TensorBoard."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
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
        ax.set_title(title, fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close(fig)
        buf.close()
        
        return img_array
    
    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        print(f"EmbeddingVisualizationCallback initialized. Will log every {self.log_every_n_epochs} epochs.")
        
        # Create plots directory if saving plots
        if self.save_plots and self.plots_dir is None:
            log_dir = Path(trainer.log_dir) if trainer.log_dir else Path("./logs")
            self.plots_dir = log_dir / "embedding_plots"
            self.plots_dir.mkdir(parents=True, exist_ok=True)


class EmbeddingComparisonCallback(Callback):
    """
    Callback for comparing embeddings across different epochs.
    
    This callback tracks embedding evolution during training and creates
    comparison visualizations.
    """
    
    def __init__(self,
                 dataloader,
                 transcripts_df: pd.DataFrame,
                 comparison_epochs: List[int],
                 max_batches_per_comparison: int = 80,
                 gene_types_dict: Optional[Dict] = None,
                 cell_types_dict: Optional[Dict] = None,
                 config: Optional[EmbeddingVisualizationConfig] = None):
        """
        Initialize the embedding comparison callback.
        
        Args:
            dataloader: DataLoader to extract embeddings from
            transcripts_df: DataFrame containing transcript information
            comparison_epochs: List of epochs to compare
            max_batches_per_comparison: Maximum batches to process for comparison
            gene_types_dict: Mapping from gene name to gene type
            cell_types_dict: Mapping from cell ID to cell type
            config: Visualization configuration
        """
        super().__init__()
        
        self.dataloader = dataloader
        self.transcripts_df = transcripts_df
        self.comparison_epochs = set(comparison_epochs)
        self.max_batches_per_comparison = max_batches_per_comparison
        self.gene_types_dict = gene_types_dict
        self.cell_types_dict = cell_types_dict
        self.config = config or EmbeddingVisualizationConfig()
        
        # Initialize components
        self.extractor = EmbeddingExtractor()
        self.visualizer = EmbeddingVisualizer(self.config)
        
        # Create gene names dictionary
        self.gene_names_dict = dict(zip(transcripts_df['transcript_id'], transcripts_df['feature_name']))
        
        # Store embeddings for comparison
        self.stored_embeddings = {}
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Store embeddings for comparison epochs."""
        current_epoch = trainer.current_epoch
        
        if current_epoch in self.comparison_epochs:
            print(f"Storing embeddings for comparison at epoch {current_epoch}...")
            
            try:
                embeddings_data = self.extractor.extract_embeddings_from_batches(
                    model=pl_module.model,
                    dataloader=self.dataloader,
                    max_batches=self.max_batches_per_comparison,
                    gene_names_dict=self.gene_names_dict,
                    cell_types_dict=self.cell_types_dict,
                    transcripts_df=self.transcripts_df
                )
                
                self.stored_embeddings[current_epoch] = embeddings_data
                
            except Exception as e:
                print(f"Error storing embeddings at epoch {current_epoch}: {str(e)}")
    
    def on_train_end(self, trainer, pl_module):
        """Create comparison visualizations at the end of training."""
        if len(self.stored_embeddings) < 2:
            print("Not enough epochs stored for comparison.")
            return
            
        print("Creating embedding comparison visualizations...")
        
        try:
            # Set up save directory
            log_dir = Path(trainer.log_dir) if trainer.log_dir else Path("./logs")
            save_dir = log_dir / "embedding_comparisons"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comparison plots for each node type
            for node_type in ['tx', 'bd']:
                if node_type not in self.stored_embeddings[list(self.stored_embeddings.keys())[0]]:
                    continue
                    
                self._create_comparison_plot(node_type, save_dir)
                
        except Exception as e:
            print(f"Error creating comparison visualizations: {str(e)}")
    
    def _create_comparison_plot(self, node_type: str, save_dir: Path):
        """Create comparison plot for a specific node type."""
        epochs = sorted(self.stored_embeddings.keys())
        n_epochs = len(epochs)
        
        fig, axes = plt.subplots(1, n_epochs, figsize=(5 * n_epochs, 5))
        if n_epochs == 1:
            axes = [axes]
        
        for i, epoch in enumerate(epochs):
            embeddings = self.stored_embeddings[epoch][node_type]['embeddings']
            metadata = self.stored_embeddings[epoch][node_type]['metadata']
            
            # Apply dimensionality reduction
            reduced_embeddings = self.visualizer._apply_dimensionality_reduction(
                embeddings, 
                node_type=node_type
            )
            
            # Determine color scheme
            if node_type == 'tx':
                if self.gene_types_dict and 'gene_name' in metadata.columns:
                    metadata['gene_type'] = metadata['gene_name'].map(self.gene_types_dict)
                    # Don't fill NA values - filter them out instead
                    valid_gene_type_mask = metadata['gene_type'].notna()
                    if valid_gene_type_mask.sum() > 0:
                        metadata = metadata[valid_gene_type_mask]
                        reduced_embeddings = reduced_embeddings[valid_gene_type_mask]
                        color_column = 'gene_type'
                    else:
                        color_column = 'gene_name'
                else:
                    color_column = 'gene_name'
            else:
                color_column = 'cell_type'
            
            # Plot
            ax = axes[i]
            unique_values = metadata[color_column].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
            
            for value, color in zip(unique_values, colors):
                mask = metadata[color_column] == value
                ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                          c=[color], label=value, s=self.config.point_size, alpha=self.config.alpha)
            
            ax.set_title(f'Epoch {epoch}')
            ax.set_xlabel(f'{self.config.method.upper()} 1')
            if i == 0:
                ax.set_ylabel(f'{self.config.method.upper()} 2')
            
            # Add legend only to the last subplot
            if i == n_epochs - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle(f'{node_type.upper()} Embedding Evolution During Training')
        plt.tight_layout()
        
        plot_path = save_dir / f'{node_type}_embedding_evolution.{self.config.save_format}'
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {node_type} comparison plot to {plot_path}")


def create_embedding_callbacks(dataloader,
                             transcripts_df: pd.DataFrame,
                             gene_types_dict: Optional[Dict] = None,
                             cell_types_dict: Optional[Dict] = None,
                             log_every_n_epochs: int = 10,
                             comparison_epochs: Optional[List[int]] = None,
                             config: Optional[EmbeddingVisualizationConfig] = None,
                             use_fixed_coordinates: bool = True,
                             reference_epoch: Optional[int] = None) -> List[Callback]:
    """
    Convenience function to create a set of embedding visualization callbacks.
    
    Args:
        dataloader: DataLoader to extract embeddings from
        transcripts_df: DataFrame containing transcript information
        gene_types_dict: Mapping from gene name to gene type
        cell_types_dict: Mapping from cell ID to cell type
        log_every_n_epochs: Frequency of logging embeddings
        comparison_epochs: List of epochs to store for comparison
        config: Visualization configuration
        use_fixed_coordinates: Whether to use consistent coordinates across epochs
        reference_epoch: Epoch to use as reference for coordinates (defaults to final epoch)
        
    Returns:
        List of callback instances
    """
    callbacks = []
    
    # Main visualization callback
    viz_callback = EmbeddingVisualizationCallback(
        dataloader=dataloader,
        transcripts_df=transcripts_df,
        log_every_n_epochs=log_every_n_epochs,
        gene_types_dict=gene_types_dict,
        cell_types_dict=cell_types_dict,
        config=config,
        use_fixed_coordinates=use_fixed_coordinates,
        reference_epoch=reference_epoch
    )
    callbacks.append(viz_callback)
    
    # Comparison callback if epochs are specified
    if comparison_epochs:
        comp_callback = EmbeddingComparisonCallback(
            dataloader=dataloader,
            transcripts_df=transcripts_df,
            comparison_epochs=comparison_epochs,
            gene_types_dict=gene_types_dict,
            cell_types_dict=cell_types_dict,
            config=config
        )
        callbacks.append(comp_callback)
    
    return callbacks
