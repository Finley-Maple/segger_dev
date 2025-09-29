"""
PyTorch Lightning Callback for Embedding Visualization

This module provides Lightning callbacks for logging and visualizing embeddings
during training, including Weights & Biases integration and periodic plot generation.

Key Features:
- Logs high-dimensional embeddings as wandb Tables for easy exploration
- Creates 2D visualization plots and logs them as wandb Images
- Supports spatial coordinate visualization with custom coloring
- Integrates seamlessly with PyTorch Lightning and WandbLogger
- Maintains backward compatibility with file-based plot generation

Migration from TensorBoard:
- Replace `log_to_tensorboard=True` with `log_to_wandb=True`
- Use `WandbLogger` instead of `TensorBoardLogger` in your trainer
- Install wandb: `pip install wandb`
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
from lightning.pytorch.loggers import WandbLogger
import warnings

try:
    import wandb
except ImportError:
    warnings.warn("Weights & Biases not available. Please install with: pip install wandb")
    wandb = None

try:
    import umap
except ImportError:
    warnings.warn("UMAP not installed. Please install with: pip install umap-learn")
    umap = None

try:
    from .embedding_visualization import EmbeddingExtractor, EmbeddingVisualizer, EmbeddingVisualizationConfig, _apply_spatial_filtering
    from .utils import load_cell_type_color_palette
except ImportError:
    # Fallback for when running as standalone module
    try:
        from embedding_visualization import EmbeddingExtractor, EmbeddingVisualizer, EmbeddingVisualizationConfig, _apply_spatial_filtering
        from utils import load_cell_type_color_palette
    except ImportError:
        warnings.warn("Could not import visualization dependencies. Some functionality may be limited.")
        # Create dummy classes for testing
        class EmbeddingExtractor:
            def extract_embeddings_from_batches(self, *args, **kwargs):
                return {}
                
        class EmbeddingVisualizer:
            def __init__(self, config=None):
                pass
            def _apply_dimensionality_reduction(self, *args, **kwargs):
                return np.array([[0, 0]])
            def visualize_embeddings(self, *args, **kwargs):
                return {}
            def save_embeddings(self, *args, **kwargs):
                pass
            def fit_reference_reducers(self, *args, **kwargs):
                pass
            def create_interactive_dashboard(self, *args, **kwargs):
                return None
                
        class EmbeddingVisualizationConfig:
            def __init__(self):
                self.point_size = 10
                self.alpha = 0.7
                self.method = 'umap'
                self.save_format = 'png'
                self.dpi = 300
                
        def load_cell_type_color_palette():
            return {}


class EmbeddingVisualizationCallback(Callback):
    """
    Lightning callback for visualizing embeddings during training.
    
    This callback extracts and visualizes node embeddings at specified intervals
    during training, supporting both Weights & Biases logging and file-based plot generation.
    
    Example:
        To use with wandb, initialize your trainer with a WandbLogger:
        
        ```python
        from lightning.pytorch.loggers import WandbLogger
        
        wandb_logger = WandbLogger(project="my_project", name="my_run")
        callback = EmbeddingVisualizationCallback(
            dataloader=val_dataloader,
            transcripts_df=transcripts_df,
            log_to_wandb=True
        )
        trainer = Trainer(logger=wandb_logger, callbacks=[callback])
        ```
    """
    
    def __init__(self,
                 dataloader,
                 transcripts_df: pd.DataFrame,
                 log_every_n_epochs: int = 10,
                 max_batches_per_log: int = 40,
                 save_plots: bool = True,
                 log_to_wandb: bool = True,
                 save_embeddings: bool = False,
                 gene_types_dict: Optional[Dict] = None,
                 cell_types_dict: Optional[Dict] = None,
                 config: Optional[EmbeddingVisualizationConfig] = None,
                 plots_dir: Optional[Path] = None,
                 use_fixed_coordinates: bool = True,
                 reference_epoch: Optional[int] = None,
                 create_interactive_plots: bool = True,
                 create_gene_level_plot: bool = True):
        """
        Initialize the embedding visualization callback.
        
        Args:
            dataloader: DataLoader to extract embeddings from
            transcripts_df: DataFrame containing transcript information
            log_every_n_epochs: Frequency of logging (in epochs)
            max_batches_per_log: Maximum number of batches to process per logging event
            save_plots: Whether to save plots to disk
            log_to_wandb: Whether to log embeddings to Weights & Biases
            save_embeddings: Whether to save raw embeddings to disk
            gene_types_dict: Mapping from gene name to gene type
            cell_types_dict: Mapping from cell ID to cell type
            config: Visualization configuration
            plots_dir: Directory to save plots (defaults to trainer.log_dir/embedding_plots)
            use_fixed_coordinates: Whether to use consistent coordinates across epochs
            reference_epoch: Epoch to use as reference for coordinates (defaults to final epoch)
            create_interactive_plots: Whether to create interactive Plotly dashboards
            create_gene_level_plot: Whether to create gene-level aggregated plot
        """
        super().__init__()
        
        self.dataloader = dataloader
        self.transcripts_df = transcripts_df
        self.log_every_n_epochs = log_every_n_epochs
        self.max_batches_per_log = max_batches_per_log
        self.save_plots = save_plots
        self.log_to_wandb = log_to_wandb
        self.save_embeddings = save_embeddings
        self.gene_types_dict = gene_types_dict
        self.cell_types_dict = cell_types_dict
        self.config = config or EmbeddingVisualizationConfig()
        self.plots_dir = plots_dir
        self.use_fixed_coordinates = use_fixed_coordinates
        self.reference_epoch = reference_epoch
        self.create_interactive_plots = create_interactive_plots
        self.create_gene_level_plot = create_gene_level_plot
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
        
        # Load cell type color palette
        self.cell_type_color_palette = load_cell_type_color_palette()
    
    def _get_gene_type_colors(self, gene_types: List[str]) -> List[str]:
        """
        Get colors for gene types using the loaded palette.
        
        Args:
            gene_types: List of gene type names
            
        Returns:
            List of color codes corresponding to the gene types
        """
        colors = []
        fallback_colors = [
            '#999999', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#e377c2', '#7f7f7f', '#bcbd22', '#aec7e8', '#17becf'
        ]
        
        fallback_index = 0
        for gene_type in gene_types:
            if gene_type in self.cell_type_color_palette:
                colors.append(self.cell_type_color_palette[gene_type])
            else:
                colors.append(fallback_colors[fallback_index % len(fallback_colors)])
                fallback_index += 1
                
        return colors
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""
        current_epoch = trainer.current_epoch
        
        # Check if we should log this epoch (add debug print)
        should_log = (current_epoch % self.log_every_n_epochs == 0 and 
                     current_epoch not in self.logged_epochs)
        
        print(f"DEBUG: Epoch {current_epoch}, should_log={should_log}, log_every_n_epochs={self.log_every_n_epochs}")
        
        if should_log:
            # Store embeddings if using fixed coordinates
            if self.use_fixed_coordinates and self.stored_embeddings is not None:
                self._store_embeddings_for_reference(trainer, pl_module, current_epoch)
            
            self.logged_epochs.add(current_epoch)
            print(f"🎯 Logging embeddings at epoch {current_epoch}")
            self._log_embeddings(trainer, pl_module, current_epoch)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        current_epoch = trainer.current_epoch
        
        # Log embeddings on validation end if it's a logging epoch
        if (current_epoch % self.log_every_n_epochs == 0 and 
            current_epoch not in self.logged_epochs):
            # Store embeddings if using fixed coordinates
            if self.use_fixed_coordinates and self.stored_embeddings is not None:
                self._store_embeddings_for_reference(trainer, pl_module, current_epoch, prefix="val_")
            
            self.logged_epochs.add(current_epoch)
            self._log_embeddings(trainer, pl_module, current_epoch, prefix="val_")
    
    def _log_embeddings(self, trainer, pl_module, epoch: int, prefix: str = ""):
        """Extract and log embeddings."""
        # Set up save directory
        if self.plots_dir is None:
            log_dir = Path(trainer.log_dir) if trainer.log_dir else Path("./logs")
            save_dir = log_dir / "embedding_plots" / f"epoch_{epoch}"
        else:
            save_dir = self.plots_dir / f"epoch_{epoch}"
            
        try:
            print(f"🔄 Extracting embeddings at epoch {epoch}...")
            
            embeddings_data = None
            
            if self.create_interactive_plots:
                print(f"🔧 Using dataloader with {len(self.dataloader)} batches, max_batches={self.max_batches_per_log}")
                
                # if already have embeddings data, use it
                if self.stored_embeddings:
                    embeddings_data = self.stored_embeddings[epoch]
                else:
                    # Extract embeddings from model
                    embeddings_data = self.extractor.extract_embeddings_from_batches(
                        model=pl_module.model,
                        dataloader=self.dataloader,
                        #max_batches=self.max_batches_per_log,
                        gene_names_dict=self.gene_names_dict,
                        cell_types_dict=self.cell_types_dict,
                        transcripts_df=self.transcripts_df
                    )
                    
                    if self.spatial_region is not None:
                        embeddings_data = _apply_spatial_filtering(embeddings_data, self.spatial_region)
                if self.save_plots:
                    title_prefix = f"{prefix}Epoch {epoch} - "
                    self.visualizer.visualize_embeddings(
                        embeddings_data=embeddings_data,
                        save_dir=save_dir,
                        title_prefix=title_prefix,
                        gene_types_dict=self.gene_types_dict,
                        create_interactive_plots=self.create_interactive_plots,
                        create_gene_level_plot=False # we don't want to create gene-level plot here
                    )
            
            if self.create_gene_level_plot:
                if self.stored_embeddings:
                    gene_embeddings_data = self.stored_embeddings[epoch]['gene']
                else:
                    # Extract gene embeddings using memory-efficient method
                    print("🧬 Extracting gene embeddings using memory-efficient method...")
                    gene_embeddings_data = self.extractor.extract_gene_embeddings_from_batches(
                        model=pl_module.model,
                        dataloader=self.dataloader,
                        # max_batches=self.max_batches_per_log,
                        gene_names_dict=self.gene_names_dict,
                        transcripts_df=self.transcripts_df
                    )
                if embeddings_data is None:
                    embeddings_data = {}
                embeddings_data['gene'] = gene_embeddings_data
                
                # Save plots if requested
                if self.save_plots:
                    title_prefix = f"{prefix}Epoch {epoch} - "
                    # Create gene-level plots using the new method
                    gene_plots = self.visualizer.create_gene_level_plots(
                        embeddings_data=embeddings_data,
                        save_dir=save_dir,
                        title_prefix=title_prefix,
                        gene_types_dict=self.gene_types_dict,
                        min_transcript_count=1,
                        exclude_unknown=True
                    )
                    if gene_plots:
                        print(f"✅ Created gene-level plots: {list(gene_plots.keys())}")
                    else:
                        print("⚠️ No gene-level plots were created")
                    print(f"Saved embedding plots to {save_dir}")

            # Log to Weights & Biases if requested
            if self.log_to_wandb and hasattr(trainer, 'logger'):
                self._log_to_wandb(trainer.logger, embeddings_data, epoch, prefix)
            
            # Save raw embeddings if requested
            if self.save_embeddings:
                embeddings_path = save_dir / f"{prefix}embeddings_epoch_{epoch}.pkl"
                self.visualizer.save_embeddings(embeddings_data, embeddings_path)
                
        except Exception as e:
            print(f"Error during embedding logging at epoch {epoch}: {str(e)}")
            # Don't raise the exception to avoid stopping training
    
    def _store_embeddings_for_reference(self, trainer, pl_module, epoch: int, prefix: str = ""):
        """Store embeddings for reference epoch fitting."""
        try:
            # Initialize embeddings_data dict
            embeddings_data = {}
            
            if self.create_interactive_plots:
                embeddings_data.update(self.extractor.extract_embeddings_from_batches(
                    model=pl_module.model,
                    dataloader=self.dataloader,
                    # max_batches=self.max_batches_per_log,
                    gene_names_dict=self.gene_names_dict,
                    cell_types_dict=self.cell_types_dict,
                    transcripts_df=self.transcripts_df
                ))
            if self.create_gene_level_plot:
                gene_embeddings_data = self.extractor.extract_gene_embeddings_from_batches(
                    model=pl_module.model,
                    dataloader=self.dataloader,
                    # max_batches=self.max_batches_per_log,
                    gene_names_dict=self.gene_names_dict,
                    transcripts_df=self.transcripts_df
                )
                embeddings_data['gene'] = gene_embeddings_data
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
                        gene_types_dict=self.gene_types_dict,
                        create_interactive_plots=self.create_interactive_plots,
                        create_gene_level_plot=self.create_gene_level_plot
                    )
                    print(f"Regenerated plots for epoch {epoch} with fixed coordinates")
                    
                except Exception as e:
                    print(f"Error regenerating plots for epoch {epoch}: {str(e)}")
    
    def _log_to_wandb(self, logger, embeddings_data: Dict, epoch: int, prefix: str = ""):
        """Log embeddings and visualization plots to Weights & Biases."""
        if not isinstance(logger, WandbLogger) or wandb is None:
            print("Weights & Biases logging not available")
            return
            
        # Only tx nodes
        if 'tx' in embeddings_data:
            node_type = 'tx'
            data = embeddings_data[node_type]
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            # Subsample for wandb (to avoid performance issues with very large datasets)
            max_points = 2000  # wandb can handle more points than TensorBoard
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
            
            # Log high-dimensional embeddings as scatter plots and tables
            self._log_wandb_embeddings(embeddings_subset, metadata_subset, node_type, epoch, prefix)
            
            # Log 2D visualization plots as images
            self._log_wandb_plots(reduced_embeddings, metadata_subset, node_type, epoch, prefix)
    
    def _log_wandb_embeddings(self, embeddings: torch.Tensor, metadata: pd.DataFrame, 
                             node_type: str, epoch: int, prefix: str = ""):
        """Log high-dimensional embeddings to Weights & Biases."""
        # Prepare data for wandb logging
        if self.gene_types_dict and 'gene_name' in metadata.columns:
            metadata['gene_type'] = metadata['gene_name'].map(self.gene_types_dict)
            # Filter out NA gene types
            valid_mask = metadata['gene_type'].notna()
            if valid_mask.sum() > 0:
                filtered_metadata = metadata[valid_mask]
                filtered_embeddings = embeddings[valid_mask]
                color_column = 'gene_type'
            else:
                filtered_metadata = metadata
                filtered_embeddings = embeddings
                color_column = 'gene_name'
        else:
            filtered_metadata = metadata
            filtered_embeddings = embeddings
            color_column = 'gene_name'
        
        # Create a table with embeddings and metadata for wandb
        embedding_table = wandb.Table(
            columns=['embedding_dim_' + str(i) for i in range(filtered_embeddings.shape[1])] + 
                   ['gene_name', color_column] + 
                   (['x', 'y'] if 'x' in filtered_metadata.columns and 'y' in filtered_metadata.columns else [])
        )
        
        # Add rows to the table
        for i, (_, row) in enumerate(filtered_metadata.iterrows()):
            table_row = list(filtered_embeddings[i].cpu().numpy()) + [row['gene_name'], row[color_column]]
            if 'x' in filtered_metadata.columns and 'y' in filtered_metadata.columns:
                table_row.extend([row['x'], row['y']])
            embedding_table.add_data(*table_row)
        
        # Log the table to wandb
        wandb.log({
            f"{prefix}embeddings/{node_type}_embeddings_table": embedding_table,
            "epoch": epoch
        })
    
    def _log_wandb_plots(self, reduced_embeddings: np.ndarray, metadata: pd.DataFrame,
                        node_type: str, epoch: int, prefix: str = ""):
        """Log 2D embedding visualization plots as images to Weights & Biases."""
        
        # Create plots for different metadata categories
        plots_to_create = []
        
        # Only process tx nodes
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
        else:
            # Skip bd nodes (bd disabled)
            return
        
        # Create and log each plot
        for plot_info in plots_to_create:
            try:
                plot_image = self._create_plot_image(plot_info)
                # Convert numpy array to PIL Image for wandb
                pil_image = Image.fromarray(plot_image.astype('uint8'))
                wandb.log({
                    plot_info['tag']: wandb.Image(pil_image, caption=plot_info['title']),
                    "epoch": epoch
                })
            except Exception as e:
                print(f"Error creating plot {plot_info['tag']}: {str(e)}")
        
        # Add spatial plots if spatial coordinates are available
        self._log_wandb_spatial_plots(reduced_embeddings, metadata, node_type, epoch, prefix)
    
    def _create_plot_image(self, plot_info: Dict) -> np.ndarray:
        """Create a plot image as numpy array for wandb logging."""
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
            # Use colors from loaded palette for gene types
            if color_column == 'gene_type':
                # Sort values for meaningful ordering
                sorted_values = sorted(unique_values)
                colors = self._get_gene_type_colors(sorted_values)
                values_to_plot = sorted_values
            else:
                # Use default colors for other columns
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
                values_to_plot = unique_values
                
            for value, color in zip(values_to_plot, colors):
                mask = metadata[color_column] == value
                ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                          c=color, label=str(value), s=self.config.point_size, alpha=self.config.alpha)
            
            # Add legend if not too many items
            if len(unique_values) <= 15:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=6.0)
        
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
    
    def _log_wandb_spatial_plots(self, reduced_embeddings: np.ndarray, metadata: pd.DataFrame,
                                node_type: str, epoch: int, prefix: str = ""):
        """Log spatial-based visualization plots to Weights & Biases for tx nodes only."""
        if node_type != 'tx' or 'x' not in metadata.columns or 'y' not in metadata.columns:
            return  # Only process tx nodes with spatial coordinates
            
        try:
            # Get spatial coordinates
            x_coords = metadata['x'].values
            y_coords = metadata['y'].values
            
            # Create spatial coordinates plot with two-channel coloring
            spatial_coords_plot = self._create_spatial_coordinates_plot(
                reduced_embeddings, x_coords, y_coords,
                f'{prefix}TX Embeddings by Spatial Coordinates - Epoch {epoch}'
            )
            
            # Convert to PIL Image and log to wandb
            pil_image = Image.fromarray(spatial_coords_plot.astype('uint8'))
            wandb.log({
                f'{prefix}plots/{node_type}_by_spatial_coordinates': wandb.Image(
                    pil_image, 
                    caption=f'{prefix}TX Embeddings by Spatial Coordinates - Epoch {epoch}'
                ),
                "epoch": epoch
            })
            
        except Exception as e:
            print(f"Error creating spatial plots for {node_type}: {str(e)}")
    
    def _create_spatial_coordinates_plot(self, reduced_embeddings: np.ndarray, 
                                       x_coords: np.ndarray, y_coords: np.ndarray, title: str) -> np.ndarray:
        """Create spatial coordinates plot with two-channel coloring for wandb."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize coordinates to [0, 1] for color mapping
        x_normalized = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) if x_coords.max() != x_coords.min() else np.zeros_like(x_coords)
        y_normalized = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) if y_coords.max() != y_coords.min() else np.zeros_like(y_coords)
        
        # Create RGB colors using x and y coordinates
        # Red channel: x coordinate, Green channel: y coordinate, Blue channel: fixed
        rgb_colors = np.column_stack([x_normalized, y_normalized, np.full_like(x_normalized, 0.5)])
        
        scatter = ax.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1],
            c=rgb_colors,
            s=self.config.point_size,
            alpha=self.config.alpha
        )
        
        ax.set_xlabel(f'{self.config.method.upper()} 1')
        ax.set_ylabel(f'{self.config.method.upper()} 2')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add a custom colorbar explanation
        ax.text(0.02, 0.98, 'Color: Red=X coord, Green=Y coord', 
                transform=ax.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
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
        print(f"✅ EmbeddingVisualizationCallback initialized and active!")
        print(f"📊 Will log embeddings every {self.log_every_n_epochs} epochs")
        print(f"💾 save_plots={self.save_plots}, log_to_wandb={self.log_to_wandb}")
        print(f"🎨 create_interactive_plots={self.create_interactive_plots}, create_gene_level_plot={self.create_gene_level_plot}")
        
        # Create plots directory if saving plots
        if self.save_plots and self.plots_dir is None:
            log_dir = Path(trainer.log_dir) if trainer.log_dir else Path("./logs")
            self.plots_dir = log_dir / "embedding_plots"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created plots directory: {self.plots_dir}")


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
                 max_batches_per_comparison: int = 40,
                 spatial_region: Optional[List[int]] = None,
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
        
        # Load cell type color palette
        self.cell_type_color_palette = load_cell_type_color_palette()
    
    def _get_gene_type_colors(self, gene_types: List[str]) -> List[str]:
        """
        Get colors for gene types using the loaded palette.
        
        Args:
            gene_types: List of gene type names
            
        Returns:
            List of color codes corresponding to the gene types
        """
        colors = []
        fallback_colors = [
            '#999999', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#e377c2', '#7f7f7f', '#bcbd22', '#aec7e8', '#17becf'
        ]
        
        fallback_index = 0
        for gene_type in gene_types:
            if gene_type in self.cell_type_color_palette:
                colors.append(self.cell_type_color_palette[gene_type])
            else:
                colors.append(fallback_colors[fallback_index % len(fallback_colors)])
                fallback_index += 1
                
        return colors
    
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
            
            # Create comparison plots only for tx nodes
            if 'tx' in self.stored_embeddings[list(self.stored_embeddings.keys())[0]]:
                self._create_comparison_plot('tx', save_dir)
                
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
            
            # Determine color scheme (only for tx nodes)
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
            
            # Plot
            ax = axes[i]
            unique_values = sorted(metadata[color_column].unique())
            
            # Use colors from loaded palette for gene types
            if color_column == 'gene_type':
                colors = self._get_gene_type_colors(unique_values)
            else:
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
            
            for value, color in zip(unique_values, colors):
                mask = metadata[color_column] == value
                ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                          c=color, label=value, s=self.config.point_size, alpha=self.config.alpha)
            
            ax.set_title(f'Epoch {epoch}')
            ax.set_xlabel(f'{self.config.method.upper()} 1')
            if i == 0:
                ax.set_ylabel(f'{self.config.method.upper()} 2')
            
            # Add legend only to the last subplot
            if i == n_epochs - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=6.0)
        
        plt.suptitle(f'{node_type.upper()} Embedding Evolution During Training')
        plt.tight_layout()
        
        plot_path = save_dir / f'{node_type}_embedding_evolution.{self.config.save_format}'
        plt.savefig(plot_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {node_type} comparison plot to {plot_path}")


def create_embedding_callbacks(dataloader,
                             transcripts_df: pd.DataFrame = None,
                             gene_types_dict: Optional[Dict] = None,
                             cell_types_dict: Optional[Dict] = None,
                             spatial_region: Optional[List[int]] = None,
                             log_every_n_epochs: int = 10,
                             comparison_epochs: Optional[List[int]] = None,
                             config: Optional[EmbeddingVisualizationConfig] = None,
                             use_fixed_coordinates: bool = True,
                             reference_epoch: Optional[int] = None,
                             create_interactive_plots: bool = True,
                             create_gene_level_plot: bool = True) -> List[Callback]:
    """
    Convenience function to create a set of embedding visualization callbacks.
    
    Args:
        dataloader: DataLoader to extract embeddings from
        transcripts_df: DataFrame containing transcript information
        spatial_region: List of spatial coordinates to visualize
        gene_types_dict: Mapping from gene name to gene type
        cell_types_dict: Mapping from cell ID to cell type
        log_every_n_epochs: Frequency of logging embeddings
        comparison_epochs: List of epochs to store for comparison
        config: Visualization configuration
        use_fixed_coordinates: Whether to use consistent coordinates across epochs
        reference_epoch: Epoch to use as reference for coordinates (defaults to final epoch)
        create_interactive_plots: Whether to create interactive Plotly dashboards
        create_gene_level_plot: Whether to create gene-level aggregated plot
    Returns:
        List of callback instances
    """
    callbacks = []
    
    # Main visualization callback
    viz_callback = EmbeddingVisualizationCallback(
        dataloader=dataloader,
        transcripts_df=transcripts_df,
        log_every_n_epochs=log_every_n_epochs,
        spatial_region=spatial_region,
        gene_types_dict=gene_types_dict,
        cell_types_dict=cell_types_dict,
        config=config,
        use_fixed_coordinates=use_fixed_coordinates,
        reference_epoch=reference_epoch,
        create_interactive_plots=create_interactive_plots,
        create_gene_level_plot=create_gene_level_plot
    )
    callbacks.append(viz_callback)
    
    # Comparison callback if epochs are specified
    if comparison_epochs:
        comp_callback = EmbeddingComparisonCallback(
            dataloader=dataloader,
            transcripts_df=transcripts_df,
            comparison_epochs=comparison_epochs,
            spatial_region=spatial_region,
            gene_types_dict=gene_types_dict,
            cell_types_dict=cell_types_dict,
            config=config
        )
        callbacks.append(comp_callback)
    
    return callbacks