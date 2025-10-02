"""
Example Script for Embedding Visualization

This script demonstrates how to use the embedding visualization tools
both for post-training analysis and during training with Lightning callbacks.
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from visualization.embedding_visualization import (
    visualize_embeddings_from_model,
    EmbeddingVisualizationConfig,
    visualize_gene_embeddings_from_model
)
from utils.utils import setup_model_and_data, load_metadata, clear_metadata_cache, get_metadata_cache_path, VisualizationConfig

# Configure paths (adjust these to your setup)
DATA_DIR = Path('/dkfz/cluster/gpu/data/OE0606/fengyun')

def post_training_visualization(config: VisualizationConfig, force_reload_metadata: bool = True, spatial_region: list = None, all_regions: bool = False, create_interactive_plots: bool = True, create_gene_level_plot: bool = True):
    """
    Example of how to visualize embeddings from a trained model using spatial region batches.
    """
    print("=== Post-Training Embedding Visualization Example ===")
    
    # Load model and data
    model, dm = setup_model_and_data(config)
    
    # Load metadata
    transcripts, gene_types_dict, cell_types_dict = load_metadata(config, config.load_scrna_gene_types, force_reload_metadata)
    
    # Create embedding visualization config
    embedding_config = EmbeddingVisualizationConfig(
        method=config.embedding_method,
        n_components=config.n_components,
        figsize=config.figsize,
        point_size=config.point_size,
        alpha=config.alpha,
        max_points_per_type=config.max_points_per_type,
        subsample_method=config.subsample_method,
        umap_n_neighbors=config.umap_n_neighbors,
        umap_min_dist=config.umap_min_dist
    )
    
    # Set up save directory
    save_dir = Path('./embedding_visualization_results') / config.dataset / config.model_type
    if config.align_loss:
        save_dir = save_dir / 'align_loss'
    else:
        save_dir = save_dir / 'orignal_loss'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating embeddings visualizations...")
    print(f"Dataset: {config.dataset}")
    print(f"Model type: {config.model_type}")
    print(f"Align loss: {config.align_loss}")
    print(f"Visualization method: {config.embedding_method}")
    print(f"Save directory: {save_dir}")
    
    # Generate visualizations
    try:
        if spatial_region:
            # Use spatial filtering to get batches
            from utils.spatial_batch_utils import get_spatial_combined_dataloader
            x_range = [spatial_region[0], spatial_region[1]]
            y_range = [spatial_region[2], spatial_region[3]]
            print(f"Using spatial filtering: x={x_range}, y={y_range}, all_regions={all_regions}")
            combined_dataloader = get_spatial_combined_dataloader(
                dm, x_range=x_range, y_range=y_range, all_regions=all_regions, save_dir=save_dir
            )
        elif all_regions:
            
            from utils.spatial_batch_utils import get_spatial_combined_dataloader
            print(f"Using all regions...")
            combined_dataloader = get_spatial_combined_dataloader(
                dm, all_regions=all_regions, save_dir=save_dir
            )
        
        if create_gene_level_plot:
            print(f"Creating gene-level plot...")
            plots = visualize_gene_embeddings_from_model(
                model=model.model,
                dataloader=combined_dataloader,
                save_dir=save_dir,
                transcripts_df=transcripts,
                gene_types_dict=gene_types_dict,
                max_batches=len(combined_dataloader),
                config=embedding_config,
                spatial_region=spatial_region,
                min_transcript_count=1,
                exclude_unknown=True
            )
        if create_interactive_plots:
            print(f"Creating interactive plots...")
            plots = visualize_embeddings_from_model(
                model=model.model,
                dataloader=combined_dataloader,
                save_dir=save_dir,
                transcripts_df=transcripts,
                gene_types_dict=gene_types_dict,
                cell_types_dict=cell_types_dict,
                max_batches=len(combined_dataloader),
                config=embedding_config,
                spatial_region=spatial_region
            )
        
        print(f"\nVisualization complete! Generated plots:")
        for plot_name, plot_path in plots.items():
            print(f"  - {plot_name}: {plot_path}")
            # Highlight the interactive dashboard
        if 'interactive_dashboard' in plots:
            print(f"\n🎯 INTERACTIVE DASHBOARD CREATED:")
            print(f"   📊 {plots['interactive_dashboard']}")
            print(f"   💡 Open this HTML file in your browser to test the interactive features!")
            print(f"   🎨 Features: Lasso selection, synchronized highlighting across plots")
            print(f"   📁 Full path: {Path(plots['interactive_dashboard']).absolute()}")
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function with example configurations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process dataset with specified model type')
    parser.add_argument('--dataset', type=str, choices=['colon', 'CRC', 'pancreas', 'breast'], required=True, default='colon',
                       help='Dataset: "colon", "CRC", or "pancreas"')
    parser.add_argument('--model_type', type=str, choices=['seq', 'no_seq'], required=True, default='seq',
                       help='Model type: "seq" or "no_seq"')
    parser.add_argument('--align_loss', action='store_true',
                        help='Align loss: if True, use align loss model directories')
    parser.add_argument('--force_reload_metadata', action='store_true',
                        help='Force reload metadata from source files, ignoring cache')
    parser.add_argument('--clear_cache', action='store_true',
                        help='Clear metadata cache and exit')
    parser.add_argument('--spatial_region', nargs=4, type=float, metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'),
                        help='Spatial region coordinates to be visualized: x_min x_max y_min y_max (e.g., --spatial_region 2000 3000 2000 2500)')
    parser.add_argument('--all_regions', action='store_true',
                        help='Visualize all regions')
    parser.add_argument('--create_interactive_plots', action='store_true',
                        help='Create interactive Plotly dashboards')
    parser.add_argument('--create_gene_level_plot', action='store_true',
                        help='Create gene-level aggregated plot')
    args = parser.parse_args()
    
    # Handle cache clearing
    if args.clear_cache:
        clear_metadata_cache(args.dataset)
        return
    
    load_scrna_gene_types = True if args.model_type == 'seq' else False
    
    config = VisualizationConfig(
        dataset=args.dataset,
        model_type=args.model_type,
        align_loss=args.align_loss,
        load_scrna_gene_types=load_scrna_gene_types,
        max_points_per_type=1000,
        embedding_method='umap'
    )
    
    print(f"Running visualizations with configuration:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Model type: {config.model_type}")
    print(f"  Align loss: {config.align_loss}")
    print(f"  Embedding method: {config.embedding_method}")
    print(f"  Load scRNAseq gene types: {config.load_scrna_gene_types}")
    print(f"  Force reload metadata: {args.force_reload_metadata}")
    print(f"  Output directory: ./embedding_visualization_results/{config.dataset}/{config.model_type}{'/' + 'align_loss' if config.align_loss else ''}")
    if args.spatial_region:
        print(f"  Spatial region: x=[{args.spatial_region[0]}, {args.spatial_region[1]}], y=[{args.spatial_region[2]}, {args.spatial_region[3]}]")
    if args.all_regions:
        print(f"  All regions: True")
    if config.load_scrna_gene_types:
        print(f"  → Will generate tx_embeddings_by_gene_type.png with scRNAseq cell types!")
    
    # Show cache information
    cache_path = get_metadata_cache_path(config, config.load_scrna_gene_types)
    if cache_path.exists() and not args.force_reload_metadata:
        cache_age = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(cache_path.stat().st_mtime)).total_seconds()
        print(f"  📁 Metadata cache available: {cache_path} (age: {cache_age/60:.1f} minutes)")
    else:
        print(f"  📁 Metadata cache: {cache_path} ({'will be created' if not args.force_reload_metadata else 'disabled'})")
    
    # Run embedding visualization
    post_training_visualization(
        config, 
        args.force_reload_metadata, 
        spatial_region=args.spatial_region,
        all_regions=args.all_regions,
        create_interactive_plots=args.create_interactive_plots,
        create_gene_level_plot=args.create_gene_level_plot
    )


if __name__ == '__main__':
    main()
