# Segger Embedding Visualization

This module provides comprehensive tools for visualizing the embedding space of trained Segger models, including both transcript ('tx') and cell boundary ('bd') node embeddings using dimensionality reduction techniques.

## Features

- **Post-training embedding extraction and visualization**
- **Real-time embedding visualization during training via Lightning callbacks**
- **Multiple dimensionality reduction methods**: UMAP, t-SNE, PCA
- **Flexible metadata integration**: gene types, cell types, batch information
- **TensorBoard integration** for interactive exploration
- **Efficient handling of large datasets** with smart subsampling
- **Embedding evolution tracking** across training epochs

## Installation

Ensure you have the required dependencies:

```bash
pip install umap-learn scikit-learn matplotlib seaborn pandas torch torchvision lightning tensorboard
```

## Quick Start

### 1. Post-Training Visualization

Extract and visualize embeddings from a trained model:

```python
from visualization.embedding_visualization import visualize_embeddings_from_model, EmbeddingVisualizationConfig

# Configure visualization
config = EmbeddingVisualizationConfig(
    method='umap',           # 'umap', 'tsne', 'pca'
    n_components=2,
    figsize=(12, 8),
    point_size=3.0,
    alpha=0.7
)

# Generate visualizations
plots = visualize_embeddings_from_model(
    model=trained_model,
    dataloader=data_loader,
    save_dir=Path('./embedding_results'),
    transcripts_df=transcripts,
    gene_types_dict=gene_types_dict,
    cell_types_dict=cell_types_dict,
    config=config
)
```

### 2. Using the Enhanced Process Dataset Script

Run embedding visualization alongside attention analysis:

```bash
python process_dataset.py \
    --dataset pancreas \
    --edge_type tx-tx \
    --model_type no_seq \
    --visualize_embeddings \
    --embedding_method umap \
    --max_embedding_batches 20
```

### 3. Training with Embedding Visualization

Enable embedding visualization during training:

```bash
python model_training.py \
    --dataset pancreas \
    --scRNAseq_file False \
    --enable_embedding_viz \
    --embedding_log_freq 10
```

This will:
- Log embeddings to TensorBoard every 10 epochs
- Save visualization plots to disk
- Create embedding evolution comparisons
- Track embedding changes during training

## Module Structure

```
visualization/
├── embedding_visualization.py    # Core embedding extraction and visualization
├── embedding_callback.py         # Lightning callbacks for training-time visualization
└── example_embedding_visualization.py  # Comprehensive examples
```

## Core Classes

### `EmbeddingExtractor`

Extracts final node embeddings from Segger model forward passes:

```python
extractor = EmbeddingExtractor()

# Extract from single batch
batch_embeddings = extractor.extract_batch_embeddings(model, batch)

# Extract from multiple batches with metadata
embeddings_data = extractor.extract_embeddings_from_batches(
    model=model,
    dataloader=dataloader,
    max_batches=20,
    gene_names_dict=gene_names_dict,
    cell_types_dict=cell_types_dict
)
```

### `EmbeddingVisualizer`

Creates dimensionality-reduced visualizations of embeddings:

```python
config = EmbeddingVisualizationConfig(method='umap', n_components=2)
visualizer = EmbeddingVisualizer(config)

plots = visualizer.visualize_embeddings(
    embeddings_data=embeddings_data,
    save_dir=save_dir,
    gene_types_dict=gene_types_dict
)
```

### `EmbeddingVisualizationCallback`

Lightning callback for training-time visualization:

```python
callback = EmbeddingVisualizationCallback(
    dataloader=dataloader,
    transcripts_df=transcripts,
    log_every_n_epochs=10,
    save_plots=True,
    log_to_tensorboard=True,
    gene_types_dict=gene_types_dict,
    cell_types_dict=cell_types_dict
)

trainer = Trainer(callbacks=[callback])
```

## Configuration Options

### `EmbeddingVisualizationConfig`

```python
config = EmbeddingVisualizationConfig(
    method='umap',                  # Dimensionality reduction method
    n_components=2,                 # Output dimensions
    figsize=(12, 8),               # Plot size
    point_size=3.0,                # Scatter plot point size
    alpha=0.7,                     # Point transparency
    max_points_per_type=5000,      # Subsampling limit
    subsample_method='balanced',    # 'random' or 'balanced'
    
    # UMAP-specific parameters
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    umap_metric='euclidean',
    
    # t-SNE-specific parameters
    tsne_perplexity=30.0,
    tsne_n_iter=1000
)
```

## Visualization Types

### For Transcript ('tx') Embeddings:
- **By gene name**: Color points by individual genes (if ≤50 unique genes)
- **By gene type**: Color points by gene functional categories
- **By batch**: Check for batch effects in embeddings

### For Boundary ('bd') Embeddings:
- **By cell type**: Color points by cell type annotations
- **By batch**: Check for batch effects in embeddings

## Output Files

The visualization generates several types of outputs:

### Plots
- `tx_embeddings_by_gene.png` - Transcript embeddings colored by gene
- `tx_embeddings_by_gene_type.png` - Transcript embeddings colored by gene type
- `tx_embeddings_by_batch.png` - Transcript embeddings colored by batch
- `bd_embeddings_by_cell_type.png` - Cell embeddings colored by cell type
- `bd_embeddings_by_batch.png` - Cell embeddings colored by batch

### Data Files
- `embeddings_data.pkl` - Raw embeddings and metadata for further analysis

### Training-Time Outputs
- `embedding_plots/epoch_N/` - Plots generated at each logging epoch
- TensorBoard logs with interactive embedding projections
- `embedding_comparisons/` - Evolution comparison plots

## Advanced Usage

### Custom Embedding Analysis

```python
from visualization.embedding_visualization import EmbeddingExtractor, EmbeddingVisualizer

# Extract embeddings with custom parameters
extractor = EmbeddingExtractor()
embeddings_data = extractor.extract_embeddings_from_batches(
    model=model,
    dataloader=dataloader,
    max_batches=50
)

# Apply custom dimensionality reduction
config = EmbeddingVisualizationConfig(
    method='tsne',
    tsne_perplexity=50.0,
    n_components=3  # 3D visualization
)

visualizer = EmbeddingVisualizer(config)
plots = visualizer.visualize_embeddings(embeddings_data, save_dir)

# Save embeddings for later analysis
visualizer.save_embeddings(embeddings_data, 'my_embeddings.pkl')
```

### Integration with Existing Analysis

The embedding visualization integrates seamlessly with your existing attention analysis pipeline:

```python
# After running attention analysis
from visualization.embedding_visualization import visualize_embeddings_from_model

# Add embedding visualization
embedding_plots = visualize_embeddings_from_model(
    model=hetero_model,
    dataloader=dm.train[:20],
    save_dir=results_dir / 'embeddings',
    transcripts_df=transcripts,
    gene_types_dict=gene_types_dict,
    cell_types_dict=cell_types_dict
)
```

## Performance Tips

1. **Subsampling**: For large datasets, use `max_points_per_type` to limit visualization points
2. **Batch limitation**: Use `max_batches` parameter to limit memory usage
3. **Method selection**: 
   - UMAP: Best for preserving global structure, slower but higher quality
   - t-SNE: Good for local structure, moderate speed
   - PCA: Fastest, linear method, good for initial exploration

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `max_points_per_type` or `max_batches`
2. **UMAP installation**: `pip install umap-learn` (not just `umap`)
3. **TensorBoard logging**: Ensure TensorBoardLogger is used in trainer
4. **Missing metadata**: Gene/cell type dictionaries are optional but enhance visualization

### Dependencies

Required packages:
- `torch` and `torch_geometric`
- `umap-learn` for UMAP
- `scikit-learn` for t-SNE and PCA
- `matplotlib` and `seaborn` for plotting
- `pandas` for data handling
- `lightning` for training callbacks

## Examples

See `example_embedding_visualization.py` for comprehensive usage examples including:
- Post-training visualization
- Training with callbacks
- Custom analysis workflows

Run the examples:

```bash
# Run all examples
python example_embedding_visualization.py --example all

# Run specific example
python example_embedding_visualization.py --example post_training
```

## TensorBoard Integration

When using embedding callbacks during training, you can explore embeddings interactively:

```bash
tensorboard --logdir path/to/your/lightning_logs
```

Navigate to the "Projector" tab to explore embeddings in an interactive 3D space with metadata-based coloring and search capabilities.
