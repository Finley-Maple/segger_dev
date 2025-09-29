import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from sklearn.metrics import silhouette_score
import scanpy as sc
import pickle
import pandas as pd

def compute_average_attention_matrix(attention_matrices, layer_idx=None, head_idx=None):
    """
    Compute the average attention matrix across all layers and heads.
    """
    # Compute average attention matrix
    avg_matrix = np.zeros_like(attention_matrices[0][0].toarray())
    
    if layer_idx is None:
        layer_idxs = range(len(attention_matrices))
    elif isinstance(layer_idx, int):
        layer_idxs = [layer_idx]
    elif isinstance(layer_idx, list):
        layer_idxs = layer_idx
    else:
        raise ValueError(f"Invalid layer_idx: {layer_idx}, type: {type(layer_idx)}")
        
    if head_idx is None:
        head_idxs = range(len(attention_matrices[0]))
    elif isinstance(head_idx, int):
        head_idxs = [head_idx]
    elif isinstance(head_idx, list):
        head_idxs = head_idx
    else:
        raise ValueError(f"Invalid head_idx: {head_idx}, type: {type(head_idx)}")
        
    for layer_idx in layer_idxs:
        for head_idx in head_idxs:
            avg_matrix += attention_matrices[layer_idx][head_idx].toarray()
            
    avg_matrix = avg_matrix / (len(layer_idxs) * len(head_idxs))
    
    return avg_matrix

class GeneCellAttentionAnalyzer:
    """
    A comprehensive pipeline for analyzing Gene-Cell spatial transcriptomics data
    using attention matrices as cell embeddings.
    """
    
    def __init__(self, num_genes=541, num_cells=410000, num_heads=4, num_layers=5, cell_type_dict=None):
        """
        Initialize the analyzer with data dimensions.
        
        Parameters:
        -----------
        num_genes : int
            Number of genes in the attention matrix
        num_cells : int
            Number of cells in the attention matrix
        num_heads : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        layer_idx : int
            Layer index to use
        head_idx : int
            Head index to use
        """
        self.num_genes = num_genes
        self.num_cells = num_cells
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cell_type_dict = cell_type_dict
        
        self.cell_ids = None
        self.gene_names = None
        self.cell_embeddings = None
        self.clusters = None
        self.umap_embedding = None
        self.attention_matrix = None
        
    def load_attention_data(self, attention_matrices, cell_ids, gene_names, layer_idx=None, head_idx=None, cell_type_dict=None):
        """
        Load attention matrices and metadata.
        
        Parameters:
        -----------
        attention_matrices : np.ndarray
            Shape: (num_layers, num_heads, num_genes, num_cells)
        cell_ids : list or np.ndarray
            Cell identifiers
        gene_names : list or np.ndarray
            Gene names
        cell_type_dict : dict
            Dictionary of cell types and their corresponding cell ids
        """
        self.cell_ids = np.array(cell_ids)
        self.gene_names = np.array(gene_names)
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.attention_matrix = compute_average_attention_matrix(attention_matrices, layer_idx, head_idx)
        self.cell_type_dict = cell_type_dict
        
        if cell_type_dict is not None:
            cell_types = []     
            for cell_id in cell_ids:
                if cell_id not in cell_type_dict.keys():
                    cell_types.append('Unknown')
                else:
                    cell_types.append(cell_type_dict[cell_id])
            self.cell_types = np.array(cell_types)
        else:
            self.cell_types = None
        
        print(f"Loaded attention data:")
        print(f"  - Attention matrices shape: {len(attention_matrices)} layers, {len(attention_matrices[0])} heads, {attention_matrices[0][0].shape[0]} genes, {attention_matrices[0][0].shape[1]} cells")
        
    def visualize_attention_distribution(self, save_path=None):
        """
        Visualize the distribution of attention weights across cells and layers.
        
        Parameters:
        -----------
        layer_idx : int
            Layer index to visualize (default: 0)
        head_idx : int
            Head index to visualize (default: 0)
        save_path : str or Path
            Path to save the plot
        """
        if self.attention_matrix is None:
            raise ValueError("Please load attention data first using load_attention_data()")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Attention Weight Distributions', fontsize=16, fontweight='bold')
        
        # 1. Distribution of cell-wise mean attention
        cell_mean_attention = np.mean(self.attention_matrix, axis=0)
        axes[0].hist(cell_mean_attention, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Mean Attention per Cell')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of Cell Mean Attention')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Distribution of gene-wise mean attention
        gene_mean_attention = np.mean(self.attention_matrix, axis=1)
        axes[1].hist(gene_mean_attention, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('Mean Attention per Gene')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Distribution of Gene Mean Attention')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention distribution plot saved to: {save_path}")
        
        return fig
    
    def filter_attention_cells(self, min_attention_threshold=None, max_attention_threshold=None, max_cells=None):
        """
        Filter out cells with attention weights outside the specified range.
        
        Parameters:
        -----------
        min_attention_threshold : float
            Minimum mean attention threshold for cells
        max_attention_threshold : float
            Maximum mean attention threshold for cells
        layer_idx : int
            Layer to use for filtering
        head_idx : int
            Head to use for filtering
        
        Returns:
        --------
        dict : Information about filtered cells
        """
        if self.attention_matrix is None:
            raise ValueError("Please load attention data first using load_attention_data()")
          
          
        # Calculate mean attention per cell for the specified layer and head
        cell_mean_attention = np.mean(self.attention_matrix, axis=0)
        
        # If no thresholds are provided, use the mean and 3 standard deviations
        if min_attention_threshold is None and max_attention_threshold is None:
            cell_sd_attention = np.std(cell_mean_attention)
            min_attention_threshold = np.max([0, np.mean(cell_mean_attention) - 2 * cell_sd_attention])
            max_attention_threshold = np.mean(cell_mean_attention) + 2 * cell_sd_attention

        print(f"Filtering cells with mean attention between {min_attention_threshold} and {max_attention_threshold}...")
        
        # Find cells above threshold
        valid_cells_mask = (cell_mean_attention >= min_attention_threshold) & (cell_mean_attention <= max_attention_threshold)
        
        # Filter data
        self.attention_matrix = self.attention_matrix[:, valid_cells_mask]        
        self.cell_ids = self.cell_ids[valid_cells_mask]
        self.cell_types = self.cell_types[valid_cells_mask] if self.cell_types is not None else None
        
        if max_cells is not None:
            self.attention_matrix = self.attention_matrix[:, :max_cells]
            self.cell_ids = self.cell_ids[:max_cells]
            self.cell_types = self.cell_types[:max_cells] if self.cell_types is not None else None
            self.num_cells = max_cells
        else:
            # Update cell count
            self.num_cells = len(self.cell_ids)
        
        filter_info = {
            'original_num_cells': len(valid_cells_mask),
            'filtered_num_cells': self.num_cells,
            'removed_cells': len(valid_cells_mask) - self.num_cells,
            'threshold_used': (min_attention_threshold, max_attention_threshold),
            'max_cells': max_cells if max_cells is not None else self.num_cells
        }
        self.filter_info = filter_info
        
        
        print(f"Cell filtering completed:")
        print(f"  - Original cells: {filter_info['original_num_cells']}")
        print(f"  - Filtered cells: {filter_info['filtered_num_cells']}")
        print(f"  - Removed cells: {filter_info['removed_cells']}")
        print(f"  - Threshold: {filter_info['threshold_used']}")

        return filter_info
    
    def normalize_attention_weights(self, method='layer_norm'):
        """
        Normalize attention weights across different dimensions.
        
        Parameters:
        -----------
        method : str
            Normalization method: 'layer_norm', 'standard'
        """
        if self.attention_matrix is None:
            raise ValueError("Please load attention data first using load_attention_data()")
            
        print(f"Normalizing attention weights using {method} method...")
        
        if method == 'layer_norm':
            # Normalize within each layer and head
            row_sums = np.sum(self.attention_matrix, axis=0, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            self.attention_matrix = self.attention_matrix / row_sums
                    
        elif method == 'standard':
            # Standardization (z-score)
            scaler = StandardScaler()
            self.attention_matrix = scaler.fit_transform(self.attention_matrix)
                    
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        print(f"Attention weights normalized using {method} method.")
    
    def create_cell_embeddings(self):
        """
        Create cell embeddings from attention matrix.
        """
        if self.attention_matrix is None:
            raise ValueError("Please load attention data first using load_attention_data()")
        self.cell_embeddings = self.attention_matrix.T
        
    def perform_umap_embedding(self, n_neighbors=15, n_components=2, min_dist=0.1, random_state=42, metric='cosine'):
        """
        Perform UMAP dimensionality reduction on cell embeddings.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors for UMAP
        min_dist : float
            Minimum distance for UMAP
        random_state : int
            Random state for reproducibility
        metric : str
            Distance metric for UMAP
        
        Returns:
        --------
        np.ndarray : UMAP embedding (num_cells, 2)
        """
        if self.cell_embeddings is None:
            raise ValueError("Please create cell embeddings first using create_cell_embeddings()")
            
        print(f"Performing UMAP embedding with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}...")
        
        # Create AnnData object for cells
        self.adata = sc.AnnData(self.cell_embeddings)
        
        # Perform PCA preprocessing
        sc.pp.pca(self.adata, n_comps=min(50, self.cell_embeddings.shape[1]), random_state=random_state)
        
        # Perform UMAP
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, random_state=random_state, metric=metric)
        sc.tl.umap(self.adata, min_dist=min_dist, random_state=random_state, n_components=n_components)
        self.umap_embedding = self.adata.obsm['X_umap']
        
        print(f"UMAP embedding completed:")
        print(f"  - Embedding shape: {self.umap_embedding.shape}")
        
        return self.umap_embedding
    
    def perform_clustering(self, method='leiden', **kwargs):
        """
        Perform clustering on cell embeddings.
        
        Parameters:
        -----------
        method : str
            Clustering method: 'leiden', 'hierarchical'
        **kwargs : additional parameters for clustering algorithms: for leiden, resolution and random_state are supported; for hierarchical, n_clusters is supported.
        Returns:
        --------
        np.ndarray : Cluster labels
        """
        if self.cell_embeddings is None:
            raise ValueError("Please create cell embeddings first using create_cell_embeddings()")
        if self.umap_embedding is None:
            raise ValueError("Please perform UMAP embedding first using perform_umap_embedding()")
            
        print(f"Performing {method} clustering...")
        
        self.clustering_method = method
        
        cell_linkage_matrix = linkage(self.cell_embeddings, method='ward')
        self.cell_leaf_order = leaves_list(cell_linkage_matrix)
        
        gene_linkage_matrix = linkage(self.attention_matrix, method='ward')
        self.gene_leaf_order = leaves_list(gene_linkage_matrix)
        
        if method == 'leiden':
            self.resolution = kwargs.get('resolution', 0.8)
            self.random_state = kwargs.get('random_state', 42)
            sc.tl.leiden(self.adata, resolution=self.resolution, random_state=self.random_state)
            self.clusters = self.adata.obs['leiden'].values
            unique_labels = np.unique(self.clusters)
            label_mapping = {label: i for i, label in enumerate(unique_labels)}
            self.clusters = np.array([label_mapping[label] for label in self.clusters])
         
        elif method == 'hierarchical':
            self.n_clusters = kwargs.get('n_clusters', 10)
            self.random_state = kwargs.get('random_state', 42)
            
            self.clusters = fcluster(cell_linkage_matrix, t=self.n_clusters, criterion='maxclust') - 1 # -1 because the cluster labels start from 0
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(self.cell_embeddings, self.clusters)
            print(f"  - Number of clusters: {self.n_clusters}")
            print(f"  - Silhouette score: {silhouette_avg:.3f}")
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        print(f"Clustering completed with {len(np.unique(self.clusters))} clusters")
        # print the number of cells in each cluster
        print(f"Number of cells in each cluster: {np.bincount(self.clusters)}")
        
        return self.clusters
    
    def get_top_genes_per_cluster(self, top_k=10, threshold=0.001, true_cell_type=False):
        """
        Get top genes with highest attention weights for each cluster.
        
        Parameters:
        -----------
        top_k : int
            Number of top genes to return per cluster
        threshold : float
            Threshold for attention weights to be considered
        true_cell_type : bool
            Whether to use true cell types or cluster labels
        Returns:
        --------
        dict : Dictionary with cluster labels as keys and top genes as values
        """
        if self.clusters is None and true_cell_type is False:
            raise ValueError("Please perform clustering first using perform_clustering()")
        if true_cell_type is True and self.cell_type_dict is None:
            raise ValueError("Please provide cell type dictionary using load_attention_data()")
        
        if true_cell_type is True:
            print(f"Using true cell types to get top genes per true cell type...")
        else:
            print(f"Finding top {top_k} genes per cluster...")
        
        attention_matrix = self.attention_matrix
        if true_cell_type is True:
            unique_clusters = np.unique(self.cell_types)
        else:
            unique_clusters = np.unique(self.clusters)
        top_genes_per_cluster = {}
        
        for cluster_id in unique_clusters:
            # Get cells in this cluster
            if true_cell_type is True:
                cluster_cells = self.cell_types == cluster_id
            else:
                cluster_cells = self.clusters == cluster_id
            
            # Calculate mean attention for each gene in this cluster
            cluster_attention = attention_matrix[:, cluster_cells]
            gene_mean_attention = np.mean(cluster_attention, axis=1)
            
            # Get top k genes
            top_gene_indices = np.argsort(gene_mean_attention)[-top_k:][::-1]
            top_genes = [(self.gene_names[idx], gene_mean_attention[idx]) 
                        for idx in top_gene_indices if gene_mean_attention[idx] > threshold] # filter out genes with mean attention less than threshold
            
            top_genes_per_cluster[cluster_id] = top_genes
            
        print(f"Top genes identified for {len(top_genes_per_cluster)} clusters")
        
        return top_genes_per_cluster
      
    def visualize_umap_clusters(self, save_path=None, figsize=(12, 10)):
        """
        Visualize UMAP embedding colored by clusters.
        
        Parameters:
        -----------
        save_path : str or Path
            Path to save the plot
        figsize : tuple
            Figure size
        
        Returns:
        --------
        matplotlib.figure.Figure : The created figure
        """
        if self.umap_embedding is None or self.clusters is None:
            raise ValueError("Please perform UMAP embedding and clustering first")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        unique_clusters = np.unique(self.clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = self.clusters == cluster_id
            ax.scatter(self.umap_embedding[mask, 0], self.umap_embedding[mask, 1], 
                      c=[colors[i]], alpha=0.7, s=2, label=f'Cluster {cluster_id}')
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title('UMAP Embedding Colored by Clusters', fontsize=14, fontweight='bold')
        
        # Add legend with smaller font and outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"UMAP cluster plot saved to: {save_path}")
            
        return fig
    
    def visualize_top_genes_cell_heatmap(self, top_genes_dict, top_k=5, save_path=None, figsize=(12, 8), vmin=0, vmax=0.2):
        """
        Visualize top genes per cluster as a heatmap, arranged by hierarchical clustering order.
        Parameters:
        -----------
        top_genes_dict : dict
            Dictionary from get_top_genes_per_cluster()
        top_k : int
            Number of top genes to show per cluster
        save_path : str or Path
            Path to save the plot
        figsize : tuple
            Figure size
        vmin : float
            Minimum value for the colorbar
        vmax : float
            Maximum value for the colorbar
        """
        if top_genes_dict is None:
            raise ValueError("Please get top genes per cluster first using get_top_genes_per_cluster()")
        if self.cell_leaf_order is None:
            raise ValueError("Please perform clustering first using perform_clustering()")
        if self.gene_leaf_order is None:
            raise ValueError("Please perform clustering first using perform_clustering()")
        
        # Collect all unique genes from top genes
        all_genes = set()
        for cluster_genes in top_genes_dict.values():
            for gene, _ in cluster_genes[:top_k]:
                all_genes.add(gene)
        
        # Get the original gene indices for the genes we want to show
        gene_indices = []
        for gene in all_genes:
            gene_idx = np.where(self.gene_names == gene)[0]
            if len(gene_idx) > 0:
                gene_indices.append(gene_idx[0])
        
        # Arrange the genes in the order of the gene leaf order and the cells in the order of the cell leaf order
        # First, get the attention matrix for the selected genes
        selected_attention_matrix = self.attention_matrix[gene_indices, :]
        
        # Then reorder according to hierarchical clustering
        # Find the positions of our selected genes in the gene leaf order
        gene_leaf_positions = []
        for gene_idx in gene_indices:
            pos = np.where(self.gene_leaf_order == gene_idx)[0]
            if len(pos) > 0:
                gene_leaf_positions.append(pos[0])
        
        # Sort the genes by their position in the leaf order
        sorted_indices = np.argsort(gene_leaf_positions)
        sorted_gene_indices = [gene_indices[i] for i in sorted_indices]
        
        # Create the final sorted attention matrix
        sorted_attention_matrix = selected_attention_matrix[sorted_indices][:, self.cell_leaf_order]
        sorted_gene_names = self.gene_names[sorted_gene_indices]
                    
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(sorted_attention_matrix, cmap='Reds', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set ticks and labels
        ax.set_yticks(range(len(sorted_gene_names)))
        ax.set_yticklabels(sorted_gene_names, fontsize=8)
        
        ax.set_xlabel('Cells (hierarchical order)', fontsize=12)
        ax.set_ylabel('Genes (hierarchical order)', fontsize=12)
        ax.set_title(f'Top {top_k} Genes per Cluster (Attention Weights)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Attention Weight', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Top genes cell heatmap saved to: {save_path}")
            
        return fig
    
    def visualize_top_genes_cluster_heatmap(self, top_genes_dict, top_k=5, save_path=None, figsize=(12, 8), vmin=0, vmax=0.2, true_cell_type=False):
        """
        Visualize top genes per cluster as a heatmap, with the genes arranged by hierarchical clustering order.
        If true_cell_type is True, the clusters are the true cell types.
        Parameters:
        -----------
        top_genes_dict : dict
            Dictionary from get_top_genes_per_cluster()
        top_k : int
            Number of top genes to show per cluster
        save_path : str or Path
            Path to save the plot
        figsize : tuple
            Figure size
        vmin : float
            Minimum value for the colorbar
        vmax : float
            Maximum value for the colorbar
        true_cell_type : bool
            Whether the clusters are the true cell types
        Returns:
        --------
        matplotlib.figure.Figure : The created figure
        """
        
        if self.gene_leaf_order is None:
            raise ValueError("Please perform clustering first using perform_clustering()")
        
        print("Creating top genes heatmap...")
        
        # Collect all unique genes from top genes
        all_genes = set()
        for cluster_genes in top_genes_dict.values():
            for gene, _ in cluster_genes[:top_k]:
                all_genes.add(gene)
        
        # Get the original gene indices for the genes we want to show
        gene_indices = []
        for gene in all_genes:
            gene_idx = np.where(self.gene_names == gene)[0]
            if len(gene_idx) > 0:
                gene_indices.append(gene_idx[0])
        
        # Find the positions of our selected genes in the gene leaf order
        gene_leaf_positions = []
        for gene_idx in gene_indices:
            pos = np.where(self.gene_leaf_order == gene_idx)[0]
            if len(pos) > 0:
                gene_leaf_positions.append(pos[0])
        
        # Sort the genes by their position in the leaf order
        sorted_indices = np.argsort(gene_leaf_positions)
        sorted_gene_indices = [gene_indices[i] for i in sorted_indices]
        sorted_gene_names = self.gene_names[sorted_gene_indices]
        
        clusters = sorted(top_genes_dict.keys())
        
        # Create attention matrix for heatmap with genes in hierarchical order
        heatmap_data = np.zeros((len(sorted_gene_names), len(clusters)))
        
        for j, cluster_id in enumerate(clusters):
            cluster_genes = dict(top_genes_dict[cluster_id][:top_k])
            for i, gene in enumerate(sorted_gene_names):
                if gene in cluster_genes:
                    heatmap_data[i, j] = cluster_genes[gene]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set ticks and labels
        ax.set_xticks(range(len(clusters)))
        if true_cell_type is True:
            ax.set_xticklabels([f'{c}' for c in clusters], rotation=45)
        else:
            ax.set_xticklabels([f'Cluster {c}' for c in clusters], rotation=45)
        ax.set_yticks(range(len(sorted_gene_names)))
        ax.set_yticklabels(sorted_gene_names, fontsize=8)
        
        if true_cell_type is False:
            ax.set_xlabel('Clusters', fontsize=12)
        else:
            ax.set_xlabel('Cell Types', fontsize=12)
        ax.set_ylabel('Genes (hierarchical order)', fontsize=12)
        
        if true_cell_type is False:
            ax.set_title(f'Top {top_k} Genes per Cluster (Attention Weights)', fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Top {top_k} Genes per Cell Type (Attention Weights)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Attention Weight', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Top genes heatmap saved to: {save_path}")
            
        return fig
    
    def visualize_dotplot(self, marker_genes_dict):
        """
        Visualize dotplot of top genes per cluster.
        """
        if marker_genes_dict is None:
            raise ValueError("Please get marker genes per cluster first using get_top_genes_per_cluster()")
        
        # map integer cluster labels to string cluster labels
        cluster_labels = {i: f'Cluster {i}' for i in range(len(np.unique(self.clusters)))}
        clusters_names = [cluster_labels[cluster] for cluster in self.clusters]
        
        self.adata.var_names = self.gene_names
        self.adata.obs['cluster'] = pd.Categorical(clusters_names)
        self.adata.obs['cell_ids'] = self.cell_ids
        
        sc.pl.dotplot(self.adata, marker_genes_dict, groupby='cluster', 
                      dendrogram=True, show=True,
                      colorbar_title = 'Mean Attention Weights in Cluster')
            
        return None
    
    def run_complete_analysis(self, **kwargs):
        """
        Run the complete analysis pipeline.
        """
        attention_matrices = kwargs.get('attention_matrices', None)
        cell_ids = kwargs.get('cell_ids', None)
        cell_type_dict = kwargs.get('cell_type_dict', None)
        gene_names = kwargs.get('gene_names', None)
        max_cells = kwargs.get('max_cells', 10000)
        min_attention_threshold = kwargs.get('min_attention_threshold', None)
        max_attention_threshold = kwargs.get('max_attention_threshold', None)
        layer_idx = kwargs.get('layer_idx', 0)
        head_idx = kwargs.get('head_idx', None)
        clustering_method = kwargs.get('clustering_method', 'leiden')
        resolution = kwargs.get('resolution', 0.5)
        random_state = kwargs.get('random_state', 42)
        top_k = kwargs.get('top_k', 5)
        threshold = kwargs.get('threshold', 0.001)
        marker_genes_dict = kwargs.get('marker_genes_dict', None)
        save_path_umap = kwargs.get('save_path_umap', None)
        save_path_top_genes_cluster_heatmap = kwargs.get('save_path_top_genes_cluster_heatmap', None)
        save_path_top_genes_cell_heatmap = kwargs.get('save_path_top_genes_cell_heatmap', None)
        save_path_attention_distribution = kwargs.get('save_path_attention_distribution', None)
        
        self.load_attention_data(attention_matrices, cell_ids, gene_names, layer_idx=layer_idx, head_idx=head_idx, cell_type_dict=cell_type_dict)
        self.filter_attention_cells(max_cells=max_cells, min_attention_threshold=min_attention_threshold, max_attention_threshold=max_attention_threshold)
        self.visualize_attention_distribution(save_path=save_path_attention_distribution)
        self.normalize_attention_weights(method='layer_norm')
        self.create_cell_embeddings()
        self.perform_umap_embedding(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42, metric='cosine')
        self.perform_clustering(method=clustering_method, resolution=resolution, random_state=random_state)
        self.top_genes_per_cluster = self.get_top_genes_per_cluster(top_k=top_k, threshold=threshold)
        if self.cell_type_dict is not None:
            self.top_genes_per_cell_type = self.get_top_genes_per_cluster(top_k=top_k, threshold=threshold, true_cell_type=True)
            self.visualize_top_genes_cluster_heatmap(top_genes_dict=self.top_genes_per_cell_type, top_k=top_k, save_path=save_path_top_genes_cell_heatmap, true_cell_type=True)
        self.visualize_umap_clusters(save_path=save_path_umap)
        self.visualize_top_genes_cluster_heatmap(top_genes_dict=self.top_genes_per_cluster, top_k=top_k, save_path=save_path_top_genes_cluster_heatmap)
        self.visualize_top_genes_cell_heatmap(top_genes_dict=self.top_genes_per_cluster, top_k=top_k, save_path=save_path_top_genes_cell_heatmap)
        if marker_genes_dict is not None:
            self.visualize_dotplot(marker_genes_dict=marker_genes_dict)
        
        return None
    
    def save_analysis_results(self, save_path=None):
        """
        Save the analysis results.
        """
        if save_path is None:
            save_path = f"analysis_results_{self.layer_idx}_{self.head_idx}.pkl"
            
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
            
    def load_analysis_results(self, save_path=None):
        """
        Load the analysis results.
        """
        if save_path is None:
            save_path = f"analysis_results_{self.layer_idx}_{self.head_idx}.pkl"
            
        with open(save_path, 'rb') as f:
            self = pickle.load(f)