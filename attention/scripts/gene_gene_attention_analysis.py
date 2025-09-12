import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from sklearn.metrics import silhouette_score
import scanpy as sc
import pickle
import pandas as pd

def compute_average_gene_attention_matrix(attention_matrices, layer_idx=None, head_idx=None):
    """
    Compute the average gene-gene attention matrix across all layers and heads.
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

class GeneGeneAttentionAnalyzer:
    """
    A comprehensive pipeline for analyzing gene-gene attention patterns
    from attention matrices.
    """
    
    def __init__(self, num_genes=541, num_heads=4, num_layers=5):
        """
        Initialize the analyzer with data dimensions.
        
        Parameters:
        -----------
        num_genes : int
            Number of genes in the attention matrix
        num_heads : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        """
        self.num_genes = num_genes
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.gene_names = None
        self.gene_embeddings = None
        self.clusters = None
        self.umap_embedding = None
        self.attention_matrix = None
        
    def load_attention_data(self, attention_matrices, gene_names, layer_idx=None, head_idx=None):
        """
        Load attention matrices and metadata.
        
        Parameters:
        -----------
        attention_matrices : np.ndarray
            Shape: (num_layers, num_heads, num_genes, num_cells)
        gene_names : list or np.ndarray
            Gene names
        layer_idx : int, list, or None
            Layer index(es) to use
        head_idx : int, list, or None
            Head index(es) to use
        """
        self.gene_names = np.array(gene_names)
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        
        # Compute average attention matrix across genes to get gene-gene attention
        self.attention_matrix = compute_average_gene_attention_matrix(attention_matrices, layer_idx, head_idx)
        
        print(f"Loaded gene-gene attention data:")
        print(f"  - Attention matrices shape: {len(attention_matrices)} layers, {len(attention_matrices[0])} heads, {attention_matrices[0][0].shape[0]} genes")
        
    def visualize_attention_distribution(self, save_path=None):
        """
        Visualize the distribution of gene-gene attention weights.
        
        Parameters:
        -----------
        save_path : str or Path
            Path to save the plot
        """
        if self.attention_matrix is None:
            raise ValueError("Please load attention data first using load_attention_data()")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Gene-Gene Attention Weight Distributions', fontsize=16, fontweight='bold')
        
        # 1. Distribution of gene-wise mean attention (rows)
        gene_mean_attention = np.mean(self.attention_matrix, axis=1)
        axes[0].hist(gene_mean_attention, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Mean Attention per Gene (Row)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of Gene Mean Attention (Rows)')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Distribution of gene-wise mean attention (columns)
        gene_mean_attention_cols = np.mean(self.attention_matrix, axis=0)
        axes[1].hist(gene_mean_attention_cols, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('Mean Attention per Gene (Column)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Distribution of Gene Mean Attention (Columns)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention distribution plot saved to: {save_path}")
        
        return fig
    
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
            # Normalize within each row (gene)
            row_sums = np.sum(self.attention_matrix, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            self.attention_matrix = self.attention_matrix / row_sums
                    
        elif method == 'standard':
            # Standardization (z-score)
            scaler = StandardScaler()
            self.attention_matrix = scaler.fit_transform(self.attention_matrix)
                    
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        print(f"Attention weights normalized using {method} method.")
    
    def create_gene_embeddings(self):
        """
        Create gene embeddings from attention matrix.
        """
        if self.attention_matrix is None:
            raise ValueError("Please load attention data first using load_attention_data()")
        self.gene_embeddings = self.attention_matrix
        
    def perform_umap_embedding(self, n_neighbors=15, n_components=2, min_dist=0.1, random_state=42, metric='cosine'):
        """
        Perform UMAP dimensionality reduction on gene embeddings.
        
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
        np.ndarray : UMAP embedding (num_genes, n_components)
        """
        if self.gene_embeddings is None:
            raise ValueError("Please create gene embeddings first using create_gene_embeddings()")
            
        print(f"Performing UMAP embedding with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}...")
        
        # Create AnnData object for genes
        self.adata = sc.AnnData(self.gene_embeddings)
        
        # Perform PCA preprocessing
        sc.pp.pca(self.adata, n_comps=min(50, self.gene_embeddings.shape[1]), random_state=random_state)
        
        # Perform UMAP
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, random_state=random_state, metric=metric)
        sc.tl.umap(self.adata, min_dist=min_dist, random_state=random_state, n_components=n_components)
        self.umap_embedding = self.adata.obsm['X_umap']
        
        print(f"UMAP embedding completed:")
        print(f"  - Embedding shape: {self.umap_embedding.shape}")
        
        return self.umap_embedding
    
    def perform_clustering(self, method='leiden', **kwargs):
        """
        Perform clustering on gene embeddings.
        
        Parameters:
        -----------
        method : str
            Clustering method: 'leiden', 'hierarchical'
        **kwargs : additional parameters for clustering algorithms
        
        Returns:
        --------
        np.ndarray : Cluster labels
        """
        if self.gene_embeddings is None:
            raise ValueError("Please create gene embeddings first using create_gene_embeddings()")
        if self.umap_embedding is None:
            raise ValueError("Please perform UMAP embedding first using perform_umap_embedding()")
            
        print(f"Performing {method} clustering...")
        
        self.clustering_method = method
        
        gene_linkage_matrix = linkage(self.gene_embeddings, method='ward')
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
            
            self.clusters = fcluster(gene_linkage_matrix, t=self.n_clusters, criterion='maxclust') - 1
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(self.gene_embeddings, self.clusters)
            print(f"  - Number of clusters: {self.n_clusters}")
            print(f"  - Silhouette score: {silhouette_avg:.3f}")
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        print(f"Clustering completed with {len(np.unique(self.clusters))} clusters")
        print(f"Number of genes in each cluster: {np.bincount(self.clusters)}")
        
        # print 5 genes in each cluster
        for cluster_id in np.unique(self.clusters):
            print(f"Cluster {cluster_id}: {self.gene_names[self.clusters == cluster_id][:5]}")
        
        return self.clusters
    
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
                      c=[colors[i]], alpha=0.7, s=20, label=f'Cluster {cluster_id}')
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title('Gene UMAP Embedding Colored by Clusters', fontsize=14, fontweight='bold')
        
        # Add legend with smaller font and outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"UMAP cluster plot saved to: {save_path}")
            
        return fig
    
    def visualize_gene_gene_heatmap(self, top_k=None, threshold = 0.001, save_path=None, figsize=(15, 12), vmin=None, vmax=None, max_genes=None):
        """
        Visualize gene-gene attention heatmap with hierarchical clustering.
        
        Parameters:
        -----------
        top_k : int or None
            If provided, only show the top k attention weights per gene, setting others to 0
        threshold : float
            Threshold for attention weights to be considered
        save_path : str or Path
            Path to save the plot
        figsize : tuple
            Figure size
        vmin, vmax : float
            Min and max values for color scaling
        max_genes : int
            Maximum number of genes to show (for performance)
        
        Returns:
        --------
        matplotlib.figure.Figure : The created figure
        """
        if self.attention_matrix is None:
            raise ValueError("Please load attention data first using load_attention_data()")
        if self.gene_leaf_order is None:
            raise ValueError("Please perform clustering first using perform_clustering()")
        
        print("Creating gene-gene attention heatmap...")
        
        # Limit number of genes if specified
        if max_genes is not None and max_genes < len(self.gene_names):
            # Take a subset of genes for visualization
            subset_indices = np.linspace(0, len(self.gene_names)-1, max_genes, dtype=int)
            attention_subset = self.attention_matrix[subset_indices][:, subset_indices]
            gene_names_subset = self.gene_names[subset_indices]
            
            # Recompute hierarchical clustering for subset
            gene_linkage_matrix = linkage(attention_subset, method='ward')
            gene_leaf_order = leaves_list(gene_linkage_matrix)
            
            # Reorder data
            sorted_attention_matrix = attention_subset[gene_leaf_order][:, gene_leaf_order]
            sorted_gene_names = gene_names_subset[gene_leaf_order]
        else:
            # Use all genes
            sorted_attention_matrix = self.attention_matrix[self.gene_leaf_order][:, self.gene_leaf_order]
            sorted_gene_names = self.gene_names[self.gene_leaf_order]
        
        # If top_k is not None, create a modified attention matrix with only top k connections
        if top_k is not None:
            # Create a new matrix with same shape as sorted_attention_matrix
            modified_attention_matrix = np.zeros_like(sorted_attention_matrix)
            
            # For each gene, keep only the top k attention weights above threshold
            
            for i in range(len(sorted_gene_names)):
                # Get attention weights for this gene (excluding self-attention)
                attention_weights = sorted_attention_matrix[i].copy()
                # attention_weights[i] = 0  # Exclude self-attention
                
                # Filter attention weights above threshold
                above_threshold = attention_weights > threshold
                if np.sum(above_threshold) > 0:
                    # Get top k indices among those above threshold
                    above_threshold_indices = np.where(above_threshold)[0]
                    if len(above_threshold_indices) > top_k:
                        # Get top k attention weights among those above threshold
                        top_indices = above_threshold_indices[np.argsort(attention_weights[above_threshold_indices])[-top_k:][::-1]]
                    else:
                        # Use all indices above threshold
                        top_indices = above_threshold_indices
                else:
                    # If no weights above threshold, set top k to 0
                    top_indices = np.zeros(len(attention_weights))
                
                # Set only top k attention weights to 1 for highlighting, rest to 0
                for j in range(len(attention_weights)):
                    if j in top_indices:
                        modified_attention_matrix[i, j] = 1  # Highlight top k attention weights
                    else:
                        modified_attention_matrix[i, j] = 0
            
            sorted_attention_matrix = modified_attention_matrix
        
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        if top_k is not None:
            # For top_k highlighting, use binary scale (0 to 1)
            vmin = 0
            vmax = 1
        else:
            # For regular visualization, use percentile-based scaling
            if vmin is None:
                vmin = np.percentile(sorted_attention_matrix, 5)
            if vmax is None:
                vmax = np.percentile(sorted_attention_matrix, 95)
            vmin = np.max(vmin, 0)
        
        im = ax.imshow(sorted_attention_matrix, cmap='Reds', aspect='equal', vmin=vmin, vmax=vmax)
        
        # Set ticks and labels
        tick_step = max(1, len(sorted_gene_names) // 20)  # Show every nth gene name
        tick_positions = np.arange(0, len(sorted_gene_names), tick_step)
        tick_labels = sorted_gene_names[tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha='right')
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=8)
        
        ax.set_xlabel('Genes (hierarchical order)', fontsize=12)
        ax.set_ylabel('Genes (hierarchical order)', fontsize=12)
        if top_k is not None:
            ax.set_title(f'Gene-Gene Attention Matrix (Top {top_k} per gene)', fontsize=14, fontweight='bold')
        else:
            ax.set_title('Gene-Gene Attention Matrix', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gene-gene heatmap saved to: {save_path}")
            
        return fig
    
    def get_top_genes_per_gene(self, top_k=5, threshold=0.001):
        """
        Get top genes per gene.
        
        Parameters:
        -----------
        top_k : int
            Number of top genes to return per gene
        threshold : float
            Threshold for attention weights to be considered
            
        Returns:
        --------
        dict : Dictionary with gene names as keys and top genes as values
        """
        if self.attention_matrix is None:
            raise ValueError("Please load attention data first using load_attention_data()")
        if self.gene_leaf_order is None:
            raise ValueError("Please perform clustering first using perform_clustering()")
        
        print(f"Finding top {top_k} genes per gene...")
        
        top_genes_per_gene = {}
        
        for i, gene_idx in enumerate(self.gene_leaf_order):
            gene_name = self.gene_names[gene_idx]
            
            # Get attention weights for this gene (excluding self-attention)
            attention_weights = self.attention_matrix[gene_idx].copy()
            # attention_weights[gene_idx] = 0  # Exclude self-attention
            
            # Get top k genes above threshold
            above_threshold = attention_weights > threshold
            if np.sum(above_threshold) > 0:
                above_threshold_indices = np.where(above_threshold)[0]
                if len(above_threshold_indices) > top_k:
                    # Get top k attention weights among those above threshold
                    top_indices = above_threshold_indices[np.argsort(attention_weights[above_threshold_indices])[-top_k:][::-1]]
                else:
                    # Use all indices above threshold
                    top_indices = above_threshold_indices
            else:
                # If no weights above threshold, get top k overall
                top_indices = np.argsort(attention_weights)[-top_k:][::-1]
            
            # Get top genes with their attention weights
            top_genes = [(self.gene_names[idx], attention_weights[idx]) 
                        for idx in top_indices if attention_weights[idx] > threshold]
            
            top_genes_per_gene[gene_name] = top_genes
        
        print(f"Top genes identified for {len(top_genes_per_gene)} genes")
        
        return top_genes_per_gene
    
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
        self.adata.obs['cell_ids'] = self.gene_names
        
        sc.pl.matrixplot(self.adata, marker_genes_dict, groupby='cluster', 
                      dendrogram=True, show=True,
                      colorbar_title = 'Mean Attention Weights in Cluster')
            
        return None
    
    def run_complete_analysis(self, **kwargs):
        """
        Run the complete gene-gene attention analysis pipeline.
        """
        attention_matrices = kwargs.get('attention_matrices', None)
        gene_names = kwargs.get('gene_names', None)
        layer_idx = kwargs.get('layer_idx', 0)
        head_idx = kwargs.get('head_idx', None)
        clustering_method = kwargs.get('clustering_method', 'leiden')
        resolution = kwargs.get('resolution', 0.5)
        random_state = kwargs.get('random_state', 42)
        max_genes = kwargs.get('max_genes', None)
        marker_genes_dict = kwargs.get('marker_genes_dict', None)
        threshold = kwargs.get('threshold', 0.001)
        save_path_umap = kwargs.get('save_path_umap', None)
        save_path_gene_gene_heatmap = kwargs.get('save_path_gene_gene_heatmap', None)
        save_path_attention_distribution = kwargs.get('save_path_attention_distribution', None)
        
        self.load_attention_data(attention_matrices, gene_names, layer_idx=layer_idx, head_idx=head_idx)
        self.visualize_attention_distribution(save_path=save_path_attention_distribution)
        self.normalize_attention_weights(method='layer_norm')
        self.create_gene_embeddings()
        self.perform_umap_embedding(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42, metric='cosine')
        self.perform_clustering(method=clustering_method, resolution=resolution, random_state=random_state)
        self.visualize_umap_clusters(save_path=save_path_umap)
        self.visualize_gene_gene_heatmap(max_genes=max_genes, threshold=threshold, save_path=save_path_gene_gene_heatmap)
        # Visualize with top 5 genes per gene
        self.visualize_gene_gene_heatmap(top_k=5, threshold=threshold, save_path=save_path_gene_gene_heatmap.replace('.png', '_top5.png') if save_path_gene_gene_heatmap else None)
        if marker_genes_dict is not None:
            self.visualize_dotplot(marker_genes_dict)
        
        return None
    
    def save_analysis_results(self, save_path=None):
        """
        Save the analysis results.
        """
        if save_path is None:
            save_path = f"gene_gene_analysis_results_{self.layer_idx}_{self.head_idx}.pkl"
            
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
            
    def load_analysis_results(self, save_path=None):
        """
        Load the analysis results.
        """
        if save_path is None:
            save_path = f"gene_gene_analysis_results_{self.layer_idx}_{self.head_idx}.pkl"
            
        with open(save_path, 'rb') as f:
            self = pickle.load(f) 