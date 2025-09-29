"""
Simple Gene-to-Cell-Type Mapping from scRNAseq

This module extracts which genes are markers for which cell types
directly from scRNAseq data, without complex spatial assignment.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict, Optional


def extract_gene_celltype_dict(scrnaseq_adata, 
                              celltype_column: str = "Level1",
                              method: str = "differential_expression",
                              top_n_genes: int = 50,
                              min_expression_ratio: float = 1.5) -> Dict[str, str]:
    """
    Extract gene-to-cell-type mapping from scRNAseq data.
    
    Args:
        scrnaseq_adata: AnnData object with scRNAseq data
        celltype_column: Column name containing cell types
        method: Method to determine gene-cell type association
        top_n_genes: Number of top genes per cell type
        min_expression_ratio: Minimum ratio vs other cell types
        
    Returns:
        Dictionary mapping gene names to cell types
    """
    print(f"Extracting gene-to-cell-type mapping using method: {method}")
    
    # Make gene names unique
    scrnaseq_adata.var_names_make_unique()
    
    # Get cell types
    cell_types = scrnaseq_adata.obs[celltype_column].unique()
    print(f"Found {len(cell_types)} cell types: {list(cell_types)}")
    
    gene_celltype_dict = {}
    
    if method == "top_expressed":
        # For each cell type, find genes with highest average expression
        for cell_type in cell_types:
            print(f"Processing cell type: {cell_type}")
            
            # Get cells of this type
            cell_mask = scrnaseq_adata.obs[celltype_column] == cell_type
            cell_indices = np.where(cell_mask.values)[0]  # Convert to numpy indices
            
            if len(cell_indices) == 0:
                continue
                
            # Calculate mean expression for this cell type
            if hasattr(scrnaseq_adata.X, 'toarray'):
                cell_type_expr = scrnaseq_adata.X[cell_indices].toarray()
            else:
                cell_type_expr = scrnaseq_adata.X[cell_indices]
            
            mean_expr_this_type = np.mean(cell_type_expr, axis=0)
            
            # Calculate mean expression for other cell types
            other_cell_mask = ~cell_mask
            other_cell_indices = np.where(other_cell_mask.values)[0]
            if len(other_cell_indices) > 0:
                if hasattr(scrnaseq_adata.X, 'toarray'):
                    other_expr = scrnaseq_adata.X[other_cell_indices].toarray()
                else:
                    other_expr = scrnaseq_adata.X[other_cell_indices]
                mean_expr_others = np.mean(other_expr, axis=0)
            else:
                mean_expr_others = np.zeros_like(mean_expr_this_type)
            
            # Calculate expression ratio
            expr_ratio = np.divide(mean_expr_this_type, mean_expr_others + 1e-6)
            
            # Get top genes for this cell type
            # Filter by minimum expression and ratio
            valid_genes_mask = (mean_expr_this_type > 0.1) & (expr_ratio > min_expression_ratio)
            
            if valid_genes_mask.sum() > 0:
                valid_indices = np.where(valid_genes_mask)[0]
                valid_ratios = expr_ratio[valid_indices]
                
                # Sort by expression ratio and take top N
                top_indices = valid_indices[np.argsort(valid_ratios)[-top_n_genes:]]
                
                # Add to dictionary
                for idx in top_indices:
                    gene_name = scrnaseq_adata.var_names[idx]
                    # Only assign if not already assigned to a higher-ratio cell type
                    if gene_name not in gene_celltype_dict or expr_ratio[idx] > gene_celltype_dict.get(f"{gene_name}_ratio", 0):
                        gene_celltype_dict[gene_name] = cell_type
                        gene_celltype_dict[f"{gene_name}_ratio"] = expr_ratio[idx]  # Store ratio for comparison
                
                print(f"  Added {len(top_indices)} marker genes for {cell_type}")
    
    elif method == "differential_expression":
        # Use scanpy's rank_genes_groups for more sophisticated analysis
        print("Computing differential expression...")
        
        # Run differential expression analysis
        sc.tl.rank_genes_groups(scrnaseq_adata, celltype_column, method='wilcoxon')
        
        # Extract top genes for each cell type
        for cell_type in cell_types:
            if cell_type in scrnaseq_adata.uns['rank_genes_groups']['names'].dtype.names:
                top_genes = scrnaseq_adata.uns['rank_genes_groups']['names'][cell_type][:top_n_genes]
                for gene in top_genes:
                    if gene not in gene_celltype_dict:  # First come, first served
                        gene_celltype_dict[gene] = cell_type
                
                print(f"  Added {len(top_genes)} marker genes for {cell_type}")
    
    # Clean up ratio entries
    gene_celltype_dict = {k: v for k, v in gene_celltype_dict.items() if not k.endswith('_ratio')}
    
    print(f"Created gene-to-cell-type mapping for {len(gene_celltype_dict)} genes")
    print(f"Cell type distribution:")
    celltype_counts = pd.Series(list(gene_celltype_dict.values())).value_counts()
    for ct, count in celltype_counts.items():
        print(f"  {ct}: {count} genes")
    
    return gene_celltype_dict


def create_gene_celltype_dict_simple(scrnaseq_file: str, 
                                   celltype_column: str = "Level1") -> Dict[str, str]:
    """
    Simple function to create gene-to-cell-type dictionary from scRNAseq file.
    
    Args:
        scrnaseq_file: Path to scRNAseq h5ad file
        celltype_column: Column name for cell types
        
    Returns:
        Dictionary mapping gene names to cell types
    """
    print(f"Loading scRNAseq data from {scrnaseq_file}")
    
    # Load data
    adata = sc.read(scrnaseq_file)
    print(f"Loaded scRNAseq data: {adata.shape}")
    
    # Subsample if too large
    if adata.n_obs > 50000:
        print("Subsampling large dataset...")
        sc.pp.subsample(adata, 0.1, random_state=42)
    
    # Basic preprocessing
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    # Extract gene-cell type mapping
    gene_celltype_dict = extract_gene_celltype_dict(
        adata, 
        celltype_column=celltype_column,
        method="differential_expression"
    )
    
    return gene_celltype_dict
