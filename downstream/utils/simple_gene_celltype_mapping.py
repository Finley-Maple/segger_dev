"""
Simple Gene-to-Cell-Type Mapping from scRNAseq

This module extracts which genes are markers for which cell types
directly from scRNAseq data, without complex spatial assignment.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict, Optional, Union, Tuple
from pathlib import Path


def extract_gene_celltype_dict(scrnaseq_adata, 
                               celltype_column: str = "Level1",
                               method: str = "differential_expression",
                               top_n_genes: int = 200,  # Increased default
                               pval_cutoff: float = 0.05,  # for threshold-based
                               logfc_min: float = 0.5,  # for threshold-based
                               min_expression_ratio: float = 1.5,
                               return_celltype_counts: bool = False) -> Union[Dict[str, str], Tuple[Dict[str, str], Dict[str, int]]]:
    """
    Extract gene-to-cell-type mapping from scRNAseq data.
    
    Args:
        scrnaseq_adata: AnnData object with scRNAseq data
        celltype_column: Column name containing cell types
        method: Method to determine gene-cell type association
        top_n_genes: Number of top genes per cell type (used if threshold yields too many)
        pval_cutoff: Adjusted p-value threshold for DE significance
        logfc_min: Minimum log fold-change for DE significance
        return_celltype_counts: If True, also return count of cell types each gene was a candidate for
        
    Returns:
        Dictionary mapping gene names to cell types, optionally with celltype counts dict
    """
    print(f"Extracting gene-to-cell-type mapping using method: {method}")
    
    # Make gene names unique
    scrnaseq_adata.var_names_make_unique()
    
    # Get cell types
    cell_types = scrnaseq_adata.obs[celltype_column].unique()
    print(f"Found {len(cell_types)} cell types: {list(cell_types)}")
    
    gene_celltype_dict = {}
    
    if method == "differential_expression":
        print("Computing differential expression...")
        
        # Run differential expression analysis
        sc.tl.rank_genes_groups(scrnaseq_adata, celltype_column, method='wilcoxon')
        
        # Collect candidates: gene -> list of (cell_type, score)
        candidates = {}
        
        for cell_type in cell_types:
            if cell_type not in scrnaseq_adata.uns['rank_genes_groups']['names'].dtype.names:
                continue
            
            # Get DE dataframe for this cell type with thresholds
            de_df = sc.get.rank_genes_groups_df(scrnaseq_adata, group=cell_type)
            filtered_df = de_df[(de_df['pvals_adj'] < pval_cutoff) & (de_df['logfoldchanges'] > logfc_min)]
            
            # Optionally cap to top N if too many
            if len(filtered_df) > top_n_genes:
                filtered_df = filtered_df.sort_values('scores', ascending=False).head(top_n_genes)
            
            for _, row in filtered_df.iterrows():
                gene = row['names']
                score = row['scores']  # z-score; higher = more specific
                if gene not in candidates:
                    candidates[gene] = []
                candidates[gene].append((cell_type, score))
            
            print(f"  Found {len(filtered_df)} candidate marker genes for {cell_type} (after thresholds)")
        
        # Count how many cell types each gene was a candidate for
        gene_celltype_counts = {gene: len(type_scores) for gene, type_scores in candidates.items()}
        
        # Resolve assignments: pick cell type with max score for each gene
        for gene, type_scores in candidates.items():
            if gene not in gene_celltype_dict:  # Avoid duplicates, though unlikely
                best_type = max(type_scores, key=lambda x: x[1])[0]
                gene_celltype_dict[gene] = best_type
        
        print(f"Resolved assignments for {len(gene_celltype_dict)} unique genes after conflict resolution")
    
    print(f"Created gene-to-cell-type mapping for {len(gene_celltype_dict)} genes")
    print(f"Cell type distribution:")
    celltype_counts = pd.Series(list(gene_celltype_dict.values())).value_counts()
    for ct, count in celltype_counts.items():
        print(f"  {ct}: {count} genes")
    
    if return_celltype_counts:
        print(f"Cell type candidacy distribution:")
        candidacy_counts = pd.Series(list(gene_celltype_counts.values())).value_counts().sort_index()
        for count, num_genes in candidacy_counts.items():
            print(f"  {num_genes} genes were candidates for {count} cell type(s)")
        return gene_celltype_dict, gene_celltype_counts
    else:
        return gene_celltype_dict


def compute_mutually_exclusive_gene_counts(mutually_exclusive_gene_pairs, common_genes=None) -> Dict[str, int]:
    """
    Compute the number of mutually exclusive relationships for each gene.
    
    Args:
        mutually_exclusive_gene_pairs: Set or list of tuples containing gene pairs
        common_genes: Optional list of genes to include (with 0 count if no exclusive relationships)
        
    Returns:
        Dictionary mapping gene names to count of mutually exclusive relationships
    """
    gene_exclusive_counts = {}
    
    # Initialize all common genes with 0 count if provided
    if common_genes is not None:
        for gene in common_genes:
            gene_exclusive_counts[gene] = 0
    
    # Count exclusive relationships
    for gene1, gene2 in mutually_exclusive_gene_pairs:
        gene_exclusive_counts[gene1] = gene_exclusive_counts.get(gene1, 0) + 1
        gene_exclusive_counts[gene2] = gene_exclusive_counts.get(gene2, 0) + 1
    
    print(f"Computed mutually exclusive counts for {len(gene_exclusive_counts)} genes")
    if gene_exclusive_counts:
        exclusive_counts = pd.Series(list(gene_exclusive_counts.values())).value_counts().sort_index()
        print(f"Mutually exclusive distribution:")
        for count, num_genes in exclusive_counts.items():
            print(f"  {num_genes} genes have {count} mutually exclusive relationship(s)")
    
    return gene_exclusive_counts


def create_gene_celltype_dict_simple(scrnaseq_file: str, 
                                     celltype_column: str = "Level1",
                                     top_n_genes: int = 200,
                                     subsample_fraction: float = 0.5) -> Dict[str, str]:  # Increased default subsample
    """
    Simple function to create gene-to-cell-type dictionary from scRNAseq file.
    
    Args:
        scrnaseq_file: Path to scRNAseq h5ad file
        celltype_column: Column name for cell types
        subsample_fraction: Fraction to subsample if large (set to 1.0 to disable)
        
    Returns:
        Dictionary mapping gene names to cell types
    """
    print(f"Loading scRNAseq data from {scrnaseq_file}")
    
    # Load data
    adata = sc.read(scrnaseq_file)
    print(f"Loaded scRNAseq data: {adata.shape}")
    
    # View the scRNAseq data
    print(adata.obs.head())
    print(adata.var.head())
    
    # Basic QC filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(adata.obs.head())
    print(adata.var.head())
    print(f"After QC: {adata.shape}")
    
    # Subsample if too large (less aggressive)
    if adata.n_obs > 50000 and subsample_fraction < 1.0:
        print(f"Subsampling large dataset to {subsample_fraction*100}%...")
        sc.pp.subsample(adata, subsample_fraction, random_state=42)
    
    # Store raw counts if needed (for DE)
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()
    
    # Basic preprocessing (normalize on copy)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    # For DE, switch back to raw counts
    adata.X = adata.layers['counts'].copy()
    
    # Extract gene-cell type mapping
    gene_celltype_dict = extract_gene_celltype_dict(
        adata, 
        celltype_column=celltype_column,
        method="differential_expression",
        top_n_genes=top_n_genes,  # Higher default
        pval_cutoff=0.05,
        logfc_min=0.5  # Adjust lower for more genes, higher for stricter
    )
    
    return gene_celltype_dict


def main():
    """
    Generate gene type mapping CSV.
    
    This script creates a CSV file containing gene names and their assigned cell types
    based on the scRNAseq data analysis method used in the visualization pipeline.
    """
    
    # Configuration
    dataset = 'colon'  # Change this to 'pancreas' or 'CRC' or 'breast' as needed
    scrnaseq_file = f'/dkfz/cluster/gpu/data/OE0606/fengyun/xenium_data/xenium_{dataset}/scRNAseq.h5ad'
    output_file = f'gene_type_mapping_{dataset}.csv'
    
    print(f"=== Gene Type Mapping Generation ===")
    print(f"Dataset: {dataset}")
    print(f"scRNAseq file: {scrnaseq_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Check if file exists
    if not Path(scrnaseq_file).exists():
        print(f"Error: scRNAseq file not found: {scrnaseq_file}")
        return
    
    try:
        if dataset == 'colon' or dataset == 'CRC':
            gene_types_dict = create_gene_celltype_dict_simple(
                scrnaseq_file=scrnaseq_file,
                celltype_column="Level1",
                top_n_genes=50
            )
        elif dataset == 'breast':
            gene_types_dict = create_gene_celltype_dict_simple(
                scrnaseq_file=scrnaseq_file,
                celltype_column="celltype_major",
                top_n_genes=200
            )
        
        # Convert to DataFrame
        gene_type_df = pd.DataFrame([
            {'gene_name': gene, 'gene_type': cell_type}
            for gene, cell_type in gene_types_dict.items()
        ])
        
        # Sort by gene type, then by gene name for easier examination
        gene_type_df = gene_type_df.sort_values(['gene_type', 'gene_name']).reset_index(drop=True)
        
        # Save to CSV
        gene_type_df.to_csv(output_file, index=False)
        
        print(f"\n=== Results Summary ===")
        print(f"Total genes mapped: {len(gene_type_df)}")
        print(f"Unique cell types: {gene_type_df['gene_type'].nunique()}")
        print(f"CSV saved to: {output_file}")
        
        print(f"\n=== Cell Type Distribution ===")
        celltype_counts = gene_type_df['gene_type'].value_counts()
        for cell_type, count in celltype_counts.items():
            print(f"  {cell_type}: {count} genes")
        
        print(f"\n=== Sample Gene Mappings ===")
        print(gene_type_df.head(10).to_string(index=False))
        print("...")
        
    except Exception as e:
        print(f"Error generating gene type mapping: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()