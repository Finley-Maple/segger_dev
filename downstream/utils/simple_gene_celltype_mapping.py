"""
Simple Gene-to-Cell-Type Mapping from scRNA-seq data.

The helpers in this module load and pre-process an scRNA dataset,
score genes per cell type, and return a mapping that can be reused by
visualisation pipelines.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import scanpy as sc
from anndata import AnnData


def extract_gene_celltype_dict(
    scrnaseq_adata: AnnData,
    celltype_column: str = "Level1",
    method: str = "differential_expression",
    top_n_genes: int = 200,
    pval_cutoff: float = 0.05,
    logfc_min: float = 0.5,
    min_expression_ratio: float = 1.5,
    return_celltype_counts: bool = False,
) -> Union[Dict[str, str], Tuple[Dict[str, str], Dict[str, int]]]:
    """Derive a cell-type assignment per gene from a preprocessed AnnData object."""

    _ = min_expression_ratio  # retained for backwards compatibility

    if celltype_column not in scrnaseq_adata.obs:
        raise KeyError(f"Column '{celltype_column}' not found in AnnData.obs")

    print(f"Extracting gene-to-cell-type mapping using method: {method}")

    adata = scrnaseq_adata  # operate in-place; caller can pass a copy if needed
    adata.var_names_make_unique()

    cell_types = adata.obs[celltype_column].astype("category").cat.categories.tolist()
    print(f"Found {len(cell_types)} cell types: {cell_types}")

    if method != "differential_expression":
        raise ValueError(f"Unsupported method '{method}'. Only 'differential_expression' is implemented.")

    print("Computing differential expression (wilcoxon)...")
    sc.tl.rank_genes_groups(
        adata,
        groupby=celltype_column,
        method="wilcoxon",
        use_raw=False,
    )

    candidates: Dict[str, List[Tuple[str, float]]] = {}
    for cell_type in cell_types:
        if cell_type not in adata.uns["rank_genes_groups"]["names"].dtype.names:
            print(f"  Warning: no DE results for cell type '{cell_type}'")
            continue

        de_df = sc.get.rank_genes_groups_df(adata, group=cell_type)
        filtered_df = de_df[(de_df["pvals_adj"] < pval_cutoff) & (de_df["logfoldchanges"] > logfc_min)]
        if len(filtered_df) > top_n_genes:
            filtered_df = filtered_df.sort_values("scores", ascending=False).head(top_n_genes)

        for _, row in filtered_df.iterrows():
            candidates.setdefault(row["names"], []).append((cell_type, row["scores"]))

        print(f"  {cell_type}: {len(filtered_df)} marker candidates after filtering")

    gene_celltype_counts = {gene: len(type_scores) for gene, type_scores in candidates.items()}
    gene_celltype_dict = {
        gene: max(type_scores, key=lambda x: x[1])[0]
        for gene, type_scores in candidates.items()
    }

    print(f"Resolved assignments for {len(gene_celltype_dict)} genes")
    if gene_celltype_dict:
        celltype_counts = pd.Series(gene_celltype_dict.values()).value_counts()
        print("Cell type distribution:")
        for ct, count in celltype_counts.items():
            print(f"  {ct}: {count} genes")

    if return_celltype_counts:
        if gene_celltype_counts:
            candidacy_counts = pd.Series(gene_celltype_counts.values()).value_counts().sort_index()
            print("Cell type candidacy distribution:")
            for count, num_genes in candidacy_counts.items():
                print(f"  {num_genes} genes were candidates for {count} cell type(s)")
        return gene_celltype_dict, gene_celltype_counts

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


def _load_and_prepare_scrnaseq(
    scrnaseq_file: str,
    celltype_column: str,
    subsample_fraction: float = 0.2,
) -> AnnData:
    """Load scRNA-seq data and return a normalised AnnData ready for DE analysis."""

    print(f"Loading scRNAseq data from {scrnaseq_file}")
    adata = sc.read(scrnaseq_file)
    adata.var_names_make_unique()
    print(f"Loaded scRNAseq data: {adata.shape}")

    required_obs = {celltype_column}
    if missing := required_obs - set(adata.obs.columns):
        raise KeyError(f"Missing expected columns in AnnData.obs: {sorted(missing)}")

    print(adata.obs[[celltype_column]].head())
    print(adata.var.head())

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"After QC: {adata.shape}")

    if adata.n_obs > 50000 and subsample_fraction < 1.0:
        target = int(adata.n_obs * subsample_fraction)
        print(f"Subsampling to ~{target} cells ({subsample_fraction*100:.1f}% of dataset)...")
        sc.pp.subsample(adata, fraction=subsample_fraction, random_state=42)

    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata  # keep log-normalised expression as reference for DE

    return adata


def create_gene_celltype_dict(
    scrnaseq_file: str,
    celltype_column: str = "Level1",
    top_n_genes: int = 200,
    subsample_fraction: float = 0.2,
) -> Dict[str, str]:
    """Create a gene→cell-type dictionary from an scRNA-seq file."""

    adata = _load_and_prepare_scrnaseq(
        scrnaseq_file=scrnaseq_file,
        celltype_column=celltype_column,
        subsample_fraction=subsample_fraction,
    )

    mapping, candidacy_counts = extract_gene_celltype_dict(
        adata,
        celltype_column=celltype_column,
        method="differential_expression",
        top_n_genes=top_n_genes,
        pval_cutoff=0.05,
        logfc_min=0.5,
        return_celltype_counts=True,
    )

    filtered_mapping = {
        gene: cell_type
        for gene, cell_type in mapping.items()
        if candidacy_counts.get(gene, 0) == 1
    }
    removed = len(mapping) - len(filtered_mapping)
    print(f"Removed {removed} genes with ambiguous cell-type candidacy")
    print(f"Final gene count: {len(filtered_mapping)}")

    return filtered_mapping


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
            gene_types_dict = create_gene_celltype_dict(
                scrnaseq_file=scrnaseq_file,
                celltype_column="Level1",
                top_n_genes=200
            )
        elif dataset == 'breast':
            gene_types_dict = create_gene_celltype_dict(
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
