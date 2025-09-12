#!/usr/bin/env python3
"""
Generate Gene Type Mapping CSV for Expert Examination

This script creates a CSV file containing gene names and their assigned cell types
based on the scRNAseq data analysis method used in the visualization pipeline.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.simple_gene_celltype_mapping import create_gene_celltype_dict_simple

def main():
    """Generate gene type mapping CSV."""
    
    # Configuration
    dataset = 'colon'  # Change this to 'pancreas' or 'CRC' as needed
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
        # Generate gene-to-cell-type mapping
        gene_types_dict = create_gene_celltype_dict_simple(
            scrnaseq_file=scrnaseq_file,
            celltype_column="Level1"
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
