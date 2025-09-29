"""
Spatial Batch Utilities

This module provides utilities to filter batches based on spatial coordinates
and create combined dataloaders from train/test/val datasets.

Usage:
    from utils.spatial_batch_utils import get_spatial_combined_dataloader
    
    # Get combined dataloader for a specific spatial region
    combined_batches = get_spatial_combined_dataloader(
        dm,
        x_range=[2000, 4000],
        y_range=[2000, 3000]
    )
"""

from typing import List, Tuple, Dict
from pathlib import Path
import pickle

class SpatialBatchFilter:
    """Utility class for filtering batches based on spatial coordinates."""
    
    def __init__(self, data_module):
        """Initialize the spatial batch filter."""
        self.dm = data_module
        
    def get_batches_in_region(self, 
                             dataset: str,
                             x_range: Tuple[float, float],
                             y_range: Tuple[float, float],
                             min_transcripts: int = 1) -> List[int]:
        """Get batch indices that contain transcripts in the specified spatial region."""
        if dataset not in ['train', 'test', 'val']:
            raise ValueError("dataset must be 'train', 'test', or 'val'")
            
        dataloader = getattr(self.dm, dataset)
        batch_indices = []
        
        print(f"Scanning {len(dataloader)} batches in {dataset} dataset...")
        
        for batch_idx in range(len(dataloader)):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}/{len(dataloader)}")
                
            batch = dataloader[batch_idx]
            transcript_count = self._count_transcripts_in_region(batch, x_range, y_range)
            
            if transcript_count >= min_transcripts:
                batch_indices.append(batch_idx)
                
        print(f"Found {len(batch_indices)} batches with transcripts in region "
              f"x={x_range}, y={y_range}")
        
        return batch_indices
    
    def get_all_region_batches(self,
                              x_range: Tuple[float, float],
                              y_range: Tuple[float, float],
                              min_transcripts: int = 1) -> Dict[str, List[int]]:
        """Get batch indices for all datasets (train/test/val) in the specified region."""
        batch_indices = {}
        
        for dataset in ['train', 'test', 'val']:
            batch_indices[dataset] = self.get_batches_in_region(
                dataset, x_range, y_range, min_transcripts
            )
        
        return batch_indices
    
    def _count_transcripts_in_region(self,
                                   batch,
                                   x_range: Tuple[float, float],
                                   y_range: Tuple[float, float]) -> int:
        """Count transcripts in a batch that fall within the specified spatial region."""
        try:
            # Extract transcript positions
            if hasattr(batch, 'x_dict') and 'tx' in batch.x_dict:
                tx_pos = batch['tx'].pos.cpu().numpy()
            elif 'tx' in batch and hasattr(batch['tx'], 'pos'):
                tx_pos = batch['tx'].pos.cpu().numpy()
            else:
                return 0
                
            if len(tx_pos) == 0:
                return 0
                
            x_coords = tx_pos[:, 0]
            y_coords = tx_pos[:, 1]
            
            # Filter by spatial region
            x_mask = (x_coords >= x_range[0]) & (x_coords <= x_range[1])
            y_mask = (y_coords >= y_range[0]) & (y_coords <= y_range[1])
            region_mask = x_mask & y_mask
            
            return int(region_mask.sum())
            
        except Exception as e:
            print(f"Warning: Error processing batch: {e}")
            return 0


def get_spatial_combined_dataloader(data_module,
                                   x_range: Tuple[float, float],
                                   y_range: Tuple[float, float],
                                   all_regions: bool = False, save_dir: Path = None) -> List:
    """
    Create a combined dataloader from train/test/val datasets for a specific spatial region.
    
    Args:
        data_module: SeggerDataModule instance
        x_range: Tuple of (min_x, max_x) coordinates
        y_range: Tuple of (min_y, max_y) coordinates
        min_transcripts: Minimum number of transcripts required in region
    
    Returns:
        List of combined batches from all datasets that contain transcripts in the region
    """
    if all_regions:
        # directly combine all batches
        combined_batches = []
        for dataset in ['train', 'test', 'val']:
            dataloader = getattr(data_module, dataset)
            combined_batches.extend(dataloader)
        print(f"Created combined dataloader with {len(combined_batches)} total batches")
        return combined_batches
    
    # Examine if the batch indices are already saved, if so, load them directly
    
    save_path = save_dir / f'batch_indices_x_min_{x_range[0]}_x_max_{x_range[1]}_y_min_{y_range[0]}_y_max_{y_range[1]}.pkl'
    
    if save_path.exists():
        with open(save_path, 'rb') as f:
            batch_indices = pickle.load(f)
        print(f"Loaded batch indices from {save_path}")
    else:
        spatial_filter = SpatialBatchFilter(data_module)
        batch_indices = spatial_filter.get_all_region_batches(x_range, y_range, min_transcripts=1)
        
        if save_dir:
            with open(save_dir / f'batch_indices_x_min_{x_range[0]}_x_max_{x_range[1]}_y_min_{y_range[0]}_y_max_{y_range[1]}.pkl', 'wb') as f:
                pickle.dump(batch_indices, f)
    
    # Create combined dataloader
    combined_batches = []
    
    for dataset, indices in batch_indices.items():
        if not indices:
            continue
            
        dataloader = getattr(data_module, dataset)
        print(f"Adding {len(indices)} batches from {dataset} dataset")
        
        for batch_idx in indices:
            if batch_idx < len(dataloader):
                combined_batches.append(dataloader[batch_idx])
            else:
                print(f"Warning: batch_idx {batch_idx} >= {dataset} dataset size {len(dataloader)}")
    
    print(f"Created combined dataloader with {len(combined_batches)} total batches")
    return combined_batches