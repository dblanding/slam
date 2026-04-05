"""
Examime contents of cells of np grid to see
if they are binary or probabilty (0-100)
"""

import numpy as np
from occupancy_grid import OccupancyGridMap

ogm = OccupancyGridMap.load('maps/raw_map.npz')

print("Unique values:", np.unique(ogm.data))
print("\nValue counts:")
unique, counts = np.unique(ogm.data, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  Value {val}: {count} cells ({100*count/ogm.data.size:.1f}%)")
