import numpy as np
from occupancy_grid import OccupancyGridMap

print("Loading and converting map...")

# Load probability grid
with np.load('maps/raw_map.npz') as npz:
    prob_data = npz['data']
    resolution = float(npz['resolution'])
    origin = (float(npz['origin_x']), float(npz['origin_y']))

# Convert to standard format
ogm = OccupancyGridMap.from_probability_grid(prob_data, resolution, origin)
ogm.get_info()

# Save png image of raw map
ogm.save_image('maps/unaligned_map.png')

# Interactive alignment
print("\n" + "="*60)
print("INTERACTIVE ALIGNMENT")
print("="*60)
ogm_aligned = ogm.align_interactive()
'''
# Inflate obstacles
print("\nInflating obstacles...")
ogm_aligned.inflate_obstacles(0.15)
'''
# Save
print("\nSaving final map...")
ogm_aligned.save('maps/final_map.npz')
ogm_aligned.save_image('maps/final_map.png')

print("\n" + "="*60)
print("DONE!")
print("="*60)
ogm_aligned.get_info()
