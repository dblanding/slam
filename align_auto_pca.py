# process_map.py - Complete map processing pipeline

from occupancy_grid import OccupancyGridMap, visualize_map
import matplotlib.pyplot as plt
import numpy as np

input_file = 'maps/raw_map.py'

def process_slam_map(input_file: str, output_file: str, 
                     inflation_radius: float = 0.15,
                     auto_align: bool = True,
                     manual_rotation: float = 0.0,
                     visualize: bool = True):
    """
    Complete map processing pipeline.
    
    Args:
        input_file: Path to raw map from SLAM
        output_file: Path to save processed map
        inflation_radius: Robot safety margin in meters
        auto_align: Use PCA to auto-align (overrides manual_rotation)
        manual_rotation: Manual rotation angle in degrees
        visualize: Show before/after comparison
    """
    print("=" * 60)
    print("MAP PROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Load raw map
    # Load the probability grid
    with np.load('maps/raw_map.npz') as npz:
        prob_data = npz['data']
        resolution = float(npz['resolution'])
        origin = (float(npz['origin_x']), float(npz['origin_y']))
    # Convert to standard format
    ogm = OccupancyGridMap.from_probability_grid(prob_data, resolution, origin)
    ogm.get_info()  # Should now show occupied cells!

    '''
    print("\n1. Loading raw map...")
    ogm = OccupancyGridMap.load(input_file)
    ogm.get_info()
    '''

    # 2. Alignment
    print("\n2. Aligning map...")
    if auto_align:
        ogm_aligned = ogm.align_to_principal_axes()
    elif manual_rotation != 0.0:
        ogm_aligned = ogm.rotate(manual_rotation)
    else:
        ogm_aligned = ogm.copy()
        print("No rotation applied")
    '''
    # 3. Inflation
    print("\n3. Inflating obstacles...")
    ogm_aligned.inflate_obstacles(inflation_radius)
    '''
    # 4. Save
    print("\n4. Saving processed map...")
    ogm_aligned.save(output_file)
    ogm_aligned.save_image(output_file.replace('.npz', '.png'))
    
    # 5. Visualization
    if visualize:
        print("\n5. Generating comparison visualization...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Before
        img1 = np.zeros_like(ogm.data, dtype=np.uint8)
        img1[ogm.data == 0] = 255
        img1[ogm.data == 100] = 0
        img1[ogm.data == -1] = 128
        ax1.imshow(img1, cmap='gray', origin='lower')
        ax1.set_title('Raw Map from SLAM', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # After
        img2 = np.zeros_like(ogm_aligned.data, dtype=np.uint8)
        img2[ogm_aligned.data == 0] = 255
        img2[ogm_aligned.data == 100] = 0
        img2[ogm_aligned.data == -1] = 128
        ax2.imshow(img2, cmap='gray', origin='lower')
        ax2.set_title(f'Processed (aligned + {inflation_radius}m inflation)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_file = output_file.replace('.npz', '_comparison.png')
        plt.savefig(comparison_file, dpi=150)
        print(f"Saved comparison: {comparison_file}")
        plt.show()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    
    return ogm_aligned


if __name__ == '__main__':
    # Process your map
    processed_map = process_slam_map(
        input_file='maps/raw_map.npz',
        output_file='maps/final_map.npz',
        #inflation_radius=0.15,      # 15cm safety margin
        auto_align=True,            # Automatic PCA alignment
        visualize=True
    )
    
    # Optional: Try different parameters
    # processed_map_conservative = process_slam_map(
    #     input_file='maps/raw_map.npz',
    #     output_file='maps/conservative_map.npz',
    #     inflation_radius=0.25,    # 25cm safety margin (more conservative)
    #     auto_align=True,
    #     visualize=False
    # )
