"""
Occupancy Grid Map for Robot Path Planning

Standalone class for loading, manipulating, and saving occupancy grid maps.
Can be used independently from SLAM code.
"""

import numpy as np
from typing import Tuple, Optional


class OccupancyGridMap:
    """
    2D Occupancy Grid Map for robot navigation and path planning.
    
    Grid values:
        0 = Free space (navigable)
        100 = Occupied (obstacle)
        -1 = Unknown (unexplored)
    
    Attributes:
        2D numpy array of occupancy values
        resolution: Grid cell size in meters
        origin_x, origin_y: World coordinates of grid[0,0]
        width, height: Grid dimensions in cells
    """
    
    def __init__(self, data: np.ndarray, resolution: float, 
                 origin_x: float, origin_y: float):
        """
        Initialize occupancy grid map.
        
        Args:
            2D array of occupancy values (0=free, 100=occupied, -1=unknown)
            resolution: Grid cell size in meters
            origin_x: X coordinate of grid origin (bottom-left) in meters
            origin_y: Y coordinate of grid origin (bottom-left) in meters
        """
        self.data = data.astype(np.int8)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.height, self.width = data.shape
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.origin_x + (grid_x + 0.5) * self.resolution
        y = self.origin_y + (grid_y + 0.5) * self.resolution
        return x, y
    

    @staticmethod
    def from_probability_grid(data: np.ndarray, resolution: float, 
                             origin: tuple, free_thresh: int = 30, 
                             occupied_thresh: int = 70):
        """
        Convert probability grid (0-100) to standard format.
        
        Args:
            Grid with values 0-100 (probability of occupancy)
            resolution: Cell size in meters
            origin: (x, y) origin in meters
            free_thresh: Values <= this are free (default 30)
            occupied_thresh: Values >= this are occupied (default 70)
        
        Returns:
            OccupancyGridMap with standard values (0=free, 100=occupied, -1=unknown)
        """
        standard_data = np.full_like(data, -1, dtype=np.int8)
        standard_data[data <= free_thresh] = 0      # Free
        standard_data[data >= occupied_thresh] = 100  # Occupied
        # Everything else stays -1 (unknown)
        
        return OccupancyGridMap(standard_data, resolution, origin[0], origin[1])




    def is_free(self, grid_x: int, grid_y: int, threshold: int = 50) -> bool:
        """
        Check if a grid cell is free for navigation.
        
        Args:
            grid_x, grid_y: Grid coordinates
            threshold: Occupancy threshold (cells < threshold are free)
        
        Returns:
            True if cell is free, False otherwise
        """
        if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):
            return False
        return self.data[grid_y, grid_x] < threshold

    def align_interactive(self):
        """
        Interactively rotate the map to align it.
        Use arrow keys to rotate, Enter to accept.
        
        Controls:
            Left/Right arrows: Rotate by 1°
            Up/Down arrows: Rotate by 0.1°
            Enter: Accept current rotation
            Escape: Cancel (return original)
        
        Returns:
            Rotated OccupancyGridMap
        """
        import matplotlib.pyplot as plt
        
        current_angle = [0.0]  # Use list to modify in nested function
        accepted = [False]
        
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.subplots_adjust(bottom=0.15)
        
        def update_display():
            ax.clear()
            
            # Create rotated map
            rotated_ogm = self.rotate(current_angle[0])
            
            # Convert to image
            img = np.zeros_like(rotated_ogm.data, dtype=np.uint8)
            img[rotated_ogm.data == 0] = 255
            img[rotated_ogm.data == 100] = 0
            img[rotated_ogm.data == -1] = 128
            
            ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=255)
            ax.set_title(f'Rotation: {current_angle[0]:.1f}°\n' + 
                        '←/→: ±1°  ↑/↓: ±0.1°  Enter: Accept  Esc: Cancel',
                        fontsize=12)
            ax.grid(True, alpha=0.3, color='red', linewidth=0.5)
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            
            plt.draw()
        
        def on_key(event):
            if event.key == 'left':
                current_angle[0] -= 1.0
                update_display()
            elif event.key == 'right':
                current_angle[0] += 1.0
                update_display()
            elif event.key == 'down':
                current_angle[0] -= 0.1
                update_display()
            elif event.key == 'up':
                current_angle[0] += 0.1
                update_display()
            elif event.key == 'enter':
                accepted[0] = True
                plt.close()
            elif event.key == 'escape':
                current_angle[0] = 0.0
                plt.close()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display()
        
        print("\nInteractive Alignment:")
        print("  Use arrow keys to rotate")
        print("  Press Enter when aligned")
        print("  Press Escape to cancel\n")
        
        plt.show()
        
        if accepted[0]:
            print(f"Applied rotation: {current_angle[0]:.1f}°")
            return self.rotate(current_angle[0])
        else:
            print("Rotation cancelled")
            return self.copy()

    def save(self, filename: str):
        """
        Save occupancy grid to .npz file.
        
        Args:
            filename: Path to save file (e.g., 'map.npz')
        """
        if not filename.endswith('.npz'):
            filename = filename + '.npz'
        
        np.savez_compressed(
            filename,
            data=self.data,
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            width=self.width,
            height=self.height
        )
        
        print(f"Saved: {filename} ({self.width}×{self.height}, {self.resolution}m resolution)")
    
    @staticmethod
    def load(filename: str) -> 'OccupancyGridMap':
        """
        Load occupancy grid from .npz file.
        
        Args:
            filename: Path to .npz file
        
        Returns:
            OccupancyGridMap object
        """
        if not filename.endswith('.npz'):
            filename = filename + '.npz'
        
        with np.load(filename) as npz:
            data = npz['data']
            resolution = float(npz['resolution'])
            origin_x = float(npz['origin_x'])
            origin_y = float(npz['origin_y'])
        
        ogm = OccupancyGridMap(data, resolution, origin_x, origin_y)
        
        print(f"Loaded: {filename} ({ogm.width}×{ogm.height}, {ogm.resolution}m resolution)")
        
        return ogm
    
    def save_image(self, filename: str):
        """
        Save as PNG image for visualization.
        White = free, Black = occupied, Gray = unknown
        
        Args:
            filename: Path to save image (e.g., 'map.png')
        """
        import matplotlib.pyplot as plt
        
        if not filename.endswith('.png'):
            filename = filename + '.png'
        
        # Convert to image format
        img = np.zeros_like(self.data, dtype=np.uint8)
        img[self.data == 0] = 255      # Free → white
        img[self.data == 100] = 0      # Occupied → black
        img[self.data == -1] = 128     # Unknown → gray
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=255)
        plt.title(f'Occupancy Grid ({self.width}×{self.height}, {self.resolution}m/cell)')
        plt.xlabel(f'X (origin: {self.origin_x:.2f}m)')
        plt.ylabel(f'Y (origin: {self.origin_y:.2f}m)')
        plt.colorbar(label='Occupancy')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved image: {filename}")
    
    def inflate_obstacles(self, radius: float):
        """
        Inflate obstacles by radius for robot safety margin.
        Modifies the map in-place.
        
        Args:
            radius: Inflation radius in meters (robot radius + safety margin)
        """
        from scipy.ndimage import binary_dilation
        
        kernel_size = int(np.ceil(radius / self.resolution))
        
        if kernel_size == 0:
            print("Warning: Inflation radius too small for grid resolution")
            return
        
        # Create circular kernel
        y, x = np.ogrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
        kernel = x**2 + y**2 <= kernel_size**2
        
        print(f"Inflating obstacles by {radius}m ({kernel_size} cells)")
        
        # Dilate occupied cells
        occupied = (self.data == 100)
        inflated = binary_dilation(occupied, structure=kernel)
        
        # Apply inflation
        newly_inflated = inflated & (self.data != 100)
        self.data[newly_inflated] = 100
        
        num_inflated = np.sum(newly_inflated)
        print(f"Inflated {num_inflated} additional cells")
    
    def rotate(self, angle_degrees: float) -> 'OccupancyGridMap':
        """
        Rotate the map by specified angle.
        
        Args:
            angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        
        Returns:
            New rotated OccupancyGridMap
        """
        from scipy.ndimage import rotate as scipy_rotate
        
        print(f"Rotating map by {angle_degrees:.1f}°")
        
        # Rotate the data
        rotated_data = scipy_rotate(
            self.data,
            angle_degrees,
            reshape=True,
            order=0,      # Nearest neighbor (preserves discrete values)
            cval=-1       # Fill with unknown
        )
        
        # Calculate new origin (rotation is around center)
        center_x = self.origin_x + (self.width * self.resolution) / 2
        center_y = self.origin_y + (self.height * self.resolution) / 2
        
        new_width = rotated_data.shape[1]
        new_height = rotated_data.shape[0]
        
        new_origin_x = center_x - (new_width * self.resolution) / 2
        new_origin_y = center_y - (new_height * self.resolution) / 2
        
        return OccupancyGridMap(
            rotated_data,
            self.resolution,
            new_origin_x,
            new_origin_y
        )
    
    def align_to_principal_axes(self) -> 'OccupancyGridMap':
        """
        Rotate map to align with principal axes (makes walls horizontal/vertical).
        Uses PCA to find dominant direction of obstacles.
        
        Returns:
            New aligned OccupancyGridMap
        """
        # Find occupied cells
        occupied_y, occupied_x = np.where(self.data == 100)
        
        if len(occupied_x) < 10:
            print("Not enough obstacles to determine alignment")
            return self
        
        # PCA: find principal direction
        mean_x = np.mean(occupied_x)
        mean_y = np.mean(occupied_y)
        
        centered_x = occupied_x - mean_x
        centered_y = occupied_y - mean_y
        
        coords = np.vstack([centered_x, centered_y])
        cov_matrix = np.cov(coords)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        principal_idx = np.argmax(eigenvalues)
        principal_direction = eigenvectors[:, principal_idx]
        
        # Calculate rotation angle
        angle_rad = np.arctan2(principal_direction[1], principal_direction[0])
        angle_deg = np.degrees(angle_rad)
        
        # Round to nearest 90° (assumes orthogonal walls)
        angle_deg_rounded = round(angle_deg / 90) * 90
        rotation_needed = angle_deg_rounded - angle_deg
        
        print(f"Detected dominant angle: {angle_deg:.1f}°")
        print(f"Rotating by {rotation_needed:.1f}° to align with axes")
        
        return self.rotate(rotation_needed)
    
    def copy(self) -> 'OccupancyGridMap':
        """Create a deep copy of the map."""
        return OccupancyGridMap(
            self.data.copy(),
            self.resolution,
            self.origin_x,
            self.origin_y
        )
    
    def get_info(self):
        """Print map information."""
        num_free = np.sum(self.data == 0)
        num_occupied = np.sum(self.data == 100)
        num_unknown = np.sum(self.data == -1)
        total = self.width * self.height
        
        print(f"\nOccupancy Grid Map Info:")
        print(f"  Size: {self.width} × {self.height} cells")
        print(f"  Resolution: {self.resolution} m/cell")
        print(f"  Origin: ({self.origin_x:.2f}, {self.origin_y:.2f}) m")
        print(f"  Bounds: X=[{self.origin_x:.2f}, {self.origin_x + self.width*self.resolution:.2f}]")
        print(f"          Y=[{self.origin_y:.2f}, {self.origin_y + self.height*self.resolution:.2f}]")
        print(f"  Free cells: {num_free} ({100*num_free/total:.1f}%)")
        print(f"  Occupied cells: {num_occupied} ({100*num_occupied/total:.1f}%)")
        print(f"  Unknown cells: {num_unknown} ({100*num_unknown/total:.1f}%)")


# Convenience function for quick visualization
def visualize_map(ogm: OccupancyGridMap, title: str = "Occupancy Grid Map"):
    """
    Display occupancy grid map in a window.
    
    Args:
        ogm: OccupancyGridMap to visualize
        title: Window title
    """
    import matplotlib.pyplot as plt
    
    img = np.zeros_like(ogm.data, dtype=np.uint8)
    img[ogm.data == 0] = 255
    img[ogm.data == 100] = 0
    img[ogm.data == -1] = 128
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=255)
    plt.title(title)
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.colorbar(label='Occupancy')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    ogm = OccupancyGridMap.load('maps/raw_map.npz')
    ogm.get_info()  # Should now show occupied cells
    ogm_aligned = ogm.align_interactive()  # or align_to_principal_axes()
    ogm_aligned.inflate_obstacles(0.15)
    ogm_aligned.save('maps/final_map.npz')
    
