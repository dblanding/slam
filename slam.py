import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from collections import defaultdict

# Input filenames
pose_data_file = "Scan_Pose_Data/pose_data.npz"
scan_data_file = "Scan_Pose_Data/scan_data.npz"

# Output filenames
slam_map_file = "Scan_Pose_Data/my_slam_map.png"
save_ogm_file = "Scan_Pose_Data/occupancy_map.npy"
save_opt_poses_file = "Scan_Pose_Data/optimized_poses.csv"

@dataclass
class Pose2D:
    """Represents a 2D pose with x, y position and theta orientation"""
    x: float
    y: float
    theta: float
    
    def to_matrix(self) -> np.ndarray:
        """Convert to homogeneous transformation matrix"""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s, c, self.y],
            [0, 0, 1]
        ])
    
    @staticmethod
    def from_matrix(matrix: np.ndarray) -> 'Pose2D':
        """Create pose from transformation matrix"""
        x = matrix[0, 2]
        y = matrix[1, 2]
        theta = np.arctan2(matrix[1, 0], matrix[0, 0])
        return Pose2D(x, y, theta)
    
    def inverse(self) -> 'Pose2D':
        """Compute inverse pose"""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        x_inv = -c * self.x - s * self.y
        y_inv = s * self.x - c * self.y
        return Pose2D(x_inv, y_inv, -self.theta)
    
    def compose(self, other: 'Pose2D') -> 'Pose2D':
        """Compose two poses (self * other)"""
        result = self.to_matrix() @ other.to_matrix()
        return Pose2D.from_matrix(result)
    
    def to_vector(self) -> np.ndarray:
        """Convert to vector [x, y, theta]"""
        return np.array([self.x, self.y, self.theta])
    
    @staticmethod
    def from_vector(vec: np.ndarray) -> 'Pose2D':
        """Create from vector [x, y, theta]"""
        return Pose2D(vec[0], vec[1], vec[2])

@dataclass
class LidarScan:
    """Represents a lidar scan with ranges and angles"""
    ranges: np.ndarray
    angles: np.ndarray
    max_range: float = 10.0
    
    def get_points(self, pose: Pose2D) -> np.ndarray:
        """Convert scan to 2D points in world frame"""
        valid_mask = (self.ranges > 0) & (self.ranges < self.max_range) & ~np.isnan(self.ranges)
        valid_ranges = self.ranges[valid_mask]
        valid_angles = self.angles[valid_mask]
        
        # Points in sensor frame
        local_x = valid_ranges * np.cos(valid_angles)
        local_y = valid_ranges * np.sin(valid_angles)
        
        # Transform to world frame
        c = np.cos(pose.theta)
        s = np.sin(pose.theta)
        
        world_x = pose.x + c * local_x - s * local_y
        world_y = pose.y + s * local_x + c * local_y
        
        return np.column_stack([world_x, world_y])

@dataclass
class PoseConstraint:
    """Represents a constraint between two poses"""
    from_idx: int
    to_idx: int
    transform: Pose2D  # Relative transformation from 'from_idx' to 'to_idx'
    information: np.ndarray  # 3x3 information matrix (inverse covariance)
    constraint_type: str = "odometry"  # "odometry" or "loop_closure"

class ICP:
    """Iterative Closest Point algorithm for scan matching"""
    
    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-5, 
                 max_correspondence_distance: float = 0.5):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
    
    def align(self, source_points: np.ndarray, target_points: np.ndarray, 
              initial_pose: Pose2D = None) -> Tuple[Pose2D, float, bool]:
        """
        Align source points to target points using ICP
        
        Returns:
            (optimized_pose, fitness_score, converged)
        """
        if initial_pose is None:
            initial_pose = Pose2D(0, 0, 0)
        
        current_pose = initial_pose
        prev_error = float('inf')
        
        # Build KD-tree for target points
        target_tree = KDTree(target_points)
        
        for iteration in range(self.max_iterations):
            # Transform source points
            transformed_source = self._transform_points(source_points, current_pose)
            
            # Find correspondences
            distances, indices = target_tree.query(transformed_source)
            
            # Filter correspondences by distance
            valid_mask = distances < self.max_correspondence_distance
            
            if np.sum(valid_mask) < 3:
                return current_pose, 0.0, False
            
            valid_source = transformed_source[valid_mask]
            valid_target = target_points[indices[valid_mask]]
            
            # Compute error
            error = np.mean(distances[valid_mask])
            
            # Check convergence
            if abs(prev_error - error) < self.tolerance:
                fitness = np.sum(valid_mask) / len(source_points)
                return current_pose, fitness, True
            
            prev_error = error
            
            # Compute transformation update using SVD
            delta_pose = self._compute_transformation(valid_source, valid_target)
            
            # Update current pose
            current_pose = delta_pose.compose(current_pose)
        
        fitness = np.sum(valid_mask) / len(source_points)
        return current_pose, fitness, False
    
    def _transform_points(self, points: np.ndarray, pose: Pose2D) -> np.ndarray:
        """Transform points by pose"""
        c = np.cos(pose.theta)
        s = np.sin(pose.theta)
        
        rotation = np.array([[c, -s], [s, c]])
        translation = np.array([pose.x, pose.y])
        
        return (rotation @ points.T).T + translation
    
    def _compute_transformation(self, source: np.ndarray, target: np.ndarray) -> Pose2D:
        """Compute transformation from source to target using SVD"""
        # Compute centroids
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        
        # Center the points
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        
        # Compute covariance matrix
        H = source_centered.T @ target_centered
        
        # SVD
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = target_centroid - R @ source_centroid
        
        # Extract angle
        theta = np.arctan2(R[1, 0], R[0, 0])
        
        return Pose2D(t[0], t[1], theta)

class PoseGraph:
    """Pose graph for SLAM with optimization"""
    
    def __init__(self):
        self.poses: List[Pose2D] = []
        self.constraints: List[PoseConstraint] = []
        self.scans: List[LidarScan] = []
        
    def add_pose(self, pose: Pose2D, scan: LidarScan = None):
        """Add a new pose to the graph"""
        self.poses.append(pose)
        if scan is not None:
            self.scans.append(scan)
    
    def add_constraint(self, constraint: PoseConstraint):
        """Add a constraint between poses"""
        self.constraints.append(constraint)
    
    def optimize(self, max_iterations: int = 100) -> bool:
        """Optimize the pose graph using least squares"""
        if len(self.poses) < 2:
            return False
        
        # Convert poses to parameter vector
        x0 = np.zeros(len(self.poses) * 3)
        for i, pose in enumerate(self.poses):
            x0[i*3:(i+1)*3] = pose.to_vector()
        
        # Fix the first pose (anchor)
        fixed_indices = [0, 1, 2]  # x, y, theta of first pose
        
        # Optimize
        result = least_squares(
            self._residuals,
            x0,
            max_nfev=max_iterations,
            verbose=0
        )
        
        # Update poses with optimized values
        optimized_params = result.x
        for i in range(len(self.poses)):
            self.poses[i] = Pose2D.from_vector(optimized_params[i*3:(i+1)*3])
        
        return result.success
    
    def _residuals(self, params: np.ndarray) -> np.ndarray:
        """Compute residuals for all constraints"""
        residuals = []
        
        for constraint in self.constraints:
            # Extract poses
            i = constraint.from_idx
            j = constraint.to_idx
            
            pose_i = Pose2D.from_vector(params[i*3:(i+1)*3])
            pose_j = Pose2D.from_vector(params[j*3:(j+1)*3])
            
            # Compute predicted relative transformation
            predicted_transform = pose_i.inverse().compose(pose_j)
            
            # Compute error
            error = self._pose_difference(predicted_transform, constraint.transform)
            
            # Weight by information matrix
            weighted_error = np.sqrt(constraint.information.diagonal()) * error
            
            residuals.extend(weighted_error)
        
        # Add anchor constraint for first pose
        anchor_error = params[0:3] - self.poses[0].to_vector()
        residuals.extend(anchor_error * 1000)  # Strong weight
        
        return np.array(residuals)
    
    def _pose_difference(self, pose1: Pose2D, pose2: Pose2D) -> np.ndarray:
        """Compute difference between two poses"""
        dx = pose1.x - pose2.x
        dy = pose1.y - pose2.y
        dtheta = self._normalize_angle(pose1.theta - pose2.theta)
        return np.array([dx, dy, dtheta])
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

class OccupancyGridMap:
    """2D Occupancy Grid Map"""
    
    def __init__(self, width: int, height: int, resolution: float, origin: Tuple[float, float]):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array(origin)
        self.log_odds = np.zeros((height, width))
        self.l_occ = np.log(0.7 / 0.3)
        self.l_free = np.log(0.3 / 0.7)
        self.l_min = -5.0
        self.l_max = 5.0
    
    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        mx = int((x - self.origin[0]) / self.resolution)
        my = int((y - self.origin[1]) / self.resolution)
        return mx, my
    
    def is_valid(self, mx: int, my: int) -> bool:
        return 0 <= mx < self.width and 0 <= my < self.height
    
    def update_cell(self, mx: int, my: int, occupied: bool):
        if not self.is_valid(mx, my):
            return
        if occupied:
            self.log_odds[my, mx] += self.l_occ
        else:
            self.log_odds[my, mx] += self.l_free
        self.log_odds[my, mx] = np.clip(self.log_odds[my, mx], self.l_min, self.l_max)
    
    def get_occupancy_grid(self) -> np.ndarray:
        odds = np.exp(self.log_odds)
        prob = odds / (1 + odds)
        return (prob * 100).astype(np.int8)

class BresenhamRayCast:
    """Bresenham's line algorithm for ray casting"""
    
    @staticmethod
    def get_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

class GraphSLAM:
    """Graph-based SLAM with ICP and pose graph optimization"""
    
    def __init__(self, map_width: int = 500, map_height: int = 500,
                 resolution: float = 0.05, origin: Tuple[float, float] = (-12.5, -12.5)):
        self.map = OccupancyGridMap(map_width, map_height, resolution, origin)
        self.pose_graph = PoseGraph()
        self.icp = ICP(max_iterations=50, tolerance=1e-5, max_correspondence_distance=0.3)
        self.ray_caster = BresenhamRayCast()
        
        # Parameters
        self.loop_closure_distance_threshold = 1.0  # meters (was 2.0)
        self.loop_closure_fitness_threshold = 0.6  # ICP fitness score
        self.min_scans_between_loop_closure = 20  # Avoid checking recent scans
        
        # Information matrices (inverse covariance)
        self.odometry_information = np.diag([100.0, 100.0, 50.0])  # x, y, theta
        self.loop_closure_information = np.diag([200.0, 200.0, 100.0])  # Higher confidence
        
        # Track when last optimization occurred
        self.last_optimization_size = 0
        self.optimization_interval = 10  # Optimize every N poses
    
    def process_scan(self, pose: Pose2D, scan: LidarScan, use_icp: bool = True):
        """
        Process a new lidar scan with optional ICP refinement
        
        Args:
            pose: Initial pose estimate (e.g., from odometry)
            scan: Lidar scan data
            use_icp: Whether to refine pose using ICP
        """
        refined_pose = pose
        
        # If this is not the first scan, use ICP to refine the pose
        if use_icp and len(self.pose_graph.poses) > 0:
            refined_pose = self._refine_pose_with_icp(pose, scan)
        
        # Add pose and scan to graph
        pose_idx = len(self.pose_graph.poses)
        self.pose_graph.add_pose(refined_pose, scan)
        
        # Add odometry constraint from previous pose
        if pose_idx > 0:
            relative_transform = self.pose_graph.poses[pose_idx - 1].inverse().compose(refined_pose)
            constraint = PoseConstraint(
                from_idx=pose_idx - 1,
                to_idx=pose_idx,
                transform=relative_transform,
                information=self.odometry_information,
                constraint_type="odometry"
            )
            self.pose_graph.add_constraint(constraint)
        
        # Detect and add loop closures
        self._detect_loop_closures(pose_idx, scan)
        
        # Periodically optimize the pose graph
        if len(self.pose_graph.poses) - self.last_optimization_size >= self.optimization_interval:
            print(f"Optimizing pose graph with {len(self.pose_graph.poses)} poses...")
            success = self.pose_graph.optimize()
            if success:
                print("Optimization successful!")
                self.last_optimization_size = len(self.pose_graph.poses)
                # Rebuild map after optimization
                self._rebuild_map()
            else:
                print("Optimization failed!")
        else:
            # Just update map incrementally
            self._update_map_with_scan(refined_pose, scan)
    
    def _refine_pose_with_icp(self, initial_pose: Pose2D, current_scan: LidarScan) -> Pose2D:
        """Refine pose estimate using ICP against recent scans"""
        if len(self.pose_graph.poses) == 0:
            return initial_pose
        
        # Get the most recent scan
        prev_pose = self.pose_graph.poses[-1]
        prev_scan = self.pose_graph.scans[-1]
        
        # Convert scans to point clouds
        current_points = current_scan.get_points(initial_pose)
        prev_points = prev_scan.get_points(prev_pose)
        
        if len(current_points) < 10 or len(prev_points) < 10:
            return initial_pose
        
        # Compute initial relative transformation guess
        relative_guess = prev_pose.inverse().compose(initial_pose)
        
        # Run ICP in the local frame
        current_local = current_scan.get_points(Pose2D(0, 0, 0))
        prev_local = prev_scan.get_points(Pose2D(0, 0, 0))
        
        refined_relative, fitness, converged = self.icp.align(
            current_local, prev_local, relative_guess
        )
        
        if converged and fitness > 0.3:
            # Transform back to world frame
            refined_pose = prev_pose.compose(refined_relative)
            return refined_pose
        else:
            return initial_pose
    
    def _detect_loop_closures(self, current_idx: int, current_scan: LidarScan):
        """Detect loop closures and add constraints"""
        if current_idx < self.min_scans_between_loop_closure:
            return
        
        current_pose = self.pose_graph.poses[current_idx]
        current_points = current_scan.get_points(current_pose)
        
        if len(current_points) < 10:
            return
        
        # Check against older poses
        for old_idx in range(current_idx - self.min_scans_between_loop_closure):
            old_pose = self.pose_graph.poses[old_idx]
            
            # Check distance threshold
            distance = np.sqrt((current_pose.x - old_pose.x)**2 + 
                             (current_pose.y - old_pose.y)**2)
            
            if distance < self.loop_closure_distance_threshold:
                # Try ICP alignment
                old_scan = self.pose_graph.scans[old_idx]
                old_points = old_scan.get_points(old_pose)
                
                if len(old_points) < 10:
                    continue
                
                # Initial guess for relative transform
                relative_guess = old_pose.inverse().compose(current_pose)
                
                # ICP in local frame
                current_local = current_scan.get_points(Pose2D(0, 0, 0))
                old_local = old_scan.get_points(Pose2D(0, 0, 0))
                
                refined_relative, fitness, converged = self.icp.align(
                    current_local, old_local, relative_guess
                )
                
                if converged and fitness > self.loop_closure_fitness_threshold:
                    print(f"Loop closure detected! Pose {current_idx} <-> {old_idx} "
                          f"(fitness: {fitness:.3f})")
                    
                    # Add loop closure constraint
                    constraint = PoseConstraint(
                        from_idx=old_idx,
                        to_idx=current_idx,
                        transform=refined_relative,
                        information=self.loop_closure_information,
                        constraint_type="loop_closure"
                    )
                    self.pose_graph.add_constraint(constraint)
    
    def _update_map_with_scan(self, pose: Pose2D, scan: LidarScan):
        """Update occupancy grid with a single scan"""
        robot_mx, robot_my = self.map.world_to_map(pose.x, pose.y)
        
        if not self.map.is_valid(robot_mx, robot_my):
            return
        
        for range_val, angle in zip(scan.ranges, scan.angles):
            if range_val <= 0 or range_val >= scan.max_range or np.isnan(range_val):
                continue
            
            beam_angle = pose.theta + angle
            end_x = pose.x + range_val * np.cos(beam_angle)
            end_y = pose.y + range_val * np.sin(beam_angle)
            
            end_mx, end_my = self.map.world_to_map(end_x, end_y)
            
            ray_cells = self.ray_caster.get_line(robot_mx, robot_my, end_mx, end_my)
            
            for j, (mx, my) in enumerate(ray_cells):
                if not self.map.is_valid(mx, my):
                    continue
                
                if j < len(ray_cells) - 1:
                    self.map.update_cell(mx, my, occupied=False)
                else:
                    self.map.update_cell(mx, my, occupied=True)
    
    def _rebuild_map(self):
        """Rebuild the entire map from scratch using optimized poses"""
        print("Rebuilding map with optimized poses...")
        
        # Clear the map
        self.map.log_odds = np.zeros((self.map.height, self.map.width))
        
        # Reprocess all scans with optimized poses
        for pose, scan in zip(self.pose_graph.poses, self.pose_graph.scans):
            self._update_map_with_scan(pose, scan)
    
    def optimize_full(self):
        """Perform a full pose graph optimization"""
        print(f"Running full optimization on {len(self.pose_graph.poses)} poses...")
        success = self.pose_graph.optimize(max_iterations=200)
        
        if success:
            print("Full optimization successful!")
            self._rebuild_map()
        else:
            print("Full optimization failed!")
        
        return success
    
    def get_map(self) -> np.ndarray:
        """Get the current occupancy grid map"""
        return self.map.get_occupancy_grid()
    
    def get_poses(self) -> List[Pose2D]:
        """Get all poses in the graph"""
        return self.pose_graph.poses
    
    def get_constraints(self) -> List[PoseConstraint]:
        """Get all constraints in the graph"""
        return self.pose_graph.constraints
    
    def visualize(self, show_trajectory: bool = True, show_constraints: bool = True):
        """Visualize the map, trajectory, and constraints"""
        occupancy = self.get_map()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Occupancy map with trajectory
        ax1.imshow(occupancy, cmap='gray_r', origin='lower',
                   extent=[self.map.origin[0],
                          self.map.origin[0] + self.map.width * self.map.resolution,
                          self.map.origin[1],
                          self.map.origin[1] + self.map.height * self.map.resolution])
        
        if show_trajectory and len(self.pose_graph.poses) > 0:
            trajectory_x = [p.x for p in self.pose_graph.poses]
            trajectory_y = [p.y for p in self.pose_graph.poses]
            ax1.plot(trajectory_x, trajectory_y, 'b-', linewidth=2, label='Trajectory', alpha=0.7)
            ax1.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=10, label='Start')
            ax1.plot(trajectory_x[-1], trajectory_y[-1], 'ro', markersize=10, label='Current')
        
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('Occupancy Grid Map with Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Pose graph with constraints
        if show_constraints:
            poses = self.pose_graph.poses
            
            # Draw constraints
            for constraint in self.pose_graph.constraints:
                p1 = poses[constraint.from_idx]
                p2 = poses[constraint.to_idx]
                
                if constraint.constraint_type == "odometry":
                    ax2.plot([p1.x, p2.x], [p1.y, p2.y], 'b-', alpha=0.3, linewidth=0.5)
                else:  # loop closure
                    ax2.plot([p1.x, p2.x], [p1.y, p2.y], 'r-', alpha=0.8, linewidth=2)
            
            # Draw poses
            if len(poses) > 0:
                pose_x = [p.x for p in poses]
                pose_y = [p.y for p in poses]
                ax2.plot(pose_x, pose_y, 'ko', markersize=3, label='Poses')
                ax2.plot(pose_x[0], pose_y[0], 'go', markersize=10, label='Start')
                ax2.plot(pose_x[-1], pose_y[-1], 'ro', markersize=10, label='Current')
            
            # Create custom legend entries
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='b', linewidth=1, label='Odometry'),
                Line2D([0], [0], color='r', linewidth=2, label='Loop Closure'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
                       markersize=10, label='Start'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='r', 
                       markersize=10, label='Current')
            ]
            
            ax2.legend(handles=legend_elements)
            ax2.set_xlabel('X (meters)')
            ax2.set_ylabel('Y (meters)')
            ax2.set_title('Pose Graph with Constraints')
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
        
        plt.tight_layout()
        return fig


# Example of loading data from common formats
def load_poses_from_npz(filename: str) -> List[Pose2D]:
    """
    Load poses from .npz file
    """
    with np.load(filename) as pose_data:
        num_poses = pose_data['num_poses']
        poses = pose_data['poses'].tolist()
    # Convert poses to instances of Pose2D class
    pose_vals = []
    for pose in poses:
        p = Pose2D.from_vector(pose)
        pose_vals.append(p)
    return pose_vals

def load_scans_from_npz(filename: str) -> List[LidarScan]:
    """
    Load lidar scans from .npz file
    """
    with np.load(filename) as scan_data:
        all_angles = scan_data['angles']
        all_ranges = scan_data['ranges']
        num_scans = scan_data['num_scans']
        scan_lengths = scan_data['scan_lengths']

    # Reconstruct individual scans
    scans = []
    offset = 0
    for i, length in enumerate(scan_lengths):
        scan = {
            'angles': all_angles[offset:offset+length], 
            'ranges': all_ranges[offset:offset+length]
            }
        scans.append(scan)
        offset += length

    # Convert scans to instances of LidarScan class
    lidar_scans = []
    for scan in scans:
        ls = LidarScan(scan['ranges'], scan['angles'])
        lidar_scans.append(ls)
    return lidar_scans

def quick_start_template():
    """
    Quick start template for using your data
    """
    # 1. Initialize SLAM
    slam = GraphSLAM(
        map_width=300,  # pixels
        map_height=300,  # pixels
        resolution=0.05,  # meters
        origin=(-4, -7)  # lower left corner (meters)
    )
    
    # 2. Load your data (replace with your data loading code)
    poses = load_poses_from_npz(pose_data_file)
    scans = load_scans_from_npz(scan_data_file)
    
    # 3. Process data
    for i, (pose, scan) in enumerate(zip(poses, scans)):
        slam.process_scan(pose, scan, use_icp=True)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(poses)} scans")
    
    # 4. Final optimization
    slam.optimize_full()
    
    # 5. Get results
    final_map = slam.get_map()
    optimized_poses = slam.get_poses()
    constraints = slam.get_constraints()
    
    # 6. Visualize and save
    slam.visualize(show_trajectory=True, show_constraints=True)
    plt.savefig(slam_map_file, dpi=200, bbox_inches='tight')
    plt.show()
    
    # 7. Optional: Save map to file
    np.save(save_ogm_file, final_map)
    
    # 8. Optional: Save optimized poses
    with open(save_opt_poses_file, 'w') as f:
        for i, pose in enumerate(optimized_poses):
            f.write(f"{i},{pose.x},{pose.y},{pose.theta}\n")
    

# Main execution
if __name__ == "__main__":
        
    quick_start_template()
