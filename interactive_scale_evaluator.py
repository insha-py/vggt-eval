import open3d as o3d
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud_path", required=True, help="Path to points.ply")
    args = parser.parse_args()

    try:
        pcd = o3d.io.read_point_cloud(args.cloud_path)
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return

    if pcd.is_empty():
        print(f"Failed to load point cloud from {args.cloud_path} (File might be empty or invalid).")
        return

    print("=" * 60)
    print("Interactive Scale Evaluator Instructions:")
    print(" 1. A 3D window will open. Rotate and zoom to find the object.")
    print(" 2. Hold [Shift] + Left Click to pick exactly 2 points spanning the object's real-life dimension.")
    print(" 3. Close the window (press 'Q' or X button) when done.")
    print("=" * 60)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # blocks until closed
    vis.destroy_window()

    picked_indices = vis.get_picked_points()
    if len(picked_indices) != 2:
        print(f"Error: You selected {len(picked_indices)} points. You must select exactly 2 points. Note: Shift+Click to select a point. Try again.")
        return

    print(f"\nSuccessfully selected 2 points: {picked_indices}")
    # Compute Reconstructed Distance
    p1 = np.asarray(pcd.points)[picked_indices[0]]
    p2 = np.asarray(pcd.points)[picked_indices[1]]
    recon_dist = np.linalg.norm(p1 - p2)
    print(f"Reconstructed Distance: {recon_dist:.4f} units")

    try:
        actual_dist = float(input("\nEnter the actual distance between these two selected points (in meters): "))
    except ValueError:
        print("Invalid input. Must be a number.")
        return

    if actual_dist <= 0:
        print("Distance must be > 0.")
        return
        
    scale_factor = recon_dist / actual_dist
    print("-" * 60)
    print("SCALE EVALUATION:")
    print(f" VGGT Output Scale Ratio: {scale_factor:.4f}")
    if abs(scale_factor - 1.0) < 0.1:
        print(" Result: The reconstruction is close to true metric scale! (error < 10%).")
    else:
        print(" Result: The reconstruction is NOT metric scale.")
        print(f" To scale the output reconstruction to match true physical metric scale, multiply all coordinates by: {1.0/scale_factor:.6f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
