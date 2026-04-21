import os
import shutil
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run VGGT local pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing your raw images")
    parser.add_argument("--vggt_dir", type=str, default="vggt", help="Directory where VGGT is cloned")
    parser.add_argument("--visualize", action="store_true", help="Launch interactive 3D visualization window after inference")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    vggt_dir = os.path.abspath(args.vggt_dir)
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory {input_dir} not found.")
        sys.exit(1)
        
    if not os.path.isdir(vggt_dir):
        print(f"Error: VGGT directory {vggt_dir} not found. Please run setup.bat first.")
        sys.exit(1)

    # Create scene directory formatted for VGGT
    # VGGT needs a top-level scene folder with an `images/` subfolder.
    scene_dir = os.path.join(os.path.dirname(input_dir), f"{os.path.basename(input_dir)}_vggt_scene")
    scene_images_dir = os.path.join(scene_dir, "images")
    os.makedirs(scene_images_dir, exist_ok=True)
    
    print(f"Formatting and resizing images into {scene_images_dir}...")
    from PIL import Image
    valid_exts = {".jpg", ".jpeg", ".png"}
    copied = 0
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            src = os.path.join(input_dir, f)
            dst = os.path.join(scene_images_dir, f)
            try:
                with Image.open(src) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Resize to max 512px maintaining aspect ratio
                    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    img.save(dst, quality=95)
                    copied += 1
            except Exception as e:
                print(f"Error processing {f}: {e}")
            
    print(f"Processed and resized {copied} images.")
    if copied == 0:
        print("No valid images found or processing failed. Exiting.")
        sys.exit(1)
            
    print(f"Executing VGGT inference on {scene_dir}...")
    demo_script = os.path.join(vggt_dir, "demo_colmap.py")
    
    # Run the subprocess
    try:
        subprocess.run(["python", demo_script, "--scene_dir", scene_dir], cwd=vggt_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"VGGT Inference failed: {e}")
        sys.exit(1)
        
    print("Inference complete!")
    points_path = os.path.join(scene_dir, "sparse", "points.ply")
    if os.path.isfile(points_path):
        print(f"Saved 3D reconstruction to {points_path}")
        
        if args.visualize:
            print("Launching visualization window...")
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(points_path)
            if not pcd.is_empty():
                 o3d.visualization.draw_geometries([pcd], window_name="VGGT Reconstruction", width=1280, height=720)
            else:
                print("Warning: Point cloud is empty, cannot visualize.")

        print(f"\nNext Step: Run the scale evaluator")
        print(f"python interact_scale_evaluator.py --cloud_path \"{points_path}\"")
    else:
        print("Error: Reconstruction output not found. Something went wrong.")

if __name__ == "__main__":
    main()
