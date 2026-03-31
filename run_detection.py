"""
Interactive Image Tampering Detection Launcher
Usage: python run_detection.py
"""
import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("    Four-Branch Image Tampering Detection System")
    print("=" * 60)
    print()

    # Get image path from user
    image_path = input("Enter image path (or drag image here): ").strip()

    # Remove extra quotes
    image_path = image_path.strip('"').strip("'")

    # Validate path
    if not image_path:
        print("[ERROR] Path cannot be empty!")
        input("Press Enter to exit...")
        return

    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        input("Press Enter to exit...")
        return

    # Check file extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_extensions:
        print(f"[WARNING] File format {ext} may not be supported. Use JPG or PNG.")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            return

    print()
    print(f"[INFO] Preparing detection: {os.path.basename(image_path)}")
    print("-" * 60)

    # Run detection
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        predict_script = os.path.join(script_dir, "predict_all.py")

        result = subprocess.run(
            [sys.executable, predict_script, image_path],
            cwd=script_dir,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print()
            print("=" * 60)
            print("[OK] Detection completed!")
            print("=" * 60)
        else:
            print()
            print("[ERROR] Detection failed")

    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")

    print()
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
