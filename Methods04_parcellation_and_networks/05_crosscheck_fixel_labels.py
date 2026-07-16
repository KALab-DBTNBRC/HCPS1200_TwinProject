import os
import subprocess

# --- THE INVESTIGATION ---
# Check the labels we generated in Step 2
labels = [
    os.path.join(os.environ.get("PROJECT_ROOT", "."), "study_template/label_maps/dk_fixel_labels.mif"),
    os.path.join(os.environ.get("PROJECT_ROOT", "."), "study_template/label_maps/glasser_fixel_labels.mif")
]

def inspect_labels():
    print("DIAGNOSTIC REPORT:")
    print("-" * 40)
    for f in labels:
        try:
            # mrstats gives us the min and max values in the file
            # If min is 1 and max is 1, it's just a brain mask, not a label map!
            stats = subprocess.check_output(["mrstats", f]).decode()
            print(f"FILE: {f}")
            print(stats)
            print("-" * 40)
        except:
            print(f"Could not find or open: {f}")

if __name__ == "__main__":
    inspect_labels()