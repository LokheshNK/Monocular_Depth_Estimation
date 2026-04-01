"""
find_sequences.py
─────────────────────────────────────────────────────────────
Auto-discovers ALL matching KITTI sequence pairs from your
dataset folder and prints them — no manual listing needed.

Run this first to verify what sequences are found:
    python find_sequences.py

Then the SEQUENCE_PAIRS it prints are automatically used
by train_hybrid.py and train_dpt.py via auto_find_sequences().
─────────────────────────────────────────────────────────────

Expected folder structure:
    depth_dataset/
    ├── raw_rgb/
    │   └── 2011_09_26/
    │       ├── 2011_09_26_drive_0001_sync/image_02/data/
    │       ├── 2011_09_26_drive_0009_sync/image_02/data/
    │       └── ...
    └── data_depth_annotated/
        └── train/
            ├── 2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/
            ├── 2011_09_26_drive_0009_sync/proj_depth/groundtruth/image_02/
            └── ...
"""

import os


def auto_find_sequences(base_dir: str,
                        rgb_root:   str = "raw_rgb",
                        depth_root: str = r"data_depth_annotated\train",
                        verbose:    bool = True) -> list:
    """
    Walks the KITTI folder structure and returns all (rgb_dir, depth_dir)
    pairs where BOTH the RGB folder and the depth GT folder exist AND
    contain at least one matching filename.

    Parameters
    ----------
    base_dir   : root of your dataset folder (where train_dpt.py lives)
    rgb_root   : subfolder containing raw RGB frames
    depth_root : subfolder containing depth ground truth
    verbose    : print a summary table

    Returns
    -------
    list of (rgb_dir, depth_dir) tuples — ready to pass to SEQUENCE_PAIRS
    """

    pairs   = []
    skipped = []

    rgb_base   = os.path.join(base_dir, rgb_root)
    depth_base = os.path.join(base_dir, depth_root)

    if not os.path.isdir(rgb_base):
        print(f"ERROR: RGB root not found: {rgb_base}")
        return pairs
    if not os.path.isdir(depth_base):
        print(f"ERROR: Depth root not found: {depth_base}")
        return pairs

    # Walk every date folder inside raw_rgb/
    for date_folder in sorted(os.listdir(rgb_base)):
        date_path = os.path.join(rgb_base, date_folder)
        if not os.path.isdir(date_path):
            continue

        # Walk every drive folder inside the date folder
        for drive_folder in sorted(os.listdir(date_path)):
            # Expected RGB path:
            #   raw_rgb/<date>/<drive>/image_02/data/
            rgb_dir = os.path.join(date_path, drive_folder, "image_02", "data")

            # Expected depth path:
            #   data_depth_annotated/train/<drive>/proj_depth/groundtruth/image_02/
            depth_dir = os.path.join(depth_base, drive_folder,
                                     "proj_depth", "groundtruth", "image_02")

            # Both must exist
            if not os.path.isdir(rgb_dir):
                skipped.append((drive_folder, f"RGB missing: {rgb_dir}"))
                continue
            if not os.path.isdir(depth_dir):
                skipped.append((drive_folder, f"Depth missing: {depth_dir}"))
                continue

            # Count matching files
            rgb_files   = set(os.listdir(rgb_dir))
            depth_files = set(os.listdir(depth_dir))
            common      = rgb_files & depth_files

            if not common:
                skipped.append((drive_folder,
                                f"No matching filenames "
                                f"(RGB={len(rgb_files)}, Depth={len(depth_files)})"))
                continue

            pairs.append((rgb_dir, depth_dir))

            if verbose:
                print(f"  ✓  {drive_folder:45s}  "
                      f"RGB={len(rgb_files):4d}  "
                      f"Depth={len(depth_files):4d}  "
                      f"Matched={len(common):4d}")

    if verbose:
        print()
        if skipped:
            print(f"  Skipped ({len(skipped)}):")
            for name, reason in skipped:
                print(f"    ✗  {name}: {reason}")
            print()

        total_matched = 0
        for rgb_dir, depth_dir in pairs:
            rgb_f   = set(os.listdir(rgb_dir))
            dep_f   = set(os.listdir(depth_dir))
            total_matched += len(rgb_f & dep_f)

        print(f"  {'─'*55}")
        print(f"  Found  : {len(pairs)} valid sequence pairs")
        print(f"  Total matched image pairs: {total_matched}")
        print(f"  {'─'*55}")
        print()
        print("  WHY matched < RGB total:")
        print("  KITTI depth GT is SPARSE — LiDAR only annotates ~26% of")
        print("  RGB frames. Only frames with BOTH RGB and depth count.")
        print()

    return pairs


# ── Run standalone to preview what will be found ──────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    print(f"Scanning: {base}\n")
    print(f"{'Drive':<45}  {'RGB':>5}  {'Depth':>6}  {'Match':>6}")
    print("─" * 70)
    pairs = auto_find_sequences(base)

    if pairs:
        print("\nCopy-paste into SEQUENCE_PAIRS if you want manual control:\n")
        print("SEQUENCE_PAIRS = [")
        for rgb_dir, depth_dir in pairs:
            # Print relative paths for readability
            rgb_rel   = os.path.relpath(rgb_dir,   base)
            depth_rel = os.path.relpath(depth_dir, base)
            print(f'    (os.path.join(BASE_DIR, r"{rgb_rel}"),')
            print(f'     os.path.join(BASE_DIR, r"{depth_rel}")),')
        print("]")