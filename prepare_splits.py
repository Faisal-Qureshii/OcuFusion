# prepare_splits.py
import os, argparse, glob, shutil
from pathlib import Path

def read_labels(path):
    items=[]
    with open(path,'r',encoding='utf-8',errors='replace') as f:
        for line in f:
            s=line.strip()
            if not s: continue
            parts = s.split()
            if len(parts) >= 2:
                items.append((parts[0], parts[1]))
            else:
                # fallback if only one token per line (some of your files contain blocks)
                # skip malformed
                continue
    return items

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def find_and_copy(name, src_dirs, dst_dir):
    # search for file by name (with any extension) in src_dirs
    patterns = [f"{name}.*", f"{name}*.*"]
    for src in src_dirs:
        for pat in patterns:
            matches = glob.glob(os.path.join(src, pat))
            if matches:
                # pick first match
                src_path = matches[0]
                dst_path = os.path.join(dst_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)
                return os.path.basename(src_path)
    # try searching by relaxed substring match (rare)
    for src in src_dirs:
        matches = glob.glob(os.path.join(src, f"*{name}*"))
        if matches:
            src_path = matches[0]
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            return os.path.basename(src_path)
    return None

def prepare(label_file, src_dirs, out_subset_dir):
    items = read_labels(label_file)
    if len(items)==0:
        print(f"[WARN] No labels found in {label_file}")
        return
    ensure_dir(out_subset_dir)
    img_root = os.path.join(out_subset_dir, "ImageData")
    ensure_dir(img_root)
    new_label_path = os.path.join(out_subset_dir, os.path.basename(label_file))
    written = 0
    with open(new_label_path, 'w', encoding='utf-8') as wf:
        for name,label in items:
            copied = find_and_copy(name, src_dirs, img_root)
            if copied:
                wf.write(f"{copied} {label}\n")
                written += 1
            else:
                print(f"[MISSING] {name} not found in any src dirs.")
    print(f"Prepared {out_subset_dir}: copied {written}/{len(items)} files. Labels -> {new_label_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fundus-train-src", nargs='+', help="Folders where fundus train images live")
    parser.add_argument("--fundus-train-labels", help="Path to large9cls.txt (fundus train labels)")
    parser.add_argument("--fundus-test-src", nargs='+', help="Folders where fundus test images live")
    parser.add_argument("--fundus-test-labels", help="Path to testlarge9cls.txt (fundus test labels)")
    parser.add_argument("--oct-train-src", nargs='+')
    parser.add_argument("--oct-train-labels")
    parser.add_argument("--oct-test-src", nargs='+')
    parser.add_argument("--oct-test-labels")
    parser.add_argument("--out-base", required=True, help="Output base folder (e.g. /Users/you/Desktop/mulyeye_backend/data/assemble )")
    args = parser.parse_args()

    # FUNDUS
    if args.fundus_train_labels and args.fundus_train_src:
        prepare(args.fundus_train_labels, args.fundus_train_src, os.path.join(args.out_base, "train"))
    if args.fundus_test_labels and args.fundus_test_src:
        prepare(args.fundus_test_labels, args.fundus_test_src, os.path.join(args.out_base, "test"))

    # OCT (separate dataset base: assemble_oct)
    if args.oct_train_labels and args.oct_train_src:
        prepare(args.oct_train_labels, args.oct_train_src, os.path.join(args.out_base, "train"))
    if args.oct_test_labels and args.oct_test_src:
        prepare(args.oct_test_labels, args.oct_test_src, os.path.join(args.out_base, "test"))
