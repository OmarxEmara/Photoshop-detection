#!/usr/bin/env python3
import os, glob, random, shutil, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with all real ID fronts")
    ap.add_argument("--out-root", required=True, help="Output root to create splits under")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio", type=str, default="0.2,0.5,0.3",
                    help="calib,train_fonts,val_real ratios; must sum to 1.0")
    args = ap.parse_args()

    r = [float(x) for x in args.ratio.split(",")]
    assert abs(sum(r)-1.0) < 1e-6, "ratios must sum to 1.0"

    imgs = sorted([p for p in glob.glob(os.path.join(args.images,"*"))
                   if p.lower().endswith((".jpg",".jpeg",".png"))])
    if not imgs:
        raise SystemExit("No images found.")

    random.seed(args.seed); random.shuffle(imgs)
    n = len(imgs); n1 = int(r[0]*n); n2 = n1 + int(r[1]*n)
    splits = [("calib", imgs[:n1]),
              ("train_fonts", imgs[n1:n2]),
              ("val_real", imgs[n2:])]

    for name, items in splits:
        outdir = os.path.join(args.out_root, name)
        os.makedirs(outdir, exist_ok=True)
        for p in items:
            shutil.copy2(p, outdir)
        print(f"{name}: {len(items)}")

if __name__ == "__main__":
    main()
