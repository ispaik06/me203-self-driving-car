# rename_and_annotate.py
import argparse, csv, os, re, shutil, sys
from pathlib import Path

SAFE = re.compile(r"[^a-z0-9_]+")

def safe_prefix(name: str) -> str:
    return SAFE.sub("_", name.lower()).strip("_")

def iter_class_dirs(root: Path):
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        yield d

def main():
    ap = argparse.ArgumentParser(description="Rename PNGs by class and create annotations.csv")
    ap.add_argument("--src", required=True, help="source root (has class dirs, e.g., red/ green/ straight_line/)")
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy",
                    help="copy or move files (default: copy)")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    class_dirs = list(iter_class_dirs(src_root))
    if not class_dirs:
        print(f"[ERR] No class directories under: {src_root}")
        sys.exit(1)

    # 라벨 ID: 폴더명 알파벳 정렬 기준으로 고정
    class_names_sorted = [d.name for d in class_dirs]
    label2id = {name: i for i, name in enumerate(class_names_sorted)}
    print("[INFO] label mapping =", label2id)

    # CSV 준비 (절대 경로만 기록)
    csv_path = out_root / "annotations.csv"
    rows = []

    counters = {name: 0 for name in class_names_sorted}
    valid_exts = {".png"}

    for class_dir in class_dirs:
        label_name = class_dir.name
        label_id = label2id[label_name]
        prefix = safe_prefix(label_name)
        dst_class_dir = out_root / prefix
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        # 재현성을 위해 정렬
        for src in sorted(class_dir.rglob("*")):
            if not src.is_file():
                continue
            ext = src.suffix.lower()
            if ext not in valid_exts:
                continue

            counters[label_name] += 1
            idx = counters[label_name]

            dst_name = f"{prefix}_{idx:06d}.png"  # 확장자 통일 유지
            dst = dst_class_dir / dst_name

            # 이름 충돌 방지(희귀 케이스)
            k = 0
            while dst.exists():
                k += 1
                dst = dst_class_dir / f"{prefix}_{idx:06d}_{k}.png"

            action = "COPY" if args.mode == "copy" else "MOVE"
            print(f"[{action}] {src} -> {dst}")

            if args.mode == "copy":
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

            # CSV 절대 경로로 기록
            rows.append({
                "filepath": str(dst.resolve()),
                "label_id": label_id,
                "label": label_name,
            })

    # CSV 쓰기
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "label_id", "label"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote {csv_path} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
