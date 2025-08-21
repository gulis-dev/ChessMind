import os, glob, random

INPUT_DIR = "../../data/shards_v1"
OUTPUT_DIR = "../../data/shards_v1_shuffled"
PATTERN = "positions_shard_*.csv"
ROWS_PER_SHARD = 100000
RANDOM_SEED = 73

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

files = sorted(glob.glob(os.path.join(INPUT_DIR, PATTERN)))
if not files:
    print("No files found.")
    raise SystemExit

all_lines = []
header = None

print("Reading shards...")
for f in files:
    with open(f, "r", encoding="utf-8") as fh:
        lines = fh.read().strip().split("\n")
        if header is None:
            header = lines[0]
        else:
            if lines[0] != header:
                print("WARNING: different header in", f)
        data_part = lines[1:]
        all_lines.extend(data_part)

print("Total rows (without header):", len(all_lines))
print("Shuffling...")
random.shuffle(all_lines)

print("Writing new shuffled shards...")
count = 0
shard_idx = 0
while count < len(all_lines):
    chunk = all_lines[count:count + ROWS_PER_SHARD]
    out_path = os.path.join(OUTPUT_DIR, f"positions_shard_{shard_idx:05d}.csv")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(header + "\n")
        out.write("\n".join(chunk))
    print("Wrote", out_path, "rows:", len(chunk))
    count += ROWS_PER_SHARD
    shard_idx += 1

print("Done. New shard count:", shard_idx)