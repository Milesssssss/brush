import argparse
import json
import math
from pathlib import Path
import random
import struct
import zlib

parser = argparse.ArgumentParser(description="Generate the Issue #7 synthetic COLMAP depth fixtures")
parser.add_argument("output", type=Path)
root = parser.parse_args().output
size = 48
focal = 55
positions = (-0.46, -0.31, -0.16, 0.0, 0.16, 0.31, 0.46, 0.10)
panels = (
    (3.7, -0.89, 0.91, -0.62, 0.60, (46, 97, 146)),
    (2.8, -0.58, -0.02, -0.42, 0.40, (190, 79, 66)),
    (3.13, 0.11, 0.59, -0.45, 0.34, (72, 174, 86)),
)


def rgb(x, y, base):
    light = 0.94 + 0.20 * math.sin(33 * x) * math.cos(28 * y)
    return [round(min(255, c * light)) for c in base]


def observation(cx, px, py):
    ray_x = (px + 0.5 - size / 2) / focal
    ray_y = (py + 0.5 - size / 2) / focal
    for z, xmin, xmax, ymin, ymax, base in sorted(panels):
        x, y = cx + ray_x * z, ray_y * z
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return rgb(x, y, base) + [255], z
    return [0, 0, 0, 0], math.nan


def png(path, rows):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    pixels = b"".join(b"\x00" + bytes(row) for row in rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


def tiff(path, depths):
    offset = 8 + 2 + 11 * 12 + 4
    tags = (
        (256, 4, size), (257, 4, size), (258, 3, 32), (259, 3, 1),
        (262, 3, 1), (273, 4, offset), (277, 3, 1), (278, 4, size),
        (279, 4, 4 * len(depths)), (284, 3, 1), (339, 3, 3),
    )
    path.write_bytes(
        struct.pack("<2sHIH", b"II", 42, 8, len(tags))
        + b"".join(struct.pack("<HHII", tag, kind, 1, val) for tag, kind, val in tags)
        + struct.pack("<I", 0)
        + struct.pack(f"<{len(depths)}f", *depths)
    )


generator = random.Random(721)
seeds = []
for z, xmin, xmax, ymin, ymax, base in panels:
    for yy in range(9):
        for xx in range(11):
            x = xmin + (xmax - xmin) * (xx + 0.5) / 11
            y = ymin + (ymax - ymin) * (yy + 0.5) / 9
            dz = generator.uniform(-0.27, 0.27)
            seeds.append(f"{x} {y} {z + dz} " + " ".join(map(str, rgb(x, y, base))))
for x, y, z in [(-0.49, -0.81, 3.25), (0.08, 0.87, 3.50), (0.88, -0.63, 2.99)]:
    seeds.append(f"{x} {y} {z} 180 180 180")

initial_ply = "\n".join([
    "ply", "format ascii 1.0", f"element vertex {len(seeds)}",
    "property float x", "property float y", "property float z",
    "property uchar red", "property uchar green", "property uchar blue",
    "end_header", *seeds, "",
])

for name in ("metric", "relative"):
    dataset = root / name
    for subdir in ("images", "depth", "sparse/0"):
        (dataset / subdir).mkdir(parents=True, exist_ok=True)
    (dataset / "init.ply").write_text(initial_ply)
    (dataset / "sparse/0/cameras.txt").write_text(
        f"1 PINHOLE {size} {size} {focal} {focal} {size / 2} {size / 2}\n"
    )
    (dataset / "sparse/0/images.txt").write_text("".join(
        f"{index + 1} 1 0 0 0 {-cx:.6f} 0 0 1 view_{index:02d}.png\n\n"
        for index, cx in enumerate(positions)
    ))
    for index, cx in enumerate(positions):
        observations = [
            observation(cx, x, y) for y in range(size) for x in range(size)
        ]
        pixels, depths = zip(*observations)
        png(dataset / "images" / f"view_{index:02d}.png", [
            [c for pixel in pixels[start:start + size] for c in pixel]
            for start in range(0, len(pixels), size)
        ])
        if name == "relative":
            depths = [2.4 / depth - 0.75 + index * 0.12 for depth in depths]
        tiff(dataset / "depth" / f"view_{index:02d}.tiff", depths)

print(json.dumps({"image_size": size, "views": len(positions), "seeds": len(seeds),
                  "split_every": 4, "initialization_seed": 721}))
