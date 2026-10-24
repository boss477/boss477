#!/usr/bin/env python3
"""Generate an animated, depth-layered point-cloud portrait for the GitHub profile README.

The visual method follows EricJujianZou/EricJujianZou/tools/face.py:
brightness-weighted point sampling, a shallow 3D dome, spatial neighbour links,
and SMIL yaw animation split across depth layers.
"""
import math
import os
import random
from PIL import Image, ImageFilter
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "assets", "fawaz.png")
OUTPUT = os.path.join(ROOT, "assets", "banner-dark.svg")
G = 192
POINT_COUNT = 3000
LAYERS = 24
FRAME_STEPS = 48
DURATION = 14
INK = "#e8e4da"
SOFT = "#9a9690"

def box_blur(values, radius):
    image = Image.new("L", (G, G))
    image.putdata([max(0, min(255, round(value * 255))) for value in values])
    image = image.filter(ImageFilter.BoxBlur(radius))
    return [value / 255 for value in image.getdata()]

def seeded_random(seed):
    state = [seed & 0xffffffff]
    def next_random():
        state[0] = (state[0] + 0x6D2B79F5) & 0xffffffff
        value = state[0]
        value = ((value ^ (value >> 15)) * (1 | value)) & 0xffffffff
        value = (value + (((value ^ (value >> 7)) * (61 | value)) & 0xffffffff) ^ value) & 0xffffffff
        return ((value ^ (value >> 14)) & 0xffffffff) / 4294967296
    return next_random

def load_cutout(path):
    """Remove the outdoor background with GrabCut; return a clean RGBA crop."""
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    # Keep the full head and shoulders inside the foreground initialization box.
    rect = (max(1, int(width * 0.04)), max(1, int(height * 0.10)),
            int(width * 0.92), int(height * 0.88))
    cv2.grabCut(bgr, mask, rect, bg_model, fg_model, 8, cv2.GC_INIT_WITH_RECT)
    alpha = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype("uint8")
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.45)
    ys, xs = np.where(alpha > 32)
    if len(xs) == 0:
        raise RuntimeError("Background removal found no foreground pixels")
    x0, x1 = max(0, int(xs.min()) - 3), min(width, int(xs.max()) + 4)
    y0, y1 = max(0, int(ys.min()) - 3), min(height, int(ys.max()) + 4)
    foreground = np.dstack((rgb, alpha))[y0:y1, x0:x1]
    crop = Image.fromarray(foreground.astype("uint8"), "RGBA")
    side = max(crop.size) + 8
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.alpha_composite(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square.resize((G, G), Image.Resampling.LANCZOS)

def sample_points(image, count=POINT_COUNT, seed=7, dome=0.34):
    pixels = image.load()
    luminance = [0.0] * (G * G)
    alpha = [0.0] * (G * G)
    for y in range(G):
        for x in range(G):
            red, green, blue, a = pixels[x, y]
            index = y * G + x
            alpha[index] = a / 255
            luminance[index] = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255

    histogram = sorted(luminance[i] for i in range(0, G * G, 3) if alpha[i] > 0.5)
    if not histogram:
        raise RuntimeError("The cutout contains no opaque pixels")
    low = histogram[int(len(histogram) * 0.03)]
    high = histogram[int(len(histogram) * 0.985)]
    luminance = [max(0, min(1, (v - low) / max(0.05, high - low))) for v in luminance]
    blurred = box_blur(luminance, 5)
    weight = [
        max(0, min(1, 0.55 * luminance[i] + 2.6 * (luminance[i] - blurred[i]) + 0.12))
        for i in range(G * G)
    ]

    random_value = seeded_random(seed)
    points = []
    taken = bytearray(G * G)
    attempts = 0
    while len(points) < count and attempts < count * 150:
        attempts += 1
        u, v = random_value(), random_value()
        px, py = min(G - 1, int(u * G)), min(G - 1, int(v * G))
        index = py * G + px
        if taken[index] or alpha[index] < 0.4:
            continue
        intensity = weight[index]
        if random_value() > alpha[index] * (0.035 + 0.965 * intensity * intensity):
            continue
        taken[index] = 1
        dx, dy = (u - 0.5) / 0.44, (v - 0.47) / 0.52
        radius_squared = dx * dx + dy * dy
        z = dome * math.sqrt(max(0, 1 - radius_squared)) + (intensity - 0.5) * 0.12
        points.append(((u - 0.5) * 2.3, -(v - 0.47) * 2.3, z, 0.2 + 0.8 * intensity))

    if len(points) < min(1800, count):
        for index in range(G * G):
            if len(points) >= count:
                break
            if alpha[index] > 0.55 and not taken[index]:
                y, x = divmod(index, G)
                u, v = (x + 0.5) / G, (y + 0.5) / G
                intensity = weight[index]
                dx, dy = (u - 0.5) / 0.44, (v - 0.47) / 0.52
                z = dome * math.sqrt(max(0, 1 - dx * dx - dy * dy)) + (intensity - 0.5) * 0.12
                points.append(((u - 0.5) * 2.3, -(v - 0.47) * 2.3, z, 0.2 + 0.8 * intensity))
                taken[index] = 1

    # Build the same short-range neighbour graph as the reference generator.
    cell = 0.09
    grid = {}
    edges = []
    for index, (x, y, z, tone) in enumerate(points):
        grid.setdefault((math.floor(x / cell), math.floor(y / cell)), []).append(index)
    for index, (ax, ay, az, tone) in enumerate(points):
        gx, gy = math.floor(ax / cell), math.floor(ay / cell)
        best = [(1e9, -1), (1e9, -1)]
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for other in grid.get((gx + ox, gy + oy), ()):
                    if other == index:
                        continue
                    dd = (points[other][0] - ax) ** 2 + (points[other][1] - ay) ** 2
                    if dd < best[0][0]:
                        best = [(dd, other), best[0]]
                    elif dd < best[1][0]:
                        best[1] = (dd, other)
        for dd, other in best:
            if other > index and dd < cell * cell:
                edges.append((index, other))
    return points, edges

FONT = {
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    " ": ["00000"] * 7,
}

def pixel_title(text, x, y, cell):
    rectangles = []
    cursor = x
    for character in text:
        for row_index, row in enumerate(FONT.get(character, FONT[" "])):
            for col_index, bit in enumerate(row):
                if bit == "1":
                    xx, yy = cursor + col_index * cell, y + row_index * cell
                    size = cell * 0.88
                    rectangles.append(
                        f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{size:.1f}" height="{size:.1f}"/>'
                    )
        cursor += 5 * cell + cell * 0.58
    return '<g fill="' + INK + '">' + "".join(rectangles) + "</g>"

def build_svg(points, edges):
    z_values = [point[2] for point in points]
    z_min, z_max = min(z_values), max(z_values)
    def point_layer(z):
        return min(LAYERS - 1, int((z - z_min) / (z_max - z_min + 1e-9) * LAYERS))
    layer_centres = [
        z_min + (layer_index + 0.5) * (z_max - z_min) / LAYERS
        for layer_index in range(LAYERS)
    ]
    def yaw(t):
        return 0.55 * math.sin(2 * math.pi * t) ** 3
    yaws = [yaw(k / FRAME_STEPS) for k in range(FRAME_STEPS)] + [yaw(0)]
    key_times = ";".join(f"{k / FRAME_STEPS:.4f}" for k in range(FRAME_STEPS + 1))
    face_scale = 118
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="880" height="300" viewBox="0 0 880 300">',
        '<title>Fawaz Ahamed — animated point-cloud portrait</title>',
        '<defs><radialGradient id="bg"><stop stop-color="#111214"/><stop offset="1" stop-color="#030405"/></radialGradient></defs>',
        '<rect width="880" height="300" fill="#030405"/><rect width="880" height="300" fill="url(#bg)"/>',
        '<g opacity=".22" fill="none" stroke="#8b8881" stroke-width=".6"><ellipse cx="202" cy="150" rx="166" ry="76" transform="rotate(-14 202 150)"/><ellipse cx="202" cy="150" rx="164" ry="78" transform="rotate(17 202 150)"/></g>',
        '<g transform="translate(200 150)"><animate attributeName="opacity" values=".92;1;.92" dur="8s" repeatCount="indefinite"/>'
    ]
    for layer_index in range(LAYERS):
        translations = ";".join(f"{layer_centres[layer_index] * math.sin(angle) * face_scale:.1f} 0" for angle in yaws)
        scales = ";".join(f"{math.cos(angle):.4f} 1" for angle in yaws)
        svg.append(
            f'<g><animateTransform attributeName="transform" type="translate" values="{translations}" keyTimes="{key_times}" dur="{DURATION}s" repeatCount="indefinite"/>'
            f'<g><animateTransform attributeName="transform" type="scale" values="{scales}" keyTimes="{key_times}" dur="{DURATION}s" repeatCount="indefinite"/>'
        )
        layer_points = [point for point in points if point_layer(point[2]) == layer_index]
        layer_edges = [
            (i, j) for i, j in edges
            if point_layer((points[i][2] + points[j][2]) / 2) == layer_index
        ]
        segments = [
            f"M{points[i][0] * face_scale:.0f} {-points[i][1] * face_scale:.0f}L{points[j][0] * face_scale:.0f} {-points[j][1] * face_scale:.0f}"
            for i, j in layer_edges
        ]
        if segments:
            svg.append(f'<path d="{"".join(segments)}" stroke="{INK}" stroke-opacity=".20" stroke-width=".6"/>')
        circles = []
        for x, y, z, tone in layer_points:
            radius = 0.7 + 1.1 * tone
            opacity = 0.35 + 0.65 * tone
            circles.append(
                f'<circle cx="{x * face_scale:.0f}" cy="{-y * face_scale:.0f}" r="{radius:.1f}" fill="{INK}" fill-opacity="{opacity:.2f}"/>'
            )
        svg.append("".join(circles) + "</g></g>")
    svg.append("</g>")
    svg.append('<path d="M414 70V231" stroke="#393834" stroke-width="1"/>')
    svg.append(f'<text x="446" y="78" font-family="monospace" font-size="12" fill="{SOFT}" letter-spacing="2.5">COMPUTER SCIENCE  /  APPLIED AI</text>')
    svg.append(pixel_title("FAWAZ AHAMED", 418, 100, 6.1))
    svg.append(f'<text x="420" y="177" font-family="monospace" font-size="13" fill="{INK}">CO-FOUNDER &amp; CTO  ·  SYMMET</text>')
    svg.append(f'<text x="420" y="201" font-family="monospace" font-size="11" fill="{SOFT}">AGENTIC AI  ·  MULTIMODAL RAG  ·  PRODUCT ENGINEERING</text>')
    svg.append(f'<circle cx="421" cy="229" r="3" fill="{INK}"><animate attributeName="opacity" values="1;.25;1" dur="1.4s" repeatCount="indefinite"/></circle>')
    svg.append(f'<text x="433" y="233" font-family="monospace" font-size="11" fill="{SOFT}">BUILDING SYSTEMS THAT SHIP</text>')
    svg.append("</svg>")
    return "".join(svg)

def main():
    if not os.path.isfile(SOURCE):
        raise FileNotFoundError(f"Portrait source not found: {SOURCE}")
    portrait = load_cutout(SOURCE)
    points, edges = sample_points(portrait)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    result = build_svg(points, edges)
    with open(OUTPUT, "w", encoding="utf-8") as file:
        file.write(result)
    print(f"Generated {OUTPUT} ({len(result):,} characters; {len(points)} points; {len(edges)} links)")

if __name__ == "__main__":
    main()
