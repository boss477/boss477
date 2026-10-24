#!/usr/bin/env python3
"""Generate the profile banner with a cutout portrait and animated depth layers.

Sampling, spatial links, and yaw keyframes follow the method in
EricJujianZou/EricJujianZou/tools/face.py. The source image is assets/fawaz.png.
"""
import math
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "assets", "fawaz.png")
OUTPUT = os.path.join(ROOT, "assets", "banner-dark.svg")

G = 224
POINT_COUNT = 9000
LAYERS = 26
STEPS = 48
DURATION = 14
INK = "#e8e4da"
SOFT = "#96938c"

# Conservative silhouette around the supplied 200x200 image. GrabCut refines
# the edge to remove the wall and plants without sampling them as portrait points.
SILHOUETTE = np.array([
    (74,49),(74,45),(82,42),(94,41),(105,42),(116,45),(123,50),(127,56),
    (129,62),(128,69),(131,74),(127,78),(125,84),(121,90),(115,95),(109,99),
    (110,103),(120,105),(134,109),(149,115),(160,122),(169,130),(175,139),
    (177,147),(176,155),(172,165),(171,173),(175,185),(178,198),
    (25,198),(25,185),(26,171),(29,156),(33,142),(40,131),(48,123),(58,116),
    (69,110),(79,105),(80,101),(74,96),(68,91),(64,86),(60,82),(58,76),
    (57,69),(59,62),(62,56),(68,51)
], np.int32)


def rng(seed):
    state = [seed & 0xffffffff]
    def random_value():
        state[0] = (state[0] + 0x6D2B79F5) & 0xffffffff
        value = state[0]
        value = ((value ^ (value >> 15)) * (1 | value)) & 0xffffffff
        value = (value + (((value ^ (value >> 7)) * (61 | value)) & 0xffffffff) ^ value) & 0xffffffff
        return ((value ^ (value >> 14)) & 0xffffffff) / 4294967296
    return random_value


def cutout(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    height, width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    silhouette = np.zeros((height, width), np.uint8)
    cv2.fillPoly(silhouette, [SILHOUETTE], 1)
    mask = np.full((height, width), cv2.GC_BGD, np.uint8)
    mask[silhouette == 1] = cv2.GC_PR_FGD
    inner = cv2.erode(silhouette, np.ones((5, 5), np.uint8), iterations=1)
    mask[(silhouette == 1) & (inner == 0)] = cv2.GC_PR_BGD

    # Seed only interior hair, face, and suit regions, never the contour edge.
    cv2.ellipse(mask, (96,58), (20,10), 0, 0, 360, cv2.GC_FGD, -1)
    cv2.ellipse(mask, (101,75), (16,20), 0, 0, 360, cv2.GC_FGD, -1)
    cv2.ellipse(mask, (94,147), (26,28), 0, 0, 360, cv2.GC_FGD, -1)
    cv2.ellipse(mask, (134,155), (30,24), 0, 0, 360, cv2.GC_FGD, -1)

    background_model = np.zeros((1,65), np.float64)
    foreground_model = np.zeros((1,65), np.float64)
    cv2.grabCut(bgr, mask, None, background_model, foreground_model, 10, cv2.GC_INIT_WITH_MASK)
    alpha = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Keep the connected component containing the face; remove stray background islands.
    _, labels, _, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), 8)
    face_label = labels[75,101]
    if face_label > 0:
        alpha = np.where(labels == face_label, 255, 0).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (3,3), 0.55)

    # Fade the lower torso into sparse particles for a cleaner, face-forward banner.
    fade = np.ones(height, np.float32)
    for y in range(145, height):
        fade[y] = max(0.0, min(1.0, (168 - y) / 23.0))
    alpha = (alpha.astype(np.float32) * fade[:,None]).astype(np.uint8)
    rgba = Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")
    crop = rgba.crop((22,34,180,168))
    side = max(crop.size) + 8
    square = Image.new("RGBA", (side,side), (0,0,0,0))
    square.alpha_composite(crop, ((side-crop.width)//2, (side-crop.height)//2))
    return square.resize((G,G), Image.Resampling.LANCZOS)


def sample_points(image, count=POINT_COUNT, seed=7, dome=0.40):
    pixels = image.load()
    luminance = [0.0] * (G*G)
    alpha = [0.0] * (G*G)
    for y in range(G):
        for x in range(G):
            red, green, blue, a = pixels[x,y]
            index = y*G+x
            alpha[index] = a/255
            luminance[index] = (0.2126*red + 0.7152*green + 0.0722*blue)/255

    histogram = sorted(luminance[i] for i in range(0,G*G,3) if alpha[i] > 0.5)
    low = histogram[int(len(histogram)*0.03)] if histogram else 0
    high = histogram[int(len(histogram)*0.985)] if histogram else 1
    luminance = [max(0,min(1,(v-low)/max(0.05,high-low))) for v in luminance]

    # Local contrast preserves details from the glasses, beard, and hairline.
    gray = Image.new("L", (G,G))
    gray.putdata([round(v*255) for v in luminance])
    enhanced = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8)).apply(np.asarray(gray))
    luminance = [v/255 for v in enhanced.reshape(-1)]
    blurred = [v/255 for v in gray_from(enhanced).filter(ImageFilter.BoxBlur(5)).getdata()]
    weight = [
        max(0,min(1,0.55*luminance[i] + 2.6*(luminance[i]-blurred[i]) + 0.12))
        for i in range(G*G)
    ]

    random_value = rng(seed)
    points = []
    taken = bytearray(G*G)
    attempts = 0
    while len(points) < count and attempts < count*150:
        attempts += 1
        u, v = random_value(), random_value()
        px, py = min(G-1,int(u*G)), min(G-1,int(v*G))
        index = py*G+px
        if taken[index] or alpha[index] < 0.4:
            continue
        tone = weight[index]
        if random_value() > alpha[index] * (0.18 + 0.82*tone*tone):
            continue
        taken[index] = 1
        dx, dy = (u-0.5)/0.44, (v-0.47)/0.52
        radius2 = dx*dx + dy*dy
        z = dome*math.sqrt(max(0,1-radius2)) + (tone-0.5)*0.12
        points.append(((u-0.5)*2.3, -(v-0.47)*2.3, z, 0.25+0.75*tone))

    if len(points) < count:
        for index in range(G*G):
            if len(points) >= count:
                break
            if alpha[index] > 0.55 and not taken[index]:
                py, px = divmod(index, G)
                u, v = (px+0.5)/G, (py+0.5)/G
                tone = weight[index]
                dx, dy = (u-0.5)/0.44, (v-0.47)/0.52
                z = dome*math.sqrt(max(0,1-dx*dx-dy*dy)) + (tone-0.5)*0.12
                points.append(((u-0.5)*2.3, -(v-0.47)*2.3, z, 0.25+0.75*tone))
                taken[index] = 1

    # Sparse short links between spatial neighbors, matching the source generator.
    cell = 0.075
    grid = {}
    edges = []
    for i,(x,y,z,tone) in enumerate(points):
        grid.setdefault((math.floor(x/cell),math.floor(y/cell)),[]).append(i)
    for i,(ax,ay,az,at) in enumerate(points):
        cx,cy = math.floor(ax/cell),math.floor(ay/cell)
        best = [(1e9,-1),(1e9,-1)]
        for ox in (-1,0,1):
            for oy in (-1,0,1):
                for j in grid.get((cx+ox,cy+oy),()):
                    if j == i:
                        continue
                    distance = (points[j][0]-ax)**2 + (points[j][1]-ay)**2
                    if distance < best[0][0]:
                        best = [(distance,j),best[0]]
                    elif distance < best[1][0]:
                        best[1] = (distance,j)
        for distance,j in best:
            if j > i and distance < cell*cell:
                edges.append((i,j))
    return points, edges


def gray_from(array):
    return Image.fromarray(array.astype(np.uint8), "L")


FONT = {
    "F":["11111","10000","11110","10000","10000","10000","10000"],
    "A":["01110","10001","10001","11111","10001","10001","10001"],
    "W":["10001","10001","10001","10101","10101","10101","01010"],
    "Z":["11111","00001","00010","00100","01000","10000","11111"],
    "H":["10001","10001","10001","11111","10001","10001","10001"],
    "M":["10001","11011","10101","10101","10001","10001","10001"],
    "E":["11111","10000","10000","11110","10000","10000","11111"],
    "D":["11110","10001","10001","10001","10001","10001","11110"],
    " ":["00000"]*7,
}


def pixel_title(text, x, y, cell):
    paths = []
    cursor = x
    for character in text:
        for row_index,row in enumerate(FONT.get(character,FONT[" "])):
            for col_index,bit in enumerate(row):
                if bit == "1":
                    xx, yy = cursor+col_index*cell, y+row_index*cell
                    size = cell*0.88
                    paths.append(f"M{xx:.1f} {yy:.1f}h{size:.1f}v{size:.1f}h-{size:.1f}z")
        cursor += 5*cell + cell*0.58
    return f'<path d="{"".join(paths)}" fill="{INK}"/>'


def build_svg(points, edges):
    width, height = 1000, 380
    center_x, center_y, scale = 500, 133, 130
    depths = [point[2] for point in points]
    min_z, max_z = min(depths), max(depths)
    def layer(z):
        return min(LAYERS-1, int((z-min_z)/(max_z-min_z+1e-9)*LAYERS))
    layer_centers = [min_z+(k+0.5)*(max_z-min_z)/LAYERS for k in range(LAYERS)]
    def yaw(t):
        return 0.55*math.sin(2*math.pi*t)**3
    yaws = [yaw(k/STEPS) for k in range(STEPS)] + [yaw(0)]
    key_times = ";".join(f"{k/STEPS:.4f}" for k in range(STEPS+1))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title>Fawaz Ahamed — animated point-cloud portrait</title>',
        '<defs><radialGradient id="bg"><stop stop-color="#101113"/><stop offset="1" stop-color="#020304"/></radialGradient></defs>',
        f'<rect width="{width}" height="{height}" fill="#020304"/><rect width="{width}" height="{height}" fill="url(#bg)"/>',
        '<g fill="none" stroke="#6f6c66" stroke-width=".55" opacity=".32"><ellipse cx="500" cy="135" rx="230" ry="56" transform="rotate(-12 500 135)"/><ellipse cx="500" cy="135" rx="228" ry="57" transform="rotate(14 500 135)"/></g>',
        '<g fill="#a7a39b" opacity=".66>'
    ]
    random_value = rng(33)
    for _ in range(110):
        x, y = random_value()*width, random_value()*height
        if 360 < x < 640 and 10 < y < 275:
            continue
        radius = 0.45+random_value()*0.65
        opacity = 0.18+random_value()*0.5
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.2f}" opacity="{opacity:.2f}"/>')
    svg.append('</g>')
    svg.append('<ellipse cx="500" cy="145" rx="230" ry="75" transform="rotate(-15 500 145)" fill="none" stroke="#aaa69d" stroke-opacity=".25" stroke-width=".7"/>')
    svg.append(f'<g transform="translate({center_x} {center_y})"><animate attributeName="opacity" values=".94;1;.94" dur="8s" repeatCount="indefinite"/>')

    for k in range(LAYERS):
        translations = ";".join(f"{layer_centers[k]*math.sin(angle)*scale:.1f} 0" for angle in yaws)
        scales = ";".join(f"{math.cos(angle):.4f} 1" for angle in yaws)
        svg.append(
            f'<g><animateTransform attributeName="transform" type="translate" values="{translations}" keyTimes="{key_times}" dur="{DURATION}s" repeatCount="indefinite"/>'
            f'<g><animateTransform attributeName="transform" type="scale" values="{scales}" keyTimes="{key_times}" dur="{DURATION}s" repeatCount="indefinite"/>'
        )
        layer_indices = [i for i,p in enumerate(points) if layer(p[2]) == k]
        segments = [
            f'M{points[i][0]*scale:.0f} {-points[i][1]*scale:.0f}L{points[j][0]*scale:.0f} {-points[j][1]*scale:.0f}'
            for i,j in edges if layer((points[i][2]+points[j][2])*0.5) == k
        ]
        if segments:
            svg.append(f'<path d="{"".join(segments)}" stroke="{INK}" stroke-opacity=".15" stroke-width=".5"/>')
        for cutoff,color,opacity,radius in [
            (0.45,"#aaa79f",0.52,0.8),
            (0.72,"#d6d2c9",0.76,1.05),
            (1.01,"#fffdf5",0.96,1.3),
        ]:
            dots = []
            for i in layer_indices:
                x,y,z,tone = points[i]
                if cutoff == 0.45 and tone >= 0.45:
                    continue
                if cutoff == 0.72 and not (0.45 <= tone < 0.72):
                    continue
                if cutoff == 1.01 and tone < 0.72:
                    continue
                dots.append(f'<circle cx="{x*scale:.1f}" cy="{-y*scale:.1f}" r="{radius}" fill="{color}" fill-opacity="{opacity:.2f}"/>')
            svg.extend(dots)
        svg.append('</g></g>')
    svg.append('</g>')

    cell = 5.8
    title_width = len("FAWAZ AHAMED")*(5*cell+cell*0.58)-cell*0.58
    svg.append(pixel_title("FAWAZ AHAMED",(width-title_width)/2,278,cell))
    svg.append(f'<text x="{width/2}" y="342" text-anchor="middle" font-family="monospace" font-size="12.5" fill="{SOFT}" letter-spacing="3">COMPUTER SCIENCE  ·  APPLIED AI  ·  SYMMET</text>')
    svg.append('<g fill="none" stroke="#9a968e" stroke-width=".7" stroke-opacity=".5"><path d="M385 359h62M553 359h62"/><path d="M490 354v10M500 354v10M510 354v10"/></g>')
    svg.append('</svg>')
    return "".join(svg)


def main():
    if not os.path.isfile(SOURCE):
        raise FileNotFoundError(f"Missing portrait source: {SOURCE}")
    portrait = cutout(SOURCE)
    points, edges = sample_points(portrait)
    banner = build_svg(points, edges)
    with open(OUTPUT,"w",encoding="utf-8") as file:
        file.write(banner)
    print(f"Generated {OUTPUT}: {len(points)} particles, {len(edges)} links, {len(banner):,} characters")


if __name__ == "__main__":
    main()
