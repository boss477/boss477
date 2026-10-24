#!/usr/bin/env python3
"""Deterministic photo-derived particle banner; no raster or script in the SVG.

Depth-sliced SMIL yaw follows the approach demonstrated by EricJujianZou's
profile tools/face.py. Geometry here is sampled from Fawaz's own portrait.
Run: python tools/generate_particle_banner.py
The 200px original limits facial resolution; replace it with the original photo
and update the normalized silhouette if its framing changes.
"""
from pathlib import Path
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'assets/fawaz.png'
OUTPUT = ROOT / 'assets/banner-dark.svg'
INK = '#e8e4da'
LAYERS = 24
# Manually traced foreground, in the original photo's 200x200 coordinates.
# Excludes the second person behind the left shoulder, wall, plants and sky.
OUTLINE = [(72,50),(76,45),(83,42),(91,41),(96,39),(103,40),(108,40),
 (113,43),(118,43),(123,47),(126,51),(126,56),(123,61),(124,68),
 (130,72),(129,78),(125,82),(123,87),(121,94),(118,100),
 (115,104),(125,108),(139,112),(149,115),(155,124),(162,142),
 (172,161),(179,176),(180,184),(176,197),(24,199),(27,179),
 (31,160),(34,144),(38,128),(48,124),(64,116),(80,108),
 (82,103),(81,92),(78,88),(75,84),(74,77),(73,72),(71,65),(71,57)]


def cutout():
    im = Image.open(SOURCE).convert('RGB')
    w,h = im.size
    # Trace at 4x resolution for a feathered, deterministic contour.
    mask = Image.new('L',(w*4,h*4))
    ImageDraw.Draw(mask).polygon([(x*w/50,y*h/50) for x,y in OUTLINE],fill=255)
    mask = mask.resize(im.size, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(.35))
    rgba = im.convert('RGBA'); rgba.putalpha(mask)
    # Tight upper-body crop: hair and chin occupy a much larger canvas fraction.
    box = tuple(round(v*s) for v,s in zip((24/200,37/200,181/200,166/200),(w,h,w,h)))
    return rgba.crop(box)


def sample(image):
    # Interpolation does not invent facial detail. A fine sampling grid avoids
    # the old one-large-dot-per-source-pixel clumping and preserves dark glasses.
    im = np.asarray(image.resize((730,600),Image.Resampling.LANCZOS))
    alpha = im[:,:,3]/255.
    gray = cv2.cvtColor(im[:,:,:3],cv2.COLOR_RGB2GRAY).astype(float)/255
    valid = alpha>.8
    lo,hi = np.quantile(gray[valid],[.02,.99])
    lum = np.clip((gray-lo)/(hi-lo),0,1)
    blur = cv2.GaussianBlur(lum,(0,0),2.0)
    detail = np.clip(lum-blur,-.2,.2)
    tone = np.clip(lum+1.3*detail,0,1)
    edge = np.clip(np.hypot(cv2.Sobel(lum,cv2.CV_64F,1,0),cv2.Sobel(lum,cv2.CV_64F,0,1)),0,1)
    random = np.random.default_rng(477)
    points=[]
    # Jittered stratified sampling prevents overlaps and keeps negative space.
    for y in np.arange(0,600,2.0):
        for x in np.arange(0,730,2.0):
            px=min(729,int(x+random.uniform(0,2))); py=min(599,int(y+random.uniform(0,2)))
            a=alpha[py,px]; t=tone[py,px]; e=edge[py,px]
            fade=np.clip((600-py)/95,0,1)
            density=(.02+.93*math.sqrt(t)+.19*e)*a*fade
            if random.random()>density: continue
            # The image maps to a 560x460 upper-body portrait.
            xx=(px/730-.5)*560; yy=py/600*460
            # Synthetic depth relief, not a reconstructed head mesh.
            # Wider separation makes parallax visible at README display sizes.
            dome=math.sqrt(max(0,1-(xx/300)**2-((yy-205)/280)**2))
            z=105*dome+24*t
            radius=.38+.57*math.sqrt(t)
            opacity=.5+.48*math.sqrt(t)
            points.append((xx,yy,z,t,radius,opacity))
    return points

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


def build_svg(points):
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520" viewBox="0 0 1200 520" role="img" aria-labelledby="title desc">',
    '<title id="title">Fawaz Ahamed — animated particle portrait</title>',
    '<desc id="desc">Photo-derived monochrome portrait with glasses, hair, beard and suit, in twenty-four turning depth layers, above pixel lettering.</desc>',
    '<rect width="1200" height="520" fill="#000"/>']
    r=np.random.default_rng(38)
    out.append('<g fill="#e8e4da">')
    for _ in range(95):
        x,y=r.uniform(60,1140),r.uniform(25,465)
        if 350<x<870 and y<435: continue
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r.uniform(.3,.9):.2f}" opacity="{r.uniform(.15,.55):.2f}"/>')
    out.append('</g>')
    # Sparse orbital paths behind the portrait; no visual UI decorations.
    for angle,rx,ry,dur in [(-13,435,92,32),(12,370,113,39)]:
        out.append(f'<g transform="translate(600 235) rotate({angle})"><ellipse rx="{rx}" ry="{ry}" fill="none" stroke="#e8e4da" stroke-width=".55" opacity=".2"/>')
        for j in range(65):
            a=2*math.pi*j/65
            out.append(f'<circle cx="{rx*math.cos(a):.1f}" cy="{ry*math.sin(a):.1f}" r=".65" fill="{INK}" opacity=".4"/>')
        out.append(f'<g><animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="{dur}s" repeatCount="indefinite"/><circle cx="{rx}" cy="0" r="2.1" fill="{INK}"/></g></g>')
    out.append('<g transform="translate(600 10)">')
    zmin,zmax=min(p[2] for p in points),max(p[2] for p in points)
    for k in range(LAYERS):
        layer=[p for p in points if min(LAYERS-1,int((p[2]-zmin)/(zmax-zmin)*LAYERS))==k]
        depth=zmin+(k+.5)*(zmax-zmin)/LAYERS
        ts=[i/48 for i in range(49)]
        yaw=[.55*math.sin(2*math.pi*t) for t in ts]
        kt=';'.join(f'{t:.5f}' for t in ts)
        tv=';'.join(f'{depth*math.sin(a):.2f} 0' for a in yaw)
        sv=';'.join(f'{math.cos(a):.5f} 1' for a in yaw)
        out.append(f'<g><animateTransform attributeName="transform" type="translate" values="{tv}" keyTimes="{kt}" dur="10s" repeatCount="indefinite"/><g><animateTransform attributeName="transform" type="scale" values="{sv}" keyTimes="{kt}" dur="10s" repeatCount="indefinite"/>')
        # Each layer owns its geometry, so links move with their endpoints.
        from scipy.spatial import cKDTree
        jacket=[p for p in layer if p[1]>275 or p[1]<95]
        if len(jacket)>1:
            pairs=sorted(cKDTree([(p[0],p[1]) for p in jacket]).query_pairs(4.5))
            seg=''.join(f'M{jacket[i][0]:.1f} {jacket[i][1]:.1f}L{jacket[j][0]:.1f} {jacket[j][1]:.1f}' for i,j in pairs[::2])
            out.append(f'<path d="{seg}" fill="none" stroke="{INK}" stroke-opacity=".18" stroke-width=".45"/>')
        # Batch equal-tone dots as round-capped paths: far fewer DOM elements.
        for band in range(12):
            dots=[p for p in layer if min(11,int(p[3]*12))==band]
            if not dots: continue
            t=(band+.5)/12
            radius=.38+.57*math.sqrt(t)
            opacity=.5+.48*math.sqrt(t)
            d=''.join(f'M{x:.2f} {y:.2f}h.01' for x,y,*_ in dots)
            out.append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="{2*radius:.2f}" stroke-linecap="round" opacity="{opacity:.2f}"/>')
        out.append('</g></g>')
    out.append('</g>')
    # Mask the fading torso under the title with solid black.
    out.append('<rect y="438" width="1200" height="82" fill="#000"/>')
    cell=12.0
    tw=len('FAWAZ AHAMED')*(5.58*cell)-.58*cell
    out.append(pixel_title('FAWAZ AHAMED',(1200-tw)/2,426,cell))
    out.append('</svg>')
    return ''.join(out)


def main():
    image=cutout()
    image.save(ROOT/'assets/fawaz-cutout.png')
    points=sample(image)
    OUTPUT.write_text(build_svg(points))
    print(f'{len(points):,} particles, {LAYERS} depth layers, {OUTPUT.stat().st_size:,} bytes')

if __name__=='__main__': main()
