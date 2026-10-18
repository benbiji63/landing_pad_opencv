from pathlib import Path
from PIL import Image, ImageDraw
import random

out = Path('test_images')
out.mkdir(exist_ok=True)

def make(name, bg, fg, center, radius, ring=False):
    im = Image.new('RGB', (640, 480), bg)
    d = ImageDraw.Draw(im)
    random.seed(42)
    for _ in range(500):
        x, y = random.randrange(640), random.randrange(480)
        delta = random.choice([-8, -4, 0, 4, 8])
        c = tuple(max(0, min(255, v + delta)) for v in bg)
        d.ellipse((x, y, x+2, y+2), fill=c)
    box = (center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius)
    d.ellipse(box, fill=fg)
    if ring:
        pad = max(8, radius // 7)
        d.ellipse((box[0]+pad, box[1]+pad, box[2]-pad, box[3]-pad), fill=bg)
    im.save(out / name)

make('filled_circle.png', (105, 125, 92), (235, 235, 220), (335, 225), 100)
make('ring.png', (150, 140, 125), (240, 240, 240), (300, 250), 95, True)
make('off_center_circle.png', (70, 85, 70), (210, 210, 210), (450, 160), 70)
print('Generated', len(list(out.glob('*.png'))), 'synthetic images in', out)
