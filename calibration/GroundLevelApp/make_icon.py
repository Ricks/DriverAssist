#!/usr/bin/env python3
"""Generates the GroundLevel app icon -- a hardware-store spirit level --
as a flat 1024x1024 opaque PNG into the asset catalog. Re-run after edits:

    python3 calibration/GroundLevelApp/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 1024
OUT = Path(__file__).with_name("Assets.xcassets") / "AppIcon.appiconset" / "icon-1024.png"


def lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def vgrad(size, top, bot):
    w, h = size
    col = Image.new("RGB", (1, h))
    for y in range(h):
        col.putpixel((0, y), lerp(top, bot, y / max(1, h - 1)))
    return col.resize((w, h))


img = vgrad((S, S), (0x26, 0x2C, 0x38), (0x13, 0x15, 0x1A))
d = ImageDraw.Draw(img, "RGBA")

# ---- level body ----
body_w, body_h = int(S * 0.84), int(S * 0.42)
bx0, by0 = (S - body_w) // 2, (S - body_h) // 2
bx1, by1 = bx0 + body_w, by0 + body_h
br = int(body_h * 0.16)
cy = (by0 + by1) // 2

# drop shadow
sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ImageDraw.Draw(sh).rounded_rectangle((bx0, by0 + 30, bx1, by1 + 30), br, fill=(0, 0, 0, 160))
sh = sh.filter(ImageFilter.GaussianBlur(28))
img.paste(sh, (0, 0), sh)

# body: level-yellow with a vertical sheen
body = vgrad((body_w, body_h), (0xFF, 0xDD, 0x5E), (0xDC, 0x9F, 0x1C))
m = Image.new("L", (body_w, body_h), 0)
ImageDraw.Draw(m).rounded_rectangle((0, 0, body_w - 1, body_h - 1), br, fill=255)
img.paste(body, (bx0, by0), m)
d.rounded_rectangle((bx0, by0, bx1, by1), br, outline=(0x7C, 0x55, 0x0D, 255), width=7)
d.line((bx0 + br, by0 + 18, bx1 - br, by0 + 18), fill=(255, 255, 255, 60), width=6)

# dark end caps
cap_w = int(body_w * 0.08)
for x in (bx0, bx1 - cap_w):
    d.rounded_rectangle((x, by0, x + cap_w, by1), br, fill=(0x2A, 0x2D, 0x35, 255))
d.rounded_rectangle((bx0, by0, bx1, by1), br, outline=(0x2A, 0x2D, 0x35, 255), width=4)

# ---- small end sight-vials (short horizontal capsules, well clear of centre) ----
sv_w, sv_h = int(body_w * 0.11), int(body_h * 0.17)
for scx in (bx0 + cap_w + int(body_w * 0.075), bx1 - cap_w - int(body_w * 0.075)):
    box = (scx - sv_w // 2, cy - sv_h // 2, scx + sv_w // 2, cy + sv_h // 2)
    d.rounded_rectangle(box, sv_h // 2, fill=(0xEC, 0xF6, 0xEF, 255),
                        outline=(0x33, 0x37, 0x3F, 255), width=5)
    d.ellipse((scx - sv_h // 2 + 4, cy - sv_h // 2 + 4, scx + sv_h // 2 - 4, cy + sv_h // 2 - 4),
              fill=(0x74, 0xCE, 0x66, 255), outline=(0x45, 0x93, 0x39, 255), width=3)

# ---- centre vial: a WIDE, short horizontal capsule + a centred bubble ----
vw, vh = int(body_w * 0.46), int(body_h * 0.34)
vbox = (S // 2 - vw // 2, cy - vh // 2, S // 2 + vw // 2, cy + vh // 2)
d.rounded_rectangle(vbox, vh // 2, fill=(0xEF, 0xFA, 0xF2, 255),
                    outline=(0x2C, 0x30, 0x38, 255), width=8)

bd = int(body_h * 0.28)                       # bubble diameter -- fits inside the vial
bbox = (S // 2 - bd // 2, cy - bd // 2, S // 2 + bd // 2, cy + bd // 2)
d.ellipse(bbox, fill=(0x63, 0xC8, 0x54, 255), outline=(0x3D, 0x8C, 0x33, 255), width=6)
d.ellipse((S // 2 - int(bd * 0.34), cy - int(bd * 0.36),
           S // 2 - int(bd * 0.02), cy - int(bd * 0.02)),
          fill=(255, 255, 255, 120))

# graduation lines, just outside the bubble
lx = int(bd * 0.62)
for dx in (-lx, lx):
    d.line((S // 2 + dx, cy - vh // 2 + 10, S // 2 + dx, cy + vh // 2 - 10),
           fill=(0x22, 0x26, 0x2E, 255), width=8)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.convert("RGB").save(OUT, "PNG")
print("wrote", OUT, img.size)
