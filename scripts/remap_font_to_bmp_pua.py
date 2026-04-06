"""
Remap VoynichUnicode.ttf high code points (Supplementary PUA-B, U+FF400–FF51F)
to the BMP Private Use Area (U+E000–E11F).

Produces:
  - voynich_fonts/Voynich/BMPVoynichUnicode.ttf
  - voynich-attack/transcription/unicode_dict_bmp.json

Existing files are left untouched.
"""

import json
from pathlib import Path
from fontTools.ttLib import TTFont

# ==============================================================================
# Config
# ==============================================================================
ROOT = Path(__file__).resolve().parent.parent

SRC_FONT = ROOT / "voynich_fonts/Voynich/VoynichUnicode.ttf"
DST_FONT = ROOT / "voynich_fonts/Voynich/BMPVoynichUnicode.ttf"

SRC_DICT = ROOT / "voynich-attack/transcription/unicode_dict.json"
DST_DICT = ROOT / "voynich-attack/transcription/unicode_dict_bmp.json"

# Shift: U+FF400 → U+E000  (subtract 0x11400)
HIGH_START = 0xFF400
HIGH_END   = 0xFF51F
BMP_START  = 0xE000
OFFSET     = HIGH_START - BMP_START  # 0x11400


# ==============================================================================
# Font remapping
# ==============================================================================
def remap_font():
    font = TTFont(str(SRC_FONT))

    # Remap all cmap subtables
    for table in font["cmap"].tables:
        new_cmap = {}
        for cp, glyph_name in table.cmap.items():
            if HIGH_START <= cp <= HIGH_END:
                new_cmap[cp - OFFSET] = glyph_name
            else:
                new_cmap[cp] = glyph_name
        table.cmap = new_cmap

    font.save(str(DST_FONT))
    print(f"Saved remapped font: {DST_FONT}")

    # Verify
    check = TTFont(str(DST_FONT))
    cmap = check.getBestCmap()
    high = [cp for cp in cmap if cp >= 0x10000]
    bmp_pua = [cp for cp in cmap if BMP_START <= cp < BMP_START + (HIGH_END - HIGH_START + 1)]
    print(f"  BMP PUA glyphs (E000-E11F): {len(bmp_pua)}")
    print(f"  Remaining high-plane glyphs: {len(high)}")
    if high:
        print(f"  WARNING: unexpected high-plane glyphs: {[f'U+{cp:X}' for cp in sorted(high)[:5]]}")


# ==============================================================================
# Dict remapping
# ==============================================================================
def remap_dict():
    with open(SRC_DICT, "r") as f:
        src = json.load(f)

    dst = {}
    remapped = 0
    for key, value in src.items():
        new_chars = []
        for ch in value:
            cp = ord(ch)
            if HIGH_START <= cp <= HIGH_END:
                new_chars.append(chr(cp - OFFSET))
                remapped += 1
            else:
                new_chars.append(ch)
        dst[key] = "".join(new_chars)

    with open(DST_DICT, "w") as f:
        json.dump(dst, f, indent=2, ensure_ascii=False)

    print(f"Saved remapped dict: {DST_DICT}")
    print(f"  Total keys: {len(dst)}")
    print(f"  Characters remapped: {remapped}")


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    print("Remapping VoynichUnicode from Supplementary PUA-B to BMP PUA...")
    print(f"  Shift: U+{HIGH_START:X}–U+{HIGH_END:X}  →  U+{BMP_START:X}–U+{BMP_START + HIGH_END - HIGH_START:X}")
    print()
    remap_font()
    print()
    remap_dict()
