#!/bin/bash
# Compress acra_paper/figures/fig_teaser.pdf for submission (2026-10-04): images are downsampled to 600 ppi at print size and
# stored as JPEG (quality 92); text and vector shapes are unchanged. Run it LAST, after make_teaser.py has produced the final
# teaser. The full-resolution original is kept at acra_paper/figures/_hires/fig_teaser_hires.pdf (not used by main.tex).
# Usage: scripts/compress_teaser.sh [ppi]     (default 600)
set -e
cd "$(dirname "$0")/.."
PPI=${1:-600}
F=acra_paper/figures/fig_teaser.pdf
mkdir -p acra_paper/figures/_hires
cp "$F" acra_paper/figures/_hires/fig_teaser_hires.pdf
gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 \
   -dDownsampleColorImages=true -dColorImageResolution=$PPI -dColorImageDownsampleType=/Bicubic -dColorImageDownsampleThreshold=1.0 \
   -dAutoFilterColorImages=false -dColorImageFilter=/DCTEncode -dJPEGQ=92 -dEmbedAllFonts=true \
   -sOutputFile=/tmp/fig_teaser_small.pdf acra_paper/figures/_hires/fig_teaser_hires.pdf
mv /tmp/fig_teaser_small.pdf "$F"
ls -la acra_paper/figures/_hires/fig_teaser_hires.pdf "$F" | awk '{printf "%-55s %.2f MB\n",$9,$5/1e6}'
