"""Convert all manuscript PNG figures to PDF for faster LaTeX compilation."""
from PIL import Image
import os

figs = [
    'final_methodology_workflow_v2.png',
    'wavkan_micro_architecture_v2.png',
    'fig_seed_stability.png',
    'fig_convergence_curves.png',
    'final_confusion_matrix_main.png',
    'fig_rr_ablation_v2.png',
    'final_learned_wavelets.png',
]

for f in figs:
    pdf_name = f.replace('.png', '.pdf')
    img = Image.open(f)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert('RGB')
    img.save(pdf_name, 'PDF', resolution=300)
    size_kb = os.path.getsize(pdf_name) // 1024
    print(f"  {f} -> {pdf_name} ({size_kb} KB)")

print("\nDone! All 7 figures converted to PDF.")
