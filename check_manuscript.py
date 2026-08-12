
import re
import os
import sys

# Fixed 2026-08-12 (surfaced by the H3 path fix -- this bug was latent since the script
# always crashed on a missing file before reaching any print statement): the ✅/❌ emoji
# below crash with UnicodeEncodeError on Windows consoles using the default cp1252
# encoding. Force UTF-8 stdout so the script can actually finish running.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Fixed per AUDIT_FINDINGS.md H3 (2026-08-12): this used to hardcode
# 'manuscript_complete.tex' (a stale draft, 11 days older than and materially different
# from the actual submission) and 'references.bib' at repo root, which doesn't exist
# anywhere -- running this script as originally written either checked the wrong
# document or crashed with FileNotFoundError before any check ran.
#
# `Submission_JBHI/ieee_manuscript.tex` is the confirmed-canonical live manuscript
# (CLAUDE.md section 5). It has no references.bib of its own (AUDIT_FINDINGS.md C7 --
# still open); `paper/IEEE PAPER/references.bib` is the more corrected of the two
# candidate bib files as of the C14 citation-integrity fixes, though it still has one
# unresolved entry (xiao_deep_2023 -- see the note in that file).
tex_file = os.path.join('Submission_JBHI', 'ieee_manuscript.tex')
bib_file = os.path.join('paper', 'IEEE PAPER', 'references.bib')

def check_consistency():
    with open(tex_file, 'r', encoding='utf-8') as f:
        tex_content = f.read()
    
    with open(bib_file, 'r', encoding='utf-8') as f:
        bib_content = f.read()

    # 1. Check Citations
    citations = set(re.findall(r'\\cite\{([^}]+)\}', tex_content))
    # Handle multiple citations like \cite{ref1, ref2}
    individual_citations = set()
    for c in citations:
        keys = c.split(',')
        for k in keys:
            individual_citations.add(k.strip())
            
    bib_entries = set(re.findall(r'@\w+\{([^,]+),', bib_content))
    
    missing_citations = individual_citations - bib_entries
    
    print("-" * 20)
    print(f"Found {len(individual_citations)} unique citations in text.")
    if missing_citations:
        print(f"❌ MISSING {len(missing_citations)} citations in .bib file:")
        for m in missing_citations:
            print(f"  - {m}")
    else:
        print("✅ All citations present in .bib file.")

    # 2. Check Figures/References
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex_content))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', tex_content))
    
    missing_refs = refs - labels
    
    print("-" * 20)
    print(f"Found {len(refs)} references to {len(labels)} defined labels.")
    if missing_refs:
        print(f"❌ MISSING {len(missing_refs)} labels for references:")
        for m in missing_refs:
            print(f"  - {m}")
    else:
        print("✅ All \\ref targets are defined.")

    # 3. Check Image Files
    graphics = set(re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', tex_content))
    print("-" * 20)
    print(f"Found {len(graphics)} included graphics.")
    for g in graphics:
        if os.path.exists(g):
            print(f"  - {g} [FOUND]")
        else:
            print(f"  - {g} [MISSING ❌]")

if __name__ == "__main__":
    check_consistency()
