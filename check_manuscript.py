
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
# Repointed 2026-09-01: `Submission_JBHI/ieee_manuscript.tex` (the original draft this
# script used to check) was removed during a repo cleanup -- superseded by
# `ieee_manuscript_v2.tex` (CLAUDE.md section 5), now the sole live manuscript in that
# folder. `Submission_JBHI/references.bib` is the corrected, in-place-fixed bib file
# (all C14/C7 citation fixes applied directly to it) -- `paper/IEEE PAPER/references.bib`
# was a duplicate copy, also removed in the same cleanup.
# Repointed 2026-09-30 to the Array submission (Submission_JBHI/ is superseded).
tex_file = os.path.join('Submission_Array', 'manuscript.tex')
bib_file = os.path.join('Submission_Array', 'references.bib')

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
    # Fixed 2026-09-01: was `os.path.exists(g)`, checking relative to the process's cwd
    # (repo root) instead of the manuscript's own directory -- LaTeX resolves a bare
    # \includegraphics{name.pdf} relative to the .tex file, not the shell's cwd. This
    # silently "passed" for years only because repo-root duplicate copies of 4 of these
    # figures happened to exist under the same bare filenames (removed 2026-09-01 as
    # part of a repo cleanup of superseded old-draft manuscripts) -- once those
    # coincidental duplicates were gone, every real figure reported MISSING despite all
    # of them being present in Submission_JBHI/.
    tex_dir = os.path.dirname(tex_file)
    graphics = set(re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', tex_content))
    print("-" * 20)
    print(f"Found {len(graphics)} included graphics.")
    for g in graphics:
        if os.path.exists(os.path.join(tex_dir, g)):
            print(f"  - {g} [FOUND]")
        else:
            print(f"  - {g} [MISSING ❌]")

if __name__ == "__main__":
    check_consistency()
