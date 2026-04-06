
import re
import os

tex_file = 'manuscript_complete.tex'
bib_file = 'references.bib'

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
