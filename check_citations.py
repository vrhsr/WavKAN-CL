import re

tex = open('manuscript_complete.tex', encoding='utf-8').read()
bib = open('references.bib', encoding='utf-8').read()

# Extract all cite keys from tex
cite_matches = re.findall(r'\\cite\{([^}]+)\}', tex)
tex_keys = set()
for m in cite_matches:
    for k in m.split(','):
        tex_keys.add(k.strip())

# Extract all entry keys from bib
bib_keys = set(re.findall(r'@\w+\{(\w+)', bib))

missing = tex_keys - bib_keys
unused = bib_keys - tex_keys

print(f"Citations in manuscript: {len(tex_keys)}")
print(f"Entries in bib file: {len(bib_keys)}")
print(f"Missing from bib: {missing if missing else 'NONE'}")
print(f"Unused bib entries: {unused if unused else 'NONE'}")
print()
for k in sorted(tex_keys):
    status = "OK" if k in bib_keys else "MISSING"
    print(f"  [{status}] {k}")
