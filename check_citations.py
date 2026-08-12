import os
import re

# Fixed per AUDIT_FINDINGS.md H3/H4 (2026-08-12): see check_manuscript.py for the same
# fix and rationale. NOTE (H4, still true after this fix): this script only checks that
# a \cite{} key has *some* matching .bib entry -- it has no DOI/venue/external-existence
# check, so it cannot detect a fabricated-but-key-matched citation (see AUDIT_FINDINGS.md
# C14, where 9 of 21 real citation keys had fabricated or substantially wrong content
# that this kind of check-by-key-matching alone would never have caught).
tex = open(os.path.join('Submission_JBHI', 'ieee_manuscript.tex'), encoding='utf-8').read()
bib = open(os.path.join('paper', 'IEEE PAPER', 'references.bib'), encoding='utf-8').read()

# Extract all cite keys from tex
cite_matches = re.findall(r'\\cite\{([^}]+)\}', tex)
tex_keys = set()
for m in cite_matches:
    for k in m.split(','):
        tex_keys.add(k.strip())

# Extract all entry keys from bib.
# Fixed 2026-08-12 (surfaced by the H3 path fix -- this bug was latent since the script
# always crashed on a missing file before reaching this line): `\w+` does not match
# hyphens, so every hyphenated key (e.g. `takalo-mattila_inter-patient_2018`,
# `zhao_mak-net_2025`) was truncated at its first hyphen and reported as missing even
# when present.
bib_keys = set(re.findall(r'@\w+\{([\w-]+)', bib))

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
