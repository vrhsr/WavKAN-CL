"""
build_highlighted_version.py -- the "highlighted version showing all changes" requested by Array
for the revision of ARRAY-D-26-02633.

Added 2026-10-01. Compares the originally submitted manuscript with the current
Submission_Array/manuscript.tex using latexdiff, and builds Submission_Array/manuscript_highlighted.pdf.

- Original: elsarticle_manuscript.tex as it stood before commit 66683a7 (the Elsevier-format file
  with the reviewed title, numbers and author list). Confirm against the PDF in Editorial Manager.
- Three citation keys were renamed after submission, for the same papers. The script maps them in
  the original so the removed text still resolves: bozorgasl_liu_2024 -> bozorgasl_wavkan_2024,
  taleban_explainable_2026 -> taleban_explainable_2025, moody_mit-bih_1992 -> moody_impact_2001.
- latexdiff options: colour markup (CFONT, no strike-out, which breaks around headings), section
  headings unmarked, and tables treated as whole blocks, because word-level merging of two
  unrelated tables breaks their structure.
- latexdiff can split a command across a change boundary and leave a brace unclosed. Deleted
  segments, which come from the fixed original, are closed before \\DIFdelend when their braces
  do not balance; added text is left alone, and any remaining LaTeX error stops the build.
- Removed text can contain \\ref to tables and figures that no longer exist; these print as ??,
  and a note on the first page says so.

Usage (from the repository root; needs git, perl, pdflatex and bibtex):
    python src/build_highlighted_version.py
"""
import glob
import os
import re
import shutil
import subprocess
import sys

ORIGINAL_REF = "66683a7^:elsarticle_manuscript.tex"
SUB = "Submission_Array"
BUILD = os.path.join("build", "highlighted")
KEY_MAP = {"bozorgasl_liu_2024": "bozorgasl_wavkan_2024",
           "taleban_explainable_2026": "taleban_explainable_2025",
           "moody_mit-bih_1992": "moody_impact_2001"}
NOTE = (r"\noindent\fbox{\parbox{\dimexpr\textwidth-2\fboxsep\relax}{\small\textbf{Highlighted version.} "
        r"Text added in the revision is shown in blue; text of the original submission that was removed is "
        r"shown in red, in small type. The manuscript was substantially rewritten, so most of the text is "
        r"marked. Cross-references inside removed text point to tables and figures of the original "
        r"submission that no longer exist and print as ??. The clean version is the one to read.}}"
        "\n\\medskip\n")


def find_latexdiff():
    for cand in ("latexdiff-so", "latexdiff-so.pl"):
        p = shutil.which(cand)
        if p and not p.lower().endswith(".exe"):
            return ["perl", p]
    for base in (os.path.expandvars(r"%LOCALAPPDATA%\Programs\MiKTeX\scripts\latexdiff"),
                 r"C:\Program Files\MiKTeX\scripts\latexdiff", "/usr/share/texlive/texmf-dist/scripts/latexdiff",
                 "/usr/local/texlive/texmf-dist/scripts/latexdiff"):
        p = os.path.join(base, "latexdiff-so")
        if os.path.isfile(p):
            return ["perl", p]
    sys.exit("latexdiff-so not found (it bundles Algorithm::Diff, which plain latexdiff needs separately)")


def close_unbalanced(tex):
    """Append a closing brace to \\DIFdel{...}\\DIFdelend / \\DIFadd{...}\\DIFaddend segments left open."""
    fixed = 0
    for kind in ("del",):  # deleted text comes from the fixed original; added text is left alone
        pat = re.compile(r"\\DIF%s\{(.*?)(\\DIF%send)" % (kind, kind), flags=re.S)
        out, pos = [], 0
        for m in pat.finditer(tex):
            seg = m.group(1)
            depth = seg.count("{") - seg.count("}")
            out.append(tex[pos:m.start(2)])
            if depth == 0:          # the segment's own closing brace is missing
                out.append("}")
                fixed += 1
            pos = m.start(2)
        out.append(tex[pos:])
        tex = "".join(out)
    return tex, fixed


def main():
    os.makedirs(BUILD, exist_ok=True)
    original = subprocess.run(["git", "show", ORIGINAL_REF], capture_output=True, text=True, encoding="utf-8",
                              check=True).stdout
    for old, new in KEY_MAP.items():
        original = original.replace(old, new)
    with open(os.path.join(BUILD, "original.tex"), "w", encoding="utf-8") as f:
        f.write(original)
    for fp in [os.path.join(SUB, "manuscript.tex"), os.path.join(SUB, "references.bib")] + glob.glob(os.path.join(SUB, "*.pdf")):
        if not os.path.basename(fp).startswith(("manuscript", "response", "graphical")) or fp.endswith(".tex"):
            shutil.copy(fp, BUILD)
    shutil.copy(os.path.join(SUB, "references.bib"), BUILD)

    cmd = find_latexdiff() + [
        "--type=CFONT", "--graphics-markup=none", "--math-markup=whole",
        "--exclude-textcmd=section,subsection,subsubsection,paragraph",
        r"--config=PICTUREENV=(?:picture|DIFnomarkup|tabular|tabularx|longtable|algorithm|algorithmic)[\w\d*@]*",
        "original.tex", "manuscript.tex"]
    res = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL)
    if res.returncode != 0 or not res.stdout.strip():
        sys.exit("latexdiff failed:\n" + res.stderr[-2000:])
    tex, fixed = close_unbalanced(res.stdout)
    tex = tex.replace("\\begin{document}", "\\begin{document}\n" + NOTE, 1)
    with open(os.path.join(BUILD, "manuscript_highlighted.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

    def run(*a):
        subprocess.run(list(a), cwd=BUILD, capture_output=True, stdin=subprocess.DEVNULL)
    for ext in (".aux", ".bbl"):
        p = os.path.join(BUILD, "manuscript_highlighted" + ext)
        if os.path.exists(p):
            os.remove(p)
    run("pdflatex", "-interaction=nonstopmode", "manuscript_highlighted.tex")
    run("bibtex", "manuscript_highlighted")
    run("pdflatex", "-interaction=nonstopmode", "manuscript_highlighted.tex")
    run("pdflatex", "-interaction=nonstopmode", "manuscript_highlighted.tex")
    log = open(os.path.join(BUILD, "manuscript_highlighted.log"), encoding="latin-1").read()
    errors = len(re.findall(r"^!", log, flags=re.M))
    pages = re.search(r"Output written on .*?\((\d+) pages", log)
    print(f"unbalanced segments closed: {fixed}; LaTeX errors: {errors}; pages: {pages.group(1) if pages else '?'}")
    if errors or not pages:
        sys.exit("highlighted build failed; see " + os.path.join(BUILD, "manuscript_highlighted.log"))
    shutil.copy(os.path.join(BUILD, "manuscript_highlighted.pdf"), os.path.join(SUB, "manuscript_highlighted.pdf"))
    print("Saved -> " + os.path.join(SUB, "manuscript_highlighted.pdf"))


if __name__ == "__main__":
    main()
