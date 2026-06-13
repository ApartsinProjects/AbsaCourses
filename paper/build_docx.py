"""Regenerate ALL derived DOCX artifacts from the one HTML source via html2doc,
with content-canary verification (figures embedded, tables converted). House
style: Georgia serif, camera-ready-generic (1-col) + two-column (2-col).

Run from anywhere: python paper/build_docx.py
"""
import os
import re
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = r"E:\Projects\claude-skills\html2doc"
SRC = os.path.join(HERE, "course_absa_manuscript.html")
MATHML = os.path.join(HERE, "_course_absa_mathml.html")

ENV = dict(os.environ)
ENV["NODE_PATH"] = os.path.join(SKILL, "node_modules")
ENV["PYTHONUTF8"] = "1"
ENV["PYTHONIOENCODING"] = "utf-8"

N_IMG = len(re.findall(r"<img ", open(SRC, encoding="utf-8").read()))


def run(cmd):
    print("  $", " ".join(os.path.basename(c) if c.endswith((".py", ".js")) else c for c in cmd), flush=True)
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True, cwd=HERE)
    out = (r.stdout or "") + (r.stderr or "")
    for ln in out.splitlines():
        if re.search(r"Figures:|Tables:|embedded|referenced|converted|WARN|Error|Traceback|missing", ln, re.I):
            print("    |", ln, flush=True)
    if r.returncode != 0:
        print("    !! FAILED\n", out[-2000:], flush=True)
        sys.exit(1)
    return out


def media_count(docx):
    with zipfile.ZipFile(docx) as z:
        imgs = [n for n in z.namelist() if n.startswith("word/media/")]
    return len(imgs)


def build(profile, columns, out_name):
    print(f"\n=== building {out_name} (profile={profile}, columns={columns}) ===", flush=True)
    conv = os.path.join(HERE, f"_conv_{columns}col.docx")
    final = os.path.join(HERE, out_name)
    run([sys.executable, os.path.join(SKILL, "scripts", "convert_to_docx.py"),
         "--input", MATHML, "--output", conv, "--profile", profile,
         "--resource-path", HERE])
    run([sys.executable, os.path.join(SKILL, "scripts", "apply_academic_style.py"),
         "--input", conv, "--output", final, "--profile", profile,
         "--font-family", "Georgia"])
    n_media = media_count(final)
    try:
        from docx import Document
        n_tables = len(Document(final).tables)
    except Exception as e:
        n_tables = f"?({e})"
    print(f"  CANARY {out_name}: media(figures)={n_media}/{N_IMG}  tables={n_tables}", flush=True)
    if n_media < N_IMG:
        print(f"  !! FIGURE CANARY FAIL: {n_media} < {N_IMG}", flush=True)
        sys.exit(2)
    os.remove(conv)
    return n_media, n_tables


def main():
    print("Stage 1: KaTeX -> MathML", flush=True)
    run(["node", os.path.join(SKILL, "scripts", "katex_to_mathml.js"),
         "--input", SRC, "--output", MATHML])
    results = []
    results.append(("_1col", *build("camera-ready-generic", 1, "course_absa_manuscript_1col.docx")))
    results.append(("_2col", *build("two-column", 2, "course_absa_manuscript_2col.docx")))
    # base .docx mirrors the reliable single-column deliverable
    import shutil
    shutil.copyfile(os.path.join(HERE, "course_absa_manuscript_1col.docx"),
                    os.path.join(HERE, "course_absa_manuscript.docx"))
    print("  base course_absa_manuscript.docx <- 1col copy", flush=True)
    os.remove(MATHML)
    print("\n=== SUMMARY ===", flush=True)
    for name, media, tables in results:
        print(f"  {name}: figures={media}/{N_IMG} tables={tables}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
