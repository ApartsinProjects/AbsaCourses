from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "outputs" / "figures"


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def synthetic_generation_svg() -> str:
    return r"""
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="820" viewBox="0 0 1600 820" role="img" aria-label="Synthetic educational review generation pipeline">
  <defs>
    <style>
      .bg { fill: #fcfbf7; }
      .frame { fill: #fffefe; stroke: #d8d3c9; stroke-width: 1.2; }
      .label { font: 700 30px Georgia, "Times New Roman", serif; fill: #16324a; }
      .body { font: 17px Georgia, "Times New Roman", serif; fill: #31414f; }
      .chip { font: 700 13px Arial, sans-serif; fill: #4d6277; letter-spacing: 0.08em; text-transform: uppercase; }
      .arrow { stroke: #6a7886; stroke-width: 4; fill: none; marker-end: url(#arrow); }
      .dashed { stroke-dasharray: 12 10; }
      .panelA { fill: #edf3f7; stroke: #d7e2ea; stroke-width: 1.4; }
      .panelB { fill: #f2f5ee; stroke: #dde4d5; stroke-width: 1.4; }
      .panelC { fill: #f8f1e9; stroke: #eadbcc; stroke-width: 1.4; }
      .panelD { fill: #edf4ee; stroke: #d5e3d7; stroke-width: 1.4; }
      .side { fill: #f7f7f3; stroke: #ddd7cc; stroke-width: 1.2; }
      .center { text-anchor: middle; }
    </style>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#6a7886"/>
    </marker>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="#d9d3ca" flood-opacity="0.28"/>
    </filter>
  </defs>
  <rect class="bg" width="1600" height="820"/>
  <rect x="34" y="34" width="1532" height="752" class="frame"/>

  <rect x="116" y="180" width="320" height="132" rx="24" class="panelA" filter="url(#shadow)"/>
  <text x="276" y="236" class="label center">Target labels</text>
  <text x="276" y="274" class="body center">1-3 aspect sentiments</text>

  <rect x="116" y="416" width="320" height="132" rx="24" class="panelB" filter="url(#shadow)"/>
  <text x="276" y="472" class="label center">Context state</text>
  <text x="276" y="510" class="body center">course, student, style</text>

  <rect x="624" y="262" width="348" height="204" rx="26" class="panelC" filter="url(#shadow)"/>
  <text x="798" y="330" class="label center">Stabilized prompt</text>
  <text x="798" y="372" class="body center">labels + context + realism rule</text>

  <rect x="1154" y="262" width="330" height="204" rx="26" class="panelD" filter="url(#shadow)"/>
  <text x="1319" y="330" class="label center">Benchmark record</text>
  <text x="1319" y="372" class="body center">review + labels + attributes</text>

  <path class="arrow" d="M436 246 C520 246, 540 304, 624 328"/>
  <path class="arrow" d="M436 482 C520 482, 540 424, 624 400"/>
  <path class="arrow" d="M972 364 L1154 364"/>

  <rect x="650" y="612" width="296" height="80" rx="18" class="side"/>
  <text x="798" y="644" class="chip center">Inter-cycle update</text>
  <text x="798" y="673" class="body center">realism loop revises prompt</text>
  <path class="arrow dashed" d="M1230 466 C1230 604, 930 604, 930 612"/>
  <path class="arrow dashed" d="M650 612 C650 566, 700 514, 734 466"/>

  <rect x="104" y="648" width="332" height="56" rx="16" class="side"/>
  <text x="270" y="682" class="body center">separate supervision and variation</text>

  <rect x="1148" y="648" width="342" height="56" rx="16" class="side"/>
  <text x="1319" y="682" class="body center">export one review-level training row</text>
</svg>
"""


def realism_validation_svg() -> str:
    return r"""
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="860" viewBox="0 0 1600 860" role="img" aria-label="Realism validation and prompt refinement pipeline">
  <defs>
    <style>
      .bg { fill: #fcfbf7; }
      .frame { fill: #fffefe; stroke: #d8d3c9; stroke-width: 1.2; }
      .label { font: 700 30px Georgia, "Times New Roman", serif; fill: #16324a; }
      .body { font: 17px Georgia, "Times New Roman", serif; fill: #31414f; }
      .chip { font: 700 13px Arial, sans-serif; fill: #4d6277; letter-spacing: 0.08em; text-transform: uppercase; }
      .arrow { stroke: #6a7886; stroke-width: 4.2; fill: none; marker-end: url(#arrow); }
      .dashed { stroke-dasharray: 12 10; }
      .panelA { fill: #eef3f6; stroke: #d7e2ea; stroke-width: 1.4; }
      .panelB { fill: #f2f5ee; stroke: #dde4d5; stroke-width: 1.4; }
      .panelC { fill: #f8f1e9; stroke: #eadbcc; stroke-width: 1.4; }
      .panelD { fill: #edf4ee; stroke: #d5e3d7; stroke-width: 1.4; }
      .side { fill: #f7f7f3; stroke: #ddd7cc; stroke-width: 1.2; }
      .center { text-anchor: middle; }
    </style>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#6a7886"/>
    </marker>
    <filter id="shadow3" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="#d9d3ca" flood-opacity="0.28"/>
    </filter>
  </defs>
  <rect class="bg" width="1600" height="860"/>
  <rect x="34" y="34" width="1532" height="792" class="frame"/>

  <rect x="174" y="334" width="310" height="140" rx="24" class="panelA" filter="url(#shadow3)"/>
  <text x="329" y="394" class="label center">Data pool</text>
  <text x="329" y="432" class="body center">30 synthetic + 30 real reviews</text>

  <rect x="642" y="144" width="316" height="148" rx="28" class="panelB" filter="url(#shadow3)"/>
  <text x="800" y="208" class="label center">Realism judge</text>
  <text x="800" y="246" class="body center">binary real-vs-synthetic decisions</text>

  <rect x="1116" y="334" width="290" height="140" rx="24" class="panelC" filter="url(#shadow3)"/>
  <text x="1261" y="394" class="label center">Evaluation</text>
  <text x="1261" y="432" class="body center">judge outputs and cue evidence</text>

  <rect x="642" y="580" width="316" height="148" rx="28" class="panelD" filter="url(#shadow3)"/>
  <text x="800" y="644" class="label center">Prompt revision</text>
  <text x="800" y="682" class="body center">editor updates the next cycle</text>

  <path class="arrow" d="M484 404 C560 404, 586 304, 642 236"/>
  <path class="arrow" d="M958 236 C1062 252, 1146 312, 1162 334"/>
  <path class="arrow" d="M1261 474 C1261 544, 1034 620, 958 642"/>
  <path class="arrow dashed" d="M642 642 C510 622, 364 548, 312 474"/>

  <rect x="1110" y="160" width="320" height="82" rx="18" class="side"/>
  <text x="1270" y="191" class="chip center">Cycle metrics</text>
  <text x="1270" y="221" class="body center">accuracy, confusion, entropy, cue tags</text>

  <rect x="188" y="610" width="278" height="58" rx="16" class="side"/>
  <text x="327" y="645" class="body center">one fixed prompt per cycle</text>
</svg>
"""


def benchmark_overview_svg() -> str:
    return r"""
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="860" viewBox="0 0 1600 860" role="img" aria-label="Benchmark overview for educational ABSA">
  <defs>
    <style>
      .bg { fill: #fcfbf7; }
      .frame { fill: #fffefe; stroke: #d8d3c9; stroke-width: 1.2; }
      .label { font: 700 30px Georgia, "Times New Roman", serif; fill: #16324a; }
      .body { font: 17px Georgia, "Times New Roman", serif; fill: #31414f; }
      .chip { font: 700 13px Arial, sans-serif; fill: #4d6277; letter-spacing: 0.08em; text-transform: uppercase; }
      .arrow { stroke: #6a7886; stroke-width: 4; fill: none; marker-end: url(#arrow); }
      .panelA { fill: #edf3f7; stroke: #d7e2ea; stroke-width: 1.3; }
      .panelB { fill: #f2f5ee; stroke: #dde4d5; stroke-width: 1.3; }
      .panelC { fill: #f8f1e9; stroke: #eadbcc; stroke-width: 1.3; }
      .panelD { fill: #edf4ee; stroke: #d5e3d7; stroke-width: 1.3; }
      .side { fill: #f7f7f3; stroke: #ddd7cc; stroke-width: 1.2; }
      .center { text-anchor: middle; }
    </style>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#6a7886"/>
    </marker>
    <filter id="shadow2" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="#d9d3ca" flood-opacity="0.28"/>
    </filter>
  </defs>
  <rect class="bg" width="1600" height="860"/>
  <rect x="34" y="34" width="1532" height="792" class="frame"/>

  <rect x="530" y="108" width="540" height="118" rx="24" class="panelA" filter="url(#shadow2)"/>
  <text x="800" y="163" class="label center">Source corpus</text>
  <text x="800" y="198" class="body center">10K reviews under one 20-aspect contract</text>

  <rect x="530" y="290" width="540" height="118" rx="24" class="panelB" filter="url(#shadow2)"/>
  <text x="800" y="345" class="label center">Data splits</text>
  <text x="800" y="380" class="body center">8k train / 1k validation / 1k test</text>

  <rect x="530" y="472" width="540" height="118" rx="24" class="panelC" filter="url(#shadow2)"/>
  <text x="800" y="527" class="label center">Unified evaluation</text>
  <text x="800" y="562" class="body center">local models and GPT methods share one scoring contract</text>

  <rect x="530" y="654" width="540" height="118" rx="24" class="panelD" filter="url(#shadow2)"/>
  <text x="800" y="709" class="label center">Held-out report</text>
  <text x="800" y="744" class="body center">test-set metrics reported once</text>

  <path class="arrow" d="M800 226 L800 290"/>
  <path class="arrow" d="M800 408 L800 472"/>
  <path class="arrow" d="M800 590 L800 654"/>

  <rect x="1154" y="320" width="278" height="82" rx="16" class="side"/>
  <text x="1293" y="351" class="chip center">Validation use</text>
  <text x="1293" y="381" class="body center">thresholds and model choice</text>

  <rect x="1154" y="522" width="278" height="82" rx="16" class="side"/>
  <text x="1293" y="553" class="chip center">Complementary checks</text>
  <text x="1293" y="583" class="body center">mapped real transfer and generator validation</text>
</svg>
"""


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write(FIG_DIR / "synthetic_data_generation_pipeline.svg", synthetic_generation_svg())
    write(FIG_DIR / "realism_validation_pipeline.svg", realism_validation_svg())
    write(FIG_DIR / "absa_analysis_pipeline.svg", benchmark_overview_svg())
    print("Updated conceptual SVGs.")


if __name__ == "__main__":
    main()
