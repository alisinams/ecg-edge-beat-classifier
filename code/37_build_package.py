"""Assemble the rest of the BSPC submission package.

Produces the title page, the highlights file, the competing-interest
declaration and the cover letter as separate .docx files, as the guide requires,
and stages the five main figures and the supplementary ones under their
submission names. `33_build_bspc.py` builds the manuscript itself.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Biomedical Signal Processing and Control"
FIGSRC = ROOT / "Folder Figures"

TITLE = ("A Deep Learning ECG Arrhythmia Classifier for Microcontroller-Class "
         "Wearables: Inter-Patient Evaluation, External Validation and Embedded "
         "Deployment Analysis")

AFFILIATION = ("Computer Department, Islamic Azad University, Zanjan Branch, "
               "Zanjan, Iran")

# The five main figures, in the order they are cited, mapped from the files the
# analysis produced. Figure 4 is promoted from the supplement because the
# protocol contrast is the paper's central claim.
MAIN_FIGURES = {
    "Figure_1": "Figure_1_pipeline",
    "Figure_2": "Figure_2_confusion_matrices",
    "Figure_3": "Figure_3_precision_recall",
    "Figure_4": "Figure_S4-4_protocol_slopegraph",
    "Figure_5": "Figure_7_size_accuracy",
}

# Everything the condensed manuscript no longer shows in the main text.
SUPPLEMENTARY_FIGURES = {
    "Figure_S1_reliability": "Figure_4_reliability",
    "Figure_S2_coverage": "Figure_5_coverage",
    "Figure_S3_noise": "Figure_6_noise",
    "Figure_S4_optimiser_convergence": "Figure_S4-1_optimiser_convergence",
    "Figure_S5_roc": "Figure_S4-2_roc",
    "Figure_S6_external_confusion": "Figure_S4-3_external_confusion",
}

HIGHLIGHTS = [
    "Single-lead beat classifier of 4,516 parameters, 4.5 kB in INT8",
    "Inter-patient macro-F1 0.338, held on two external databases, no retraining",
    "Patient-mixed evaluation adds 0.204 macro-F1, and 5.9 times more to some arms",
    "Distillation from an ECG foundation model changes macro-F1 by -0.005",
    "Computed 111.3 uJ per inference and 886 us median on an STM32F446",
]


def new_doc(pt: int = 11) -> Document:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(pt)
    style.paragraph_format.space_after = Pt(6)
    return doc


def para(doc, text: str, bold: bool = False, size: int | None = None,
         italic: bool = False, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    return p


def build_title_page() -> None:
    doc = new_doc()
    para(doc, TITLE, bold=True, size=14)
    doc.add_paragraph()

    para(doc, "Authors", bold=True)
    para(doc, "Mahdi Bazargani ᵃ,* (مهدی "
              "بازرگانی)")
    para(doc, "ORCID: [[AUTHOR ACTION: Mahdi Bazargani's ORCID iD. No public "
              "record could be found under this name at ORCID, Crossref or "
              "Europe PMC; register or supply the identifier before "
              "submission.]]", size=10, italic=True)
    para(doc, "Alisina Mousavi ᵃ (علیسینا "
              "موسوی)")
    para(doc, "ORCID: https://orcid.org/0009-0003-8480-7134", size=10)
    doc.add_paragraph()

    para(doc, "Affiliation", bold=True)
    para(doc, f"ᵃ {AFFILIATION}")
    doc.add_paragraph()

    para(doc, "Corresponding author", bold=True)
    para(doc, "Mahdi Bazargani")
    para(doc, f"{AFFILIATION}")
    para(doc, "Email: Mbzir@iauz.ac.ir")
    para(doc, "[[AUTHOR ACTION: full postal address, including street and "
              "postcode, and a telephone number.]]", size=10, italic=True)
    doc.add_paragraph()

    para(doc, "Co-author email", bold=True)
    para(doc, "Alisina Mousavi: alisina.mousavi@iau.ir")
    doc.add_paragraph()

    para(doc, "Article type", bold=True)
    para(doc, "Full paper")
    doc.add_paragraph()

    para(doc, "Keywords", bold=True)
    para(doc, "Electrocardiography; Arrhythmia; Knowledge distillation; "
              "Inter-patient evaluation; Quantisation; Wearables; Embedded systems")

    doc.save(OUT / "01_Title_Page.docx")


def build_highlights() -> None:
    doc = new_doc()
    para(doc, "Highlights", bold=True, size=13)
    para(doc, TITLE, italic=True, size=10)
    doc.add_paragraph()
    for line in HIGHLIGHTS:
        assert len(line) <= 85, f"{len(line)} chars, over the 85 limit: {line}"
        doc.add_paragraph(line, style="List Bullet")
    doc.save(OUT / "03_Highlights.docx")
    print("  highlight lengths:", [len(h) for h in HIGHLIGHTS])


def build_declaration() -> None:
    doc = new_doc()
    para(doc, "Declaration of competing interest", bold=True, size=13)
    doc.add_paragraph()
    para(doc, "Manuscript title: " + TITLE)
    doc.add_paragraph()
    para(doc, "The authors whose names are listed immediately below certify "
              "that they have no affiliations with or involvement in any "
              "organization or entity with any financial interest, or "
              "non-financial interest, in the subject matter or materials "
              "discussed in this manuscript.")
    doc.add_paragraph()
    para(doc, "Author names:", bold=True)
    para(doc, "Mahdi Bazargani")
    para(doc, "Alisina Mousavi")
    doc.add_paragraph()
    para(doc, "Funding: This research did not receive any specific grant from "
              "funding agencies in the public, commercial, or not-for-profit "
              "sectors.")
    doc.save(OUT / "04_Declaration_of_Competing_Interest.docx")


def build_cover_letter() -> None:
    doc = new_doc()
    para(doc, "[[AUTHOR ACTION: date]]", size=10)
    doc.add_paragraph()
    para(doc, "The Editor-in-Chief")
    para(doc, "Biomedical Signal Processing and Control")
    doc.add_paragraph()
    para(doc, "Dear Editor,")
    doc.add_paragraph()
    para(doc, "We submit for your consideration our manuscript, “" + TITLE
              + "”, as a Full paper.")
    doc.add_paragraph()
    para(doc, "Reported accuracy in ECG arrhythmia classification is governed "
              "by the evaluation protocol rather than by the classifier: of 122 "
              "systematically reviewed studies, only about 4 % partitioned "
              "between patients, assessed embedded feasibility and used AAMI "
              "class definitions at once. This manuscript reports a single-lead "
              "beat classifier built for that intersection. It is distilled from "
              "a public ECG foundation model, quantised to INT8, and occupies "
              "4.5 kB with 8.5 kB of peak static RAM; it reaches a macro-F1 of "
              "0.338 under strict inter-patient partitioning and holds that "
              "value on two external databases with no retraining.")
    doc.add_paragraph()
    para(doc, "Two results may interest your readership beyond the model "
              "itself. First, refitting all seven arms under a patient-mixed "
              "partition of the same beats raised every one of them, by 0.204 "
              "macro-F1 on average, but by amounts differing 5.9-fold, which "
              "quantifies how far two published intra-patient figures can be "
              "compared. Second, we report a null result with the same "
              "prominence as a positive one: distillation from the foundation "
              "model changed macro-F1 by -0.005, an interval containing zero.")
    doc.add_paragraph()
    para(doc, "The work also reimplements a published two-stage "
              "polynomial-sigmoid and Takagi-Sugeno fuzzy classifier and tests "
              "three things its source never tested, including its "
              "metaheuristic optimisers against metaphor-free controls at a "
              "matched evaluation budget. The manuscript falls within the "
              "journal's scope in wearable and embedded health monitoring, "
              "real-time systems and machine learning for biomedical signal "
              "analysis.")
    doc.add_paragraph()
    para(doc, "The work is original, has not been published previously and is "
              "not under consideration elsewhere. All authors have approved the "
              "manuscript and agree to its submission. We declare no competing "
              "interests. All four databases used are public and distributed "
              "through PhysioNet, and no new human data were collected, so "
              "ethics approval was not required.")
    doc.add_paragraph()
    para(doc, "[[AUTHOR ACTION: suggested reviewers, if you wish to propose "
              "any, and any reviewers to exclude.]]", size=10, italic=True)
    doc.add_paragraph()
    para(doc, "Yours sincerely,")
    doc.add_paragraph()
    para(doc, "Mahdi Bazargani")
    para(doc, "Corresponding author")
    para(doc, AFFILIATION)
    para(doc, "Mbzir@iauz.ac.ir")
    doc.save(OUT / "05_Cover_Letter.docx")


def stage_figures() -> None:
    """Copy figures under submission names, per the artwork instructions."""
    figs = OUT / "Figures"
    supp = OUT / "Supplementary"
    figs.mkdir(exist_ok=True)
    supp.mkdir(exist_ok=True)
    for dest_dir, mapping in ((figs, MAIN_FIGURES), (supp, SUPPLEMENTARY_FIGURES)):
        for dest, src in mapping.items():
            for ext in (".pdf", ".tif"):
                s = FIGSRC / (src + ext)
                assert s.exists(), f"missing source figure {s.name}"
                shutil.copy2(s, dest_dir / (dest + ext))
    print(f"  staged {len(MAIN_FIGURES)} main and "
          f"{len(SUPPLEMENTARY_FIGURES)} supplementary figures, PDF and TIFF")


def main() -> int:
    build_title_page()
    build_highlights()
    build_declaration()
    build_cover_letter()
    stage_figures()
    print("package files written to", OUT.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
