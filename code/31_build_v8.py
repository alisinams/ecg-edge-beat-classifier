"""Build 12_Manuscript_v8.docx from 11_Manuscript_v7.docx.

Two groups of changes. The first was already drafted into v8.txt and adds the
two Bazargani references (#93, #94) to sections 5.5, 8.4 and 8.6 and to the
reference list. The second follows a full reading of Abbaszadeh and Bazargani
2024 (Heliyon 10(23) e40537, ref #12), the source architecture reimplemented as
arm B, and puts its reported protocol, configuration and numbers into the text
so that the reimplementation and the protocol argument are both auditable.

Every replacement is anchored on a unique substring of the v7 paragraph and
asserts that it matched exactly once, so a silent miss is impossible.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
W_PPR, W_R = W_NS + "pPr", W_NS + "r"

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "11_Manuscript_v7.docx"
DST = ROOT / "12_Manuscript_v8.docx"

# ---------------------------------------------------------------- edits
# (anchor, old fragment, new fragment). The anchor identifies the paragraph;
# the fragment is replaced inside it. Both must be unique.
EDITS: list[tuple[str, str, str]] = []

# --- group 1: carried over from the v8 draft ---------------------------

EDITS.append((
    "Federated averaging keeps waveforms on the device",
    "on-device learning has been combined with wearable ECG and smartphone "
    "photoplethysmography in an adaptive monitoring pipeline {Kartali, 2026 #56}. "
    "Sharing updates instead of data",
    "on-device learning has been combined with wearable ECG and smartphone "
    "photoplethysmography in an adaptive monitoring pipeline {Kartali, 2026 #56}. "
    "The same combination has been assembled outside the cardiac domain under the "
    "constraint that matters here: a framework pairing federated self-supervised "
    "pretraining with a lightweight student network was trained across serverless "
    "edge nodes for attack detection, adopted in that setting because the "
    "participating devices could hold neither the full model nor each other's data, "
    "which is the pair of constraints a wearable ECG deployment also works under "
    "{Bazargani, 2025 #93}. Sharing updates instead of data",
))

EDITS.append((
    "Waveform data are personal data.",
    "and a deployed system still has to state what it logs, for how long, and who "
    "can read it. The lifecycle obligations",
    "and a deployed system still has to state what it logs, for how long, and who "
    "can read it. The channel by which a device reports its decisions is a second "
    "exposure, and work on anomaly detection in Internet-of-Things networks has "
    "coupled federated aggregation to a blockchain-anchored record of model updates "
    "so that no participant surrenders raw data and every update stays auditable "
    "{Bazargani, 2026 #94}; a wearable that transmits nothing but class labels still "
    "needs that link secured and that record accountable. The lifecycle obligations",
))

EDITS.append((
    "Four experiments follow directly",
    "and a differential privacy budget stated explicitly {He, 2026 #52; Islam, 2025 "
    "#53; Elmir, 2025 #54; Bokhari, 2025 #55}.",
    "and a differential privacy budget stated explicitly {He, 2026 #52; Islam, 2025 "
    "#53; Elmir, 2025 #54; Bokhari, 2025 #55; Bazargani, 2025 #93}.",
))

# Front matter: v8 adds 791 words over v7 by the count in this repo, so the
# author's own stated figure moves by the same amount.
EDITS.append((
    "Word count.",
    "roughly 17,800 words",
    "roughly 18,600 words",
))

# --- group 2: from the full reading of ref #12 -------------------------

# 5.2, related work: give the source's own protocol and its own numbers.
EDITS.append((
    "Intra-patient evaluation inflates reported performance",
    "including our own earlier two-stage classifier {Abbaszadeh, 2024 #12}, which is "
    "reimplemented here as arm B and ablated as arm C.",
    "including our own earlier two-stage classifier {Abbaszadeh, 2024 #12}, which "
    "reported 98.58 per cent accuracy, 98.13 per cent sensitivity and 96.47 per cent "
    "specificity over seven rhythm types after splitting 3.9-second segments at "
    "random into 85 per cent for training and 15 per cent for testing, a partition "
    "that puts segments from one recording on both sides of the boundary; it is "
    "reimplemented here as arm B and ablated as arm C.",
))

# 5.6, metaheuristics: the budgets the source gave its optimisers.
EDITS.append((
    "The two optimisers used in the classifier we reimplement",
    "We adopt the abbreviation WHOA for the first, because WHO in a cardiovascular "
    "paper means the World Health Organization.",
    "We adopt the abbreviation WHOA for the first, because WHO in a cardiovascular "
    "paper means the World Health Organization. The source states the budgets it "
    "gave them: 100 candidates over 500 iterations for WHOA on the first-stage "
    "coefficients, 30 over 30 for GPC on the fuzzy membership parameters, and a Grey "
    "Wolf pass that reduced 22 extracted descriptors to 13 {Abbaszadeh, 2024 #12}. "
    "It reports a single run of each, with no spread over seeds, which is the "
    "practice section 7.6 puts to the test.",
))

# 6.5, arm B: what the reimplementation retargets, and where it deviates. This
# goes in as its own paragraph because the paragraph it follows carries inline
# OMML equations that must not be rewritten.
INSERTS: list[tuple[str, str]] = []

INSERTS.append((
    "are the cubic, quadratic and linear coefficients",
    "Four properties of that reimplementation should be stated, because the source "
    "solves a different problem with the same machinery {Abbaszadeh, 2024 #12}. It "
    "classifies seven rhythm types over 3.9-second segments resampled to 128 Hz, so "
    "neither its label set nor its unit of analysis is the one used here, and the "
    "architecture is retargeted to five AAMI superclasses over single beats rather "
    "than replicated. Its Grey Wolf stage reduced 22 extracted descriptors to 13; "
    "ours searches the 36-descriptor pool of section 6.3 and keeps that same count "
    "of 13. Its WHOA run was 100 candidates over 500 iterations, 50,000 evaluations, "
    "while arm B is fitted at the 15,000-evaluation budget of section 6.7 so that "
    "every optimiser in Table 7 is compared at one budget; the convergence curves in "
    "supplement S4 show what the remaining evaluations would and would not buy. And "
    "its fuzzy stage carries one input per class, so its input count follows its "
    "class count and not ours. No figure reported for arm B in this paper is a "
    "reproduction of a figure reported in the source, and none is offered as one.",
))

# 8.2, comparison with prior work: the concrete head-to-head.
EDITS.append((
    "Comparison numbers in this subsection are not like-for-like",
    "Under the inter-patient protocol adopted here, performance is not directly "
    "comparable to our earlier intra-patient evaluation {Abbaszadeh, 2024 #12}.",
    "Our own earlier evaluation is the sharpest case {Abbaszadeh, 2024 #12}. It "
    "reported 98.58 per cent accuracy over seven rhythm types on segments split at "
    "random, and arm B, which reimplements that architecture, reaches a macro-F1 of "
    "0.087 (0.067 to 0.122) on DS2 and 0.146 (0.115 to 0.177) on the patient-mixed "
    "partition of the same beats. The distance between those figures is not a failed "
    "replication and should not be read as one. Accuracy over seven rhythm classes "
    "and a macro-F1 averaged over five AAMI superclasses, one of which has 7 test "
    "beats, are different quantities; the unit of analysis is a 3.9-second segment "
    "there and a single beat here; and section 7.6 shows that the optimiser the "
    "source used leaves the stage-one objective 11 times higher than gradient "
    "descent reaches on the same 205 parameters. What the pair does establish is how "
    "little of a headline accuracy is about the classifier once the question is put "
    "in the form a wearable actually poses it, and that the earlier figure is a "
    "statement about that protocol rather than about a new wearer.",
))

# 8.2, edge group: the source's latency was not a microcontroller figure.
EDITS.append((
    "Against the inter-patient group the comparison is meaningful",
    "which is a statement about the substrate and not about the model "
    "{Mommen, 2026 #42}.",
    "which is a statement about the substrate and not about the model "
    "{Mommen, 2026 #42}. The same distinction applies to our own earlier system, "
    "whose 0.32 ms per classification was obtained on a Raspberry Pi, a Linux-class "
    "single-board computer with megabytes of memory rather than the kilobytes "
    "budgeted in Table 8, so it is not a microcontroller figure and Table 1 records "
    "it as the source reports it {Abbaszadeh, 2024 #12}.",
))

# 8.3, clinical interpretation: the source found the same class hardest.
EDITS.append((
    "The error profile matters more than its summary.",
    "and that weakness is tolerable for a device whose alarm is ventricular and "
    "unacceptable for one whose purpose is atrial fibrillation screening.",
    "and that weakness is tolerable for a device whose alarm is ventricular and "
    "unacceptable for one whose purpose is atrial fibrillation screening. That the "
    "supraventricular class is the hard one is not a property of this classifier "
    "alone: the architecture reimplemented as arm B named a supraventricular rhythm, "
    "the class for which it held the fewest examples, as its own worst, and it did "
    "so under a protocol far more forgiving than this one {Abbaszadeh, 2024 #12}.",
))

# ---------------------------------------------------------------- refs
NEW_REFS = [
    "Bazargani M., Tarkesh Esfahani S., Heidari B., Hosseinzadeh Shabestary R., "
    "Forghani M. (2025) AFedSLL-LDL: a framework based-on federated self-supervised "
    "learning and lightweight deep learning for attack detection in serverless edge "
    "computing. Cluster Computing, 28(15), 975. "
    "https://doi.org/10.1007/s10586-025-05673-7  [#93]",
    "Bazargani M., Fareghzadeh N., Najafi M.H. (2026) A Novel Intrusion Detection "
    "System to Enhance Internet of Things Security, Based on the Integration of "
    "Evolutionary Approach, Blockchain, and Federated Learning. Journal of Soft "
    "Computing and Information Technology, e251446, in press. "
    "https://doi.org/10.22034/jscit.2026.531689.2133  [#94]",
]
REF_INSERT_AFTER = "Barthels M., Verhofstadt E."


def set_text(par: Paragraph, text: str) -> None:
    """Replace paragraph text, keeping the first run's character formatting."""
    runs = par.runs
    if not runs:
        par.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


def main() -> int:
    doc = Document(SRC)
    paras = doc.paragraphs

    for anchor, old, new in EDITS:
        hits = [p for p in paras if anchor in p.text]
        assert len(hits) == 1, f"anchor {anchor!r} matched {len(hits)} paragraphs"
        par = hits[0]
        assert par.text.count(old) == 1, (
            f"fragment matched {par.text.count(old)} times under {anchor!r}")
        set_text(par, par.text.replace(old, new))
        print(f"edited  : {anchor[:56]}")

    for anchor, text in INSERTS:
        hits = [p for p in paras if anchor in p.text]
        assert len(hits) == 1, f"anchor {anchor!r} matched {len(hits)} paragraphs"
        after = hits[0]
        element = copy.deepcopy(after._element)
        after._element.addnext(element)
        new_par = Paragraph(element, after._parent)
        # The clone inherits the source paragraph's inline equations. Keep the
        # paragraph properties and the first plain run; drop everything else.
        kept_run = False
        for child in list(element):
            if child.tag == W_PPR:
                continue
            if child.tag == W_R and not kept_run:
                kept_run = True
                continue
            element.remove(child)
        assert kept_run, f"no plain run to clone under {anchor!r}"
        set_text(new_par, text)
        print(f"inserted: after {anchor[:46]}")

    anchors = [p for p in paras if p.text.strip().startswith(REF_INSERT_AFTER)]
    assert len(anchors) == 1, f"reference anchor matched {len(anchors)} paragraphs"
    cursor = anchors[0]
    for ref in NEW_REFS:
        element = copy.deepcopy(cursor._element)
        cursor._element.addnext(element)
        cursor = Paragraph(element, cursor._parent)
        set_text(cursor, ref)
        print(f"ref add : {ref[:56]}")

    doc.save(DST)
    print(f"\nwrote {DST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
