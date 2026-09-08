"""Re-run the AI-tell sweep of 07_Self_Audit_and_AI_Tell_Sweep.md on the final text.

Machine checks only, for the rules that are decidable by string and pattern:
H1 banned vocabulary, H2 banned constructions, H3 banned formatting, H4 banned
markup artefacts. Writes results/style_audit.json and prints a summary.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RESULTS, jdump

DOCS = [ROOT / "05_Manuscript.md", ROOT / "06_Supplementary.md"]

# ---- H1, one regular expression per banned form, inflections included
H1 = [
    r"\badditionally\b", r"\baligns? with\b", r"\balign(?:ed|ing) with\b",
    r"\bboasts?\b", r"\bbolster(?:ed|ing|s)?\b", r"\bcrucial(?:ly)?\b",
    r"\bdeep dive\b", r"\bdelv(?:e|es|ed|ing)\b", r"\bemphasis(?:e|es|ed|ing)\b",
    r"\benduring\b", r"\benhanc(?:e|es|ed|ing|ement|ements)\b",
    r"\bfoster(?:s|ed|ing)?\b", r"\bgarner(?:s|ed|ing)?\b",
    r"\bhighlight(?:s|ed|ing)?\b", r"\binterplay\b",
    r"\bintricat(?:e|ely)\b", r"\bintricac(?:y|ies)\b",
    r"\blandscape\b", r"\bkey (?:role|factor|feature|challenge|insight|component|advantage)\b",
    r"\bmeticulous(?:ly)?\b", r"\bpivotal\b", r"\brealm\b",
    r"\bshowcas(?:e|es|ed|ing)\b", r"\btapestry\b", r"\btestament\b",
    r"\bunderscor(?:e|es|ed|ing)\b", r"\bvaluable\b", r"\bvibrant\b",
    r"\bseamless(?:ly)?\b", r"\bgroundbreaking\b", r"\brenowned\b",
    r"\bcutting[- ]edge\b", r"\bharness(?:es|ed|ing)?\b",
    r"\bleverag(?:e|es|ed|ing)\b", r"\bunlock(?:s|ed|ing)?\b",
    r"\bprofound(?:ly)?\b", r"\bvital\b", r"\bholistic\b",
    r"\bparadigm shift\b", r"\bgame[- ]chang(?:er|ing)\b",
    r"\bever[- ]evolving\b", r"\brapidly evolving\b", r"\bin the realm of\b",
    r"\bin today's world\b", r"\bit is worth noting that\b",
    r"\bit is important to note that\b",
    # self-description ban
    r"\bnovel\b", r"\brevolutionary\b", r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bexcellent\b", r"\bremarkable\b", r"\boutperforms all\b",
    r"\bsuperior\b", r"\bunprecedented\b", r"\bfirst[- ]of[- ]its[- ]kind\b",
]
# "robust" is exempt only in the noise or distribution-shift sense
H1_CONDITIONAL = [(r"\brobust(?:ness|ly)?\b",
                   r"robustness (?:to noise|to distribution|was assessed)|noise robustness")]

H2 = {
    "negative parallelism, rather than": (r"\brather than\b", 1),
    "negative parallelism, not only but also": (r"\bnot only\b[^.]{0,80}\bbut also\b", 0),
    "copula avoidance": (r"\b(?:serves? as|stands? as|functions? as|operates? as|"
                         r"constitutes?|acts? as|boasts?|refers? to)\b", 0),
    "participial tail": (r",\s+(?:highlighting|ensuring|reflecting|contributing to|"
                         r"thereby enabling|ultimately improving|showcasing|underscoring)\b", 0),
    "vague attribution": (r"\b(?:experts argue|several studies show|research indicates|"
                          r"it is widely believed|some critics)\b", 0),
    "promotional register": (r"\b(?:plays a key role|sets the stage|diverse array|"
                             r"wide range of|commitment to|gateway to|exemplifies)\b", 0),
}

H3 = {
    "em dash": (r"—", 30),
    "en dash": (r"–", 0),
    "curly double quote": (r"[“”]", 0),
    "curly apostrophe": (r"[‘’]", 0),
    "emoji or symbol": (r"[\U0001F300-\U0001FAFF✀-➿←-⇿✓✔]", 0),
    "horizontal rule": (r"(?m)^\s*(?:---|\*\*\*|___)\s*$", 0),
}

H4 = {
    "unresolved RESULT placeholder": (r"\[\[RESULT", 0),
    "unresolved VERIFY placeholder": (r"\[\[VERIFY", 0),
    "TODO or FIXME": (r"\b(?:TODO|FIXME|XXX)\b", 0),
    "doubled space inside a sentence": (r"[a-z]  [a-z]", 0),
    "placeholder ellipsis": (r"\.\.\.\s*\]", 0),
    "generation artefact": (
        r"(?:contentReference|oaicite|oai_citation|attributableIndex|turn\d+search\d+"
        r"|\[cite:|span_\d+|grok_card|grok_render|attached_file|ppl-ai-file-upload"
        r"|as an AI language model|\[TBD\]|\[Insert|\[Your name\])", 0),
    "self-duplicated citation group": (r"\{([^{}]+?#\d+);\s*\1\}", 0),
}

# H6, conversational and filler behaviour, from the same audit
H6 = {
    "knowledge-cutoff disclaimer": (r"as of my (?:last )?(?:update|knowledge)", 0),
    "reader address": (r"\b(?:as we can see|let us now consider|you will notice"
                       r"|it is worth mentioning)\b", 0),
    "canned assurance": (r"\b(?:all citations preserved|fully compliant|tone improved"
                         r"|I hope this helps)\b", 0),
    "meta-commentary about writing": (r"\b(?:in this (?:section|paper) (?:we will|I will)"
                                      r"|the following section will)\b", 0),
}


def context(text, m, w=60):
    a, b = max(0, m.start() - w), min(len(text), m.end() + w)
    return re.sub(r"\s+", " ", text[a:b]).strip()


def main() -> int:
    report, failures = {}, 0
    for doc in DOCS:
        if not doc.exists():
            print(f"  skip {doc.name}, not built")
            continue
        text = doc.read_text(encoding="utf-8")
        words = len(re.findall(r"\b[\w'-]+\b", text))
        d = {"words": words, "H1": [], "H2": {}, "H3": {}, "H4": {}, "H6": {}}

        for pat in H1:
            for m in re.finditer(pat, text, re.I):
                d["H1"].append({"pattern": pat, "match": m.group(0),
                                "context": context(text, m)})
        for pat, exempt in H1_CONDITIONAL:
            for m in re.finditer(pat, text, re.I):
                c = context(text, m, 80)
                if not re.search(exempt, c, re.I):
                    d["H1"].append({"pattern": pat, "match": m.group(0), "context": c})

        for name, (pat, allow) in H2.items():
            hits = [context(text, m) for m in re.finditer(pat, text, re.I)]
            d["H2"][name] = {"count": len(hits), "allowance": allow,
                             "over": max(0, len(hits) - allow),
                             "examples": hits[:5]}
        for name, (pat, allow) in H3.items():
            n = len(re.findall(pat, text))
            d["H3"][name] = {"count": n, "allowance": allow, "over": max(0, n - allow)}
        for grp, table in (("H4", H4), ("H6", H6)):
            for name, (pat, allow) in table.items():
                hits = [context(text, m) for m in re.finditer(pat, text, re.I)]
                d[grp][name] = {"count": len(hits), "allowance": allow,
                                "over": max(0, len(hits) - allow), "examples": hits[:5]}
        d["balance"] = {"double_open": text.count("[["), "double_close": text.count("]]"),
                        "brace_open": text.count("{"), "brace_close": text.count("}")}
        if d["balance"]["double_open"] != d["balance"]["double_close"]:
            d["H4"]["bracket balance"] = {"count": 1, "allowance": 0, "over": 1,
                                          "examples": ["[[ and ]] counts differ"]}
        if d["balance"]["brace_open"] != d["balance"]["brace_close"]:
            d["H4"]["brace balance"] = {"count": 1, "allowance": 0, "over": 1,
                                        "examples": ["{ and } counts differ"]}

        over = (len(d["H1"]) + sum(v["over"] for v in d["H2"].values())
                + sum(v["over"] for v in d["H3"].values())
                + sum(v["over"] for v in d["H4"].values())
                + sum(v["over"] for v in d["H6"].values()))
        d["violations"] = over
        failures += over
        report[doc.name] = d

        print(f"{doc.name}: {words} words, {over} violations")
        if d["H1"]:
            print(f"  H1 banned vocabulary: {len(d['H1'])}")
            for h in d["H1"][:10]:
                print(f"    '{h['match']}' :: {h['context'][:110]}")
        for grp in ("H2", "H3", "H4", "H6"):
            for name, v in d[grp].items():
                if v["over"]:
                    print(f"  {grp} {name}: {v['count']} against an allowance of {v['allowance']}")
                    for e in v.get("examples", [])[:4]:
                        print(f"    {e[:110]}")

    jdump(report, RESULTS / "style_audit.json")
    print("STYLE_AUDIT_CLEAN" if failures == 0 else f"STYLE_AUDIT_VIOLATIONS {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
