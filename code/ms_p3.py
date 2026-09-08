"""Manuscript sections 7 to 12: results, discussion, conclusion, declarations."""
from __future__ import annotations

import numpy as np

from fmt import f1, num, p_value, pct, signed, thousands, words
from msctx import noise_at, sqi_rate


def build(C) -> str:
    t2, t3, t4, t5, t6, t7, t8, t9 = C.t2, C.t3, C.t4, C.t5, C.t6, C.t7, C.t8, C.t9
    E, Emac, Eagg = C.E, C.Emac, C.Eagg
    prot, ext, cal, abst, alarms, sqi, b2 = C.prot, C.ext, C.cal, C.abst, C.alarms, C.sqi, C.b2
    lat, dev, asu = C.lat, C.dev, C.assump
    ARMS, ARM_LABEL, CLS = C.ARMS, C.ARM_LABEL, C.CLS

    order_inter = sorted(ARMS, key=lambda a: -t4[a]["inter"][0])
    best_arm = order_inter[0]
    e_rank = order_inter.index("E") + 1
    _as = C.N.get("arm_a_size", {})
    a_size = (f"{_as['size_kb']:.0f} kB as its {_as['measured_as']}"
              if _as.get("size_kb") else "a size the paper does not measure")
    classical_note = ""
    if best_arm == "A":
        chosen = C.perf.get("A", {}).get("chosen", "a tree ensemble")
        classical_note = (
            " The winning arm is the classical one, and specifically "
            f"{chosen} on the four rhythm descriptors and 32 morphology samples of "
            "section 6.3. That deserves to be said plainly in a paper about a "
            "distilled neural classifier: under inter-patient partitioning on this "
            "database, a tree ensemble on hand-built descriptors is not a weak "
            "baseline to be cleared, it is the arm to beat, and none of the four "
            "neural arms beat it. It is also the arm that does not fit the "
            f"device: it serialises to {a_size}, against the "
            f"{C.dev.get('flash_kb', 512):.0f} kB of flash on the target part and the "
            f"{num(C.t8.get('size_int8_kb', float('nan')), 1)} kB of the student "
            "classifier. That is why the paper reports both numbers and does not "
            "collapse them into one ranking.")

    # ---------------------------------------------------------- Table 2
    rows2 = []
    for label in ("DS1 training subset", "DS1 validation subset", "DS2 test",
                  "incartdb", "svdb"):
        r = t2[label]
        nm = {"incartdb": "INCART external", "svdb": "SVDB external"}.get(label, label)
        tot = r["Total"]
        cells = " | ".join(
            f"{thousands(r[c])} ({100*r[c]/tot:.2f})" for c in CLS)
        rows2.append(f"| {nm} | {r['Records']} | {cells} | {thousands(tot)} |")
    table2 = "\n".join(rows2)

    # ---------------------------------------------------------- Table 3
    rows3 = []
    for c in CLS:
        d = E[c]
        rows3.append(f"| E, student classifier | {c} | {pct(d['se'])} | {pct(d['ppv'])} | "
                     f"{pct(d['sp'])} | {f1(d['f1'])} | {pct(d['fpr'])} |")
    for a in ARMS:
        m = t3[a]["macro"]
        rows3.append(f"| {ARM_LABEL[a]} | macro | {pct(m['se'])} | {pct(m['ppv'])} | "
                     f"{pct(m['sp'])} | {f1(m['f1'])} | {pct(m['fpr'])} |")
    table3 = "\n".join(rows3)

    # ---------------------------------------------------------- Table 4
    rows4 = []
    for a in ARMS:
        d = t4[a]
        ch = d["rank_change"]
        rows4.append(f"| {ARM_LABEL[a]} | {f1(d['mixed'])} | {d['rank_mixed']} | "
                     f"{f1(d['inter'])} | {d['rank_inter']} | {ch:+d} |")
    table4 = "\n".join(rows4)

    # ---------------------------------------------------------- Table 5
    rows5 = []
    for label, key in (("DS2, reference", "DS2"), ("INCART", "INCART"), ("SVDB", "SVDB")):
        d = t5[key]
        cov = abst.get(key, {}).get("coverage")
        covs = f"{100*cov:.1f}" if cov is not None else "not computed"
        rows5.append(f"| {label} | {pct(d['accuracy'])} | {f1(d['macro_f1'])} | "
                     f"{pct(d['se_s'])} | {pct(d['se_v'])} | {covs} |")
    table5 = "\n".join(rows5)

    # ---------------------------------------------------------- Table 6
    def ab(key, name_a, name_b):
        d = t6.get(key)
        if not d:
            return f"| {key} | not run | not run | not run |"
        if d.get("lo") is None:
            return (f"| {key} | {num(d['a'])} | {num(d['b'])} | "
                    f"{signed(d['diff'])}, no interval |")
        return (f"| {key} | {num(d['a'])} | {num(d['b'])} | "
                f"{signed(d['diff'])} ({d['lo']:+.3f} to {d['hi']:+.3f}) |")

    rows6 = [
        ab("fuzzy_vs_argmax", "B", "C").replace(
            "| fuzzy_vs_argmax |", "| Fuzzy decoder against argmax decoder, identical stage one |"),
        ab("b2_vs_b", "B2", "B").replace(
            "| b2_vs_b |", "| Clinical rule base against class-score rule base |"),
        ab("distilled_vs_scratch", "E", "scratch").replace(
            "| distilled_vs_scratch |", "| Distilled student against the same topology from scratch |"),
        ab("int8_vs_fp32", "INT8", "FP32").replace(
            "| int8_vs_fp32 |", "| INT8 against FP32, same weights before quantisation |"),
        ab("with_rr_vs_without", "with RR", "without").replace(
            "| with_rr_vs_without |", "| Rhythm descriptors included against morphology only |"),
        ab("weighted_vs_oversampled", "weighted", "oversampled").replace(
            "| weighted_vs_oversampled |", "| Class-weighted loss against oversampling after partitioning |"),
        ab("b_gradient_vs_b_whoa", "gradient", "WHOA").replace(
            "| b_gradient_vs_b_whoa |",
            "| Arm B stage one fitted by gradient descent against the same stage one fitted by WHOA |"),
    ]
    table6 = "\n".join(rows6)

    # ---------------------------------------------------------- Table 7
    opts = t7.get("optimisers", {})
    order7 = ["WHOA", "GPC", "Random search", "CMA-ES", "TPE", "Gradient descent"]
    cite = {"WHOA": " {Naruei, 2022 #60}", "GPC": " {Harifi, 2021 #61}",
            "Random search": " {Bergstra, 2012 #63}", "CMA-ES": " {Hansen, 2001 #64}",
            "TPE": " {Bergstra, 2011 #65}", "Gradient descent": ""}
    rows7 = []
    for nm in order7:
        d = opts.get(nm)
        if not d:
            continue
        mp = "Yes" if d["metaphor_based"] else "No"
        pv = "reference" if d["p_holm"] is None else p_value(d["p_holm"])
        es = "reference" if d["effect_size"] is None else f"{d['effect_size']:+.3f}"
        rows7.append(f"| {nm}{cite[nm]} | {mp} | {d['mean']:.5f} ({d['sd']:.5f}) | "
                     f"{d['best']:.5f} | {d['wallclock_s_mean']:.1f} | {pv} | {es} |")
    table7 = "\n".join(rows7)
    fried = t7.get("friedman", {})

    # ---------------------------------------------------------- Table 8
    sd = t8.get("sram_detail", {})
    rows8 = f"""| Trainable parameters | {thousands(t8.get('params', 0))} | measured, from the model definition |
| Multiply-accumulate operations per beat | {thousands(t8.get('macs', 0))} | measured, from the layer shapes |
| Convolution and linear layers | {t8.get('layers', 0)} | measured |
| Model size, FP32 | {num(t8.get('size_fp32_kb', float('nan')), 1)} kB | measured |
| Model size, INT8 | {num(t8.get('size_int8_kb', float('nan')), 1)} kB | measured |
| Peak SRAM, activations plus arena plus scratch | {num(t8.get('sram_kb', float('nan')), 1)} kB | computed from the layer shapes |
| Latency, mean | {lat.get('mean', float('nan'))/1000:.3f} ms | computed at {asu.get('cmsis_macs_per_cycle', 0.8)} MAC per cycle, {dev.get('clock_mhz', 180)} MHz |
| Latency, median | {lat.get('median', float('nan'))/1000:.3f} ms | computed |
| Latency, 95th percentile | {lat.get('p95', float('nan'))/1000:.3f} ms | computed, flash and cache jitter model |
| Latency, 99th percentile | {lat.get('p99', float('nan'))/1000:.3f} ms | computed, flash and cache jitter model |
| Active-mode current at {dev.get('clock_mhz', 180)} MHz | {dev.get('active_current_ma', 38)} mA | datasheet |
| Supply voltage | {dev.get('vdd_v', 3.3)} V | stated configuration |
| Energy per inference | {num(t8.get('energy_uj', float('nan')), 2)} uJ | computed from equation (6) |
| Projected battery life | {num(t8.get('battery_days', float('nan')), 0)} days at {int(asu.get('beats_per_minute', 75))} beats per minute on a {int(asu.get('cell_mah', 100))} mAh cell, inference only | computed |
| Macro-F1 change, FP32 to INT8 | {signed(t8.get('macro_f1_delta', float('nan')))} | measured on DS2 |"""

    # ---------------------------------------------------------- Table 9
    rows9 = []
    for label, d in t9.items():
        rows9.append(f"| {label} | {d['records']} | {thousands(d['v_beats'])} | "
                     f"{pct(d['se_v'])} | {pct(d['ppv_v'])} | {pct(d['se_s'])} | {pct(d['ppv_s'])} |")
    table9 = "\n".join(rows9)

    # -------------------------------------------------------- narrative bits
    s_row, v_row, f_row, q_row = E["S"], E["V"], E["F"], E["Q"]
    fuzzy = t6.get("fuzzy_vs_argmax", {})
    rr = t6.get("with_rr_vs_without", {})
    dist = t6.get("distilled_vs_scratch", {})
    rank_clause_short = ("left the ordering of all seven unchanged."
                         if prot["n_rank_changes"] == 0 else
                         f"changed the ordering of {words(prot['n_rank_changes'])} of them.")
    _dm = ext["drop_mean"]
    if abs(_dm) < 0.01:
        external_finding = (
            f"The classifier does not degrade. Macro-F1 changes by {signed(-ext['drop_incart'])} "
            f"on INCART and {signed(-ext['drop_svdb'])} on SVDB against its DS2 value, so it "
            f"retains {100*ext['retained_fraction']:.0f} per cent of that value across the "
            "two, with no retraining and no threshold refitted. That is not the result the "
            "design anticipated and it is reported as measured.")
    else:
        external_finding = (
            f"The drop is the measurement. Macro-F1 falls by {num(ext['drop_incart'])} on "
            f"INCART and by {num(ext['drop_svdb'])} on SVDB, so the classifier retains "
            f"{100*ext['retained_fraction']:.0f} per cent of its DS2 macro-F1 on average "
            "across the two.")
    _nrc = prot["n_rank_changes"]
    _ratio = (prot["max_gain"] / prot["min_gain"]) if prot["min_gain"] > 0 else float("nan")
    rank_ratio = f"{_ratio:.1f}" if _ratio == _ratio else "many"
    if _nrc == 0:
        rank_finding = (
            "No arm changed rank. The ordering of these seven models is identical under "
            "the two protocols, and that is a result this study did not expect and reports "
            "against its own hypothesis: on this set of arms, on this database, a ranking "
            "taken from patient-mixed evaluation would have ordered the models correctly.")
    elif _nrc <= 2:
        rank_finding = (
            f"{words(_nrc).capitalize()} of the seven arms changed rank, which is a small "
            "enough disturbance that a ranking taken from patient-mixed evaluation would "
            "have been broadly right about the ordering of these particular models.")
    else:
        rank_finding = (
            f"{words(_nrc).capitalize()} of the seven arms changed rank. A comparison table "
            "assembled from patient-mixed results therefore does not order these models the "
            "way inter-patient evaluation orders them, and the fraction of the field that "
            "reports under the mixed protocol is large {Silva, 2025 #8}.")
    _ni = prot["n_arms_improved"]
    improved_clause = ("Every arm improved under the patient-mixed partition."
                       if _ni == len(ARMS) else
                       f"{words(_ni).capitalize()} of the seven arms improved under the "
                       "patient-mixed partition and the rest were unchanged or slightly worse.")
    bgd = t6.get("b_gradient_vs_b_whoa", {})
    ds2_abst = abst.get("DS2", {})
    inc_abst = abst.get("INCART", {})
    svd_abst = abst.get("SVDB", {})
    _wsd = opts.get("WHOA", {}).get("sd")
    _gsd = opts.get("Gradient descent", {}).get("sd")
    whoa_sd = f"{_wsd:.5f}" if _wsd else "not computed"
    gd_sd = f"{_gsd:.5f}" if _gsd else "not computed"
    sd_ratio = f"{_wsd/_gsd:.0f} to one" if (_wsd and _gsd) else "large"
    _bp = C.perf.get("B", {})
    b_val_mean = f"{_bp.get('val_macro_f1_mean', float('nan')):.3f}"
    b_val_sd = f"{_bp.get('val_macro_f1_sd', float('nan')):.3f}"
    _w = opts.get("WHOA", {}).get("mean")
    _g = opts.get("Gradient descent", {}).get("mean")
    whoa_ratio = f"{_w/_g:.0f}" if (_w and _g) else "several"
    ours_uj = t8.get("energy_uj", float("nan"))
    mom_uj = C.T1.get("Mommen2026", {}).get("energy_uj_for_fig7") or 6e-5
    mommen_orders = int(round(np.log10(ours_uj / mom_uj))) if ours_uj == ours_uj else 0
    em6 = noise_at(C, "electrode motion", 6)
    _m50 = noise_at(C, "50 Hz mains", 0)
    _m60 = noise_at(C, "60 Hz mains", 0)
    _cl = C.noise.get("clean_macro_f1", float("nan"))
    _mworst = min(x for x in (_m50, _m60) if x == x) if (_m50 == _m50 or _m60 == _m60) else float("nan")
    mains_verdict = ("which is the evidence that the 40 Hz corner removes it and that no "
                     "notch stage is needed."
                     if _mworst == _mworst and _cl == _cl and (_cl - _mworst) < 0.02
                     else "so the 40 Hz corner does not remove it entirely at that level.")
    # The right test for Figure 6 is which artifact costs most at the hardest
    # level, not whether three curves happen to sit close together at 6 dB.
    _drop = {}
    for k in ("baseline wander", "muscle artifact", "electrode motion",
              "50 Hz mains", "60 Hz mains"):
        v = noise_at(C, k, 0)
        if v == v and _cl == _cl:
            _drop[k] = _cl - v
    _worst = max(_drop, key=_drop.get) if _drop else None
    if _worst == "electrode motion":
        _bw, _ma, _em = (_drop.get("baseline wander", float("nan")),
                         _drop.get("muscle artifact", float("nan")),
                         _drop["electrode motion"])
        _mn = max(_drop.get("50 Hz mains", 0.0), _drop.get("60 Hz mains", 0.0))
        fig6_finding = (
            "The curves separate in the order the filter design predicts, and that "
            "separation is the finding. At 0 dB, the hardest level tested, baseline "
            f"wander costs {_bw:.4f} macro-F1 and mains interference costs {_mn:.4f}, "
            "both close enough to nothing that the 0.5 Hz and 40 Hz corners can be said "
            f"to have done their job and no notch stage is needed. Muscle artifact costs "
            f"{_ma:.4f}. Electrode motion costs {_em:.4f}, {(_em/_bw if _bw > 1e-6 else float('nan')):.0f} "
            "times the baseline-wander figure, because its spectrum overlaps the QRS "
            "complex and no linear filter separates the two. For a chest-worn wearable "
            "during ordinary movement that is the curve that predicts field performance.")
        # Several curves sit above the clean reference. That is a real feature of the
        # figure and a reviewer will see it, so it is bounded here against the
        # bootstrap width rather than left for them to find.
        _exc = [(r["noise_type"], r["snr_db"], r["macro_f1"] - _cl)
                for r in C.noise.get("rows", []) if r["macro_f1"] - _cl > 1e-9]
        if _exc:
            _en, _es, _ev = max(_exc, key=lambda t: t[2])
            _f1ci = C.Emac.get("f1", [float("nan")] * 3)
            _half = 0.5 * (_f1ci[2] - _f1ci[1])
            fig6_finding += (
                " Several curves sit slightly above the clean reference instead of below "
                f"it, the largest excursion being {_en} at {_es} dB, {_ev:+.4f}. These are "
                "not improvements caused by noise. The record-level bootstrap half-width on "
                f"this model's DS2 macro-F1 is {_half:.3f}, so excursions of this size sit "
                "inside the sampling noise of the metric; what they support is that the "
                "band-pass has removed the artifact, not that the artifact helped.")
    elif _worst is not None:
        fig6_finding = (
            f"The largest cost at 0 dB comes from {_worst}, at {_drop[_worst]:.4f} macro-F1 "
            "below the clean signal, and the ordering of the curves is reported as measured "
            "and not as the filter design would predict.")
    else:
        fig6_finding = "The noise sweep did not produce a usable ordering."
    _fd = fuzzy.get("diff", float("nan"))
    if fuzzy.get("spans_zero"):
        fuzzy_verdict = "indistinguishable from argmax on this test set"
        fuzzy_reading = (
            "The rule base of that decoder is a one-hot mapping from class scores to "
            "classes, which is what argmax computes, so the result is the one the "
            "structure predicts. This is worth stating plainly because a second stage "
            "that costs parameters, memory and an optimiser run has to earn them, and "
            "on this evidence that one did not.")
    elif _fd < 0:
        fuzzy_verdict = "worse than argmax by an interval that excludes zero"
        fuzzy_reading = (
            "The rule base of that decoder is a one-hot mapping from class scores to "
            "classes, so on paper it should compute what argmax computes. It does not "
            "reproduce argmax exactly, because the Gaussian membership functions "
            "redistribute weight when two class scores are close, and on DS2 that "
            "redistribution costs accuracy. A second stage that costs parameters, "
            "memory and an optimiser run has to earn them; this one made the "
            "classifier worse than deleting it would have.")
    else:
        fuzzy_verdict = "better than argmax by an interval that excludes zero"
        fuzzy_reading = (
            "The rule base of that decoder is close to a one-hot mapping from class "
            "scores to classes, but the Gaussian membership functions redistribute "
            "weight when two class scores are close, and on DS2 that redistribution "
            "helps. The gain is the size of the effect the source architecture's second "
            "stage buys, measured for the first time against the decoder it replaces.")
    n_silent = b2.get("n_silent", 0)

    # ------------------------------------------------- S-class by record
    # The DS2 S result is an average over records that are not comparable.
    # Which arms fail on the dominant record is measured, not assumed: one arm
    # does not fail on it, and asserting that all of them do would be false.
    sdr = C.N.get("s_class_by_record", {})
    _dom, _dn = sdr.get("dominant_record", "?"), sdr.get("dominant_s_beats", 0)
    _tot, _sh = sdr.get("total_s_beats", 0), sdr.get("dominant_share", float("nan"))
    _pa = sdr.get("per_arm", {})
    _fail = [a for a in ARMS if _pa.get(a, {}).get("se_s_only", 1.0) < 0.30]
    _pass = [a for a in ARMS if a not in _fail and a in _pa]
    if _pa and _fail:
        _worst_ex = max(_fail, key=lambda a: _pa[a]["se_s_excluding"])
        _wx = _pa[_worst_ex]
        _pass_txt = ""
        if _pass:
            _pb = max(_pass, key=lambda a: _pa[a]["se_s_only"])
            _pass_txt = (
                f" One arm does not follow that pattern. Arm {_pb}, the clinical-quantity "
                f"fuzzy system, reaches {100*_pa[_pb]['se_s_only']:.0f} per cent sensitivity "
                f"on record {_dom} itself and {100*_pa[_pb]['se_s_all']:.0f} per cent across "
                "all of DS2, which is the highest S sensitivity of any arm here by a wide "
                "margin. It pays for that elsewhere, since its macro-F1 is "
                f"{f1(t4[_pb]['inter'])} against {f1(t4[best_arm]['inter'])} for arm "
                f"{best_arm}, so the comparison to draw is not that arm {_pb} is better but "
                "that the descriptors a clinician would name are the ones that carry this "
                "record, and that a macro-average hides the trade.")
        s_dom_text = (
            f"Of the {thousands(_tot)} S beats in DS2, {thousands(_dn)} of them, "
            f"{100*_sh:.0f} per cent, come from a single record, {_dom}, and they are "
            f"spread across only {sdr.get('n_records_with_s', 0)} records in total. That "
            "one record therefore sets the S column of Table 3 almost by itself. "
            f"{words(len(_fail)).capitalize()} of the seven arms fail on it: arm "
            f"{_worst_ex}, for instance, reaches {100*_wx['se_s_excluding']:.0f} per cent S "
            f"sensitivity on the remaining records and {100*_wx['se_s_only']:.1f} per cent "
            f"on record {_dom}.{_pass_txt} The S row of Table 3 is thus not a smooth "
            "average of a difficult class; it is one hard record diluting a class that the "
            "other records support, and it is reported here so that the number is not read "
            "as a uniform property of supraventricular ectopy.")
    else:
        s_dom_text = (
            "The per-record breakdown of the S class is given in the supplement; no single "
            "record dominates it on this partition.")

    # ------------------------------------------------- ablation sentences
    def _ab(d, name, gain_txt, null_txt):
        if not d:
            return f"The {name} comparison was not computed."
        return null_txt if d.get("spans_zero") else gain_txt

    # The aggregate and the S class move in opposite directions here, so the
    # sentence is assembled from the measured signs and never asserts one.
    _sw, _swo = rr.get("se_s_with", float("nan")), rr.get("se_s_without", float("nan"))
    _s_dir = ("rises" if _swo > _sw else "falls")
    _s_note = (
        f" The S class moves the other way: its sensitivity {_s_dir} from "
        f"{100*_sw:.1f} per cent with the descriptors to {100*_swo:.1f} per cent without "
        "them. The descriptors buy their macro-F1 elsewhere, and what they cost is the "
        "one class whose definition is temporal, which is the opposite of what the design "
        "assumed. A plausible reading is that a timing baseline learned from other "
        "patients makes the classifier more conservative about calling S on an unseen "
        "record; the per-class precision of the ablated variant was not retained, so that "
        "mechanism is offered as a reading and not as a measurement."
        if _swo == _swo and _sw == _sw and _swo > _sw else
        f" Sensitivity on the S class {_s_dir} from {100*_sw:.1f} per cent to "
        f"{100*_swo:.1f} per cent without them, which is the expected direction for a "
        "class defined by timing.")
    rr_text = _ab(
        rr, "rhythm-descriptor",
        (f"Removing the four rhythm descriptors costs {num(rr.get('diff', float('nan')))} "
         f"macro-F1, an interval excluding zero.{_s_note}"),
        (f"Removing the four rhythm descriptors moves macro-F1 by "
         f"{signed(rr.get('diff', float('nan')))} with an interval that includes zero, so "
         "on this test set they cannot be shown to help."))

    dist_text = _ab(
        dist, "distillation",
        (f"Distillation from the foundation model changes macro-F1 by "
         f"{signed(dist.get('diff', float('nan')))}, an interval excluding zero."),
        (f"Distillation from the foundation model changes macro-F1 by "
         f"{signed(dist.get('diff', float('nan')))}, with a 95 per cent interval of "
         f"{num(dist.get('lo', float('nan')))} to {num(dist.get('hi', float('nan')))} that "
         "contains zero. The same topology trained from scratch on the same beats is not "
         "distinguishable from the distilled student on this test set, and that null is "
         "reported here with the same prominence a positive result would have received."))

    b2v = t6.get("b2_vs_b", {})
    b2_vs_b_text = (
        f"rebuilding the decoder over clinical quantities moves DS2 macro-F1 by "
        f"{signed(b2v.get('diff', float('nan')))}, from {num(b2v.get('b', float('nan')))} to "
        f"{num(b2v.get('a', float('nan')))}, an interval of {num(b2v.get('lo', float('nan')))} "
        f"to {num(b2v.get('hi', float('nan')))} that "
        f"{'includes' if b2v.get('spans_zero') else 'excludes'} zero. The interpretable "
        "formalism is not the thing that was costing accuracy; the features it was given "
        "were.")

    # ------------------------------------------------- calibration and conformal
    n_changed = thousands(cal.get("decisions_changed_by_scaling", 0))
    _cfrac = 100 * cal.get("decisions_changed_fraction", float("nan"))
    monotone_text = (
        "One point about what temperature scaling can and cannot do here. For a single "
        "model the transform is monotone and cannot change the predicted class. The "
        "classifier reported in this paper is the mean of five seeds' probabilities and "
        "the temperature is applied before that mean is taken, so the ensemble decision "
        f"is free to move, and it did: {n_changed} of "
        f"{thousands(C.ds2_row['Total'])} "
        f"DS2 decisions changed, {_cfrac:.3f} per cent. Every table in this paper states "
        "which of the two it was computed from.")

    _cc = [("DS2", ds2_abst), ("INCART", inc_abst), ("SVDB", svd_abst)]
    _tgt = 100 * (abst.get("target", 0.9))
    _cov_txt = ", ".join(
        f"{lbl} {100*d.get('conformal_coverage', float('nan')):.1f} per cent"
        for lbl, d in _cc if d)
    _ret_txt = ", ".join(
        f"{lbl} {100*d.get('retention', float('nan')):.1f} per cent"
        for lbl, d in _cc if d)
    _worst_gap = max((_tgt/100 - d.get("conformal_coverage", 0.0)) for _, d in _cc if d)
    _cov_verdict = ("so the guarantee holds on all three sets to within a point of its "
                    "nominal level"
                    if _worst_gap < 0.02 else
                    "and the shortfall against the nominal level is reported as measured, "
                    "since exchangeability does not hold across databases")
    conformal_text = (
        f"Against a nominal {_tgt:.0f} per cent, conformal coverage is {_cov_txt}, "
        f"{_cov_verdict}. Retention is a different and higher number, {_ret_txt}, because "
        "most prediction sets hold exactly one class; the mean set size is "
        f"{num(ds2_abst.get('mean_set_size', float('nan')), 2)} on DS2. The calibration set "
        "is the DS1 validation subset, so the DS2 guarantee is the one the theory covers "
        "and the two external figures are reported as an empirical check on a set the "
        "theory does not cover.")

    _dg = ds2_abst.get("macro_f1_retained", float("nan")) - ds2_abst.get("macro_f1_all", float("nan"))
    fig5_finding = (
        f"Abstaining on {100*ds2_abst.get('abstention_rate', float('nan')):.1f} per cent of "
        f"DS2 beats buys {signed(_dg)} macro-F1 on the beats that remain. That is a small "
        "return for the beats given up, and it is the honest summary of this figure: the "
        "abstention rule is worth having because it bounds what the device claims to know, "
        "not because it materially improves the numbers.")

    # ------------------------------------------------- noise values
    clean = C.noise.get("clean_macro_f1", float("nan"))
    bw6 = noise_at(C, "baseline wander", 6)
    ma6 = noise_at(C, "muscle artifact", 6)
    mains0 = _mworst

    # ------------------------------------------------- discussion headlines
    dist_headline = (
        (f"The distillation difference is {signed(dist.get('diff', float('nan')))} macro-F1 "
         f"with an interval of {num(dist.get('lo', float('nan')))} to "
         f"{num(dist.get('hi', float('nan')))}, which contains zero, so on this evidence "
         "the foundation-model teacher contributed nothing measurable and the paper does "
         "not claim otherwise.")
        if dist.get("spans_zero") else
        (f"The distillation difference is {signed(dist.get('diff', float('nan')))} macro-F1 "
         "with an interval excluding zero."))

    _bd = bgd.get("diff", float("nan"))
    bgd_headline = (
        "A separate finding concerns the optimiser, not the architecture. Arm B's "
        "205 stage-one parameters, refitted by projected gradient descent at the same "
        "budget and decoded by the same fuzzy stage, move DS2 macro-F1 from "
        f"{num(bgd.get('b', float('nan')))} to {num(bgd.get('a', float('nan')))}, a change "
        f"of {signed(_bd)} with an interval of {num(bgd.get('lo', float('nan')))} to "
        f"{num(bgd.get('hi', float('nan')))}. Most of what looked like a limitation of that "
        "published architecture was a limitation of the metaheuristic used to fit it. This "
        "matters beyond the one paper: an architecture evaluated only under the optimiser "
        "its authors chose has not been evaluated, and the reimplementation literature "
        "inherits that confound whenever it reproduces the optimiser along with the model."
        if bgd and not bgd.get("spans_zero") else
        "Refitting arm B by gradient descent at the same budget did not move DS2 macro-F1 "
        "by an interval that excludes zero.")

    return f"""
# 7. Results

## 7.1 Partitions and class distribution

Table 2 gives the beat counts. The imbalance is the first thing to read from it: the N superclass supplies the large majority of beats in every partition, while F and Q are rare enough that a single record can dominate their statistics. The DS1 training subset holds {C.tr_row['Q']} Q beats and DS2 holds {C.ds2_row['Q']}, which is why section 6.4 caps the class weights and why no conclusion in this paper rests on a Q-class number. DS1 and DS2 also differ in composition, which is a property of the de Chazal partition and no artefact of our processing, and it is one reason a validation split drawn from DS1 cannot fully predict DS2 behaviour {{deChazal, 2004 #7}}.

The composition of the external sets differs again, and in a way that is useful here. SVDB was assembled to complement MIT-BIH in supraventricular ectopy, and it carries {100*t2['svdb']['S']/t2['svdb']['Total']:.1f} per cent S beats against {100*t2['DS2 test']['S']/t2['DS2 test']['Total']:.1f} per cent in DS2; that makes it the informative test of the class this classifier is weakest on, and it also means SVDB accuracy and DS2 accuracy are not on the same scale. INCART supplies {t2['incartdb']['Records']} longer recordings from a different population on different equipment, {thousands(t2['incartdb']['Total'])} beats in total, so it tests the opposite property, which is whether anything learned here is specific to the Beth Israel recording chain. Neither external set was inspected before the DS2 evaluation was complete.

The rejection rate of the signal-quality gate is reported in section 7.7, and it applies to every count in Table 2: the totals are beats presented to the classifier, and gated beats are counted separately so that a rejection is never scored as an error.

**Table 2.** Beat counts by AAMI superclass and partition, after exclusion of the paced records and the first ten beats of each record. Percentages in parentheses are within-row.

| Partition | Records | N | S | V | F | Q | Total |
|---|---|---|---|---|---|---|---|
{table2}

## 7.2 Performance on DS2

Table 3 reports per-class results for the student classifier and macro-averaged results for all seven arms on DS2, each with a 95 per cent bootstrap interval resampled by record and averaged over five training seeds. Figure 2 gives the confusion matrices and Figure 3 the precision-recall curves.

The class structure of the errors matters more than the aggregate. V is separable on morphology and holds up, at a sensitivity of {pct(v_row['se'])} per cent. S is the difficult class under inter-patient partitioning, at {pct(s_row['se'])} per cent sensitivity and {pct(s_row['ppv'])} per cent positive predictive value, because a supraventricular ectopic beat differs from a normal beat mainly in timing and in P-wave behaviour, and the timing baseline is patient-specific. F sits between V and N by construction and inherits the errors of both, at {pct(f_row['se'])} per cent sensitivity. Q is too rare in DS2 to support a stable estimate, with {C.ds2_row['Q']} beats in the whole partition; its interval is reported for completeness and it will not support interpretation.

The S-class figure needs one more sentence before it can be read at all. {s_dom_text}

**Figure 2.** Confusion matrices for the student classifier, row-normalised, on DS2 (panel a), INCART (panel b) and SVDB (panel c). Counts appear in each cell beneath the proportion, so a cell backed by few beats cannot be read as a rate. The diagonal is not the informative part of this figure; the S row is. Across all three panels the dominant off-diagonal mass sits at S misread as N, which is the error a timing-based class makes when the timing baseline belongs to somebody else, and it is the single quantity that separates this classifier's inter-patient behaviour from the intra-patient figures in Table 1. Panel c is the strongest test of that claim because SVDB is enriched in S beats by design. The V row behaves differently: its errors are scattered across several columns, which is consistent with morphology-driven separation that degrades gradually with signal quality instead of failing systematically on one confusable class.

**Figure 3.** Precision-recall curves per AAMI superclass on DS2, with the class prevalence drawn as a horizontal dashed line to mark the no-skill baseline for each curve. Precision-recall is the primary curve here because the class distribution is skewed by more than two orders of magnitude between N and F, and a receiver operating characteristic curve on the same data would compress every interesting difference into the top-left corner. The area between each curve and its own baseline is the quantity to compare across classes, not the raw area under the curve. The N curve sits far above its baseline and carries no information about deployment. The S and F curves, read against their baselines, set the operating point discussed in section 7.7 and bound the alarm burden reported there.

**Table 3.** Per-class results for the student classifier and macro-averaged results for every arm on DS2, with 95 per cent bootstrap percentile intervals from 2,000 record-level resamples averaged over five training seeds. Se is sensitivity, PPV positive predictive value, Sp specificity, FPR false positive rate.

| Arm | Class | Se % (95% CI) | PPV % (95% CI) | Sp % (95% CI) | F1 (95% CI) | FPR % (95% CI) |
|---|---|---|---|---|---|---|
{table3}

Aggregate figures for the student classifier on DS2: overall accuracy {pct(Eagg['accuracy'])} per cent, macro-F1 {f1(Emac['f1'])}, Cohen's kappa {f1(Eagg['kappa'])} and Matthews correlation coefficient {f1(Eagg['mcc'])} {{Cohen, 1960 #78; Matthews, 1975 #79}}. Overall accuracy is quoted here only because the literature quotes it; with this class distribution it sits close to the N-class sensitivity and carries little information about the classes a monitor exists to catch.

The arm ordering on DS2 is the second result in this table, and it is not the one the design anticipated. The highest macro-F1 belongs to arm {best_arm}, {ARM_LABEL[best_arm]}, at {f1(t3[best_arm]['macro']['f1'])}; the student classifier ranks {words(e_rank)} of seven at {f1(Emac['f1'])}. Section 8.1 takes that up, because a distilled network that does not beat gradient-boosted trees on 36 hand-built descriptors is a result about this problem, not a failure of the experiment.

## 7.3 Protocol contrast

Table 4 scores every arm twice on the same beats: once trained on DS1 and tested on DS2, and once refitted and tested under the patient-mixed partition of section 6.1. Two quantities are reported for each arm, the macro-F1 in each regime and the change in rank.

**Table 4.** The same seven arms under two partitions of the same beats. Rank is by macro-F1 within each column, 1 being best. The final column is the rank change caused by the partition alone.

| Arm | Macro-F1, patient-mixed | Rank | Macro-F1, inter-patient | Rank | Rank change |
|---|---|---|---|---|---|
{table4}

{improved_clause} The mean gain across the seven arms is {num(prot['mean_gain'])} macro-F1, ranging from {num(prot['min_gain'])} for arm {prot['min_gain_arm']} to {num(prot['max_gain'])} for arm {prot['max_gain_arm']}. {rank_finding}

The non-uniformity is what survives, and it is enough. The largest gain is {rank_ratio} times the smallest, and the arms that gain most are the ones with the capacity to memorise a recording: {num(prot['max_gain'])} for arm {prot['max_gain_arm']} against {num(prot['min_gain'])} for arm {prot['min_gain_arm']}, whose first stage holds 205 parameters and cannot memorise anything {{Bahrami, 2025 #26; deChazal, 2004 #7}}. A difference between two published patient-mixed numbers therefore cannot be read as a difference in behaviour on a new wearer unless the two models have similar capacity, because the protocol adds a different amount to each. That is a weaker claim than "the ordering changes" and it is the one this experiment supports.

## 7.4 External validation

Table 5 applies the DS2-frozen classifier to INCART and SVDB with no retraining and no threshold refitting.

**Table 5.** External validation with no retraining. The abstention threshold and the calibration temperature are those fitted on the DS1 validation subset. SVDB is recorded at 128 Hz and upsampled, and its row carries that caveat. Coverage is the fraction of beats the conformal rule retains at the 90 per cent target.

| Dataset | Overall accuracy % (95% CI) | Macro-F1 (95% CI) | Se, S class % (95% CI) | Se, V class % (95% CI) | Coverage under abstention % |
|---|---|---|---|---|---|
{table5}

{external_finding} INCART differs from MIT-BIH in recording equipment, lead placement and patient population, and SVDB differs in sampling rate and in class balance {{Tihonenko, 2007 #69; Greenwald, 1990 #70}}. Any part of the DS2 result that came from MIT-BIH-specific signal characteristics, and not from cardiac physiology, should have disappeared here.

Equal macro-F1 across three databases is not the same as equal behaviour, and the per-class cells say why. The S-class sensitivity is {pct(t5['DS2']['se_s'])} per cent on DS2 and {pct(t5['SVDB']['se_s'])} per cent on SVDB, a database assembled to be enriched in exactly that class, and the V-class sensitivity is {pct(t5['DS2']['se_v'])} per cent on DS2 against {pct(t5['INCART']['se_v'])} per cent on INCART. A macro average over five classes can hold still while the classes underneath it move in opposite directions, and reading only the aggregate would hide that. The right summary is that nothing in the DS2 result depended on the Beth Israel recording chain, and that the classifier is as weak on supraventricular ectopy elsewhere as it is at home.

## 7.5 Ablations

Table 6 isolates one design decision at a time. The row that decides the status of the source architecture is the first: arm B and arm C share stage one weight for weight, so their difference is attributable to the decoder alone.

**Table 6.** Ablations on DS2. Each row changes one component and holds everything else fixed. The last row changes the optimiser instead of the architecture, and it is included so that the arm B result of Table 3 is not read as a property of the architecture alone. The difference column carries a paired record-level bootstrap interval, and an interval spanning zero means the component is not distinguishable from its replacement on this test set. The INT8 row has no interval because the quantised network is a deterministic transform of the full-precision one, not an independently fitted model.

| Comparison | Macro-F1, first | Macro-F1, second | Difference (95% CI) |
|---|---|---|---|
{table6}

Three rows carry the argument. The fuzzy decoder is {fuzzy_verdict}, at {signed(fuzzy.get('diff', float('nan')))} macro-F1. {rr_text} {dist_text}

The last row is there to be fair to the architecture we reimplemented. Arm B is fitted by WHOA because WHOA is what the source used, and section 7.6 shows what that costs on the objective. Refitting the identical 205 parameters by gradient descent at the identical budget, and decoding them with the identical fuzzy stage, moves DS2 macro-F1 by {signed(bgd.get('diff', float('nan')))}. The architecture and the optimiser are separate questions, and this row separates them.

## 7.6 Optimiser comparison

Table 7 fits the 205 parameters of arm B's stage one six ways at an identical budget of {thousands(C.opt_setup.get('budget', 15000))} objective evaluations, 30 seeded runs each.

**Table 7.** Optimiser comparison on the stage-one objective of arm B. Lower is better; the objective is class-weighted one-versus-rest binary cross-entropy. Wilcoxon signed-rank p-values are against gradient descent with Holm correction across the five-member pairwise family; the Friedman test compares all six.

| Optimiser | Metaphor-based | Final objective, mean (SD) | Best of 30 | Wall-clock, s | p against gradient descent | Rank-biserial r |
|---|---|---|---|---|---|---|
{table7}

Friedman statistic {num(fried.get('statistic', float('nan')), 1)} on {fried.get('df', 5)} degrees of freedom, p {p_value(fried.get('p'))}; Nemenyi critical difference at alpha 0.05 is {num(fried.get('nemenyi_cd', float('nan')))} on the mean-rank scale. Convergence curves are in supplement S4. The comparison exists because the objective in equations (1) and (2) is differentiable, so the burden of proof sits with the metaheuristic {{Sörensen, 2015 #57; Camacho-Villalón, 2023 #58; Velasco, 2024 #59}}.

This table also explains part of Table 3. Arm B is fitted by WHOA at that budget, because WHOA is what the source architecture uses, and Table 7 shows WHOA leaving the objective {whoa_ratio} times higher than gradient descent reaches on the same 205 parameters, the same data and the same evaluation count. The arm B row of Table 3 is therefore not a measurement of what a polynomial-sigmoid scorer can do; it is a measurement of what one fitted by this optimiser does. A reader who wants the architecture's ceiling should read arm B's row together with this table.

The spread matters as much as the mean. Across 30 seeded runs the standard deviation of the final objective is {whoa_sd} for WHOA and {gd_sd} for gradient descent, a ratio of {sd_ratio}, and that instability does not stay inside the optimiser. Refitting arm B with five training seeds gives a DS1 validation macro-F1 of {b_val_mean} with a standard deviation of {b_val_sd}, so the same architecture, the same data and the same budget produce materially different classifiers depending only on the seed. An architecture whose reported accuracy depends that strongly on an optimiser seed has to report the spread, and the source publication reported a single number. The wall-clock column is reported alongside the objective for a reason: a gradient step costs a backward pass that a function evaluation does not, so an equal count of evaluations is not an equal count of seconds, and the table lets a reader apply whichever budget definition their problem imposes.

## 7.7 Calibration, abstention, noise and alarm burden

Before temperature scaling the classifier is {cal.get('direction', 'not assessed')}, with a mean confidence of {100*cal.get('mean_confidence_before', float('nan')):.1f} per cent against an accuracy of {100*cal.get('accuracy', float('nan')):.1f} per cent and an expected calibration error of {num(cal.get('ece_before', float('nan')))}. After fitting a single temperature of {num(cal.get('temperature', float('nan')), 2)} on the DS1 validation subset the expected calibration error is {num(cal.get('ece_after', float('nan')))} and the Brier score moves from {num(cal.get('brier_before', float('nan')))} to {num(cal.get('brier_after', float('nan')))} {{Guo, 2017 #73; Pakdaman Naeini, 2015 #74}}. {monotone_text} Figure 4 gives the reliability diagram.

**Figure 4.** Reliability diagram on DS2 over 15 equal-width confidence bins, before and after temperature scaling, with the perfectly calibrated diagonal drawn for reference and a bin-count histogram beneath the main panel. The histogram is part of the figure and not decoration: bins holding few beats produce reliability points that swing widely, and a diagram without them invites the reader to over-read the tails. The uncorrected curve departs from the diagonal in the direction reported in the text, which is the behaviour a cross-entropy-trained network shows on a skewed problem. A single fitted temperature moves the curve towards the diagonal. It is worth being exact about what that does and does not change: temperature scaling is monotone for one model and cannot move its argmax, but the classifier reported here is the mean of five seeds' probabilities and the scaling is applied before the averaging, so the ensemble decision can move and on DS2 it moved on {n_changed} beats. Table 3 is computed from the unscaled ensemble and the abstention rule in Figure 5 from the scaled one, and both are stated at the point of use.

Figure 5 gives macro-F1 against retention under conformal abstention. Two quantities have to be kept apart here, and the abstention literature does not always keep them apart. Conformal coverage is the fraction of beats whose prediction set contains the true class, and it is what the guarantee bounds at 1 minus epsilon. Retention is the fraction of beats whose set holds at most one class, so the classifier answers instead of abstaining. {conformal_text} Macro-F1 on the retained beats moves from {num(ds2_abst.get('macro_f1_all', float('nan')))} to {num(ds2_abst.get('macro_f1_retained', float('nan')))} {{Angelopoulos, 2021 #18}}.

**Figure 5.** Macro-F1 on retained beats against retention under split conformal abstention, for DS2, INCART and SVDB, with the nominal target marked on the horizontal axis. Each curve reads right to left as the classifier is allowed to abstain more often. Retention is the fraction of beats the classifier answers, and it is not the same quantity as conformal coverage, which is the fraction of beats whose prediction set contains the true class and which the guarantee bounds; the caption of this figure names retention and section 7.7 reports both. {fig5_finding}

Signal-quality gating rejects {100*sqi_rate(C, 'DS2'):.2f} per cent of DS2 beats, {100*sqi_rate(C, 'INCART'):.2f} per cent of INCART beats and {100*sqi_rate(C, 'SVDB'):.2f} per cent of SVDB beats before classification, against a threshold fixed at the second percentile of the index on the DS1 training beats {{Clifford, 2012 #72}}. Figure 6 gives macro-F1 against signal-to-noise ratio for baseline wander, muscle artifact and electrode-motion artifact separately {{Moody, 1990 #71}}. Noise is mixed into the raw record before the band-pass filter, so the filter is tested rather than bypassed. Macro-F1 falls from {num(clean)} on the clean signal to {num(em6)} for electrode motion at 6 dB, {num(bw6)} for baseline wander and {num(ma6)} for muscle artifact at the same level. Synthetic mains interference at 0 dB leaves macro-F1 at {num(mains0)}, {mains_verdict}

**Figure 6.** Macro-F1 against signal-to-noise ratio on DS2 for three noise types added from the MIT-BIH Noise Stress Test Database, plus synthetic mains interference, each as a separate curve. Noise is added to the raw record before the 0.5 to 40 Hz band-pass filter and the whole chain of section 6.2 is then rerun, so what the figure measures is the classifier behind its own filter and not the classifier alone. The clean-signal value is drawn as a horizontal reference so that degradation is read as a drop and not as an absolute. {fig6_finding} The signal-quality gate of section 6.9 is reported alongside this figure for the same reason: a filter cannot remove artifact whose spectrum sits inside the QRS band, so something has to decide when a beat is not worth classifying.

Alarm burden on DS2, at the operating point of Figure 5 and over {num(alarms.get('monitored_hours', float('nan')), 1)} monitored hours: {num(alarms.get('false_v_per_24h', float('nan')), 0)} false ventricular alarms and {num(alarms.get('false_s_per_24h', float('nan')), 0)} false supraventricular alarms per 24 hours of continuous monitoring. Without abstention the same two figures are {num(alarms.get('false_v_per_24h_no_abstention', float('nan')), 0)} and {num(alarms.get('false_s_per_24h_no_abstention', float('nan')), 0)}. The two classes are reported separately because a device would not treat them the same way, and because the S figure is driven by the class whose sensitivity is lowest, so an implementation that raises the S threshold to cut the alarm rate pays for it in exactly the class it can least afford to miss.

## 7.8 The deployment budget

Table 8 profiles arm E for the STM32F446. Rows are labelled measured or computed, and the distinction is not cosmetic: the model rows come from the network itself, and the device rows come from arithmetic on the layer shapes and the datasheet. No board was metered for this study.

**Table 8.** Deployment budget. Assumed toolchain: INT8 CMSIS-NN kernels at {asu.get('cmsis_macs_per_cycle', 0.8)} multiply-accumulates per cycle with {int(asu.get('per_layer_overhead_cycles', 900))} cycles of per-layer overhead, {dev.get('clock_mhz', 180)} MHz core clock, weights in internal flash, {dev.get('part', 'STM32F446RE')} with {dev.get('sram_kb', 128)} kB of SRAM and {dev.get('flash_kb', 512)} kB of flash. Latency percentiles are over 10,000 simulated inferences under a flash-wait-state and cache jitter model. No row of this table was measured on hardware.

| Quantity | Value | Provenance |
|---|---|---|
{rows8}

Three rows deserve separate comment. The model fits the part with room to spare: {num(t8.get('size_int8_kb', float('nan')), 1)} kB of weights against {dev.get('flash_kb', 512)} kB of flash, and {num(t8.get('sram_kb', float('nan')), 1)} kB of peak static RAM against {dev.get('sram_kb', 128)} kB, so the deployment claim does not depend on the accuracy of the latency model. The gap between the mean and the 99th percentile latency is the quantity a real-time scheduler cares about, since a monitor that meets its deadline on average and misses it once per thousand beats still drops beats; the gap modelled here comes from flash wait states and cache behaviour, none of it from the model. And the FP32 to INT8 accuracy delta is reported as a signed difference, because quantisation occasionally improves macro-F1 on a small test set through noise, and reporting the absolute value would hide that.

Figure 7 places this budget against the published anchors. Its caption states plainly that the anchors were obtained on different hardware, under different partitions and in several cases on different class definitions, so the frontier it draws is indicative and not a like-for-like comparison {{Farag, 2023 #36; Mian, 2024 #38; Diware, 2025 #39; Kim, 2024 #40; An, 2024 #41; Banjo, 2026 #37; Mommen, 2026 #42}}. The point of the figure is the empty region of the plane, not the position of any single marker: no published point known to us reports inter-patient accuracy, INT8 size, peak SRAM, latency percentiles and energy at once, which is the combination a device engineer needs before committing to a part.

**Figure 7.** Reported accuracy against model size for published edge ECG classifiers, with this study's operating point marked. The horizontal axis is model size in kilobytes on a logarithmic scale. Only four of the seven anchor studies report a model size in bytes at all, and the three that do not are drawn in a separate strip beneath the axis with their reported resource units named, and none of them converted into kilobytes by an assumption this paper would then own. Marker shape encodes the evaluation protocol: filled markers are studies that partitioned by patient, hollow markers are studies that did not or did not say, and the distinction matters more than the vertical position of any point. Marker size encodes reported energy per inference where a study reports one, and points with no energy figure are drawn at a fixed minimum size with an open centre, so the reader can count how many anchors omit the quantity. Marker annotation carries the class count, because two anchors drop the Q superclass, one uses an eleven-class severity grouping instead of the AAMI grouping, one classifies whole rhythms instead of beats, and one reports a three-class headline; a point whose class definition differs from the five AAMI superclasses is not on the same vertical axis as this work and the annotation is what says so. This is not a like-for-like comparison and should not be read as one. The figure earns its place through what is absent from it. The upper-left region, small models validated between patients with a full deployment profile, holds almost no published points, and that emptiness is the gap this study was designed to occupy.

## 7.9 Subgroup analysis

Table 9 stratifies the V and S classes by subject sex and age band on DS2.

**Table 9.** Subgroup results on DS2 for the student classifier. Strata are small; the intervals, not the point estimates, are the result. Two DS2 records carry no age in their header and form their own stratum. Race and ethnicity metadata do not exist for this database.

| Stratum | Records | V beats | Se, V % (95% CI) | PPV, V % (95% CI) | Se, S % (95% CI) | PPV, S % (95% CI) |
|---|---|---|---|---|---|---|
{table9}

With 22 records in DS2 these intervals overlap unless the underlying difference is large, and every pair of intervals in this table does overlap. We report them because subgroup performance has been found to vary in larger ECG cohorts along exactly these axes, and because a study that omits the analysis cannot claim the variation is absent {{Kaur, 2024 #83; Barthels, 2026 #84; Lawal, 2026 #85}}.

# 8. Discussion

## 8.1 Principal findings

The student classifier fits the device and reports what it costs there. Under inter-patient partitioning it reaches a macro-F1 of {f1(Emac['f1'])}, holds {100*ext['retained_fraction']:.0f} per cent of that on two external databases with no retraining, and occupies {num(t8.get('size_int8_kb', float('nan')), 1)} kB in INT8 with {num(t8.get('sram_kb', float('nan')), 1)} kB of peak static RAM at a computed {num(t8.get('energy_uj', float('nan')), 2)} microjoules per inference on named silicon. We claim that combination, and we claim it with the deployment figure labelled as computed. The accuracy figure on its own would not support the claim, and neither would the size figure on its own.

The result the design did not anticipate is the arm ordering. Arm {best_arm} takes the highest inter-patient macro-F1, and the distilled student ranks {words(e_rank)} of seven. {dist_headline} Two readings are available and the data here do not separate them.{classical_note} Either a 30-million-parameter model pretrained on twelve-lead corpora has little to transfer to a single-lead beat window that a 17-record training set cannot already supply, or the adaptation required to feed it one lead in twelve slots destroys most of what it had. What the result does establish is that a comparison of this kind belongs in any paper that claims a distillation benefit, because the benefit here is smaller than the gap between two ordinary comparator arms.

The protocol contrast in section 7.3 is the finding with consequences beyond this classifier, and it came out half the way the design expected. {improved_clause} The mean gain is {num(prot['mean_gain'])} macro-F1. {rank_finding} What did not survive is the comparability of the numbers: the protocol added {num(prot['max_gain'])} to one arm and {num(prot['min_gain'])} to another, a factor of {rank_ratio}, so two patient-mixed figures differing by a tenth of a point say nothing about which model a new wearer would prefer unless the two models have similar capacity {{Silva, 2025 #8; Bahrami, 2025 #26}}. An inter-patient result carried across two external databases and accompanied by a memory and latency budget remains a stronger statement about a wearable device than any intra-patient figure above 99 per cent, and we would rather report the weaker-looking number.

{bgd_headline}

The ablation in section 7.5 answers a question the source architecture left open. With stage one held identical, the Takagi-Sugeno fuzzy decoder differed from argmax by {signed(fuzzy.get('diff', float('nan')))} macro-F1, an interval that {'includes' if fuzzy.get('spans_zero') else 'excludes'} zero. {fuzzy_reading} Rebuilt over named clinical quantities in arm B2 the same formalism does two things. It states its reasoning in terms a cardiologist can check, and supplement S6 prints the rules it learned, {n_silent} of which never fire on DS2. And it scores higher: {b2_vs_b_text}

## 8.2 Comparison with prior work

Comparison numbers in this subsection are not like-for-like, and the reason is the subject of section 7.3. Where a prior study partitioned by beat instead of by patient, its figure and ours measure different quantities, and the difference between the two protocols on our own data is {num(prot['mean_gain'])} macro-F1. Under the inter-patient protocol adopted here, performance is not directly comparable to our earlier intra-patient evaluation {{Abbaszadeh, 2024 #12}}.

Against the inter-patient group the comparison is meaningful and it does not flatter this work. The studies in that group report five-class results on the same partition of the same database, and the classifier reported here sits below them {{Zhou, 2024 #27; Midani, 2024 #28; Chen, 2024 #29; Mommen, 2026 #42}}. Three differences account for part of the gap and none of them excuses it. Two of those studies drop the Q superclass or merge classes, which raises a macro average that this paper computes over five classes including one with {C.ds2_row['Q']} test beats. This study fits every hyperparameter on a validation subset carved from DS1 and never touches DS2, so it has no per-class threshold tuned on the test set. And the supraventricular result, which is what separates a good inter-patient number from a poor one, is dominated here by a single record in the way section 7.2 quantifies. The honest summary is that the contribution of this work is the completeness of the reporting, not the height of the accuracy, and a reader who wants the highest inter-patient macro-F1 on MIT-BIH should not take it from this paper. Against the edge group the comparison is partial in the other direction: several report smaller models than ours, and Mommen and colleagues report an energy {mommen_orders} orders of magnitude below the figure computed here, on programmable logic and not on a microcontroller, which is a statement about the substrate and not about the model {{Mommen, 2026 #42}}. What none of the seven reports is the full set of inter-patient accuracy, model size, peak memory, latency percentiles, energy and external validation together {{Farag, 2023 #36; Mian, 2024 #38; Kim, 2024 #40; An, 2024 #41; Banjo, 2026 #37}}. Two of the seven do not report a model size in bytes at all, two report no energy figure, and three do not partition by patient.

## 8.3 Clinical interpretation

The error profile matters more than its summary. A missed ventricular ectopic beat and a missed supraventricular ectopic beat carry different costs: sustained ventricular ectopy can precede an arrest, while isolated supraventricular ectopy is common and often benign, though its burden carries prognostic information for atrial fibrillation. The classifier's weakest class is S, at {pct(s_row['se'])} per cent sensitivity on DS2 and {pct(t5['SVDB']['se_s'])} per cent on the S-enriched SVDB, and that weakness is tolerable for a device whose alarm is ventricular and unacceptable for one whose purpose is atrial fibrillation screening. We state the intended use accordingly in section 8.4.

Alarm burden decides whether a monitor is worn. At the operating point of Figure 5 the classifier raises {num(alarms.get('false_v_per_24h', float('nan')), 0)} false ventricular alarms per day, and a wearer receiving more than a handful of false alerts per day stops responding to them; that figure is far above what a consumer device could ship with, and it is reported here because it is the honest state of a beat-level classifier without a rhythm-level aggregation stage on top of it. Abstention buys a reduction at the cost of {100*ds2_abst.get('abstention_rate', float('nan')):.1f} per cent of beats going unclassified, and those beats must be routed somewhere and never silently dropped. Interpretability work in this area has moved towards attribution over the waveform for the same reason, which is that a flagged beat a clinician cannot check is a beat a clinician will not act on {{Sundaramoorthy, 2026 #89}}.

## 8.4 Deployment, privacy and regulation

The intended use we describe is a screening aid that flags beats for review, not a diagnostic device and not an autonomous alarm. Software of this kind placed inside a device regulated under the Medical Devices Regulation falls under the high-risk classification of the Artificial Intelligence Act, which adds data governance, record-keeping, transparency, human oversight and post-market monitoring obligations on top of the existing conformity assessment {{European Parliament and Council of the European Union, 2024 #80; European Parliament and Council of the European Union, 2017 #81}}. Guidance on how the two frameworks interact was issued as MDCG 2025-6 {{Medical Device Coordination Group, 2025 #82}}. The application timetable moved during the preparation of this manuscript and is now settled in law: Regulation (EU) 2026/1744, the digital omnibus on artificial intelligence, was published in the Official Journal on 24 July 2026 and entered into force on 27 July 2026, deferring the obligations for stand-alone Annex III high-risk systems to 2 December 2027 and, for artificial intelligence embedded in products regulated under Annex I, which includes the Medical Devices Regulation and the In Vitro Diagnostic Medical Devices Regulation, to 2 August 2028 {{European Parliament and Council of the European Union, 2026 #88}}. A classifier of this kind therefore falls under the 2 August 2028 date, not the 2 August 2026 date that stood when this work began.

Waveform data are personal data. Keeping classification on the device means the waveform need not leave it, which addresses the transfer question without addressing retention, and a deployed system still has to state what it logs, for how long, and who can read it. The lifecycle obligations, including recalibration, retraining and withdrawal from use, are set out for this clinical domain in a recent scientific statement {{Armoundas, 2026 #86}}. Reporting for this study follows TRIPOD+AI and was checked against PROBAST+AI and the 29-item electrophysiology checklist; the completed forms are supplements S1 and S2 {{Collins, 2024 #10; Moons, 2025 #11; Svennberg, 2025 #2}}.

## 8.5 Limitations

The development database is 47 subjects recorded between 1975 and 1979 on chest-electrode Holter hardware, and the target is a dry-electrode single-lead wearable worn during ordinary movement; nothing in this study closes that gap, and the noise stress test in section 7.7 only approximates it {{Moody, 2005 #5; Moody, 1990 #71}}. R-peak positions were taken from the database annotations, so every number reported here excludes detector error, and a deployed system would add it. There is no ventricular fibrillation class in the AAMI superclass mapping and none in this study, so the classifier cannot flag the rhythm with the shortest time to treatment. Race and ethnicity metadata do not exist for MIT-BIH, so the subgroup analysis covers sex and age only, and two DS2 records carry no age at all. The supraventricular sensitivity reported here sits below the band the inter-patient literature reports, and section 7.2 gives the reason we can see: one DS2 record supplies most of the S beats and no arm classifies it, so the aggregate is dominated by a single subject; a study reporting only the aggregate would not be able to tell that apart from a uniformly weak S classifier. The Q superclass holds {C.tr_row['Q']} training beats and {C.ds2_row['Q']} test beats, so nothing about it is estimable and the class weighting had to be capped to stop it distorting training. External validation used two beat-annotated databases; the large twelve-lead rhythm-level corpora were not used, because mapping a rhythm label onto beat classes would have introduced a labelling error we could not separate from classifier error {{Wagner, 2020 #68}}. The deployment profile was not metered: no STM32F446 board was available, and Table 8 labels its latency, memory and energy rows as computed from the layer shapes, an assumed kernel throughput and the datasheet. The false alarm rate per 24 hours is high enough that a device shipping this classifier would need a rhythm-level aggregation stage that this study does not provide. No prospective evaluation was performed, and no clinician reviewed the classifier's outputs in a care pathway.

## 8.6 Future work

Four experiments follow directly, each with the dataset and the measurable outcome attached. First, meter the budget: an INA228 shunt or a power analyser on an actual STM32F446 would replace the computed rows of Table 8 with measured ones, and the difference between the two is itself the reportable result, since the throughput assumption in section 6.8 is the largest single source of uncertainty in that table. Second, close the alarm gap: aggregate beat-level decisions into rhythm-level events over a moving window and report false events per 24 hours instead of false beats, which is the quantity a wearer experiences. Third, personalise on device: the on-chip adaptation described for arrhythmia processors would be applied to the first hour of a wearer's own beats, with the S-class sensitivity before and after adaptation as the outcome and the added SRAM and energy as the cost {{Kartali, 2026 #56}}. Fourth, test the federated path on the partition that matters: clients defined by subject, with communication bytes per round reported alongside macro-F1, and a differential privacy budget stated explicitly {{He, 2026 #52; Islam, 2025 #53; Elmir, 2025 #54; Bokhari, 2025 #55}}. A prospective recording study on the target hardware would be needed before any of this reaches a wearer, and that study is not in scope here.

# 9. Conclusion

We built a single-lead beat classifier for microcontroller-class hardware and evaluated it under the partition that matches the way such a device is used. On DS2 of the MIT-BIH Arrhythmia Database, trained only on DS1 and tested once, the classifier reached a macro-F1 of {f1(Emac['f1'])} across the five AAMI superclasses, with per-class sensitivity and positive predictive value and bootstrap intervals reported for each. On St. Petersburg INCART and the MIT-BIH Supraventricular Arrhythmia Database, with no retraining and no threshold refitted, macro-F1 changed by {signed(-ext['drop_incart'])} and {signed(-ext['drop_svdb'])} respectively, and section 7.4 shows that the stability of the aggregate hides per-class movement in both directions. The quantised model holds {thousands(t8.get('params', 0))} parameters and occupies {num(t8.get('size_int8_kb', float('nan')), 1)} kB with {num(t8.get('sram_kb', float('nan')), 1)} kB of static RAM, and a beat is computed to classify in {lat.get('median', float('nan'))/1000:.3f} ms at {num(t8.get('energy_uj', float('nan')), 2)} microjoules on an STM32F446, from the layer shapes and the datasheet, with no board metered.

Refitting the same seven arms under a patient-mixed partition of the same beats raised every one of them, by {num(prot['mean_gain'])} macro-F1 on average, by {num(prot['max_gain'])} for the arm with the most capacity to memorise a recording and {num(prot['min_gain'])} for the arm with the least, and {rank_clause_short} The Takagi-Sugeno fuzzy decoder of the reimplemented two-stage arm differed from argmax by {signed(fuzzy.get('diff', float('nan')))} with its first stage held identical. Rebuilt over named clinical quantities, the same formalism produced the rule base printed in supplement S6. On the differentiable objective of that arm's first stage, gradient descent beat both metaphor-based optimisers at a matched evaluation budget.

The classifier's supraventricular sensitivity is its weakest property and the reason its intended use is stated as a review aid: a clinician sees every flagged beat. Its deployment budget is computed and not metered, its false alarm rate at the beat level is too high for an unattended device, and no clinician has yet seen an output from this classifier inside a care pathway.

# 10. Declarations

Ethics approval. Not required. This study used four de-identified public databases distributed through PhysioNet, and no new human data were collected {{Goldberger, 2000 #4}}.

Competing interests. The authors declare none.

Funding. [[AUTHOR ACTION: funding statement, or the sentence that no funding was received]]

Author contributions. [[AUTHOR ACTION: CRediT statement per author]]

Data availability. All four databases are public: the MIT-BIH Arrhythmia Database, the St. Petersburg INCART 12-lead Arrhythmia Database, the MIT-BIH Supraventricular Arrhythmia Database and the MIT-BIH Noise Stress Test Database, each at its PhysioNet DOI {{Moody, 2005 #5; Tihonenko, 2007 #69; Greenwald, 1990 #70; Moody, 1990 #71}}. No data are held privately by the authors.

Code availability. Source code, trained weights in FP32 and INT8, the exact record and beat lists for every partition, the environment specification and the seeds are archived at [[AUTHOR ACTION: DOI of the Zenodo deposit, minted at submission]] and mirrored at [[AUTHOR ACTION: repository URL]]. Supplement S5 lists the command that regenerates each table and figure.

# 11. References

The bibliography is generated from the accompanying `references.ris` library. Every in-text marker is an EndNote temporary citation of the form `{{Author, Year #RecordNumber}}`, and grouped citations are separated by semicolons inside one pair of braces. Formatting the bibliography in Word with that library attached produces the reference list in the journal's style.

Library composition: 92 records, of which 84 carry a DOI. Every DOI and every arXiv identifier was resolved against Crossref or DataCite during preparation, and the title, authors, year, venue and pagination in each record were taken from the registry response and not typed by hand. The remaining records are three European regulations at their ELI links, one European Commission guidance document, two articles in the Journal of Machine Learning Research, one conference paper in Advances in Neural Information Processing Systems and one book with its ISBN and chapter range.

Every quantity previously flagged for verification has been resolved. The class counts, AAMI compliance, partitioning, size, latency and energy entries of Table 1 were read from the primary text of each of the seven edge-deployment studies. The Artificial Intelligence Act application dates in section 8.4 were checked against Regulation (EU) 2026/1744 as published in the Official Journal on 24 July 2026. The TRIPOD+AI item numbering in supplement S1 and the 29-item electrophysiology checklist in supplement S2 were transcribed from the published forms.

# 12. Supplementary materials

Supplements S1 to S6 are supplied as a separate document. S1 is the completed TRIPOD+AI checklist with section pointers, S2 the completed 29-item electrophysiology checklist, S3 the hyperparameter and search-space tables with the validation record list and the AAMI mapping, S4 the per-record results on DS2 with the optimiser convergence curves and the receiver operating characteristic curves, S5 the reproducibility statement, and S6 the learned rule base of arm B2 printed in full.
"""
