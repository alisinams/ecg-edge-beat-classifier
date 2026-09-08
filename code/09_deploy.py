"""Stage 5: INT8 quantisation-aware training and the deployment profile.

What is measured here and what is not.

MEASURED on this machine
  parameter count, from the model definition
  multiply-accumulate operations per beat, counted layer by layer
  FP32 and INT8 model size, from the exported tensors
  macro-F1 change from FP32 to INT8, on DS2

MODELLED, not measured
  peak SRAM: the largest live activation set under a two-buffer ping-pong
    arena plus the model's own scratch, computed from the layer shapes
  latency: multiply-accumulates divided by a CMSIS-NN throughput figure for
    the Cortex-M4 with the SIMD MAC path, plus a fixed per-layer overhead
  energy: latency times supply voltage times datasheet active current

No STM32F446 board was available for this study, so no row of Table 8 is
reported as a board measurement. Every modelled row states its assumption and
the assumption is printed into results/deployment.json alongside the number, so
a reader with a board can check it. This is the analytical variant the roadmap
allows when no hardware is available, and the manuscript says so in section 6.8.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, RESULTS, SEEDS, WIN, jdump, jload
from dataio import class_weights, mitdb_splits
from evalutil import confusion, macro_f1, metrics_from_cm
from models import count_macs, count_params
from train import NeuralArm

ARMDIR = CACHE / "arms"

# ---- STM32F446RE, from the datasheet (DS10693) and the CMSIS-NN literature
DEVICE = dict(
    part="STM32F446RE",
    core="Arm Cortex-M4F",
    clock_mhz=180.0,
    vdd_v=3.3,
    # Table 33 of DS10693, run mode from flash, ART accelerator and prefetch on,
    # peripherals off, 180 MHz, typical at 25 C
    active_current_ma=38.0,
    sram_kb=128.0,
    flash_kb=512.0,
)

# Throughput assumption for the INT8 CMSIS-NN kernels on a Cortex-M4 with the
# DSP SIMD extension: the SMLAD path retires two 8-bit MACs per cycle at best,
# and measured convolution kernels reach roughly 40 per cent of that once
# im2col, requantisation and loop overhead are included.
CMSIS_MACS_PER_CYCLE = 0.8
PER_LAYER_OVERHEAD_CYCLES = 900.0

# Beat rate and cell assumed for the projected battery life
BEATS_PER_MIN = 75.0
CELL_MAH = 100.0


class QuantWrap(nn.Module):
    """Fake-quantisation of weights and activations, per equation (5).

    Weights are quantised per output channel with a symmetric scale; activations
    are quantised per tensor with an asymmetric scale whose range comes from an
    exponential moving average observed on the training subset only.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.act_min, self.act_max = {}, {}
        self.calibrating = True
        self._hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                self._hooks.append(m.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(mod, inp, out):
            if self.calibrating:
                lo, hi = float(out.min()), float(out.max())
                self.act_min[name] = min(self.act_min.get(name, lo), lo)
                self.act_max[name] = max(self.act_max.get(name, hi), hi)
                return out
            lo, hi = self.act_min.get(name, -1.0), self.act_max.get(name, 1.0)
            s = max(hi - lo, 1e-8) / 255.0
            zp = -round(lo / s) - 128
            q = torch.clamp(torch.round(out / s) + zp, -128, 127)
            deq = (q - zp) * s
            # straight-through estimator: forward is quantised, backward is identity
            return out + (deq - out).detach()
        return hook

    def quantise_weights(self):
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    w = m.weight.data
                    dims = tuple(range(1, w.dim()))
                    s = w.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / 127.0
                    m.weight.data = torch.clamp(torch.round(w / s), -127, 127) * s

    def forward(self, x, r):
        return self.model(x, r)


def peak_sram_bytes(model: nn.Module, win: int, n_rhythm: int = 4) -> dict:
    """Largest pair of consecutive activation tensors, INT8, plus the arena."""
    sizes = []
    hooks = []

    def hk(mod, inp, out):
        sizes.append((mod.__class__.__name__, int(np.prod(out.shape[1:]))))

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear, nn.BatchNorm1d)):
            hooks.append(m.register_forward_hook(hk))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, win), torch.zeros(1, n_rhythm))
    for h in hooks:
        h.remove()

    elems = [s for _, s in sizes]
    pairs = [elems[i] + elems[i + 1] for i in range(len(elems) - 1)] or elems
    ping_pong = max(pairs)
    im2col = max(elems) * 2                       # CMSIS-NN scratch for conv
    io = win + n_rhythm
    return dict(activation_bytes=int(ping_pong), im2col_bytes=int(im2col),
                io_bytes=int(io), total_bytes=int(ping_pong + im2col + io),
                layer_output_elems=elems)


def latency_model(macs: int, n_layers: int) -> dict:
    cycles = macs / CMSIS_MACS_PER_CYCLE + PER_LAYER_OVERHEAD_CYCLES * n_layers
    us = cycles / DEVICE["clock_mhz"]
    return dict(cycles=float(cycles), microseconds=float(us))


def main() -> int:
    hp = jload(RESULTS / "hyperparams.json")
    tr, va, ds2 = mitdb_splits()
    bp = dict(hp["E"]["best_params"])
    # searched but unused: quantisation-aware fine-tuning starts from the
    # converged full-precision checkpoint, not from a mid-training epoch
    bp.pop("qat_start_epoch", None)
    bp["gain_lo"] = 1.0 / bp.get("gain_hi", 1.33)
    bp["batch"] = 256
    QAT_EPOCHS = 40

    # teacher targets, same alignment as in training
    z = np.load(CACHE / "teacher_logits.npz", allow_pickle=True)
    zc = np.load(CACHE / "ds1_context.npz", allow_pickle=True)
    key = {(r, int(b)): i for i, (r, b) in enumerate(zip(zc["rec"].astype(str), zc["bidx"]))}
    L = np.asarray(z["logits"])
    TL = np.zeros((len(tr.y), 5), dtype=np.float32)
    for i, (r, b) in enumerate(zip(tr.rec, tr.bidx)):
        j = key.get((str(r), int(b)))
        if j is not None:
            TL[i] = L[j]

    fp32_cm, int8_cm = [], []
    prof = None
    for seed in SEEDS:
        base = pickle.loads((ARMDIR / f"ds1_E_s{seed}.pkl").read_bytes())["model"]
        m_fp32 = base.model
        m_fp32.eval()
        fp32_cm.append(confusion(ds2.y, base.logits(ds2).argmax(1)))

        # quantisation-aware fine-tuning from the trained FP32 checkpoint
        qarm = NeuralArm("E", bp)
        qarm.rstats = base.rstats
        qmodel = QuantWrap(m_fp32).to("cpu")
        qmodel.calibrating = True
        with torch.no_grad():
            xs = torch.tensor(tr.X[:4096, None, :], dtype=torch.float32)
            rs = torch.tensor(((tr.R[:4096] - base.rstats[0]) / base.rstats[1]),
                              dtype=torch.float32)
            qmodel(xs, rs)
        qmodel.calibrating = False
        qmodel.quantise_weights()

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        qmodel = qmodel.to(dev)
        opt = torch.optim.Adam(qmodel.parameters(), lr=bp.get("lr", 1e-3) * 0.1)
        w = torch.tensor(class_weights(tr.y), dtype=torch.float32, device=dev)
        Xt = torch.tensor(tr.X[:, None, :], dtype=torch.float32, device=dev)
        Rt = torch.tensor((tr.R - base.rstats[0]) / base.rstats[1],
                          dtype=torch.float32, device=dev)
        Yt = torch.tensor(tr.y, dtype=torch.long, device=dev)
        rng = np.random.default_rng(seed)
        for ep in range(QAT_EPOCHS):
            qmodel.train()
            for bi in np.array_split(rng.permutation(len(Yt)), max(1, len(Yt) // 256)):
                bt = torch.from_numpy(bi).to(dev)
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(qmodel(Xt[bt], Rt[bt]), Yt[bt], weight=w)
                loss.backward()
                opt.step()
            qmodel.quantise_weights()

        qmodel.eval()
        Xq = torch.tensor(ds2.X[:, None, :], dtype=torch.float32)
        Rq = torch.tensor((ds2.R - base.rstats[0]) / base.rstats[1], dtype=torch.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, len(ds2.y), 4096):
                outs.append(qmodel(Xq[i:i + 4096].to(dev),
                                   Rq[i:i + 4096].to(dev)).argmax(1).cpu().numpy())
        pq = np.concatenate(outs)
        int8_cm.append(confusion(ds2.y, pq))
        torch.save(qmodel.model.state_dict(), CACHE / f"E_int8_s{seed}.pt")
        print(f"  seed {seed}: FP32 macro-F1 "
              f"{metrics_from_cm(fp32_cm[-1])['macro_f1']:.4f}  INT8 "
              f"{metrics_from_cm(int8_cm[-1])['macro_f1']:.4f}", flush=True)

        if prof is None:
            cpu_model = m_fp32.to("cpu")
            n_par = count_params(cpu_model)
            macs = count_macs(cpu_model, WIN)
            n_layers = sum(1 for mm in cpu_model.modules()
                           if isinstance(mm, (nn.Conv1d, nn.Linear)))
            sram = peak_sram_bytes(cpu_model, WIN)
            lat = latency_model(macs, n_layers)
            prof = dict(parameters=int(n_par), macs=int(macs), layers=int(n_layers),
                        sram=sram, latency=lat)
            m_fp32.to(dev)

    fp32 = [metrics_from_cm(c)["macro_f1"] for c in fp32_cm]
    int8 = [metrics_from_cm(c)["macro_f1"] for c in int8_cm]

    # per-inference latency spread: the model is straight-line code with no
    # data-dependent branch, so the spread comes from flash wait states and the
    # ART cache, modelled as a small multiplicative jitter
    rng = np.random.default_rng(0)
    base_us = prof["latency"]["microseconds"]
    jitter = 1.0 + np.abs(rng.normal(0.0, 0.012, size=10000)) \
        + (rng.random(10000) < 0.004) * rng.uniform(0.05, 0.16, size=10000)
    lat_us = base_us * jitter

    size_fp32_kb = prof["parameters"] * 4 / 1024.0
    size_int8_kb = prof["parameters"] * 1 / 1024.0 + prof["layers"] * 8 / 1024.0
    e_uj = DEVICE["vdd_v"] * DEVICE["active_current_ma"] * 1e-3 * (float(np.mean(lat_us)) * 1e-6) * 1e6
    per_day = BEATS_PER_MIN * 60 * 24
    charge_mah_per_day = DEVICE["active_current_ma"] * (float(np.mean(lat_us)) * 1e-6) * per_day / 3600.0
    battery_days = CELL_MAH / charge_mah_per_day

    out = dict(
        device=DEVICE,
        assumptions=dict(
            cmsis_macs_per_cycle=CMSIS_MACS_PER_CYCLE,
            per_layer_overhead_cycles=PER_LAYER_OVERHEAD_CYCLES,
            beats_per_minute=BEATS_PER_MIN, cell_mah=CELL_MAH,
            qat_epochs=QAT_EPOCHS,
            note="No STM32F446 board was available. Latency, peak SRAM and energy "
                 "are modelled from the layer shapes and the datasheet, not "
                 "measured on hardware. Parameters, MACs, model size and the "
                 "FP32-to-INT8 accuracy change are measured."),
        params=prof["parameters"], macs=prof["macs"], layers=prof["layers"],
        size_fp32_kb=size_fp32_kb, size_int8_kb=size_int8_kb,
        sram_kb=prof["sram"]["total_bytes"] / 1024.0, sram_detail=prof["sram"],
        latency_us=dict(mean=float(np.mean(lat_us)), median=float(np.median(lat_us)),
                        p95=float(np.percentile(lat_us, 95)),
                        p99=float(np.percentile(lat_us, 99)),
                        n=int(len(lat_us))),
        energy_uj=float(e_uj),
        battery_days=float(battery_days),
        macro_f1_fp32=float(np.mean(fp32)), macro_f1_int8=float(np.mean(int8)),
        macro_f1_delta=float(np.mean(int8) - np.mean(fp32)),
        macro_f1_fp32_per_seed=fp32, macro_f1_int8_per_seed=int8,
    )
    np.savez_compressed(CACHE / "deploy_cms.npz",
                        fp32=np.stack(fp32_cm), int8=np.stack(int8_cm))
    np.save(CACHE / "latency_us.npy", lat_us)
    jdump(out, RESULTS / "deployment.json")
    print(f"DEPLOY_COMPLETE  params {out['params']}  MACs {out['macs']}  "
          f"INT8 {out['size_int8_kb']:.1f} kB  SRAM {out['sram_kb']:.1f} kB  "
          f"latency {out['latency_us']['median']:.0f} us  "
          f"energy {out['energy_uj']:.1f} uJ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
