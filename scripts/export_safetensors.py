#!/usr/bin/env python3
"""Export trained runs to safetensors packs for the Quantiloom C++ renderer.

Each run directory becomes one flat file `<band>_<geom>_<net>.safetensors`
(geom: horizontal->ground, slant_to_ground->slant, sky->sky). The file is
fully self-contained: EMA weights + per-target normalization arrays as F32
tensors (log_mask as U8), plus JSON metadata strings carrying the input
spec, band grid, targets and provenance. The C++ loader never needs the
original run dir, manifest or any Python artifact.

Self-check: every exported file is reloaded and a standalone numpy forward
(mirroring the C++ algorithm: feature assembly -> fp32 ResMLP -> float64
inverse transform) is compared against the torch reference forward on
random in-domain parameter rows.

Usage:
  python scripts/export_safetensors.py --all runs/ --out-dir export/atmos_models
  python scripts/export_safetensors.py --run-dir runs/lwir_ground_v1_tau_256x4_muon
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmoemu.manifest import slant_range_km          # noqa: E402
from atmoemu.model import build_model                 # noqa: E402
from atmoemu.transforms import InputSpec, OutputSpec  # noqa: E402

GEOM_NAME = {"horizontal": "ground", "slant_to_ground": "slant", "sky": "sky"}
OPAQUE_DELTA = 7.0
DELTA_CLAMP = 20.0


# ---------------------------------------------------------------- container --

def save_safetensors(path: Path, tensors: dict[str, np.ndarray],
                     metadata: dict[str, str]) -> None:
    """Write a safetensors file. Uses the safetensors package when present,
    otherwise writes the container by hand (8-byte LE header length + JSON
    header + raw little-endian blobs)."""
    try:
        from safetensors.numpy import save_file
        save_file(tensors, str(path), metadata=metadata)
        return
    except ImportError:
        pass
    dtype_map = {np.dtype(np.float32): "F32", np.dtype(np.uint8): "U8"}
    header: dict = {"__metadata__": metadata}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        dt = dtype_map[arr.dtype]
        nbytes = arr.nbytes
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        blobs.append(arr.tobytes())
        offset += nbytes
    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def load_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Minimal reader used by the self-check (independent of the writer)."""
    raw = path.read_bytes()
    (hlen,) = struct.unpack_from("<Q", raw, 0)
    header = json.loads(raw[8:8 + hlen].decode("utf-8"))
    meta = header.pop("__metadata__", {})
    base = 8 + hlen
    dtype_map = {"F32": np.float32, "U8": np.uint8}
    out = {}
    for name, info in header.items():
        o0, o1 = info["data_offsets"]
        arr = np.frombuffer(raw[base + o0:base + o1],
                            dtype=dtype_map[info["dtype"]])
        out[name] = arr.reshape(info["shape"])
    return out, meta


# ------------------------------------------------------------------- export --

def export_run(run_dir: Path, out_dir: Path, ckpt_name: str) -> Path:
    ck = torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=True)
    net, cfg, fp = ck["net"], ck["config"], ck["fingerprint"]
    band = fp["band"]
    geom = GEOM_NAME[fp["path_type"]]
    norm = json.loads((run_dir / f"norm_{net}.json").read_text(encoding="utf-8"))
    ospec = OutputSpec.load(run_dir / f"norm_{net}.targets.json",
                            run_dir / f"norm_{net}.npz")
    targets_meta = json.loads((run_dir / f"norm_{net}.targets.json")
                              .read_text(encoding="utf-8"))

    tensors: dict[str, np.ndarray] = {}
    for k, v in ck["model_ema"].items():
        tensors[k] = np.ascontiguousarray(v.detach().numpy().astype(np.float32))
    if cfg["pca_mode"] == "coeff":
        z = np.load(run_dir / f"pca_{net}.npz")
        tensors["pca.basis"] = np.ascontiguousarray(z["basis"].astype(np.float32))
        tensors["pca.mean"] = np.ascontiguousarray(z["mean"].astype(np.float32))
    for t, r in enumerate(ospec.rows):
        tensors[f"norm.{t}.mean"] = r.mean.astype(np.float32)
        tensors[f"norm.{t}.std"] = r.std.astype(np.float32)
        tensors[f"norm.{t}.log_eps"] = r.log_eps.astype(np.float32)
        tensors[f"norm.{t}.log_mask"] = r.log_mask.astype(np.uint8)

    model_json = {"d_in": cfg["d_in"], "d_out": cfg["d_out"],
                  "width": cfg["width"], "blocks": cfg["blocks"],
                  "pca_mode": cfg["pca_mode"],
                  "n_pc": int(tensors["proj.weight"].shape[0])
                  if "proj.weight" in tensors else 0}
    metadata = {
        "format_version": "1",
        "net": net,
        "path_type": fp["path_type"],
        "band_json": json.dumps(band),
        "model_json": json.dumps(model_json),
        "input_spec_json": json.dumps(norm["input"]),
        "targets_json": json.dumps(targets_meta),
        "feature_names_json": json.dumps(fp["feature_names"]),
        "sampled_json": json.dumps(fp["sampled"]),
        "opaque_delta": str(OPAQUE_DELTA),
        "delta_clamp": str(DELTA_CLAMP),
        "source_run": run_dir.name,
        "epoch": str(ck.get("epoch", -1)),
        "val_loss": str(ck.get("val_loss", float("nan"))),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{band['name']}_{geom}_{net}.safetensors"
    save_safetensors(out, tensors, metadata)
    return out


# --------------------------------------------------------------- self-check --

class NumpyNet:
    """Standalone forward built ONLY from the exported file; mirrors the
    C++ implementation (fp32 network, float64 transforms)."""

    def __init__(self, path: Path):
        self.tensors, self.meta = load_safetensors(path)
        self.model = json.loads(self.meta["model_json"])
        self.ispec = json.loads(self.meta["input_spec_json"])
        self.targets = json.loads(self.meta["targets_json"])
        self.K = int(self.targets["K"])

    def assemble(self, P: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        cols = []
        for e in self.ispec["entries"]:
            x = P[:, e["col"]]
            kind = e["kind"]
            if kind == "onehot":
                for v in e["values"]:
                    cols.append((np.abs(x - v) < 1e-9).astype(np.float64))
                continue
            if kind == "log":
                x = np.log(np.maximum(x, 1e-300))
            elif kind == "cos_deg":
                x = np.cos(np.radians(x))
            lo, hi = e["lo"], e["hi"]
            cols.append(2.0 * (x - lo) / (hi - lo) - 1.0)
        return np.stack(cols, axis=1).astype(np.float32)

    def net_forward(self, X: np.ndarray) -> np.ndarray:
        t = self.tensors
        h = X @ t["stem.weight"].T + t["stem.bias"]
        for i in range(self.model["blocks"]):
            p = f"blocks.{i}."
            a = h @ t[p + "fc1.weight"].T + t[p + "fc1.bias"]
            mu = a.mean(axis=1, keepdims=True)
            var = ((a - mu) ** 2).mean(axis=1, keepdims=True)
            a = (a - mu) / np.sqrt(var + 1e-5)
            a = a * t[p + "norm.weight"] + t[p + "norm.bias"]
            a = a * (1.0 / (1.0 + np.exp(-a)))  # SiLU
            a = h + (a @ t[p + "fc2.weight"].T + t[p + "fc2.bias"])
            h = a.astype(np.float32)
        if "proj.weight" in t:
            h = h @ t["proj.weight"].T + t["proj.bias"]
        if "head.weight" in t:
            h = h @ t["head.weight"].T + t["head.bias"]
        if self.model["pca_mode"] == "coeff":
            h = h @ t["pca.basis"].T + t["pca.mean"]
        return h.astype(np.float32)

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        rows = self.targets["rows"]
        Z = np.asarray(Z, dtype=np.float64).reshape(-1, len(rows), self.K)
        out = np.empty_like(Z)
        for ti, row in enumerate(rows):
            mean = self.tensors[f"norm.{ti}.mean"].astype(np.float64)
            std = self.tensors[f"norm.{ti}.std"].astype(np.float64)
            log_eps = self.tensors[f"norm.{ti}.log_eps"].astype(np.float64)
            log_mask = self.tensors[f"norm.{ti}.log_mask"].astype(bool)
            y = Z[:, ti] * std + mean
            if log_mask.any():
                y = np.where(log_mask, np.exp(y) - log_eps, y)
            if row["kind"] != "delta":
                y = np.maximum(y, 0.0)
                ceiling = np.where(log_mask, log_eps * 1e4 * 2.0, np.inf)
                y = np.minimum(y, ceiling)
            out[:, ti] = y
        return out

    def forward(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.net_forward(self.assemble(P))
        return z, self.inverse(z)


def sample_params(meta: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random in-domain parameter rows in the manifest feature layout."""
    sampled = json.loads(meta["sampled_json"])
    names = json.loads(meta["feature_names_json"])
    path_type = meta["path_type"]
    P = np.zeros((n, len(names)))
    vals = {}
    for i, (name, dist) in enumerate(sampled.items()):
        if "values" in dist:
            v = rng.choice(np.asarray(dist["values"], dtype=np.float64), size=n)
        elif "log_uniform" in dist:
            lo, hi = dist["log_uniform"]
            v = np.exp(rng.uniform(math.log(lo), math.log(hi), size=n))
        else:
            lo, hi = dist["uniform"]
            v = rng.uniform(lo, hi, size=n)
        P[:, i] = v
        vals[name] = v
    ns = len(sampled)
    h1 = vals.get("h1_km", np.full(n, 0.1))
    theta = vals.get("view_zenith_deg", np.full(n, 90.0))
    P[:, ns + 0] = h1
    if path_type == "horizontal":
        P[:, ns + 1] = h1
        P[:, ns + 2] = math.cos(math.radians(90.0))
        P[:, ns + 3] = vals.get("range_km", np.full(n, 1.0))
    elif path_type == "slant_to_ground":
        P[:, ns + 1] = 0.0
        P[:, ns + 2] = np.cos(np.radians(theta))
        P[:, ns + 3] = [slant_range_km(a, b) for a, b in zip(h1, theta)]
    else:  # sky
        P[:, ns + 1] = 100.0
        P[:, ns + 2] = np.cos(np.radians(theta))
        P[:, ns + 3] = 0.0
    return P


def torch_reference(run_dir: Path, ckpt_name: str):
    """Reference forward identical to scripts/infer.py::load_run (CPU)."""
    ck = torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=True)
    net, cfg = ck["net"], ck["config"]
    norm = json.loads((run_dir / f"norm_{net}.json").read_text(encoding="utf-8"))
    ispec = InputSpec.from_json(norm["input"])
    ospec = OutputSpec.load(run_dir / f"norm_{net}.targets.json",
                            run_dir / f"norm_{net}.npz")
    pb = pm = None
    if cfg["pca_mode"] != "none":
        z = np.load(run_dir / f"pca_{net}.npz")
        pb, pm = z["basis"], z["mean"]
    model = build_model(cfg, pca_basis=pb, pca_mean=pm)
    model.load_state_dict(ck["model_ema"])
    model.eval()

    ceilings = {}
    for t, r in enumerate(ospec.rows):
        if r.kind != "delta" and r.log_mask is not None and r.log_mask.any():
            ceilings[t] = np.where(r.log_mask, r.log_eps * 1e4 * 2.0, np.inf)

    def forward(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = torch.from_numpy(ispec.apply(P))
        with torch.no_grad():
            z = model(X).float()
            if cfg["pca_mode"] == "coeff":
                z = z @ torch.as_tensor(pb, dtype=torch.float32).T \
                    + torch.as_tensor(pm, dtype=torch.float32)
        zn = z.numpy()
        Y = ospec.inverse(zn)
        for t, r in enumerate(ospec.rows):
            if r.kind != "delta":
                Y[:, t] = np.maximum(Y[:, t], 0.0)
                if t in ceilings:
                    Y[:, t] = np.minimum(Y[:, t], ceilings[t])
        return zn, Y

    return forward


def self_check(run_dir: Path, out_path: Path, ckpt_name: str, n: int = 64,
               z_tol: float = 1e-4, seed: int = 0) -> None:
    npnet = NumpyNet(out_path)
    ref = torch_reference(run_dir, ckpt_name)
    rng = np.random.default_rng(seed)
    P = sample_params(npnet.meta, n, rng)
    z_np, Y_np = npnet.forward(P)
    z_th, Y_th = ref(P)
    dz = float(np.abs(z_np - z_th).max())
    scale = max(float(np.abs(Y_th).max()), 1e-30)
    dy = float(np.abs(Y_np - Y_th).max()) / scale
    status = "OK" if (dz <= z_tol and dy <= 1e-4) else "FAIL"
    print(f"  [check] {out_path.name}: max|dz|={dz:.3e} "
          f"max|dY|/peak={dy:.3e} ({status})")
    if status == "FAIL":
        raise SystemExit(f"self-check failed for {out_path}")


# --------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", nargs="*", default=None)
    ap.add_argument("--all", default=None, metavar="RUNS_ROOT",
                    help="export every run directory under RUNS_ROOT")
    ap.add_argument("--out-dir", default="export/atmos_models")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    if (args.run_dir is None) == (args.all is None):
        raise SystemExit("exactly one of --run-dir / --all is required")
    if args.all:
        run_dirs = sorted(p for p in Path(args.all).iterdir()
                          if (p / args.ckpt).exists())
    else:
        run_dirs = [Path(d) for d in args.run_dir]

    out_dir = Path(args.out_dir)
    for rd in run_dirs:
        out = export_run(rd, out_dir, args.ckpt)
        size_mb = out.stat().st_size / 2**20
        print(f"[export] {rd.name} -> {out.name} ({size_mb:.1f} MB)")
        if not args.no_check:
            self_check(rd, out, args.ckpt)
    print(f"[done] {len(run_dirs)} packs -> {out_dir}")


if __name__ == "__main__":
    main()
