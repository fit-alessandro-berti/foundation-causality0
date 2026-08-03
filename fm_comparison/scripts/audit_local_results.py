#!/usr/bin/env python3
"""Audit result artifacts in the uploaded repository.

Finds CSV/TSV/JSON/NPZ files containing a plausible ground-truth causal-effect
column and prediction column, then computes scale-aware error metrics. It does
not execute model training or inference.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, math, re
from typing import Iterable
import numpy as np
import pandas as pd

TRUE_PATTERNS = [
    r"^(true|gt|ground_truth)(_.*)?$", r"^(y_?true|target)$",
    r"^(true_?)?(ate|cate|ite|effect)$", r"^(ate|cate|ite)_?(true|gt)$",
]
PRED_PATTERNS = [
    r"^(pred|prediction|estimate|estimated)(_.*)?$", r"^y_?pred$",
    r"^(predicted|estimated)_?(ate|cate|ite|effect)$",
    r"^(ate|cate|ite)_?(pred|hat|estimate|estimated)$", r"^tau_?hat$",
]

def match_column(columns: Iterable[str], patterns: list[str]) -> str | None:
    cols=[str(c) for c in columns]
    for pat in patterns:
        rx=re.compile(pat,re.I)
        for c in cols:
            if rx.match(c.strip()): return c
    return None

def metrics(y: np.ndarray, p: np.ndarray) -> dict[str,float|int|None]:
    mask=np.isfinite(y)&np.isfinite(p)
    y=y[mask].astype(float); p=p[mask].astype(float)
    if len(y)==0: return {}
    e=p-y
    mae=float(np.mean(np.abs(e)))
    rmse=float(np.sqrt(np.mean(e*e)))
    bias=float(np.mean(e))
    std=float(np.std(y,ddof=0))
    mean_abs=float(np.mean(np.abs(y)))
    denom=float(np.sum((y-y.mean())**2))
    r2=(1.0-float(np.sum(e*e))/denom) if denom>0 else None
    return {
        "n":int(len(y)),"mae":mae,"rmse":rmse,"bias":bias,"r2":r2,
        "true_mean":float(np.mean(y)),"true_std":std,"true_mean_abs":mean_abs,
        "nmae_by_true_sd":mae/std if std>0 else None,
        "relative_mae_by_mean_abs_true":mae/mean_abs if mean_abs>0 else None,
    }

def audit_frame(path: Path, df: pd.DataFrame) -> list[dict]:
    out=[]
    t=match_column(df.columns,TRUE_PATTERNS)
    p=match_column(df.columns,PRED_PATTERNS)
    if t and p:
        vals=metrics(pd.to_numeric(df[t],errors="coerce").to_numpy(),
                     pd.to_numeric(df[p],errors="coerce").to_numpy())
        if vals: out.append({"path":str(path),"true_col":t,"pred_col":p,**vals})
    # Also test pairs where column names share a causal-effect stem.
    for stem in ("ate","cate","ite","effect","tau"):
        tc=[c for c in df.columns if stem in str(c).lower() and re.search(r"true|gt|actual|target",str(c),re.I)]
        pc=[c for c in df.columns if stem in str(c).lower() and re.search(r"pred|hat|estim",str(c),re.I)]
        for a in tc:
            for b in pc:
                vals=metrics(pd.to_numeric(df[a],errors="coerce").to_numpy(),
                             pd.to_numeric(df[b],errors="coerce").to_numpy())
                if vals: out.append({"path":str(path),"true_col":str(a),"pred_col":str(b),**vals})
    # deduplicate
    seen=set(); uniq=[]
    for r in out:
        k=(r['path'],r['true_col'],r['pred_col'])
        if k not in seen: seen.add(k); uniq.append(r)
    return uniq

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("repo",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    rows=[]; errors=[]
    for path in args.repo.rglob("*"):
        if not path.is_file() or path.stat().st_size>200_000_000: continue
        try:
            if path.suffix.lower()==".csv": rows += audit_frame(path,pd.read_csv(path))
            elif path.suffix.lower() in {".tsv",".tab"}: rows += audit_frame(path,pd.read_csv(path,sep="\t"))
            elif path.suffix.lower()==".json":
                obj=json.loads(path.read_text(encoding="utf-8",errors="replace"))
                if isinstance(obj,list) and obj and isinstance(obj[0],dict): rows += audit_frame(path,pd.DataFrame(obj))
                elif isinstance(obj,dict):
                    # dict of equal-length arrays
                    arr={k:v for k,v in obj.items() if isinstance(v,list)}
                    lens={len(v) for v in arr.values()}
                    if arr and len(lens)==1: rows += audit_frame(path,pd.DataFrame(arr))
            elif path.suffix.lower()==".npz":
                z=np.load(path,allow_pickle=False)
                arr={k:z[k] for k in z.files if np.asarray(z[k]).ndim==1}
                if arr and len({len(v) for v in arr.values()})==1: rows += audit_frame(path,pd.DataFrame(arr))
        except Exception as exc:
            errors.append({"path":str(path),"error":type(exc).__name__+": "+str(exc)})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({"candidates":rows,"errors":errors},indent=2),encoding="utf-8")
    csv_path=args.out.with_suffix('.csv')
    pd.DataFrame(rows).to_csv(csv_path,index=False)

if __name__=="__main__": main()
