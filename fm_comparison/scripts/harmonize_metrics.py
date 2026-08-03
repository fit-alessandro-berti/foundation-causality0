#!/usr/bin/env python3
"""Transparent conversions used in the cross-paper comparison.

These conversions do *not* turn different benchmarks into a head-to-head test.
They only express related errors on an approximate common scale.
"""
from __future__ import annotations
import argparse, math

NORMAL_MAE_TO_RMSE_RATIO = math.sqrt(2.0 / math.pi)  # 0.797884...

def rmse_to_mae_point(rmse: float) -> float:
    """MAE implied by zero-mean Gaussian errors."""
    return NORMAL_MAE_TO_RMSE_RATIO * rmse

def rmse_to_mae_sensitivity(rmse: float, low: float = 0.70, high: float = 0.90) -> tuple[float,float]:
    """A deliberately broad sensitivity band around the Gaussian conversion."""
    if not (0 <= low <= high <= 1):
        raise ValueError("Require 0 <= low <= high <= 1 because MAE <= RMSE.")
    return low * rmse, high * rmse

def pehe_to_ite_mae_proxy(pehe: float, low: float = 0.70, high: float = 0.90) -> tuple[float,float]:
    """PEHE is an ITE-error RMSE, so use the same RMSE-to-MAE sensitivity band."""
    return rmse_to_mae_sensitivity(pehe, low, high)

def ate_error_upper_bound_from_pehe(pehe: float) -> tuple[float,float]:
    """By |mean(e)| <= sqrt(mean(e^2)), absolute ATE error is at most PEHE."""
    return 0.0, pehe

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("value",type=float)
    p.add_argument("--kind",choices=["rmse","pehe"],required=True)
    p.add_argument("--low",type=float,default=0.70)
    p.add_argument("--high",type=float,default=0.90)
    a=p.parse_args()
    lo,hi=rmse_to_mae_sensitivity(a.value,a.low,a.high)
    print(f"MAE sensitivity band: [{lo:.6g}, {hi:.6g}]")
    print(f"Gaussian point conversion: {rmse_to_mae_point(a.value):.6g}")
    if a.kind=="pehe":
        print(f"ATE absolute-error bound on the same sample/scale: [0, {a.value:.6g}]")

if __name__=="__main__": main()
