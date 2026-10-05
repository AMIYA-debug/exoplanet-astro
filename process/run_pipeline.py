"""Command-line runner for the exoplanet transit detection pipeline.

Usage:
    python process/run_pipeline.py
    python process/run_pipeline.py --kic 12885212
    python process/run_pipeline.py --kic 12885212 12935144
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process.pipeline import process_kic


DEFAULT_FITS_ROOT = Path("/home/amiya/projects/astronitr/data/raw/fits")
DEFAULT_OUTPUT_ROOT = Path("/home/amiya/projects/astronitr/process/data")
DEMO_KICS = ["12885212", "12935144"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exoplanet transit detection pipeline on combined Kepler FITS."
    )
    parser.add_argument(
        "--kic",
        nargs="*",
        default=DEMO_KICS,
        help="One or more KIC IDs to process (default: 12885212 12935144).",
    )
    parser.add_argument(
        "--fits-root",
        type=Path,
        default=DEFAULT_FITS_ROOT,
        help=f"Root directory of raw FITS (default: {DEFAULT_FITS_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for processed data (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--no-mcmc",
        action="store_true",
        help="Skip MCMC sampling stage.",
    )
    parser.add_argument(
        "--mcmc-steps",
        type=int,
        default=2500,
        help="Number of MCMC steps per walker (default: 2500).",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=50000,
        help="Number of BLS period grid points (default: 50000).",
    )

    args = parser.parse_args()

    results = []
    for kic_str in args.kic:
        fits_file = args.fits_root / str(kic_str) / "combined.fits"
        if not fits_file.is_file():
            print(f"Error: FITS file not found for KIC {kic_str} at {fits_file}", file=sys.stderr)
            continue

        target_out_dir = args.output_root / str(kic_str)
        summary = process_kic(
            fits_path=fits_file,
            output_dir=target_out_dir,
            kic_id=str(kic_str),
            run_mcmc=not args.no_mcmc,
            n_periods=args.n_periods,
            mcmc_steps=args.mcmc_steps,
        )
        results.append(summary)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    for res in results:
        kic = res["kic"]
        bls = res["bls"]
        model = res["transit_model"]
        mcmc = res.get("mcmc", {})
        val = res.get("validation", {})

        print(f"\n--- KIC {kic} ---")
        print(f"  Observations:       {res['observations']['processed']:,} / {res['observations']['raw']:,}")
        print(f"  BLS Best Period:    {bls['BEST_PERIOD']:.5f} days")
        print(f"  BLS Epoch (t0):     {bls['BEST_TRANSIT_TIME']:.5f} BKJD")
        print(f"  BLS Peak Power:     {bls['BEST_POWER']:.6f}")
        print(f"  BLS Transit Depth:  {bls['BEST_DEPTH']*1e6:.1f} ppm")
        print(f"  BATMAN Rp/R*:       {model['k_rp_rstar']:.4f} +/- {model['k_err']:.4f}")
        print(f"  BATMAN Depth:       {model['transit_depth_ppm']:.1f} ppm")
        print(f"  BATMAN Duration:    {model['transit_duration_hours']:.2f} hours")
        if mcmc and "k" in mcmc:
            print(f"  MCMC Rp/R*:         {mcmc['k']['median']:.4f} (-{mcmc['k']['err_minus']:.4f}, +{mcmc['k']['err_plus']:.4f})")
            print(f"  MCMC Depth:         {mcmc['transit_depth_ppm']:.1f} ppm")
        if val and "consistent_with_catalog" in val:
            ref = val["catalog_reference"]
            print(f"  NASA Catalog Check: Consistent={val['consistent_with_catalog']} (Catalog Period: {ref['period_days']:.5f} d, Depth: {ref['depth_ppm']:.1f} ppm)")
    print("=" * 80)


if __name__ == "__main__":
    main()
