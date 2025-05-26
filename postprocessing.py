#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse
import os

def convert_and_smooth(df, mmperpix, window_length, polyorder):
    """
    - Converts px → mm, px/s → mm/s, px/s² → mm/s² in-place.
    - Computes smoothed versions in new columns `*_smooth`.
    """
    kinematic_cols = [
        ('position_x', 'mm'),
        ('position_y', 'mm'),
        ('velocity_x', 'mm/s'),
        ('velocity_y', 'mm/s'),
        ('acceleration_x', 'mm/s²'),
        ('acceleration_y', 'mm/s²'),
    ]

    # 1) Convert units
    for col, _unit in kinematic_cols:
        df[col] = df[col] * mmperpix

    # 2) Smooth
    for col, _unit in kinematic_cols:
        df[f'{col}_smooth'] = savgol_filter(df[col], window_length, polyorder)

    return df

def plot_pair(df, raw_x, raw_y, ylabel, title, outfile):
    """
    Plots raw (dashed) and smoothed (solid) for x & y on one figure.
    """
    t = df['time']
    plt.figure()
    # raw
    plt.plot(t, df[raw_x],    '--', label=f'{raw_x} raw')
    plt.plot(t, df[raw_y],    '--', label=f'{raw_y} raw')
    # smooth
    plt.plot(t, df[f'{raw_x}_smooth'], '-', label=f'{raw_x} smooth')
    plt.plot(t, df[f'{raw_y}_smooth'], '-', label=f'{raw_y} smooth')

    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {outfile}")
    
    plt.figure()
    # raw
    # plt.plot(t, df[raw_x],    '--', label=f'{raw_x} raw')
    # plt.plot(t, df[raw_y],    '--', label=f'{raw_y} raw')
    # smooth
    plt.plot(t, df[f'{raw_x}_smooth'], '-', label=f'{raw_x} smooth')
    plt.plot(t, df[f'{raw_y}_smooth'], '-', label=f'{raw_y} smooth')

    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(outfile.replace('.png', '_smoothed_only.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {outfile.replace('.png', '_smoothed_only.png')}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert px→mm, smooth kinematics, export CSV and PNGs"
    )
    parser.add_argument("csv_file",
                        help="Input CSV (time + position_x/y, velocity_x/y, acceleration_x/y in px-units)")
    parser.add_argument("--start", type=float, default=11,
                        help="Start time (s)")
    parser.add_argument("--end",   type=float, default=18,
                        help="End time (s)")
    parser.add_argument("--mmperpix", type=float, required=True,
                        help="Millimeters per pixel")
    parser.add_argument("--window", type=int, default=101,
                        help="Savgol window length (odd integer)")
    parser.add_argument("--poly",   type=int, default=3,
                        help="Savgol polynomial order (≤ window-1)")
    args = parser.parse_args()

    # — Load & time‐trim —
    df = pd.read_csv(args.csv_file, comment='#')
    df = df[(df.time >= args.start) & (df.time <= args.end)].reset_index(drop=True)
    n = len(df)

    # — Sanitize window & polyorder —
    w = args.window
    if w > n:
        w = n if n % 2 == 1 else n - 1
    p = min(args.poly, w-1)

    # — Convert & smooth —
    df = convert_and_smooth(df, args.mmperpix, w, p)

    # — Export single CSV —
    base = os.path.splitext(os.path.basename(args.csv_file))[0]
    out_csv = f"{base}_mm_converted_smoothed.csv"
    df.to_csv(out_csv, index=False)
    print(f"Exported CSV: {out_csv}")

    # — Plotting —
    plot_pair(
        df,
        raw_x='position_x', raw_y='position_y',
        ylabel='Position (mm)',
        title='Position',
        outfile=f'{base}_position.png'
    )
    plot_pair(
        df,
        raw_x='velocity_x', raw_y='velocity_y',
        ylabel='Velocity (mm/s)',
        title='Velocity',
        outfile=f'{base}_velocity.png'
    )
    plot_pair(
        df,
        raw_x='acceleration_x', raw_y='acceleration_y',
        ylabel='Acceleration (mm/s²)',
        title='Acceleration',
        outfile=f'{base}_acceleration.png'
    )

if __name__ == "__main__":
    main()
