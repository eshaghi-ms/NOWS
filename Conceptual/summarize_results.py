#!/usr/bin/env python3
"""
summarize_results.py

Scan a directory for .out files (non-recursively by default), parse hyperparameters
from filenames (e.g. _s64_m8_) and extract training/testing statistics:
  - The average testing error is ...
  - Std. deviation of testing error is ...
  - Min testing error is ...
  - Max testing error is ...

Writes summary CSV and prints a table. Optionally plots avg testing error vs modes.

Usage:
    python summarize_results.py            # current folder, no subfolders
    python summarize_results.py --dir ./outs --recursive --plot
"""

import re
import argparse
import os
import glob
import csv
from collections import OrderedDict

try:
    import pandas as pd
except Exception:
    pd = None


# -------------------------
# File discovery & parsing
# -------------------------
def find_out_files(directory, recursive=False):
    if recursive:
        pattern = os.path.join(directory, '**', '*.out')
        return glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory, '*.out')
        return glob.glob(pattern)


def parse_filename_basics(path):
    base = os.path.basename(path)
    if base.endswith('.out'):
        base_noout = base[:-4]
    else:
        base_noout = base

    idnum = None
    m = None
    s = None
    tag = None
    model = None

    parts = base_noout.rsplit('.', 1)
    if len(parts) == 2 and parts[1].isdigit():
        namepart = parts[0]
        idnum = parts[1]
    else:
        namepart = base_noout

    s_match = re.search(r'_s(\d+)', namepart)
    m_match = re.search(r'_m(\d+)', namepart)
    if s_match:
        s = int(s_match.group(1))
    if m_match:
        m = int(m_match.group(1))

    model_match = re.match(r'^(.*?)(?:\.py)?_s', namepart)
    if model_match:
        model = model_match.group(1)
    else:
        model = namepart.split('_')[0]

    tag_match = re.search(r'_m\d+_([^_]*)$', namepart)
    if tag_match:
        tag = tag_match.group(1)
    else:
        tag_match2 = re.search(r'_m\d+_(.+)$', namepart)
        if tag_match2:
            tag = tag_match2.group(1)

    return dict(model=model, s=s, m=m, tag=tag, idnum=idnum, filename=base, filepath=path)


# -------------------------
# Regex to capture stats
# -------------------------
RE_AVG_TEST = re.compile(r'The\s+average\s+testing\s+error\s+is\s+([0-9Ee+.\-]+)', re.IGNORECASE)
RE_STD_TEST = re.compile(r'Std\.?\s*(?:deviation\s+of\s+)?testing\s+error\s+(?:is|=)\s+([0-9Ee+.\-]+)', re.IGNORECASE)
RE_MIN_TEST = re.compile(r'Min(?:imum)?\s+testing\s+error\s+is\s+([0-9Ee+.\-]+)', re.IGNORECASE)
RE_MAX_TEST = re.compile(r'Max(?:imum)?\s+testing\s+error\s+is\s+([0-9Ee+.\-]+)', re.IGNORECASE)

RE_AVG_TRAIN = re.compile(r'The\s+average\s+training\s+error\s+is\s+([0-9Ee+.\-]+)', re.IGNORECASE)
RE_STD_TRAIN = re.compile(r'Std\.?\s*(?:deviation\s+of\s+)?training\s+error\s+(?:is|=)\s+([0-9Ee+.\-]+)', re.IGNORECASE)
RE_MIN_TRAIN = re.compile(r'Min(?:imum)?\s+training\s+error\s+is\s+([0-9Ee+.\-]+)', re.IGNORECASE)
RE_MAX_TRAIN = re.compile(r'Max(?:imum)?\s+training\s+error\s+is\s+([0-9Ee+.\-]+)', re.IGNORECASE)

RE_PARAMS = re.compile(r'Our model has\s+([0-9]+)\s+parameters', re.IGNORECASE)


def extract_stats_from_file(path):
    stats = {
        'avg_testing_error': None,
        'std_testing_error': None,
        'min_testing_error': None,
        'max_testing_error': None,
        'avg_training_error': None,
        'std_training_error': None,
        'min_training_error': None,
        'max_training_error': None,
        'num_parameters': None,
    }
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return stats

    m = RE_AVG_TEST.search(text)
    if m:
        stats['avg_testing_error'] = float(m.group(1))
    m = RE_STD_TEST.search(text)
    if m:
        stats['std_testing_error'] = float(m.group(1))
    m = RE_MIN_TEST.search(text)
    if m:
        stats['min_testing_error'] = float(m.group(1))
    m = RE_MAX_TEST.search(text)
    if m:
        stats['max_testing_error'] = float(m.group(1))

    m = RE_AVG_TRAIN.search(text)
    if m:
        stats['avg_training_error'] = float(m.group(1))
    m = RE_STD_TRAIN.search(text)
    if m:
        stats['std_training_error'] = float(m.group(1))
    m = RE_MIN_TRAIN.search(text)
    if m:
        stats['min_training_error'] = float(m.group(1))
    m = RE_MAX_TRAIN.search(text)
    if m:
        stats['max_training_error'] = float(m.group(1))
    m = RE_PARAMS.search(text)
    if m:
        stats['num_parameters'] = int(m.group(1))

    return stats


# -------------------------
# Main summarizer
# -------------------------
def summarize(directory, out_csv='summary_results.csv', recursive=False, top=None, plot=False):
    files = find_out_files(directory, recursive)
    if not files:
        print("No .out files found in", directory)
        return

    rows = []
    for path in sorted(files):
        basic = parse_filename_basics(path)
        stats = extract_stats_from_file(path)
        row = OrderedDict()
        row.update(basic)
        row.update(stats)
        rows.append(row)

    fieldnames = list(rows[0].keys())
    try:
        with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    except Exception as e:
        print("Failed to write CSV:", e)
        out_csv = None

    if pd:
        df = pd.DataFrame(rows)

        # Sort by s, then m, then model
        if all(col in df.columns for col in ['s', 'm', 'model']):
            df_sorted = df.sort_values(
                by=['s', 'm', 'model'],
                ascending=[True, True, True],
                na_position='last'
            )
        else:
            df_sorted = df

        pd.set_option('display.max_rows', 200)
        pd.set_option('display.max_colwidth', 50)
        print("\nSummary (first 50 rows):")
        print(df_sorted.head(50).to_string(index=False))

        if top:
            print(f"\nTop {top} rows after sorting by s, m, model:")
            print(df_sorted.head(top).to_string(index=False))
    else:
        print("\nSummary (no pandas installed):")
        for r in rows[:50]:
            print(r)

    if out_csv:
        print(f"\nCSV written to: {out_csv}")

    # -------------------------
    # Plotting
    # -------------------------
    if plot:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.size": 14,  # base font size for everything
            "axes.titlesize": 18,  # title font size
            "axes.labelsize": 14,  # x/y axis label font size
            "xtick.labelsize": 14,  # x-axis tick labels
            "ytick.labelsize": 14,  # y-axis tick labels
            "legend.fontsize": 14,  # legend font size
        })
        plt.rcParams['font.family'] = "DejaVu Serif"
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import PercentFormatter
            plt.rcParams['font.family'] = "DejaVu Serif"
            plt.rcParams['font.size'] = 14
            df_plot = pd.DataFrame(rows) if pd else None
            if df_plot is None:
                print("Plotting requires pandas + matplotlib.")
            else:
                df_plot = df_plot.dropna(subset=['avg_testing_error', 'm', 's', 'model'])
                if df_plot.empty:
                    print("No usable data found for plotting, skipping plot.")
                else:
                    df_plot['m'] = pd.to_numeric(df_plot['m'], errors='coerce')
                    df_plot['s'] = pd.to_numeric(df_plot['s'], errors='coerce')

                    # Normalize model labels
                    def model_label(row):
                        if row['model'] == 'FNO_Darcy2D':
                            return 'FNO_Darcy2D'
                        elif row['model'] == 'VINO_Darcy2D' and row['tag'] == 'phy':
                            return 'VINO_Darcy2D (phy)'
                        elif row['model'] == 'VINO_Darcy2D' and row['tag'] == 'phy_data':
                            return 'VINO_Darcy2D (phy_data)'
                        else:
                            return None

                    df_plot['model_group'] = df_plot.apply(model_label, axis=1)
                    df_plot = df_plot.dropna(subset=['model_group'])

                    # Define line styles per model
                    line_styles = {
                        'FNO_Darcy2D': '-',
                        'VINO_Darcy2D (phy)': '--',
                        'VINO_Darcy2D (phy_data)': ':'
                    }
                    colors = {
                        'FNO_Darcy2D': 'blue',
                        'VINO_Darcy2D (phy)': 'red',
                        'VINO_Darcy2D (phy_data)': 'black'
                    }
                    line_widths = {
                        'FNO_Darcy2D': 2,
                        'VINO_Darcy2D (phy)': 2,
                        'VINO_Darcy2D (phy_data)': 2.5
                    }

                    resolutions = sorted(df_plot['s'].dropna().unique())
                    ncols = min(len(resolutions), 3)
                    nrows = int((len(resolutions) + ncols - 1) / ncols)

                    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 3 * nrows), squeeze=False)

                    for idx, s_val in enumerate(resolutions):
                        ax = axes[idx // ncols, idx % ncols]
                        subdf = df_plot[df_plot['s'] == s_val]

                        for model_name, g in subdf.groupby('model_group'):
                            g_sorted = g.sort_values('m')
                            ax.plot(
                                g_sorted['m'],
                                g_sorted['avg_testing_error'] * 100,
                                line_styles.get(model_name, '-'),
                                marker='o',
                                color=colors.get(model_name, 'black'),
                                linewidth=line_widths.get(model_name, 1.5),
                                label=model_name
                            )

                        # Apply formatting
                        ax.set_title(f"Resolution s={s_val}")
                        ax.set_xlabel("Number of modes")
                        ax.set_ylabel("Average testing error")
                        ax.set_xticks(sorted(subdf['m'].unique()))  # only actual m values
                        ax.set_ylim(0, 2.0)  # fix y range
                        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                        ax.grid(True)
                        # ax.legend()

                    plt.tight_layout()
                    plot_path = os.path.join(directory, 'avg_test_error_by_model.png')
                    plt.savefig(plot_path, bbox_inches='tight', dpi=1200)
                    plt.show()
                    plt.close()
                    print("Plot saved to:", plot_path)
        except Exception as e:
            print("Plot generation failed:", e)


# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Summarize .out files with testing stats.")
    p.add_argument('--dir', '-d', default='./logs', help='Directory to search for .out files (default: current dir)')
    p.add_argument('--out', '-o', default='plots/summary_results.csv', help='CSV output filename')
    p.add_argument('--recursive', action='store_true',
                   help='Search directories recursively (enable to include subfolders)')
    p.add_argument('--top', '-t', type=int, default=None, help='Print top N results (after sorting by s, m, model)')
    p.add_argument('--plot', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True,
                   help='Enable/disable plotting (default: True). Use --plot False to turn off.')
    args = p.parse_args()

    summarize(args.dir, out_csv=args.out, recursive=args.recursive, top=args.top, plot=args.plot)


if __name__ == '__main__':
    main()
