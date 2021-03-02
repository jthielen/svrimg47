#!/usr/bin/env python
# Copyright (c) 2021 Jonathan Thielen.
# Distributed under the terms of the Apache 2.0 License
# SPDX-License-Identifier: Apache-2.0
"""
Download select MYRORSS data fields and repackage for use by SVRIMG-47.

The MYRORSS processing pipeline is severely bandwidth limited, so if you have extra machines
with a good data pipe to Dropbox/the OSF server, but not so great CPU/RAM for regridding, this
script can be used to speed up the process by subsetting the convective day tar files into new
archives with just the data needed.

Relies upon stored JSON mapping of convective days to hours we need to grab
data from for storm report samples (generatable using the
preprocess_reports_for_intermediate_files.ipynb notebook).
"""

# stdlib imports
import argparse
import json
import glob
import os
import sys
import subprocess
import warnings

# scientific library imports
import numpy as np
import pandas as pd

# Resource definitions
orig_tar_url = "https://files.osf.io/v1/resources/9gzp2/providers/dropbox/{dt:%Y}/MYRORSS_orig/{dt:%Y%m%d}.tar"
azshear_tar_url = "https://files.osf.io/v1/resources/9gzp2/providers/dropbox/{dt:%Y}/Azshear/{dt:%Y%m%d}.tar"


def search_hourly_exclude(timestamps, fieldname, return_hourly_range=False):
    """Search the base directory for file or files at this time for this field, and exclude others."""
    # Get files and their timestamps
    field_dir = f"{temp_myrorss_dir}{convective_day:%Y%m%d}/{fieldname}/00.25/"
    files = sorted(glob.glob(f"{field_dir}*.net*"))
    file_timestamps = pd.Series({pd.Timestamp(f.split("/")[-1].split(".")[0]): f for f in files})

    # Select file with nearest time to that specified
    good_files = []
    for timestamp in timestamps:
        nearest_idx = file_timestamps.index.get_loc(pd.Timestamp(timestamp), method='nearest')
        if return_hourly_range:
            nearest_idx_prior = file_timestamps.index.get_loc(pd.Timestamp(timestamp) - pd.Timedelta(1, "H"), method='nearest')
            good_files += file_timestamps.iloc[(nearest_idx_prior + 1):(nearest_idx + 1)].to_list()
        else:
            good_files.append(file_timestamps.iloc[nearest_idx])

    return list(set(file_timestamps) - set(good_files))


# Run the script
if __name__ == '__main__':
    #########################
    # Runtime Configuration #
    #########################

    parser = argparse.ArgumentParser(
        description="Repackage convective day of MYRORSS data into smaller archive for download efficency", usage=argparse.SUPPRESS
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="<Required> Output Directory",
        required=True,
        metavar="~/output/",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        help="<Required> Convective day to process",
        required=True,
        metavar="2015-07-03",
    )
    parser.add_argument(
        "-c", "--cache-dir", help="MYRORSS file directory", default="~/myrorss_cache/", metavar="~/myrorss_cache/"
    )
    parser.add_argument(
        "-H", "--hour-file", help="JSON file mapping convective days to hours to extract", default="~/research/severe_report_hours.json"
    )

    try:
        args = parser.parse_args()
        output_dir = args.output_dir
        convective_day = pd.Timestamp(args.timestamp)
        temp_myrorss_dir = args.cache_dir
        with open(args.hour_file, "r") as hour_file:
            analysis_hours = json.load(hour_file)[f"{convective_day:%Y-%m-%d}"]
    except (SystemExit, ValueError):
        parser.print_help()
        raise
    except (KeyError):
        print(f"{convective_day:%Y-%m-%d} has no severe report data to load, so skip...")
        sys.exit(0)

    ################################
    # Data Download and Extraction #
    ################################

    # Go to working directory
    os.chdir(temp_myrorss_dir)

    # Print runtime information
    print(f"\nPROCESSING {convective_day:%Y-%m-%d} ...\nHours to process:")
    for hour in analysis_hours:
        print(f"\t{hour}")
    print("\n")

    # Make local dir
    subprocess.run(["mkdir", f"{convective_day:%Y%m%d}"])

    # Download main archive
    subprocess.run(["wget", orig_tar_url.format(dt=convective_day)])

    # Extract SVRIMG47 fields from main archive
    subprocess.run(["tar", "-xvf", f"{convective_day:%Y%m%d}.tar", "-C", f"{convective_day:%Y%m%d}", "EchoTop_18", "MergedReflectivityQCComposite"])

    # Remove old tar
    subprocess.run(["rm", f"{convective_day:%Y%m%d}.tar"])

    # Download azshear products
    subprocess.run(["wget", azshear_tar_url.format(dt=convective_day)])

    # Extract SVRIMG47 fields from secondary archive
    subprocess.run(["tar", "-xvf", f"{convective_day:%Y%m%d}.tar", "-C", f"{convective_day:%Y%m%d}", "MergedLLShear", "MergedMLShear"])

    # Remove old tar
    subprocess.run(["rm", f"{convective_day:%Y%m%d}.tar"])

    # Delete files that aren't needed
    subprocess.run(["rm", *search_hourly_exclude(analysis_hours, "MergedReflectivityQCComposite")])
    subprocess.run(["rm", *search_hourly_exclude(analysis_hours, "EchoTop_18")])
    subprocess.run(["rm", *search_hourly_exclude(analysis_hours, "MergedLLShear", True)])
    subprocess.run(["rm", *search_hourly_exclude(analysis_hours, "MergedMLShear", True)])

    # tar gz the files that remain
    subprocess.run(["tar", "-czvf", f"{output_dir}{convective_day:%Y%m%d}_reduced.tar.gz", f"{convective_day:%Y%m%d}"])

    # Remove temporary directory to save space
    subprocess.run(["rm", "-r", f"{convective_day:%Y%m%d}"])

