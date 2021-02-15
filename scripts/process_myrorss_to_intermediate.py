#!/usr/bin/env python
# Copyright (c) 2021 Jonathan Thielen.
# Distributed under the terms of the Apache 2.0 License
# SPDX-License-Identifier: Apache-2.0
"""
Download and munge select MYRORSS data fields to intermediate format for use by SVRIMG-47.

Relies upon stored weight and target grid files (should be available elsewhere in the
accompanying repo), as well as a JSON mapping of convective days to hours we need to grab
data from for storm report samples (generatable using the
preprocess_reports_for_intermediate_files.ipynb notebook).
"""

# stdlib imports
import argparse
import json
import os
import sys
import subprocess
import warnings

# scientific library imports
import glob
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import sparse
import xarray as xr
import xesmf as xe
import zarr


# Resource definitions
orig_tar_url = "https://files.osf.io/v1/resources/9gzp2/providers/dropbox/{dt:%Y}/MYRORSS_orig/{dt:%Y%m%d}.tar"
azshear_tar_url = "https://files.osf.io/v1/resources/9gzp2/providers/dropbox/{dt:%Y}/Azshear/{dt:%Y%m%d}.tar"


def search_hourly(timestamp, fieldname, return_hourly_range=False):
    """Search the base directory for file or files at this time for this field."""
    # Get files and their timestamps
    field_dir = f"{temp_myrorss_dir}{convective_day:%Y%m%d}/{fieldname}/00.25/"
    files = sorted(glob.glob(f"{field_dir}*.netcdf"))
    file_timestamps = pd.Series({pd.Timestamp(f.split("/")[-1][:-7]): f for f in files})

    # Select file with nearest time to that specified
    nearest_idx = file_timestamps.index.get_loc(timestamp, method='nearest')
    if return_hourly_range:
        # Also check for range back to an hour ago if needed
        nearest_idx_prior = file_timestamps.index.get_loc(timestamp - pd.Timedelta(1, 'H'), method='nearest')
        return file_timestamps.iloc[(nearest_idx_prior + 1):(nearest_idx + 1)].to_list()
    else:
        return file_timestamps.iloc[nearest_idx]


def open_field_to_full_dataarray(f, timestamp, field):
    """Open a given field at a specified time to a dense xarray DataArray."""
    # Use netcdf-python size lat/lon size hidden on missing dimensions, which are ignored by
    # xarray
    netcdf = Dataset(f)
    lat_size = netcdf.dimensions['Lat'].size
    lon_size = netcdf.dimensions['Lon'].size
    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf))
    
    # Use sparse.COO data structure to represent the sparse field, then expand to dense field as DataArray
    field_COO = sparse.COO(np.stack([ds['pixel_x'].values, ds['pixel_y'].values]), ds[field], shape=(lat_size, lon_size), fill_value=ds.attrs['MissingData'])
    longitude = ds.attrs['Longitude'] + np.arange(lon_size) * ds.attrs['LonGridSpacing']
    latitude = ds.attrs['Latitude'] - np.arange(lat_size) * ds.attrs['LonGridSpacing']
    field_full = xr.DataArray(
        field_COO.todense(),
        coords={'lat': latitude, 'lon': longitude, 'time': timestamp},
        dims=('lat', 'lon'),
        name=field,
        attrs=ds.attrs
    )

    # Throw away missing and small values, since we only care about storm morphology
    field_full.data[field_full.data < -10] = np.nan
    
    return field_full


def open_single_field(timestamp, field):
    """Open a field that only requires consists of data at a single time."""
    f = search_hourly(timestamp, field)
    return open_field_to_full_dataarray(f, timestamp, field)


def open_max_field(timestamp, field):
    """Open a field that requires an hourly maximium of data."""
    files = search_hourly(timestamp, field, True)
    return xr.concat([open_field_to_full_dataarray(f, timestamp, field) for f in files[::-1]], 'file').max('file')


# Define control parameters for fields of interest
field_control = {
    "MergedReflectivityQCComposite": {
        'open': open_single_field,
        'coarsen': 2,
        'method': 'mean',
        'rename': 'reflectivity'
    },
    "EchoTop_18": {
        'open': open_single_field,
        'coarsen': 2,
        'method': 'mean',
        'rename': 'echo_top_height'
    },
    "MergedLLShear": {
        'open': open_max_field,
        'coarsen': 2,
        'method': 'max',
        'rename': 'low_level_rotation_track'
    },
    "MergedMLShear": {
        'open': open_max_field,
        'coarsen': 2,
        'method': 'max',
        'rename': 'mid_level_rotation_track'
    }
}


# Run the script
if __name__ == '__main__':
    #########################
    # Runtime Configuration #
    #########################

    parser = argparse.ArgumentParser(
        description="Munge convective day of MYRORSS data into hourly intermediate products for creation of SVRIMG-47 samples", usage=argparse.SUPPRESS
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
        "-g", "--target-grid", help="Dataset defining target grid", default="~/research/svrimg_2km_target_conus_grid.nc"
    )
    parser.add_argument(
        "-r", "--regridder-weights", help="Dataset defining regridder weights", default="~/research/nearest_1750x3500_svrimg_2km.nc"
    )
    parser.add_argument(
        "-H", "--hour-file", help="JSON file mapping convective days to hours to extract", default="~/research/severe_report_hours.json"
    )

    try:
        args = parser.parse_args()
        output_dir = args.output_dir
        convective_day = pd.Timestamp(args.timestamp)
        temp_myrorss_dir = args.cache_dir
        ds_target = xr.open_dataset(args.target_grid)
        weight_file = args.regridder_weights
        with open(args.hour_file, "r") as hour_file:
            analysis_hours = json.load(hour_file)[f"{convective_day:%Y-%m-%d}"]
    except (SystemExit, ValueError):
        parser.print_help()
        raise

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

    # Unzip any zipped files from the archives
    subprocess.run(["gunzip", f"{convective_day:%Y%m%d}/*/00.25/*.netcdf.gz"])

    ######################
    # Analysis Hour Loop #
    ######################

    for hour in analysis_hours:
        timestamp = pd.Timestamp(hour)
        print(f"\tRegridding {timestamp}")
        collected_fields = []
        for field in field_control:
            print(f"\t\t{field}")

            # Open the field as a dense DataArray
            field_full = field_control[field]['open'](timestamp, field)

            # Resample to reduced resolution for computational feasibility
            field_coarsened = getattr(
                field_full.coarsen({'lat': field_control[field]['coarsen'], 'lon': field_control[field]['coarsen']}, 'trim', keep_attrs=True),
                field_control[field]['method']
            )()

            # Regrid, hiding warnings like not F_CONTIGOUS
            with warnings.simplefilter("ignore"):
                regridder = xe.Regridder(
                    field_coarsened.to_dataset(name=field_control[field]['rename']),
                    ds_target,
                    method='nearest_s2d',
                    reuse_weights=True,
                    weights=weight_file
                )
                field_regridded = regridder(field_coarsened, keep_attrs=True)

            # Rename field
            field_regridded.name = field_control[field]['rename']
            
            collected_fields.append(field_regridded)

        # Merge all fields into dataset
        ds = xr.merge(collected_fields)

        # Clean up the dataset for appending to the zarr store
        ds['x'] = ds_target['x']
        ds['y'] = ds_target['y']
        ds = ds.drop_vars(['lat', 'lon'])
        ds.reflectivity.attrs = {
            'long_name': 'column_maximum_of_equivalent_reflectivity_factor',
            'units': 'dBZ'
        }
        ds.echo_top_height.attrs = {
            'long_name': 'maximum_height_of_18_dBZ_reflectivity',
            'units': 'km'
        }
        ds.low_level_rotation_track.attrs = {
            'long_name': 'hourly_maximum_of_azimuthal_shear_in_0_to_2_km_agl_layer',
            'units': 's-1'
        }
        ds.mid_level_rotation_track.attrs = {
            'long_name': 'hourly_maximum_of_azimuthal_shear_in_3_to_6_km_agl_layer',
            'units': 's-1'
        }

        ds = ds.expand_dims('time')

        zarr_store = f"{output_dir}subset_{timestamp:%Y}.zarr/"
        if os.path.isdir(zarr_store):
            # Zarr store exists, simply append
            ds.to_zarr(zarr_store, append_dim="time", consolidated=True)
        else:
            # Zarr store for this year doesn't exist, define encoding and projection
            ds.attrs = {
                'description': (
                    'SVRIMG-47 MYRORSS Intermediate Format: resampled and regridded hourly '
                    'select MYRORSS products for use in SVRIMG-47 morphology samples.'
                )
            }
            ds['projection'] = ds_target['projection']
            compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
            encoding = {
                "reflectivity": {"compressor": compressor, 'dtype': 'float32'},
                "echo_top_height": {"compressor": compressor, 'dtype': 'float32'},
                "low_level_rotation_track": {"compressor": compressor, 'dtype': 'float32'},
                "mid_level_rotation_track": {"compressor": compressor, 'dtype': 'float32'},
                "time": {"units": f"seconds since {timestamp:%Y}-01-01 00:00:00", 'dtype': 'float64'}
            }
            ds.to_zarr(zarr_store, encoding=encoding, consolidated=True)

        print("\t\t...all fields saved to zarr")

