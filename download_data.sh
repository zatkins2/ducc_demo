#!/bin/bash

# check if a directory argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

TARGET_DIR="$1"

cd "$TARGET_DIR"

# download the masks and extract
wget -t 5 -nc -w 3 "https://lambda.gsfc.nasa.gov/data/act/pspipe/windows/dr6/act_dr6.02_windows_dr6_pa6_f090.tar.gz"
tar -xzf "act_dr6.02_windows_dr6_pa6_f090.tar.gz"

# download the FITS map
wget -t 5 -nc -w 3 "https://lambda.gsfc.nasa.gov/data/act/maps/published/act-planck_dr4dr6_coadd_AA_daynight_f090_map.fits"