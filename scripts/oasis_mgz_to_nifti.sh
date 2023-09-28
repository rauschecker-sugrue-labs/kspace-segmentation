#!/bin/bash

# This script converts the FreeSurfer output (T1.mgz, brainmask.mgz and 
# aseg.mgz) to NIfTI format. Please make sure you have sourced the Freesurfer 
# environment.

# Check if the correct number of arguments are provided
# The subjects dir contains all subject dirs (e.g. OAS1_XXXX_MR1)
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <subjects_dir_path>"
    exit 1

ROOT_DIR="$1"

# Navigate to the root directory
cd "$ROOT_DIR"

# Iterate over each subject directory
for SUBJECT_DIR in OAS1_*_MR1; do
    # Check if the directory exists
    if [[ -d "$SUBJECT_DIR/mri" ]]; then
        cd "$SUBJECT_DIR/mri"
        
        # Convert T1.mgz to NIfTI format
        if [[ -f "T1.mgz" ]]; then
            mri_convert T1.mgz T1.nii.gz
        fi
        
        # Convert brainmask.mgz to NIfTI format
        if [[ -f "brainmask.mgz" ]]; then
            mri_convert brainmask.mgz brainmask.nii.gz
        fi
        
        # Convert aseg.mgz to NIfTI format
        if [[ -f "aseg.mgz" ]]; then
            mri_convert aseg.mgz aseg.nii.gz
        fi
        
        # Return to the root directory to process the next subject directory
        cd "$ROOT_DIR"
    fi
done
