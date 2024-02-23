import glob
import os

import click
import gzip
import h5py
import nibabel as nib
import numpy as np
import shutil


@click.group()
def main():
    pass


@main.command()
@click.argument(
    'src_path', default='/data/TBrecon1/miccai_challenge/train/untarred/'
)
@click.argument('dest_path', default='/data/TBrecon3/kseg/')
@click.option('--sanity-check', is_flag=True, default=False)
@click.option('--resume', is_flag=True, default=False)
def hdf2nii(
    src_path: str, dest_path: str, sanity_check: bool, resume: bool
) -> None:
    """Reads hdf files, splits each of them into image and label and save them.

    Args:
        src_path: Path of the hdf files.
        dest_path: Destination path for the nii files.
        sanity_check: Whether to check if the written files contain the same as
            the read files.
        resume: Whether to resume from an already made progress or not.
    """
    for file in glob.glob(os.path.join(src_path, 'TBrecon*.h5.gz')):
        file_name = os.path.basename(file).split('.')[0]

        # If resume flag set, check if file is already converted
        # A file is converted successfully if the seg map is present and the
        # unzipped hdf file is deleted
        if resume:
            if os.path.join(dest_path, f'{file_name}_seg.nii.gz') in glob.glob(
                os.path.join(dest_path, 'TBrecon*')
            ) and os.path.join(dest_path, f'{file_name}.h5') not in glob.glob(
                os.path.join(dest_path, 'TBrecon*')
            ):
                continue

        # Unzip data and store it to destination directory
        with gzip.open(file, 'rb') as infile:
            with open(
                os.path.join(dest_path, f'{file_name}.h5'), 'wb'
            ) as outfile:
                shutil.copyfileobj(infile, outfile)

        # Convert hdf file to two separate nii files (kspace + seg)
        file = h5py.File(os.path.join(dest_path, f'{file_name}.h5'), 'r')
        kspace, seg = file['kspace'], file['seg']

        ni_kspace = nib.Nifti1Image(kspace, affine=np.eye(4))
        nib.save(ni_kspace, os.path.join(dest_path, f'{file_name}.nii.gz'))

        ni_seg = nib.Nifti1Image(seg, affine=np.eye(4))
        nib.save(ni_seg, os.path.join(dest_path, f'{file_name}_seg.nii.gz'))

        # Clean up - Delete unzipped hdf file
        os.remove(os.path.join(dest_path, f'{file_name}.h5'))

        # Sanity check - check if written things are the same as the read ones
        if sanity_check:
            nii_kspace_file = nib.load(
                os.path.join(dest_path, f'{file_name}.nii.gz')
            )
            nii_kspace = nii_kspace_file.get_fdata(dtype=np.complex64)
            nii_seg_file = nib.load(
                os.path.join(dest_path, f'{file_name}_seg.nii.gz')
            )
            nii_seg = nii_seg_file.get_fdata()

            if (not np.array_equal(nii_kspace, kspace)) or (
                not np.array_equal(nii_seg, seg)
            ):
                print(f'Sanity check failed at file {file_name}')
                break

        print(f'Processed {file_name}')


@click.command()
@click.argument('src_path', default='/home/egosche/')
@click.argument('dest_path', default='/home/egosche/')
@click.option('--resume', is_flag=True, default=False)
def aseg2label(src_path: str, dest_path: str, resume: bool) -> None:
    """Reads freesurfer aseg files, converts them into tissue nifti labels and
    saves them.

    Args:
        src_path: Path of the aseg files.
        dest_path: Destination path for the nii files.
        resume: Whether to resume from an already made progress or not.
    """
    for file in glob.glob(os.path.join(src_path, 'file_prefix*.mgz')):
        file_name = os.path.basename(file).split('.')[0]

        # If resume flag set, check if file is already converted
        # A file is converted successfully if the seg map is present
        if resume:
            if os.path.join(dest_path, f'{file_name}_seg.nii.gz') in glob.glob(
                os.path.join(dest_path, 'file_prefix*')
            ):
                continue

        # Load data
        file = nib.load(file)
        kspace = np.array(file.dataobj)

        # Set unrelevant segmentation parts to 0
        # Relevant: Cortex, WM, CSF
        kspace[~np.isin(kspace, [2, 3, 7, 8, 24, 41, 42, 46, 47])] = 0

        # Assign class 1 to WM
        kspace[np.isin(kspace, [2, 7, 41, 46])] = 1

        # Assign class 2 to Cortex
        kspace[np.isin(kspace, [3, 8, 42, 47])] = 2

        # Assign class 3 to CSF
        kspace[np.isin(kspace, [24])] = 3

        # Save new segmentation map as nifti file
        ni_seg = nib.Nifti1Image(kspace, affine=np.eye(4))
        nib.save(ni_seg, os.path.join(dest_path, f'{file_name}_seg.nii.gz'))

        print(f'Processed {file_name}')


if __name__ == '__main__':
    main()
