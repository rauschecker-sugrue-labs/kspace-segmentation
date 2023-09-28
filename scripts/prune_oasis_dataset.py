import os
import shutil
import sys


# This script deletes everything in each downloaded OASIS disc dir expect
# the mri dir. Also, it deletes follow-up exams of patient ('MR2').

# Check if the correct number of arguments are provided
if len(sys.argv) != 2:
    print("Usage: python prune_oasis_dataset.py <path_to_discX_files>")
    sys.exit(1)

# Get the path in which the downloaded discX directories are located from CLI
base_dir = sys.argv[1]

# Iterate through the disc directories
for disc in os.listdir(base_dir):
    disc_path = os.path.join(base_dir, disc)

    # Ensure it's a directory and has the name format 'discX'
    if os.path.isdir(disc_path) and 'disc' in disc:
        # Iterate through the OAS1_XXXX_MR1 and OAS1_XXXX_MR2 directories inside the disc directories
        for oas in os.listdir(disc_path):
            oas_path = os.path.join(disc_path, oas)

            # If it's an 'OAS1_XXXX_MR2' directory, delete it completely
            if os.path.isdir(oas_path) and 'OAS1' in oas and 'MR2' in oas:
                shutil.rmtree(oas_path)
                print(f"Deleted {oas_path}")

            # For 'OAS1_XXXX_MR1' directories, go deeper to delete target dirs except 'mri'
            elif os.path.isdir(oas_path) and 'OAS1' in oas and 'MR1' in oas:
                # Iterate through the target directories inside the OAS1_XXXX_MR1 directories
                for target_dir in os.listdir(oas_path):
                    target_path = os.path.join(oas_path, target_dir)

                    # If the directory is not 'mri', delete it
                    if os.path.isdir(target_path) and target_dir != 'mri':
                        shutil.rmtree(target_path)
                        print(f"Deleted {target_path}")

print("Clean process completed.")
print(
    "Please merge the content of all discX directories into one directory now."
)
