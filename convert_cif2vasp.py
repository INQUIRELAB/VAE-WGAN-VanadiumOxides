import os
from ase.io import read, write


def convert_cif_to_vasp(directory, output_directory=None):
    if output_directory is None:
        output_directory = directory

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.cif'):
            cif_file_path = os.path.join(directory, filename)
            try:
                # Read the CIF file
                atoms = read(cif_file_path)

                # Create the output file path
                base_name = os.path.splitext(filename)[0]
                vasp_file_path = os.path.join(output_directory, base_name + '_1' + '.vasp')

                # Write the structure to the VASP file in Cartesian coordinates
                write(vasp_file_path, atoms, format='vasp', direct=False)
                print(f"Converted {filename} to {vasp_file_path}")
                os.remove(cif_file_path)
            except Exception as e:
                print(f"Error converting {cif_file_path}: {e}")


# Specify the directory to check
directory_path = './calculation_WGAN_alldata/generated_crystal_for_check/preprocessing'
# Optionally specify the output directory
output_directory_path = './calculation_WGAN_alldata/generated_crystal_for_check/preprocessing/cif'  # Set to None to save in the same directory

convert_cif_to_vasp(directory_path, output_directory_path)
