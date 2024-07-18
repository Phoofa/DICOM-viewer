from pydicom import dcmread, dcmwrite
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad
import os

def decrypt_dicom(input_file, output_file, key):
    # Read the DICOM file
    dicom_data = dcmread(input_file)
    
    # Extract pixel data
    pixel_data = dicom_data.PixelData
    
    # Create AES cipher
    cipher = AES.new(key, AES.MODE_CBC)
    
    # Encrypt the pixel data
    dencrypted_pixel_data = cipher.decrypt(pad(pixel_data, AES.block_size))
    
    # Replace the pixel data with the encrypted data
    dicom_data.PixelData = dencrypted_pixel_data
    
    # Add IV to the metadata (to ensure we can decrypt later)
    dicom_data.PrivateCreator = cipher.iv.hex()
    
    # Save the modified DICOM file
    dcmwrite(output_file, dicom_data)

# Fixed AES key
key = b'01234567890123456789012345678901'

# Paths to the input and output files
input_file = r'C:\Users\Administrator\Desktop\Code\Encryption tesy\encrypted2.dcm'
output_file = r'C:\Users\Administrator\Desktop\Code\Encryption tesy\decrypt.dcm'

# Encrypt the DICOM file
decrypt_dicom(input_file, output_file, key)
