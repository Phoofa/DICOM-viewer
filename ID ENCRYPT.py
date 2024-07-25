from pydicom import dcmread, dcmwrite
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad
import os

def encrypt_dicom_patient_id(input_file, output_file, key):
    # Read the DICOM file
    dicom_data = dcmread(input_file)
    
    # Extract Patient ID and ensure it's in bytes
    patient_id = dicom_data.PatientID.encode()
    
    # Create AES cipher in ECB mode
    cipher = AES.new(key, AES.MODE_ECB)
    
    # Encrypt the Patient ID with padding
    encrypted_patient_id = cipher.encrypt(pad(patient_id, AES.block_size))
    
    # Ensure the encrypted Patient ID is stored as bytes
    dicom_data.PatientID = encrypted_patient_id.decode('latin1')  # Use 'latin1' to ensure byte integrity
    
    # Save the modified DICOM file
    dcmwrite(output_file, dicom_data)
    print(encrypted_patient_id)
# Fixed AES key (must be 32 bytes for AES-256)
key = b'01234567890123456789012345678901'

# Paths to the input and output files
input_file = r'C:\Users\Administrator\Desktop\Code\Encryption test\Original image.dcm'
output_file = r'C:\Users\Administrator\Desktop\Code\Encryption test\Encrypted ID.dcm'

# Encrypt the Patient ID in the DICOM file
encrypt_dicom_patient_id(input_file, output_file, key)
