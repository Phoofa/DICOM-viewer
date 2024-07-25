from pydicom import dcmread, dcmwrite
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import unpad

def decrypt_dicom_patient_id(input_file, output_file, key):
    # Read the DICOM file
    dicom_data = dcmread(input_file)
    
    # Extract the encrypted Patient ID and convert it back to bytes
    encrypted_patient_id = dicom_data.PatientID.encode('latin1')
    
    # Create AES cipher in ECB mode
    cipher = AES.new(key, AES.MODE_ECB)
    
    # Decrypt the Patient ID and remove padding
    decrypted_patient_id = unpad(cipher.decrypt(encrypted_patient_id), AES.block_size).decode()
    
    # Replace the Patient ID with the decrypted data
    dicom_data.PatientID = decrypted_patient_id
    
    # Save the modified DICOM file
    dcmwrite(output_file, dicom_data)

    

# Fixed AES key (must be 32 bytes for AES-256)
key = b'01234567890123456789012345678901'

# Paths to the input and output files
encrypted_file = r'C:\Users\Administrator\Desktop\Code\Encryption test\Encrypted ID.dcm'
decrypted_file = r'C:\Users\Administrator\Desktop\Code\Encryption test\Decrypted ID.dcm'

# Decrypt the Patient ID in the DICOM file
decrypt_dicom_patient_id(encrypted_file, decrypted_file, key)
