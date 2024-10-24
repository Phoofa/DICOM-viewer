import flet as ft
from Cryptodome.Cipher import AES as PyCryptoAES
from Cryptodome.Util.Padding import pad, unpad
from pydicom import dcmread, dcmwrite

import pydicom

key = "1234567890123456" 

sbox = {
    0x00: 0x63, 0x01: 0x7C, 0x02: 0x77, 0x03: 0x7B, 0x04: 0xF2, 0x05: 0x6B, 0x06: 0x6F, 0x07: 0xC5,
    0x08: 0x30, 0x09: 0x01, 0x0A: 0x67, 0x0B: 0x2B, 0x0C: 0xFE, 0x0D: 0xD7, 0x0E: 0xAB, 0x0F: 0x76,
    0x10: 0xCA, 0x11: 0x82, 0x12: 0xC9, 0x13: 0x7D, 0x14: 0xFA, 0x15: 0x59, 0x16: 0x47, 0x17: 0xF0,
    0x18: 0xAD, 0x19: 0xD4, 0x1A: 0xA2, 0x1B: 0xAF, 0x1C: 0x9C, 0x1D: 0xA4, 0x1E: 0x72, 0x1F: 0xC0,
    0x20: 0xB7, 0x21: 0xFD, 0x22: 0x93, 0x23: 0x26, 0x24: 0x36, 0x25: 0x3F, 0x26: 0xF7, 0x27: 0xCC,
    0x28: 0x34, 0x29: 0xA5, 0x2A: 0xE5, 0x2B: 0xF1, 0x2C: 0x71, 0x2D: 0xD8, 0x2E: 0x31, 0x2F: 0x15,
    0x30: 0x04, 0x31: 0xC7, 0x32: 0x23, 0x33: 0xC3, 0x34: 0x18, 0x35: 0x96, 0x36: 0x05, 0x37: 0x9A,
    0x38: 0x07, 0x39: 0x12, 0x3A: 0x80, 0x3B: 0xE2, 0x3C: 0xEB, 0x3D: 0x27, 0x3E: 0xB2, 0x3F: 0x75,
    0x40: 0x09, 0x41: 0x83, 0x42: 0x2C, 0x43: 0x1A, 0x44: 0x1B, 0x45: 0x6E, 0x46: 0x5A, 0x47: 0xA0,
    0x48: 0x52, 0x49: 0x3B, 0x4A: 0xD6, 0x4B: 0xB3, 0x4C: 0x29, 0x4D: 0xE3, 0x4E: 0x2F, 0x4F: 0x84,
    0x50: 0x53, 0x51: 0xD1, 0x52: 0x00, 0x53: 0xED, 0x54: 0x20, 0x55: 0xFC, 0x56: 0xB1, 0x57: 0x5B,
    0x58: 0x6A, 0x59: 0xCB, 0x5A: 0xBE, 0x5B: 0x39, 0x5C: 0x4A, 0x5D: 0x4C, 0x5E: 0x58, 0x5F: 0xCF,
    0x60: 0xD0, 0x61: 0xEF, 0x62: 0xAA, 0x63: 0xFB, 0x64: 0x43, 0x65: 0x4D, 0x66: 0x33, 0x67: 0x85,
    0x68: 0x45, 0x69: 0xF9, 0x6A: 0x02, 0x6B: 0x7F, 0x6C: 0x50, 0x6D: 0x3C, 0x6E: 0x9F, 0x6F: 0xA8,
    0x70: 0x51, 0x71: 0xA3, 0x72: 0x40, 0x73: 0x8F, 0x74: 0x92, 0x75: 0x9D, 0x76: 0x38, 0x77: 0xF5,
    0x78: 0xBC, 0x79: 0xB6, 0x7A: 0xDA, 0x7B: 0x21, 0x7C: 0x10, 0x7D: 0xFF, 0x7E: 0xF3, 0x7F: 0xD2,
    0x80: 0xCD, 0x81: 0x0C, 0x82: 0x13, 0x83: 0xEC, 0x84: 0x5F, 0x85: 0x97, 0x86: 0x44, 0x87: 0x17,
    0x88: 0xC4, 0x89: 0xA7, 0x8A: 0x7E, 0x8B: 0x3D, 0x8C: 0x64, 0x8D: 0x5D, 0x8E: 0x19, 0x8F: 0x73,
    0x90: 0x60, 0x91: 0x81, 0x92: 0x4F, 0x93: 0xDC, 0x94: 0x22, 0x95: 0x2A, 0x96: 0x90, 0x97: 0x88,
    0x98: 0x46, 0x99: 0xEE, 0x9A: 0xB8, 0x9B: 0x14, 0x9C: 0xDE, 0x9D: 0x5E, 0x9E: 0x0B, 0x9F: 0xDB,
    0xA0: 0xE0, 0xA1: 0x32, 0xA2: 0x3A, 0xA3: 0x0A, 0xA4: 0x49, 0xA5: 0x06, 0xA6: 0x24, 0xA7: 0x5C,
    0xA8: 0xC2, 0xA9: 0xD3, 0xAA: 0xAC, 0xAB: 0x62, 0xAC: 0x91, 0xAD: 0x95, 0xAE: 0xE4, 0xAF: 0x79,
    0xB0: 0xE7, 0xB1: 0xC8, 0xB2: 0x37, 0xB3: 0x6D, 0xB4: 0x8D, 0xB5: 0xD5, 0xB6: 0x4E, 0xB7: 0xA9,
    0xB8: 0x6C, 0xB9: 0x56, 0xBA: 0xF4, 0xBB: 0xEA, 0xBC: 0x65, 0xBD: 0x7A, 0xBE: 0xAE, 0xBF: 0x08,
    0xC0: 0xBA, 0xC1: 0x78, 0xC2: 0x25, 0xC3: 0x2E, 0xC4: 0x1C, 0xC5: 0xA6, 0xC6: 0xB4, 0xC7: 0xC6,
    0xC8: 0xE8, 0xC9: 0xDD, 0xCA: 0x74, 0xCB: 0x1F, 0xCC: 0x4B, 0xCD: 0xBD, 0xCE: 0x8B, 0xCF: 0x8A,
    0xD0: 0x70, 0xD1: 0x3E, 0xD2: 0xB5, 0xD3: 0x66, 0xD4: 0x48, 0xD5: 0x03, 0xD6: 0xF6, 0xD7: 0x0E,
    0xD8: 0x61, 0xD9: 0x35, 0xDA: 0x57, 0xDB: 0xB9, 0xDC: 0x86, 0xDD: 0xC1, 0xDE: 0x1D, 0xDF: 0x9E,
    0xE0: 0xE1, 0xE1: 0xF8, 0xE2: 0x98, 0xE3: 0x11, 0xE4: 0x69, 0xE5: 0xD9, 0xE6: 0x8E, 0xE7: 0x94,
    0xE8: 0x9B, 0xE9: 0x1E, 0xEA: 0x87, 0xEB: 0xE9, 0xEC: 0xCE, 0xED: 0x55, 0xEE: 0x28, 0xEF: 0xDF,
    0xF0: 0x8C, 0xF1: 0xA1, 0xF2: 0x89, 0xF3: 0x0D, 0xF4: 0xBF, 0xF5: 0xE6, 0xF6: 0x42, 0xF7: 0x68,
    0xF8: 0x41, 0xF9: 0x99, 0xFA: 0x2D, 0xFB: 0x0F, 0xFC: 0xB0, 0xFD: 0x54, 0xFE: 0xBB, 0xFF: 0x16
}

def xor(data_bytes, key_bytes):
    return bytes([b1 ^ b2 for b1, b2 in zip(data_bytes, key_bytes)])

def subbytes(data, sbox):   
    return bytes([sbox.get(byte) for byte in data])

def shiftrows(data):
    order = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]    
    shiftrows_result = bytes([data[i] for i in order])    
    return shiftrows_result

def multiply(a, b):
    if a == 0x02:
        result = b << 1
        if result & 0x100: 
            result ^= 0x1B  
        return result & 0xFF 
    elif a == 0x03:
        result = (b << 1) ^ b
        if result & 0x100:  
            result ^= 0x1B 
        return result & 0xFF

def mixcolumns(shiftrows_result):
    out1 = multiply(0x02, shiftrows_result[0]) ^ multiply(0x03, shiftrows_result[1]) ^ shiftrows_result[2] ^ shiftrows_result[3]
    out2 = shiftrows_result[0] ^ multiply(0x02, shiftrows_result[1]) ^ multiply(0x03, shiftrows_result[2]) ^ shiftrows_result[3]
    out3 = shiftrows_result[0] ^ shiftrows_result[1] ^ multiply(0x02, shiftrows_result[2]) ^ multiply(0x03, shiftrows_result[3])
    out4 = multiply(0x03, shiftrows_result[0]) ^ shiftrows_result[1] ^ shiftrows_result[2] ^ multiply(0x02, shiftrows_result[3])
    out5 = multiply(0x02, shiftrows_result[4]) ^ multiply(0x03, shiftrows_result[5]) ^ shiftrows_result[6] ^ shiftrows_result[7]
    out6 = shiftrows_result[4] ^ multiply(0x02, shiftrows_result[5]) ^ multiply(0x03, shiftrows_result[6]) ^ shiftrows_result[7]
    out7 = shiftrows_result[4] ^ shiftrows_result[5] ^ multiply(0x02, shiftrows_result[6]) ^ multiply(0x03, shiftrows_result[7])
    out8 = multiply(0x03, shiftrows_result[4]) ^ shiftrows_result[5] ^ shiftrows_result[6] ^ multiply(0x02, shiftrows_result[7])
    out9 = multiply(0x02, shiftrows_result[8]) ^ multiply(0x03, shiftrows_result[9]) ^ shiftrows_result[10] ^ shiftrows_result[11]
    out10 = shiftrows_result[8] ^ multiply(0x02, shiftrows_result[9]) ^ multiply(0x03, shiftrows_result[10]) ^ shiftrows_result[11]
    out11 = shiftrows_result[8] ^ shiftrows_result[9] ^ multiply(0x02, shiftrows_result[10]) ^ multiply(0x03, shiftrows_result[11])
    out12 = multiply(0x03, shiftrows_result[8]) ^ shiftrows_result[9] ^ shiftrows_result[10] ^ multiply(0x02, shiftrows_result[11])
    out13 = multiply(0x02, shiftrows_result[12]) ^ multiply(0x03, shiftrows_result[13]) ^ shiftrows_result[14] ^ shiftrows_result[15]
    out14 = shiftrows_result[12] ^ multiply(0x02, shiftrows_result[13]) ^ multiply(0x03, shiftrows_result[14]) ^ shiftrows_result[15]
    out15 = shiftrows_result[12] ^ shiftrows_result[13] ^ multiply(0x02, shiftrows_result[14]) ^ multiply(0x03, shiftrows_result[15])
    out16 = multiply(0x03, shiftrows_result[12]) ^ shiftrows_result[13] ^ shiftrows_result[14] ^ multiply(0x02, shiftrows_result[15])
    mixcolumn_result = bytes([out1, out2, out3, out4, out5, out6, out7, out8,
                              out9, out10, out11, out12, out13, out14, out15, out16])
    return mixcolumn_result


def subkeygen(key_bytes, round_num):
    def rotword(word):
        order = [1, 2, 3, 0] 
        return bytes([word[i] for i in order])
    rcon_values = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
    rcon = rcon_values[round_num-1]
    word3 = key_bytes[-4:]
    rotated_word = rotword(word3)
    substituted_word = subbytes(rotated_word, sbox)
    xorrcon = bytes([substituted_word[0] ^ rcon]) + substituted_word[1:]
    word4 = xor(xorrcon, key_bytes[:4])
    word5 = xor(word4, key_bytes[4:8])
    word6 = xor(word5, key_bytes[8:12])
    word7 = xor(word6, key_bytes[12:16])
    gensubkey = word4 + word5 + word6 + word7
    return gensubkey

# ทำลูปปปป

def AES(data_bytes, key_bytes, rounds=10):
    state = xor(data_bytes, key_bytes)  
    for round in range(1, rounds):
        state = subbytes(state, sbox)
        state = shiftrows(state)
        if round < rounds: 
            state = mixcolumns(state)
        key_bytes = subkeygen(key_bytes, round)
        state = xor(state, key_bytes)

    state = subbytes(state, sbox)
    state = shiftrows(state)
    key_bytes = subkeygen(key_bytes, rounds)
    ciphertext = xor(state, key_bytes)
    return ciphertext

def encrypt(plaintext, key_bytes, rounds=10):
    blocksize = 16
    if isinstance(plaintext, str):
        plaintext = plaintext.encode()    
    padlength = blocksize - (len(plaintext) % blocksize)
    padding = bytes([padlength]) * padlength
    paddeddata = plaintext + padding
    data_bytes = paddeddata
    num_blocks = len(data_bytes) // blocksize
    key_bytes = key.encode()
    ciphertext_full = bytes()
    for blocknum in range(num_blocks):
        block = data_bytes[blocknum * blocksize:(blocknum + 1) * blocksize]
        ciphertext_block = AES(block, key_bytes, rounds)
        ciphertext_full = ciphertext_full + ciphertext_block 
    return ciphertext_full

def encrypt_dicom_patient_info_and_image(dicom_file_path, output_file_path, key, encrypt_patient_info=True, encrypt_image=True):
    # Read the DICOM file
    dicom_data = dcmread(dicom_file_path)

    if encrypt_patient_info:
        # Encrypt the Patient ID
        if hasattr(dicom_data, 'PatientID'):
            patient_id = dicom_data.PatientID
            encrypted_patient_id = encrypt(patient_id, key)  # Using AES encryption method
            dicom_data.PatientID = encrypted_patient_id.hex()  # Store as a hex string in the DICOM file

        # Encrypt the Patient Name
        if hasattr(dicom_data, 'PatientName'):
            patient_name = dicom_data.PatientName
            encrypted_patient_name = encrypt(str(patient_name), key)  # Using AES encryption method
            dicom_data.PatientName = encrypted_patient_name.hex()  # Store as a hex string in the DICOM file

        # Encrypt the Patient Sex
        if hasattr(dicom_data, 'PatientSex'):
            patient_sex = dicom_data.PatientSex
            encrypted_patient_sex = encrypt(patient_sex, key)  # Using AES encryption method
            dicom_data.PatientSex = encrypted_patient_sex.hex()  # Store as a hex string in the DICOM file

    if encrypt_image:
        # Encrypt the PixelData
        if hasattr(dicom_data, 'PixelData'):
            pixel_data = dicom_data.PixelData  # Raw pixel data
            encrypted_pixel_data = encrypt(pixel_data, key)  # Using AES encryption method
            dicom_data.PixelData = encrypted_pixel_data  # Store the encrypted bytes directly in PixelData

    # Save the modified DICOM file with the encrypted fields
    try:
        if not output_file_path.endswith(".dcm"):
            output_file_path += ".dcm"
        
        dcmwrite(output_file_path, dicom_data)
        print(f"Encryption complete. Output saved to {output_file_path}")
    except Exception as e:
        print(f"Error writing encrypted DICOM file: {str(e)}")

def decrypt(ciphertext, key, is_pixel_data=False):
    blocksize = 16
    
    # Use pycryptodome's AES for decryption
    aes_cipher = PyCryptoAES.new(key.encode(), PyCryptoAES.MODE_ECB)  # Use ECB mode for simplicity
    decrypted_data = aes_cipher.decrypt(ciphertext)
    
    # Only unpad if the data is not PixelData (PixelData is binary and should not be padded/unpadded)
    if not is_pixel_data:
        try:
            decrypted_data = unpad(decrypted_data, blocksize)
        except ValueError as e:
            raise ValueError("Incorrect decryption or padding.")
    
    return decrypted_data

def decrypt_dicom_patient_info_and_image(dicom_file_path, output_file_path, key, decrypt_patient_info=True, decrypt_image=True):
    # Read the encrypted DICOM file
    dicom_data = dcmread(dicom_file_path)

    # Decrypt Patient ID
    if decrypt_patient_info and hasattr(dicom_data, 'PatientID'):
        try:
            encrypted_patient_id = bytes.fromhex(dicom_data.PatientID)  # Convert hex string to bytes
            decrypted_patient_id = decrypt(encrypted_patient_id, key)  # Decrypt
            dicom_data.PatientID = decrypted_patient_id.decode()  # Decode back to string
        except Exception as e:
            print(f"Error decrypting patient ID: {str(e)}")

    # Decrypt Patient Name
    if decrypt_patient_info and hasattr(dicom_data, 'PatientName'):
        try:
            encrypted_patient_name = bytes.fromhex(str(dicom_data.PatientName))  # Convert hex string to bytes
            decrypted_patient_name = decrypt(encrypted_patient_name, key)  # Decrypt
            dicom_data.PatientName = pydicom.valuerep.PersonName(decrypted_patient_name.decode())  # Convert back to PersonName format
        except Exception as e:
            print(f"Error decrypting patient name: {str(e)}")

    # Decrypt Patient Sex
    if decrypt_patient_info and hasattr(dicom_data, 'PatientSex'):
        try:
            encrypted_patient_sex = bytes.fromhex(dicom_data.PatientSex)  # Convert hex string to bytes
            decrypted_patient_sex = decrypt(encrypted_patient_sex, key)  # Decrypt
            dicom_data.PatientSex = decrypted_patient_sex.decode()  # Decode back to string
        except Exception as e:
            print(f"Error decrypting patient sex: {str(e)}")

    # Decrypt Pixel Data
    if decrypt_image and hasattr(dicom_data, 'PixelData'):
        try:
            encrypted_pixel_data = dicom_data.PixelData
            decrypted_pixel_data = decrypt(encrypted_pixel_data, key, is_pixel_data=True)
            dicom_data.PixelData = decrypted_pixel_data  # Replace with the decrypted pixel data
        except Exception as e:
            print(f"Error decrypting image data: {str(e)}")

    # Save the modified DICOM file with the decrypted fields
    try:
        if not output_file_path.endswith(".dcm"):
            output_file_path += ".dcm"
        
        dcmwrite(output_file_path, dicom_data)
        print(f"Decryption complete. Output saved to {output_file_path}")
    except Exception as e:
        print(f"Error writing decrypted DICOM file: {str(e)}")




def encryption_window(page):
    def browse_file_dialog(e):
        file_picker.pick_files(allow_multiple=False)
        
    def browse_save_dialog(e):
        directory_picker.get_directory_path()  # Get the directory to save

    def on_file_picked(e: ft.FilePickerResultEvent):
        if e.files:
            file_path.value = e.files[0].path
            page.update()

    def on_directory_picked(e: ft.FilePickerResultEvent):
        if e.path:
            save_as_directory.value = e.path
            page.update()

    def on_mode_change(e):
        mode_text.value = f"Selected mode: {mode_dropdown.value}"
        page.update()

    def process_dicom():
        dicom_file = file_path.value
        output_file = save_as_directory.value + '/' + save_as_filename.value
        encrypt_patient_info_value = encrypt_patient_info_checkbox.value
        encrypt_image_value = encrypt_image_checkbox.value

        if mode_dropdown.value == "Encryption":
            encrypt_dicom_patient_info_and_image(dicom_file, output_file, key, encrypt_patient_info=encrypt_patient_info_value, encrypt_image=encrypt_image_value)
            page.snack_bar = ft.SnackBar(ft.Text("DICOM file encrypted successfully!"))
        elif mode_dropdown.value == "Decryption":
            decrypt_dicom_patient_info_and_image(dicom_file, output_file, key, decrypt_patient_info=encrypt_patient_info_value, decrypt_image=encrypt_image_value)
            page.snack_bar = ft.SnackBar(ft.Text("DICOM file decrypted successfully!"))
        
        page.snack_bar.open = True
        page.update()

    # File picker for file selection
    file_picker = ft.FilePicker(on_result=on_file_picked)
    page.overlay.append(file_picker)

    # Directory picker for the "Save As" functionality
    directory_picker = ft.FilePicker(on_result=on_directory_picked)
    page.overlay.append(directory_picker)

    # Labels and inputs for "File Path" and "Save As"
    file_path = ft.TextField(label="File Path", disabled=True)

    # Save As components: directory and filename
    save_as_directory = ft.TextField(label="Directory", disabled=True)
    save_as_filename = ft.TextField(label="Filename", hint_text="Enter file name")

    browse_file_button = ft.ElevatedButton("Browse", on_click=browse_file_dialog)
    browse_save_button = ft.ElevatedButton("Browse", on_click=browse_save_dialog)

    # Encryption/Decryption Mode dropdown
    mode_dropdown = ft.Dropdown(
        label="Mode",
        options=[
            ft.dropdown.Option("Encryption"),
            ft.dropdown.Option("Decryption")
        ],
        value="Encryption",  # Default selection
        on_change=on_mode_change
    )
    mode_text = ft.Text("Selected mode: Encryption")

    # Checkboxes for Encrypt/Decrypt Patient Information and Image
    encrypt_patient_info_checkbox = ft.Checkbox(label="Patient Information", value=True)
    encrypt_image_checkbox = ft.Checkbox(label="Image", value=True)

    # Process button
    process_button = ft.ElevatedButton("Process", on_click=lambda _: process_dicom())

    # Layout of the popup window
    page.add(
        ft.Container(
            content=ft.Column([
                ft.Row([file_path, browse_file_button], alignment=ft.MainAxisAlignment.START),
                ft.Row([save_as_directory, browse_save_button], alignment=ft.MainAxisAlignment.START),
                ft.Row([save_as_filename], alignment=ft.MainAxisAlignment.START),  # Filename input field
                ft.Row([mode_dropdown]),
                ft.Row([mode_text]),
                ft.Row([encrypt_patient_info_checkbox, encrypt_image_checkbox], alignment=ft.MainAxisAlignment.START),
                process_button,
                
            ]),
            width=500,
            height=450,
            padding=10,
            border_radius=ft.border_radius.all(10),
            bgcolor=ft.colors.GREY_300
        )
    )
# Main function to open the encryption window
def main(page: ft.Page):
    encryption_window(page)

ft.app(target=main)