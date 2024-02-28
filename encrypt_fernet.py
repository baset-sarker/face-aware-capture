
# import required module
from cryptography.fernet import Fernet
file_path = 'test.jpg'

def encrypt(file_path):
    #add_meta_data()
    add_metadata(file_path)

    # opening the key
    with open('filekey.key', 'rb') as filekey:
        key = filekey.read()
    
    # using the generated key
    fernet = Fernet(key)
    
    # opening the original file to encrypt
    with open(file_path, 'rb') as file:
        original = file.read()
        
    # encrypting the file
    encrypted = fernet.encrypt(original)
    
    # opening the file in write mode and 
    # writing the encrypted data
    with open(file_path+'.enc', 'wb') as encrypted_file:
        encrypted_file.write(encrypted)


def decrypt(file_path):
    # opening the key
    with open('filekey.key', 'rb') as filekey:
        key = filekey.read() 

    # using the key
    fernet = Fernet(key)
    
    # opening the encrypted file
    with open(file_path, 'rb') as enc_file:
        encrypted = enc_file.read()
    
    # decrypting the file
    decrypted = fernet.decrypt(encrypted)
    
    # opening the file in write mode and
    # writing the decrypted data
    decypted_file_name = file_path.replace('.enc', "")
    with open(decypted_file_name, 'wb') as dec_file:
        dec_file.write(decrypted)


import json
import piexif
import piexif.helper

def add_metadata(file_path):
    userdata = {
        'Name': 'Face aware',
        'Place': 'New York'
    }
    exif_dict = piexif.load(file_path)
    # insert custom data in usercomment field
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(
        json.dumps(userdata),
        encoding="unicode"
    )
    # insert mutated data (serialised into JSON) into image
    piexif.insert(
        piexif.dump(exif_dict),
        file_path
    )
    

#encrypt(file_path)
#decrypt(file_path+'.enc')
#add_metadata(file_path)