# import required module
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import AES
import json
import piexif
import piexif.helper

def encrypt(file_path):
    # add_meta_data()
    #add_metadata(file_path)

    # opening the key
    with open('keys/enc_key.key', 'rb') as filekey:
        key = filekey.read()

    # opening the iv
    with open('keys/enc_iv.iv', 'rb') as fileiv:
        iv = fileiv.read()

    # opening the original file to encrypt
    with open(file_path, 'rb') as file:
        original = file.read()

    cipher = AES.new(key, AES.MODE_CFB,iv)
    ciphertext = cipher.encrypt(pad(original,16))

    # writing the encrypted data
    with open(file_path + '.enc', 'wb') as encrypted_file:
        encrypted_file.write(ciphertext)


def decrypt(file_path):
    # opening the key
    with open('keys/enc_key.key', 'rb') as filekey:
        key = filekey.read()

    # opening the iv
    with open('keys/enc_iv.iv', 'rb') as fileiv:
        iv = fileiv.read()

    # opening the encrypted file
    with open(file_path, 'rb') as enc_file:
        encrypted = enc_file.read()

    cipher2 = AES.new(key,AES.MODE_CFB,iv)
    decrypted_data = unpad(cipher2.decrypt(encrypted),16)

    # opening the file in write mode and
    # writing the decrypted data
    decypted_file_name = file_path.replace('.enc', "")
    with open(decypted_file_name, 'wb') as dec_file:
        dec_file.write(decrypted_data)


def add_metadata(file_path,image_name):
    userdata = {
        'Name': 'Face aware',
        'Place': 'New York',
        'time' : image_name,
        'latlng':'44.6698,74.9813'
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


#file_path = 'test.jpg'
#encrypt(file_path)
#decrypt(file_path+'.enc')