

# #import required module
# from cryptography.fernet import Fernet
# # key generation
# key = Fernet.generate_key()
  
# # string the key in a file
# with open('filekey.key', 'wb') as filekey:
#    filekey.write(key)



#def encrypt_file(key, in_filename


# import hashlib

# key = hashlib.sha256(b'16-character key').digest()

# with open('filekey.key', 'wb') as filekey:
#     filekey.write(key)


import os

key = os.urandom(32)
iv = os.urandom(16)

with open('keys/enc_key.key', "wb") as mykey:
    mykey.write(key)

with open('keys/enc_iv.iv', "wb") as myiv:
    myiv.write(iv)