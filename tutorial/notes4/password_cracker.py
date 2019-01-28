from pprint import pprint
import hashlib
import time
import random
from itertools import imap, product

### Security Code  ###############################################

userpass = {}                 # Usually this a file

def digest(password):
    # XXX make the digest depend on the username as well as the password
    salt = 'the average life expectancy of a stark, lannister, or targaryen is very short'
    return hashlib.pbkdf2_hmac('sha256', password, salt, 100000)

def verify_strong(password):
    if (len(password) >= 6
        and any(imap(str.isdigit, password))
        and any(imap(str.islower, password))
        and any(imap(str.isupper, password))):
        return
    raise ValueError('Weak password:  Must be 6 character with upper and lowercase letters and digits')
        
def new_account(username, password):
    verify_strong(password)
    hashpass = digest(password)
    userpass[username] = hashpass

def verify(username, password):
    hashpass = digest(password)
    result = userpass[username] == hashpass
    time.sleep(random.random() * 0.25)       
    return result

### Session ######################################################

new_account('raymondh', 'Superman7')
new_account('sergey', '3russiA')
new_account('mike', '1Bears')
new_account('brandon', 'leaH19')
new_account('austin', 'Asdfasdf123')

print verify('raymondh', 'Superman7')
print verify('mike', '1Bears')

### Cracker code #################################################

def make_rainbow():
    rainbow_table = {}
    with open('notes4/common_passwords.txt') as f:
        for line in f:
            password = line.split(',')[0]
            hashpass = digest(password)
            rainbow_table[hashpass] = password

def rainbow_cracker(userpass, rainbow_table):
    for user, hashpass in userpass.iteritems():
        if hashpass in rainbow_table:
            print user, '-->', rainbow_table[hashpass]

passuser = {hashpass : user for user, hashpass in userpass.items()}        
prefixes = map(str, range(10)) + ['']
suffixes = map(str, range(1000)) + ['']
with open('notes4/common_passwords.txt') as f:
    for line in f:
        password = line.split(',')[0]
        case_variants = [password, password.title(), password.title().swapcase(),
                         password[:-1] + password[-1].swapcase(),
 		        (password[:-1] + password[-1].swapcase()).swapcase(),
                         password.upper()]
        for password in imap(''.join, product(prefixes, case_variants, suffixes)):
            hashpass = digest(password)
            if hashpass in passuser:
                user = passuser[hashpass]
                print user, '-->', password
                
