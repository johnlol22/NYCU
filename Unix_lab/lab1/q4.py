#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Lab sample file for the AUP course by Chun-Ying Huang

import sys
from pwn import *
from solpow import solve_pow
import base64
import zlib

def decode_msg(msg):
    try:
        # Extract message between >>> and <
        msg = msg.split(b'>>>')[1].split(b'<<<')[0].strip()
        msg = base64.b64decode(msg)
        mlen = int.from_bytes(msg[0:4], 'big')
        if len(msg)-4 != mlen:
            return None
        content = zlib.decompress(msg[4:])
        return content.decode()
    except Exception as e:
        print(f"Error decoding: {e}")
        return None

def encode_msg(msg):
    # Encode message
    zm = zlib.compress(msg.encode())
    mlen = len(zm)
    encoded = base64.b64encode(mlen.to_bytes(4, 'little') + zm)
    return encoded
def guess_num(num):
    num = num+1
    return num
def main():
    if len(sys.argv) > 1:
        ## for remote access
        r = remote('up.zoolab.org', 10155)
        solve_pow(r)
    else:
        ## for local testing
        print("using local")
        r = process('./guess_dist.py', shell=False)

    print('*** Starting solver...')
    
    while True:
        try:
            # Receive message from server
            msg = r.recvline().strip()
            if not msg:
                continue
            
            # Decode and print message
            decoded = decode_msg(msg)
            if decoded:
                print(f"Decoded message: {decoded}")
                
                # If it's asking for input, send a guess
                if "Enter your input" in decoded:
                    # Send a test guess '0000'
                    guess = '1265'
                    encoded_guess = encode_msg(guess)
                    print(f"Sending guess: {guess}")
                    r.sendline(encoded_guess)
            else:
                print(f"Raw message: {msg}")

        except EOFError:
            print("Connection closed")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    #r.interactive()

if __name__ == '__main__':
    main()

# vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4 number cindent fileencoding=utf-8 :