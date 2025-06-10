#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <stdbool.h>
#include <sched.h>

#define SYSCALL_OPCODE_0 0x0F
#define SYSCALL_OPCODE_1 0x05
#define CALL_RAX_OPCODE_0 0xFF
#define CALL_RAX_OPCODE_1 0xD0

// Debug print
#define DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)

extern int64_t trigger_syscall(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
extern int64_t handler(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
extern void asm_syscall_hook(void);
extern void syscall_addr(void);

// Define the raw assembly function for trigger_syscall and syscall_hook
void __raw_asm() {
    asm volatile(
        ".globl trigger_syscall \n"
        "trigger_syscall: \n"
        "  movq %rdi, %rax \n"    // syscall number
        "  movq %rsi, %rdi \n"    // arg1
        "  movq %rdx, %rsi \n"    // arg2
        "  movq %rcx, %rdx \n"    // arg3
        "  movq %r8, %r10  \n"    // arg4 (Linux uses r10, not rcx)
        "  movq %r9, %r8   \n"    // arg5
        "  movq 8(%rsp), %r9 \n"  // arg6 (from stack)
        ".globl syscall_addr \n"
        "syscall_addr: \n"
        "  syscall \n"
        "  ret \n"
    );

    // Syscall hook implementation
    asm volatile(
        ".globl asm_syscall_hook \n"
        "asm_syscall_hook: \n"
        "cmpq $15, %rax \n"       // rt_sigreturn
        "je do_rt_sigreturn \n"
        "pushq %rbp \n"
        "movq %rsp, %rbp \n"

        // 16 byte stack alignment for function calls
        "andq $-16, %rsp \n"
        
        // Save all registers
        "pushq %r11 \n"
        "pushq %r9 \n"
        "pushq %r8 \n"
        "pushq %rdi \n"
        "pushq %rsi \n"
        "pushq %rdx \n"
        "pushq %rcx \n"
        
        // Arguments for syscall_hook
        "pushq 136(%rbp) \n"      // return address
        "pushq %rax \n"          // syscall number
        "pushq %r10 \n"          // 4th arg for syscall
        
        "callq syscall_hook@plt \n"
        
        "popq %r10 \n"
        "addq $16, %rsp \n"       // discard arg7 and arg8
        
        "popq %rcx \n"
        "popq %rdx \n"
        "popq %rsi \n"
        "popq %rdi \n"
        "popq %r8 \n"
        "popq %r9 \n"
        "popq %r11 \n"
        
        "leaveq \n"
        "addq $128, %rsp \n"
        "retq \n"
        
        "do_rt_sigreturn: \n"
        "addq $136, %rsp \n"
        "jmp syscall_addr \n"
    );
}

// Handler function that processes syscalls
int64_t syscall_hook(int64_t rdi, int64_t rsi,
                     int64_t rdx, int64_t __rcx __attribute__((unused)),
                     int64_t r8, int64_t r9,
                     int64_t r10_on_stack /* 4th arg for syscall */,
                     int64_t rax_on_stack /* syscall number */,
                     int64_t retptr)
{
    
    // Apply leet-speak decoding for write syscalls
    if (rax_on_stack == 1 && rdi == 1) {  // write to stdout
        static char decoded[4096];
        char *buffer = (char *)rsi;
        
        memcpy(decoded, buffer, rdx);
        decoded[rdx] = '\0';  // Null-terminate
        
        // Perform leet-speak decoding
        for (int i = 0; i < rdx; i++) {
            switch(buffer[i]) {
                case '0': decoded[i] = 'o'; break;
                case '1': decoded[i] = 'i'; break;
                case '2': decoded[i] = 'z'; break;
                case '3': decoded[i] = 'e'; break;
                case '4': decoded[i] = 'a'; break;
                case '5': decoded[i] = 's'; break;
                case '6': decoded[i] = 'g'; break;
                case '7': decoded[i] = 't'; break;
                default:  decoded[i] = buffer[i]; break;
            }
        }
        
        // Call original syscall with decoded buffer
        return trigger_syscall(rax_on_stack, rdi, (int64_t)decoded, rdx, r10_on_stack, r8, r9);
    }
    
    // For all other syscalls, just pass through
    return trigger_syscall(rax_on_stack, rdi, rsi, rdx, r10_on_stack, r8, r9);
}

// Set up the trampoline at address 0
void setup_trampoline() {
    void *mem = mmap(0, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
                     
    if (mem == MAP_FAILED) {
        fprintf(stderr, "map failed\n");
        fprintf(stderr, "NOTE: /proc/sys/vm/mmap_min_addr should be set 0\n");
        exit(1);
    }
    
    // Fill with NOPs
    memset(mem, 0x90, 0x1000);
    
    // Get address of the syscall hook
    void *hook_addr = (void*)asm_syscall_hook;
    
    // Write trampoline code at offset 512
    unsigned char *trampo_addr = (unsigned char*)mem + 512;
    
    // sub $0x80,%rsp (preserve redzone)
    trampo_addr[0x00] = 0x48;
    trampo_addr[0x01] = 0x81;
    trampo_addr[0x02] = 0xec;
    trampo_addr[0x03] = 0x80;
    trampo_addr[0x04] = 0x00;
    trampo_addr[0x05] = 0x00;
    trampo_addr[0x06] = 0x00;
    
    // movabs $asm_syscall_hook,%r11
    trampo_addr[0x07] = 0x49;
    trampo_addr[0x08] = 0xbb;
    // Insert 8-byte address
    memcpy(&trampo_addr[0x09], &hook_addr, sizeof(hook_addr));
    
    // jmp *%r11
    trampo_addr[0x11] = 0x41;
    trampo_addr[0x12] = 0xff;
    trampo_addr[0x13] = 0xe3;
    
    // Make the memory executable only to prevent NULL dereference
    mprotect(mem, 0x1000, PROT_EXEC);
}

// Find and replace syscall instructions
void rewrite_syscalls() {
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) {
        fprintf(stderr, "Failed to open /proc/self/maps\n");
        exit(1);
    }
    
    char line[1024];
    while (fgets(line, sizeof(line), maps)) {
        // Skip special regions
        if (strstr(line, "[stack]") || strstr(line, "[vsyscall]") || 
            strstr(line, "[vdso]")) {
            continue;
        }
        
        // Parse address range and permissions
        uintptr_t start, end;
        char perms[5] = {0};
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3) {
            continue;
        }
        
        // Only process executable regions
        if (strchr(perms, 'x') == NULL) {
            continue;
        }
        
        // Skip regions at address 0 (our trampoline)
        if (start < 0x1000) {
            continue;
        }
        
        // Make the region writable
        int orig_prot = 0;
        if (strchr(perms, 'r')) orig_prot |= PROT_READ;
        if (strchr(perms, 'w')) orig_prot |= PROT_WRITE;
        if (strchr(perms, 'x')) orig_prot |= PROT_EXEC;
        
        if (mprotect((void*)start, end - start, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
            continue;
        }
        
        // Scan for syscall instructions
        for (uintptr_t addr = start; addr < end - 1; addr++) {
            unsigned char *ptr = (unsigned char*)addr;
            
            // Skip syscall instruction in trigger_syscall
            if ((uintptr_t)ptr == (uintptr_t)syscall_addr) {
                continue;
            }
            
            if (ptr[0] == 0x0F && ptr[1] == 0x05) {  // syscall
                ptr[0] = 0xff;  // call
                ptr[1] = 0xd0;  // *%rax
            }
        }
        
        // Restore original permissions
        mprotect((void*)start, end - start, orig_prot);
    }
    
    fclose(maps);
}

// Library constructor
__attribute__((constructor))
void init() {
    DEBUG("libzpoline.so.2 initializing");
    
    // Check mmap_min_addr
    FILE *f = fopen("/proc/sys/vm/mmap_min_addr", "r");
    if (f) {
        char buf[64];
        if (fgets(buf, sizeof(buf), f)) {
            if (atoi(buf) != 0) {
                DEBUG("Error: /proc/sys/vm/mmap_min_addr must be set to 0");
                DEBUG("Run: sudo sysctl -w vm.mmap_min_addr=0");
                exit(1);
            }
        }
        fclose(f);
    }
    
    // Set up trampoline
    setup_trampoline();
    
    // Rewrite syscalls with call *%rax
    rewrite_syscalls();
    
    DEBUG("libzpoline.so.2 initialization complete");
}