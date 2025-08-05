#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>

// NOP opcode for x86-64
#define NOP 0x90

// Function to print the message from the trampoline
void trampoline_handler() {
    printf("Hello from trampoline!\n");
}

// Initialization function that runs when the shared object is loaded
__attribute__((constructor))
void init() {
    // First, check if debugging is enabled
    if (getenv("ZDEBUG")) {
        asm("int3");  // Breakpoint for debugging
    }

    // Map memory at address 0 (must have set /proc/sys/vm/mmap_min_addr to 0)
    // We need it to be both executable and writable
    void *addr = mmap(
        (void *)0,               // Specific address: 0
        4096,                    // Size: one page (4KB)
        PROT_READ | PROT_WRITE | PROT_EXEC,  // Permissions
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,  // Flags (MAP_FIXED to force address 0)
        -1,                      // No file descriptor
        0                        // No offset
    );

    
    if (addr != (void *)0) {
        fprintf(stderr, "Failed to map memory at address 0\n");
        exit(1);
    }
    
    unsigned char *mem_ptr = (unsigned char *)(uintptr_t)addr;
    
    for (int i = 0; i < 512; i++) {
        mem_ptr[i] = NOP;
    }

    // At position 512, we'll place the code for our trampoline
    unsigned char *trampoline = mem_ptr + 512;

    // We'll place a simple call to our trampoline_handler function
    // This is a simplified approach - in practice, you'd need to write
    // proper assembly instructions to call the function
    
    // On x86-64, a direct jump to absolute address can be implemented with:
    // FF 25 followed by a 32-bit offset from the next instruction to the address where
    // the actual 64-bit target address is stored
    
    // For simplicity, let's just inject code that calls our handler:
    
    // mov rax, address_of_trampoline_handler (48 B8 followed by 8-byte address)
    trampoline[0] = 0x48;
    trampoline[1] = 0xB8;
    *(void **)(trampoline + 2) = (void *)trampoline_handler;
    
    // call rax (FF D0)
    trampoline[10] = 0xFF;
    trampoline[11] = 0xD0;
    
    // ret (C3) - to return back after handling
    trampoline[12] = 0xC3;
    
    // Memory protection: change back to read+execute only (no write)
    if (mprotect(addr, 4096, PROT_READ | PROT_EXEC) != 0) {
        fprintf(stderr, "Failed to change memory protection\n");
        exit(1);
    }
    
    // Success message
    fprintf(stderr, "Trampoline initialized at address 0\n");
}