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
#include <dlfcn.h>
#include <assert.h>
#include <linux/sched.h>  /* For __NR_clone and CLONE_VM */
#include <capstone/capstone.h>

#define SYSCALL_OPCODE_0 0x0F
#define SYSCALL_OPCODE_1 0x05
#define CALL_RAX_OPCODE_0 0xFF
#define CALL_RAX_OPCODE_1 0xD0


// For the clone syscall
#ifndef __NR_clone
#define __NR_clone 56
#endif

#ifndef CLONE_VM
#define CLONE_VM 0x00000100
#endif


// Debug print
#define DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)

typedef int64_t (*syscall_hook_fn_t)(int64_t, int64_t, int64_t, int64_t,
                                     int64_t, int64_t, int64_t);

extern int64_t trigger_syscall(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
extern void asm_syscall_hook(void);
extern void syscall_addr(void);

// Function pointer for the hook function
static syscall_hook_fn_t hook_fn = trigger_syscall;


// Define the raw assembly function for trigger_syscall and syscall_hook
void __raw_asm() {
    asm volatile(
        ".globl trigger_syscall \n"
        "trigger_syscall: \n"
        // "  movq %rdi, %rax \n"    // syscall number
        // "  movq %rsi, %rdi \n"    // arg1
        // "  movq %rdx, %rsi \n"    // arg2
        // "  movq %rcx, %rdx \n"    // arg3
        // "  movq %r8, %r10  \n"    // arg4 (Linux uses r10, not rcx)
        // "  movq %r9, %r8   \n"    // arg5
        // "  movq 8(%rsp), %r9 \n"  // arg6 (from stack)
	    "  movq %rcx, %r10    \n"
	    "  movq 8(%rsp), %rax \n"
        ".globl syscall_addr \n"
        "syscall_addr: \n"
        "  syscall \n"
        "  ret \n"
    );

    // Syscall hook implementation
    // actual trampoline code
    asm volatile(
        ".globl asm_syscall_hook \n"
        "asm_syscall_hook: \n"
        //"  cmpq $0x3a, %rax \n"       // Check if syscall is vfork (58)
        //"  jne asm_start \n"          // If not, proceed with normal processing
        //"  addq $128, %rsp \n"
        //"  popq %rsi \n"              // Pop return address into rsi
        //"  andq $-16, %rsp \n"        // 16 byte stack alignment
        //"  syscall \n"                // Execute vfork directly
        //"  pushq %rsi \n"             // Push return address back
        //"  ret \n"                    // Return to caller
            
        "asm_start: \n"
        "  cmpq $15, %rax \n"         // rt_sigreturn
        "  je do_rt_sigreturn \n"
        "  pushq %rbp \n"
        "  movq %rsp, %rbp \n"

        "  andq $-16, %rsp \n"        // 16 byte stack alignment
                
        "  pushq %r11 \n"
        "  pushq %r9 \n"
        "  pushq %r8 \n"
        "  pushq %rdi \n"
        "  pushq %rsi \n"
        "  pushq %rdx \n"
        "  pushq %rcx \n" 
                
        "  pushq 136(%rbp) \n"         // return address - CORRECTED
        "  pushq %rax \n"            // syscall number
        "  pushq %r10 \n"            // 4th arg for syscall
                
        "  callq syscall_hook@plt \n"
                
        //"  addq $8, %rsp \n"         // discard the other arguments
        "  popq %r10 \n"
        "  addq $16, %rsp \n"         // discard the other arguments
                
        "  popq %rcx \n"
        "  popq %rdx \n"
        "  popq %rsi \n"
        "  popq %rdi \n"
        "  popq %r8 \n"
        "  popq %r9 \n"
        "  popq %r11 \n"
                
        "  leaveq \n"                 // Restore %rbp and %rsp
        "  addq $128, %rsp \n\t"
        "  ret \n"                    // Return to caller
                
        "do_rt_sigreturn: \n"
        "  addq $136, %rsp \n"          // Skip only the return address
        "  jmp syscall_addr \n"
    );
}

// Handler function that processes syscalls
// actual handler
int64_t syscall_hook(int64_t rdi, int64_t rsi,
                     int64_t rdx, int64_t __rcx __attribute__((unused)),
                     int64_t r8, int64_t r9,
                     int64_t r10_on_stack /* 4th arg for syscall */,
                     int64_t rax_on_stack /* syscall number */,
                     int64_t retptr)
{
    
    if (rax_on_stack == 435 /* __NR_clone3 */)
    {
        uint64_t *ca = (uint64_t *)rdi; /* struct clone_args */
        if (ca[0] /* flags */ & CLONE_VM)
        {
            ca[6] /* stack_size */ -= sizeof(uint64_t);
            *((uint64_t *)(ca[5] /* stack */ + ca[6] /* stack_size */)) = retptr;
        }
    }

    if (rax_on_stack == __NR_clone)
    {
        if (rdi & CLONE_VM)
        { // pthread creation
            /* push return address to the stack */
            rsi -= sizeof(uint64_t);
            *((uint64_t *)rsi) = retptr;
        }
    } 
    
    // Call the hook function from the loaded library
    // if (hook_fn) {
    //     return hook_fn(rdi, rsi, rdx, r10_on_stack, r8, r9, rax_on_stack);
    // }
    
    // Fallback if hook isn't loaded - use trigger_syscall directly
    // return trigger_syscall(rax_on_stack, rdi, rsi, rdx, r10_on_stack, r8, r9);
    return hook_fn(rdi, rsi, rdx, r10_on_stack, r8, r9, rax_on_stack);
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
    // memset(mem, 0x90, 0x1000);
    int i;
	for (i = 0; i < 512; i++)
		((uint8_t *) mem)[i] = 0x90;
    
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

// void rewrite_region_with_disasm(csh handle, uintptr_t start, uintptr_t end) {
//     uint8_t *code = (uint8_t*)start;
//     size_t code_size = end - start;
//     uint64_t address = start;
//     
//     cs_insn *insn;
//     size_t count = cs_disasm(handle, code, code_size, address, 0, &insn);
//     
//     if (count > 0) {
//         for (size_t i = 0; i < count; i++) {
//             // Check if this instruction is 'syscall'
//             if (insn[i].id == X86_INS_SYSCALL) {
//                 
//                 // Skip syscall instruction in trigger_syscall
//                 if (insn[i].address == (uint64_t)syscall_addr) {
//                     continue;
//                 }
//                 
//                 // Verify it's exactly 2 bytes (0x0F 0x05)
//                 if (insn[i].size == 2) {
//                     uint8_t *patch_addr = (uint8_t*)insn[i].address;
//                     
//                     // DEBUG("Patching syscall at 0x%lx", insn[i].address);
//                     
//                     // Replace with 'call *%rax' (0xFF 0xD0)
//                     patch_addr[0] = 0xFF;
//                     patch_addr[1] = 0xD0;
//                 } else {
//                     DEBUG("Warning: syscall instruction has unexpected size %d at 0x%lx", 
//                           insn[i].size, insn[i].address);
//                 }
//             }
//         }
//         cs_free(insn, count);
//     }
// }
// 
// void rewrite_syscalls() {
//     FILE *maps = fopen("/proc/self/maps", "r");
//     if (!maps) {
//         fprintf(stderr, "Failed to open /proc/self/maps\n");
//         exit(1);
//     }
//     
//     // Initialize Capstone disassembler
//     csh handle;
//     if (cs_open(CS_ARCH_X86, CS_MODE_64, &handle) != CS_ERR_OK) {
//         fprintf(stderr, "Failed to initialize disassembler\n");
//         exit(1);
//     }
//     
//     char line[1024];
//     while (fgets(line, sizeof(line), maps)) {
//         // Skip special regions
//         if (strstr(line, "[stack]") || strstr(line, "[vsyscall]") || 
//             strstr(line, "[vdso]")) {
//             continue;
//         }
//         
//         // Parse address range and permissions
//         uintptr_t start, end;
//         char perms[5] = {0};
//         if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3) {
//             continue;
//         }
//         
//         // Only process executable regions
//         if (strchr(perms, 'x') == NULL) {
//             continue;
//         }
//         
//         // Skip regions at address 0 (our trampoline)
//         if (start < 0x1000) {
//             continue;
//         }
//         
//         // Make the region writable
//         int orig_prot = 0;
//         if (strchr(perms, 'r')) orig_prot |= PROT_READ;
//         if (strchr(perms, 'w')) orig_prot |= PROT_WRITE;
//         if (strchr(perms, 'x')) orig_prot |= PROT_EXEC;
//         
//         if (mprotect((void*)start, end - start, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
//             continue;
//         }
//         
//         // Disassemble and patch safely
//         rewrite_region_with_disasm(handle, start, end);
//         
//         // Restore original permissions
//         mprotect((void*)start, end - start, orig_prot);
//     }
//     
//     cs_close(&handle);
//     fclose(maps);
// }
// Rewrite a buffer of code using Capstone
void disassemble_and_rewrite_capstone(uint8_t *code, size_t code_size, int mem_prot)
{
    // 1. Make the memory region writable
    if (mprotect((void *)code, code_size, PROT_READ | PROT_WRITE | PROT_EXEC) != 0)
    {
        perror("mprotect RWX");
        return;
    }

    // 2. Initialize Capstone
    csh handle;
    if (cs_open(CS_ARCH_X86, CS_MODE_64, &handle) != CS_ERR_OK)
    {
        fprintf(stderr, "Failed to initialize Capstone\n");
        goto restore_prot;
    }
    cs_option(handle, CS_OPT_DETAIL, CS_OPT_ON);
    // enable skip-data so Capstone keeps going past non-code bytes
    cs_option(handle, CS_OPT_SKIPDATA, CS_OPT_ON);

    // 3. Disassemble the code region
    cs_insn *insn = NULL;
    size_t count = cs_disasm(handle, code, code_size, (uint64_t)code, 0, &insn);
    if (count == 0)
    {
        fprintf(stderr, "Capstone failed to disassemble code\n");
        goto cleanup;
    }

    // 4. Iterate over each instruction
    for (size_t i = 0; i < count; i++)
    {
        cs_insn *ci = &insn[i];
        cs_detail *detail = ci->detail;
        uint8_t *ptr = (uint8_t *)ci->address;

        // invalid instruction, e.g. data
        if (ci->id == 0 || ci->detail == NULL)
        {
            continue;
        }

        // 4a. Adjust stack-offset memory accesses
        if (detail && detail->x86.op_count > 0)
        {
            for (int op_i = 0; op_i < detail->x86.op_count; op_i++)
            {
                cs_x86_op *op = &detail->x86.operands[op_i];
                if (op->type == X86_OP_MEM && op->mem.base == X86_REG_RSP)
                {
                    int64_t disp = op->mem.disp;
                    if (disp >= -0x78 && disp < 0)
                    {
                        uint8_t off = (uint8_t)disp;
                        // Search nearby bytes for pattern 0x24, off
                        for (size_t j = 0; j < 16; j++)
                        {
                            if (ptr[j] == 0x24 && ptr[j + 1] == off)
                            {
                                ptr[j + 1] = off - 8;
                                break;
                            }
                        }
                    }
                    // else if disp < -0x80 or too small, skip
                }
            }
        }

        // 4b. Replace syscall/sysenter with call *%rax
        if (ci->id == X86_INS_SYSCALL || ci->id == X86_INS_SYSENTER)
        {
            if (ptr != (uint8_t *)(uintptr_t)syscall_addr)
            {
                ptr[0] = 0xFF; // opcode for CALL r/m64
                ptr[1] = 0xD0; // modrm: rax
            }
        }
    }

    // Free Capstone structures
    cs_free(insn, count);
cleanup:
    cs_close(&handle);

restore_prot:
    // 5. Restore original protection
    if (mprotect((void *)code, code_size, mem_prot) != 0)
    {
        perror("mprotect restore");
    }
}

/* entry point for binary rewriting */
static void rewrite_syscalls(void)
{
    FILE *fp;
    /* get memory mapping information from procfs */
    assert((fp = fopen("/proc/self/maps", "r")) != NULL);
    {
        char buf[4096];
        while (fgets(buf, sizeof(buf), fp) != NULL)
        {
            /* we do not touch stack and vsyscall memory */
            if (((strstr(buf, "[stack]\n") == NULL)/* && (strstr(buf, "[vsyscall]\n") == NULL)*/))
            {
                int i = 0;
                char addr[65] = {0};
                char *c = strtok(buf, " ");
                while (c != NULL)
                {
                    switch (i)
                    {
                    case 0:
                        strncpy(addr, c, sizeof(addr) - 1);
                        break;
                    case 1:
                    {
                        int mem_prot = 0;
                        {
                            size_t j;
                            for (j = 0; j < strlen(c); j++)
                            {
                                if (c[j] == 'r')
                                    mem_prot |= PROT_READ;
                                if (c[j] == 'w')
                                    mem_prot |= PROT_WRITE;
                                if (c[j] == 'x')
                                    mem_prot |= PROT_EXEC;
                            }
                        }
                        /* rewrite code if the memory is executable */
                        if (mem_prot & PROT_EXEC)
                        {
                            size_t k;
                            for (k = 0; k < strlen(addr); k++)
                            {
                                if (addr[k] == '-')
                                {
                                    addr[k] = '\0';
                                    break;
                                }
                            }
                            {
                                int64_t from, to;
                                from = strtol(&addr[0], NULL, 16);
                                if (from == 0)
                                {
                                    /*
                                     * this is trampoline code.
                                     * so skip it.
                                     */
                                    break;
                                }
                                to = strtol(&addr[k + 1], NULL, 16);
                                disassemble_and_rewrite_capstone((uint8_t *)from,
                                                                 (size_t)to - from,
                                                                 mem_prot);
                            }
                        }
                    }
                    break;
                    }
                    if (i == 1)
                        break;
                    c = strtok(NULL, " ");
                    i++;
                }
            }
        }
    }
    fclose(fp);
}
// Find and replace syscall instructions
// void rewrite_syscalls() {
//     
//     FILE *maps = fopen("/proc/self/maps", "r");
//     if (!maps) {
//         fprintf(stderr, "Failed to open /proc/self/maps\n");
//         exit(1);
//     }
//     
//     char line[1024];
//     while (fgets(line, sizeof(line), maps)) {
//         // Skip special regions
//         if (strstr(line, "[stack]") || strstr(line, "[vsyscall]") || 
//             strstr(line, "[vdso]")) {
//             continue;
//         }
//         
//         // Parse address range and permissions
//         uintptr_t start, end;
//         char perms[5] = {0};
//         if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3) {
//             continue;
//         }
//         
//         // Only process executable regions
//         if (strchr(perms, 'x') == NULL) {
//             continue;
//         }
//         
//         // Skip regions at address 0 (our trampoline)
//         if (start < 0x1000) {
//             continue;
//         }
//         
//         // Make the region writable
//         int orig_prot = 0;
//         if (strchr(perms, 'r')) orig_prot |= PROT_READ;
//         if (strchr(perms, 'w')) orig_prot |= PROT_WRITE;
//         if (strchr(perms, 'x')) orig_prot |= PROT_EXEC;
//         
//         if (mprotect((void*)start, end - start, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
//             perror("mprotect error");
//             return;
//         }
//         
//         for (uintptr_t addr = start; addr < end - 1; addr++) {
//             unsigned char *ptr = (unsigned char*)addr;
// 
//             // Skip syscall instruction in trigger_syscall
//             if ((uintptr_t)ptr == (uintptr_t)syscall_addr) {
//                 continue;
//             }
// 
//             if (ptr[0] == 0x0F && ptr[1] == 0x05) {  // syscall
//                 ptr[0] = 0xff;  // call
//                 ptr[1] = 0xd0;  // *%rax
//             }
//         }
//         
//         // Restore original permissions
//         mprotect((void*)start, end - start, orig_prot);
//     }
//     
//     fclose(maps);
// }


// Function to load the hook library
static void load_hook_lib(void) {
    void *handle;
    const char *filename;
    
    // Get hook library path from environment
    filename = getenv("LIBZPHOOK");
    if (!filename) {
        DEBUG("env LIBZPHOOK is empty, skipping hook library load");
        return;
    }
    // Load the hook library into a new namespace to avoid symbol conflicts
    handle = dlmopen(LM_ID_NEWLM, filename, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        DEBUG("dlmopen failed");
        fprintf(stderr, "dlmopen failed: %s\n", dlerror());
        fprintf(stderr, "NOTE: This may occur if LDFLAGS are missing or if a C++ library needs 'extern \"C\"' declarations\n");
        exit(1);
    }
    
    // Get the __hook_init function from the loaded library
    void (*hook_init)(const syscall_hook_fn_t, syscall_hook_fn_t*);
    hook_init = dlsym(handle, "__hook_init");
    
    if (!hook_init) {
        fprintf(stderr, "Failed to find __hook_init in hook library: %s\n", dlerror());
        exit(1);
    }
    
    // Initialize the hook with a NULL hook_fn initially
    hook_init(trigger_syscall, &hook_fn);
    
}

// Library constructor
__attribute__((constructor))
void init() {

    DEBUG("libzpoline.so initializing");
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

    DEBUG("finish setup trampoline");

    
    // Rewrite syscalls with call *%rax
    rewrite_syscalls();
    
    DEBUG("finish rewrite syscalls");
    // Load hook library if specified
    load_hook_lib();
    
    
    if (getenv("ZDEBUG")) {
        asm("int3");  // Breakpoint for debugging
    }
    

}
