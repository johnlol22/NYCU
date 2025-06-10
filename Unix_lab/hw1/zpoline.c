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
static syscall_hook_fn_t hook_fn = NULL;


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
    // actual trampoline code
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
// actual handler
int64_t syscall_hook(int64_t rdi, int64_t rsi,
                     int64_t rdx, int64_t __rcx __attribute__((unused)),
                     int64_t r8, int64_t r9,
                     int64_t r10_on_stack /* 4th arg for syscall */,
                     int64_t rax_on_stack /* syscall number */,
                     int64_t retptr)
{
    
    // Handle clone3 syscall
    if (rax_on_stack == 435 /* __NR_clone3 */) {
        uint64_t *ca = (uint64_t *) rdi; /* struct clone_args */
        if (ca[0] /* flags */ & CLONE_VM) {
            // Make sure we can safely access the stack pointer
            if (ca[5] != 0 && ca[6] > sizeof(uint64_t)) {
                ca[6] /* stack_size */ -= sizeof(uint64_t);
                *((uint64_t *) (ca[5] /* stack */ + ca[6] /* stack_size */)) = retptr;
            }
        }
    }
    
    // Handle clone syscall 
    if (rax_on_stack == __NR_clone) {
        if (rdi & CLONE_VM) { // pthread creation
            // Make sure stack pointer is valid before modifying
            if (rsi != 0) {
                rsi -= sizeof(uint64_t);
                *((uint64_t *) rsi) = retptr;
            }
        }
    }
    
    // Call the hook function from the loaded library
    if (hook_fn) {
        return hook_fn(rdi, rsi, rdx, r10_on_stack, r8, r9, rax_on_stack);
    }
    
    // Fallback if hook isn't loaded - use trigger_syscall directly
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
        if (strstr(line, "[stack]") || 
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
        
        // In the rewrite_syscalls function, add AVX detection:
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


void selective_rewrite() {
    // Only rewrite specific libc functions
    void *libc = dlopen("libc.so.6", RTLD_LAZY);
    if (!libc) {
        DEBUG("Failed to load libc.so.6");
        return;
    }
    
    // Find addresses of syscall wrappers
    struct func_to_patch {
        const char *name;
        void *addr;
    } funcs[] = {
        {"open", NULL},
        {"openat", NULL},
        {"read", NULL},
        {"write", NULL},
        {"connect", NULL},
        {"execve", NULL},
        {NULL, NULL}
    };
    
    // Find function addresses
    for (int i = 0; funcs[i].name != NULL; i++) {
        funcs[i].addr = dlsym(libc, funcs[i].name);
        // if (funcs[i].addr) {
        //     DEBUG("Found %s at %p", funcs[i].name, funcs[i].addr);
        // }
    }
    
    // Patch each function
    for (int i = 0; funcs[i].name != NULL; i++) {
        if (!funcs[i].addr) continue;
        
        // Make memory writable
        uintptr_t page_start = ((uintptr_t)funcs[i].addr) & ~0xFFF;
        mprotect((void*)page_start, 4096, PROT_READ | PROT_WRITE | PROT_EXEC);
        
        // Find syscall instruction (limited scan)
        unsigned char *func = (unsigned char*)funcs[i].addr;
        for (int j = 0; j < 128; j++) {
            if (func[j] == 0x0F && func[j+1] == 0x05) {  // syscall
                // DEBUG("Patching syscall in %s at offset %d", funcs[i].name, j);
                func[j] = 0xFF;  // call
                func[j+1] = 0xD0;  // *%rax
                break;
            }
        }
        
        // Restore protection
        mprotect((void*)page_start, 4096, PROT_READ | PROT_EXEC);
    }
    
    dlclose(libc);
}

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
    // DEBUG("libzpoline.so initializing");

    // Check if running under Python
    char proc_name[256] = {0};
    int len = readlink("/proc/self/exe", proc_name, sizeof(proc_name)-1);
    if (len > 0) {
        proc_name[len] = '\0';
        if (strstr(proc_name, "python")) {
            // Use selective approach for Python
            setup_trampoline();
            selective_rewrite();
            load_hook_lib();
            return;
        }
    }
    
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

    // DEBUG("finish setup trampoline");
    // Rewrite syscalls with call *%rax
    rewrite_syscalls();
    
    // DEBUG("finish rewrite syscalls");
    // Load hook library if specified
    load_hook_lib();

}