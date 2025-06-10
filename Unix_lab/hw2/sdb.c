#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <elf.h>
#include <capstone/capstone.h>
#include <errno.h>
#include <signal.h>
#include <limits.h>

#define MAX_CMD_LEN 256
#define MAX_PATH_LEN 1024
#define DISASM_COUNT 5


typedef struct {
    uint64_t start;
    uint64_t end;
} exec_region_t;

typedef struct {
    int index;
    uint64_t addr;
    unsigned char original_byte;
} breakpoint_t;

typedef struct {
    uint64_t addr;
    unsigned char new_byte;
    unsigned char original_byte;
    int is_active;  // Track if patch is currently applied
} patch_t;

typedef struct {
    uint64_t addr;
    unsigned char original_byte;
} original_memory_t;


typedef struct {
    char *program_path;
    char *program_name;
    pid_t child_pid;
    uint64_t entry_point;
    uint64_t base_address;
    int is_loaded;
    int is_pie;
    csh capstone_handle;

    // Executable regions
    exec_region_t *exec_regions;
    int num_exec_regions;

    // Breakpoints and patches
    breakpoint_t *breakpoints;
    int num_breakpoints;
    int next_breakpoint_index;

    patch_t *patches;
    original_memory_t *original_memory;
    int num_patches;
    int num_original_memory;

    // Syscall tracking
    int in_syscall;
    uint64_t syscall_nr;
    uint64_t syscall_addr;

} debugger_t;


// for reg
uint64_t get_register(pid_t pid, int reg_offset);
// for checking executable
int load_executable_regions(debugger_t *dbg);
int is_in_executable_region(debugger_t *dbg, uint64_t addr);
// for breakpoint
int add_breakpoint(debugger_t *dbg, uint64_t addr);
int delete_breakpoint(debugger_t *dbg, int id);
void info_breakpoints(debugger_t *dbg);
int is_breakpoint_at(debugger_t *dbg, uint64_t addr);
int set_breakpoint_rva(debugger_t *dbg, uint64_t offset);
int set_breakpoint(debugger_t *dbg, uint64_t addr);
// for si
int step_instruction(debugger_t *dbg);
// for cont
int continue_execution(debugger_t *dbg);
// for info reg
void info_registers(debugger_t *dbg);
// for patch
int hex_string_to_bytes(const char *hex_str, unsigned char *bytes, size_t max_bytes);
uint64_t parse_hex_address(const char *hex_str);
unsigned char get_original_byte(debugger_t *dbg, uint64_t addr);
int add_patch(debugger_t *dbg, uint64_t addr, unsigned char new_byte);
int patch_memory(debugger_t *dbg, uint64_t addr, const char *hex_str);
// for syscall
int is_syscall_instruction(debugger_t *dbg, uint64_t addr);
int execute_until_syscall_or_breakpoint(debugger_t *dbg);
// for read memory
int is_memory_accessible(debugger_t *dbg, uint64_t addr);
int read_memory_safe(pid_t pid, uint64_t addr, unsigned char *buffer, size_t size);
void read_memory(pid_t pid, uint64_t addr, unsigned char *buffer, size_t size);
void read_memory_with_patches(debugger_t *dbg, uint64_t addr, unsigned char *buffer, size_t size);
// Function prototypes
char* format_hex_bytes(unsigned char *bytes, size_t size);
void disassemble_at_rip(debugger_t *dbg, int force_disasm);
int parse_elf(const char *path, debugger_t *dbg);
int load_program(debugger_t *dbg, const char *path);
int start_program(debugger_t *dbg);
void command_loop(debugger_t *dbg);
void cleanup_debugger(debugger_t *dbg);

// Get register value from child process
uint64_t get_register(pid_t pid, int reg_offset) {
    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) < 0) {
        perror("ptrace GETREGS");
        return 0;
    }
    
    // Return specific register based on offset
    switch (reg_offset) {
        case 0: return regs.rax;    // RAX - syscall number/return value
        case 1: return regs.rbx;
        case 2: return regs.rcx;
        case 3: return regs.rdx;
        case 4: return regs.rsi;
        case 5: return regs.rdi;
        case 6: return regs.rbp;
        case 7: return regs.rsp;
        case 8: return regs.r8;
        case 9: return regs.r9;
        case 10: return regs.r10;
        case 11: return regs.r11;
        case 12: return regs.r12;
        case 13: return regs.r13;
        case 14: return regs.r14;
        case 15: return regs.r15;
        case 16: return regs.rip;
        default: return 0;
    }
}

// Load executable regions from /proc/PID/maps
int load_executable_regions(debugger_t *dbg) {
    if (dbg->child_pid <= 0) return -1;
    
    char maps_path[64];
    snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", dbg->child_pid);
    
    FILE *maps = fopen(maps_path, "r");
    if (!maps) return -1;
    
    // Free existing regions
    if (dbg->exec_regions) {
        free(dbg->exec_regions);
        dbg->exec_regions = NULL;
        dbg->num_exec_regions = 0;
    }
    
    // Count executable regions first
    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), maps)) {
        if (strstr(line, " r-x") || strstr(line, " rwx")) {
            count++;
        }
    }
    
    if (count == 0) {
        fclose(maps);
        return -1;
    }
    
    // Allocate and populate regions
    dbg->exec_regions = calloc(count, sizeof(exec_region_t));
    dbg->num_exec_regions = 0;
    
    rewind(maps);       // point the file pointer to the start of a file
    while (fgets(line, sizeof(line), maps) && dbg->num_exec_regions < count) {
        if (strstr(line, " r-x") || strstr(line, " rwx")) {
            uint64_t start, end;
            if (sscanf(line, "%lx-%lx", &start, &end) == 2) {
                dbg->exec_regions[dbg->num_exec_regions].start = start;
                dbg->exec_regions[dbg->num_exec_regions].end = end;
                dbg->num_exec_regions++;
            }
        }
    }
    
    fclose(maps);
    return 0;
}

// Check if address is in executable region
int is_in_executable_region(debugger_t *dbg, uint64_t addr) {
    for (int i = 0; i < dbg->num_exec_regions; i++) {
        if (addr >= dbg->exec_regions[i].start && addr < dbg->exec_regions[i].end) {
            return 1;
        }
    }
    return 0;
}

// Add breakpoint tracking
int add_breakpoint(debugger_t *dbg, uint64_t addr) {
    dbg->breakpoints = realloc(dbg->breakpoints, (dbg->num_breakpoints + 1) * sizeof(breakpoint_t));
    if (!dbg->breakpoints) return -1;
    
    // Read original byte
    unsigned char orig_byte;
    read_memory(dbg->child_pid, addr, &orig_byte, 1);   // read one byte
    
    
    dbg->breakpoints[dbg->num_breakpoints].index = dbg->next_breakpoint_index++;
    dbg->breakpoints[dbg->num_breakpoints].addr = addr;
    dbg->breakpoints[dbg->num_breakpoints].original_byte = orig_byte;
    dbg->num_breakpoints++;
    
    return 0;
}

// Fixed delete_breakpoint function for dynamic memory
int delete_breakpoint(debugger_t *dbg, int id) {
    if (!dbg->is_loaded || dbg->child_pid <= 0) {
        return -1;
    }
    
    // Find breakpoint by index
    int found_pos = -1;
    for (int i = 0; i < dbg->num_breakpoints; i++) {
        if (dbg->breakpoints[i].index == id) {
            found_pos = i;
            break;
        }
    }
    
    if (found_pos == -1) {
        return -1; // Breakpoint not found
    }
    
    // Restore original byte in memory
    uint64_t addr = dbg->breakpoints[found_pos].addr;
    unsigned char orig_byte = dbg->breakpoints[found_pos].original_byte;
    
    // Try ptrace first
    errno = 0;
    long data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, addr & ~7, NULL);
    if (errno == 0) {
        // ptrace works
        int byte_offset = addr & 7;
        long mask = 0xffL << (byte_offset * 8);
        long new_data = (data & ~mask) | ((long)orig_byte << (byte_offset * 8));
        if (ptrace(PTRACE_POKEDATA, dbg->child_pid, addr & ~7, new_data) < 0) {
            return -1;
        }
    } else {
        // Fall back to /proc/pid/mem
        char mem_path[64];
        snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
        
        int mem_fd = open(mem_path, O_WRONLY);
        if (mem_fd < 0) {
            return -1;
        }
        
        if (lseek(mem_fd, addr, SEEK_SET) == -1) {
            close(mem_fd);
            return -1;
        }
        
        if (write(mem_fd, &orig_byte, 1) != 1) {
            close(mem_fd);
            return -1;
        }
        
        close(mem_fd);
    }
    
    // Remove from array by shifting elements
    for (int i = found_pos; i < dbg->num_breakpoints - 1; i++) {
        dbg->breakpoints[i] = dbg->breakpoints[i + 1];
    }
    dbg->num_breakpoints--;
    
    // Realloc to smaller size if needed
    if (dbg->num_breakpoints == 0) {
        free(dbg->breakpoints);
        dbg->breakpoints = NULL;
    } else {
        dbg->breakpoints = realloc(dbg->breakpoints, dbg->num_breakpoints * sizeof(breakpoint_t));
    }
    
    return 0;
}

// Display breakpoint information
void info_breakpoints(debugger_t *dbg) {
    if (dbg->num_breakpoints == 0) {
        printf("** no breakpoints.\n");
        return;
    }
    
    printf("Num\tAddress\n");
    for (int i = 0; i < dbg->num_breakpoints; i++) {
        printf("%d\t0x%lx\n", dbg->breakpoints[i].index, dbg->breakpoints[i].addr);
    }
}

// Check if there's a breakpoint at given address
int is_breakpoint_at(debugger_t *dbg, uint64_t addr) {
    for (int i = 0; i < dbg->num_breakpoints; i++) {
        if (dbg->breakpoints[i].addr == addr) {
            return 1;
        }
    }
    return 0;
}

// Set breakpoint at RVA (relative virtual address) offset from base
int set_breakpoint_rva(debugger_t *dbg, uint64_t offset) {
    if (!dbg->is_loaded || dbg->child_pid <= 0) {
        return -1;
    }
    
    // For PIE binaries, we need the base address
    if (dbg->base_address == 0) {
        // Try to get base address from /proc/PID/maps
        char maps_path[64];
        snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", dbg->child_pid);
        
        FILE *maps = fopen(maps_path, "r");
        if (maps) {
            char line[256];
            if (fgets(line, sizeof(line), maps)) {
                sscanf(line, "%lx-", &dbg->base_address);
            }
            fclose(maps);
        }
        
        if (dbg->base_address == 0) {
            return -1; // Could not determine base address
        }
    }
    
    // Calculate actual address
    uint64_t actual_addr = dbg->base_address + offset;
    
    // Use existing breakpoint setting function
    return set_breakpoint(dbg, actual_addr);
}

// Updated set_breakpoint to handle dynamic code
int set_breakpoint(debugger_t *dbg, uint64_t addr) {
    if (!dbg->is_loaded || dbg->child_pid <= 0) {
        return -1;
    }
    
    // Load executable regions if not already loaded
    if (dbg->num_exec_regions == 0) {
        load_executable_regions(dbg);
    }
    
    // Check if address is accessible (includes dynamic code validation)
    if (!is_memory_accessible(dbg, addr)) {
        return -1;
    }
    
    // Check if breakpoint already exists
    if (is_breakpoint_at(dbg, addr)) {
        return 0;
    }
    
    // Read original byte
    unsigned char original_byte;
    if (read_memory_safe(dbg->child_pid, addr, &original_byte, 1) != 0) {
        return -1;
    }
    
    // Try ptrace first
    errno = 0;
    long data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, addr & ~7, NULL);
    if (errno == 0) {
        // ptrace works - use it
        int byte_offset = addr & 7;
        long mask = 0xffL << (byte_offset * 8);
        long new_data = (data & ~mask) | (0xccL << (byte_offset * 8));
        
        if (ptrace(PTRACE_POKEDATA, dbg->child_pid, addr & ~7, new_data) < 0) {
            return -1;
        }
    } else {
        // Fall back to /proc/pid/mem
        char mem_path[64];
        snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
        
        int mem_fd = open(mem_path, O_WRONLY);
        if (mem_fd < 0) return -1;
        
        if (lseek(mem_fd, addr, SEEK_SET) == -1) {
            close(mem_fd);
            return -1;
        }
        
        unsigned char breakpoint_byte = 0xcc;
        if (write(mem_fd, &breakpoint_byte, 1) != 1) {
            close(mem_fd);
            return -1;
        }
        close(mem_fd);
    }
    
    // Add to breakpoint tracking
    if (add_breakpoint(dbg, addr) < 0) {
        // Restore on failure (try both methods)
        errno = 0;
        data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, addr & ~7, NULL);
        if (errno == 0) {
            int byte_offset = addr & 7;
            long mask = 0xffL << (byte_offset * 8);
            long restore_data = (data & ~mask) | ((long)original_byte << (byte_offset * 8));
            ptrace(PTRACE_POKEDATA, dbg->child_pid, addr & ~7, restore_data);
        } else {
            char mem_path[64];
            snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
            int mem_fd = open(mem_path, O_WRONLY);
            if (mem_fd >= 0) {
                lseek(mem_fd, addr, SEEK_SET);
                write(mem_fd, &original_byte, 1);
                close(mem_fd);
            }
        }
        return -1;
    }
    
    // Store correct original byte
    dbg->breakpoints[dbg->num_breakpoints - 1].original_byte = original_byte;
    return 0;
}

// Update step_instruction to handle dynamic code breakpoints
int step_instruction(debugger_t *dbg) {
    if (!dbg->is_loaded || dbg->child_pid <= 0) {
        return -1;
    }
    
    // uint64_t current_rip = get_rip(dbg->child_pid);
    uint64_t current_rip = get_register(dbg->child_pid, 16);
    if (current_rip == 0) return -1;
    
    // Check if we're at a breakpoint
    for (int i = 0; i < dbg->num_breakpoints; i++) {
        if (dbg->breakpoints[i].addr == current_rip) {
            // Temporarily restore original instruction
            unsigned char orig_byte = dbg->breakpoints[i].original_byte;
            
            // Try ptrace first
            errno = 0;
            long data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, current_rip & ~7, NULL);
            if (errno == 0) {
                int byte_offset = current_rip & 7;          // byte addr of 0xcc related to that block?
                long mask = 0xffL << (byte_offset * 8);
                long orig_data = (data & ~mask) | ((long)orig_byte << (byte_offset * 8));
                ptrace(PTRACE_POKEDATA, dbg->child_pid, current_rip & ~7, orig_data);
                
                // Single step
                ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL);
                int status;
                waitpid(dbg->child_pid, &status, 0);
                
                // Restore breakpoint
                long bp_data = (data & ~mask) | (0xccL << (byte_offset * 8));
                ptrace(PTRACE_POKEDATA, dbg->child_pid, current_rip & ~7, bp_data);
            } else {
                // Use /proc/pid/mem method
                char mem_path[64];
                snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
                
                int mem_fd = open(mem_path, O_WRONLY);
                if (mem_fd >= 0) {
                    lseek(mem_fd, current_rip, SEEK_SET);       // move to rip
                    write(mem_fd, &orig_byte, 1);               // write the origal byte
                    close(mem_fd);
                    
                    // Single step
                    ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL);
                    int status;
                    waitpid(dbg->child_pid, &status, 0);
                    
                    // Restore breakpoint
                    mem_fd = open(mem_path, O_WRONLY);
                    if (mem_fd >= 0) {
                        lseek(mem_fd, current_rip, SEEK_SET);
                        unsigned char bp_byte = 0xcc;
                        write(mem_fd, &bp_byte, 1);
                        close(mem_fd);
                    }
                }
            }
            int status;
            if (WIFEXITED(status)) {
                printf("** the target program terminated.\n");
                return 1;
            }
            
            // Check for new breakpoint
            uint64_t new_rip = get_register(dbg->child_pid, 16);
            // uint64_t new_rip = get_rip(dbg->child_pid);
            for (int j = 0; j < dbg->num_breakpoints; j++) {
                if (dbg->breakpoints[j].addr == new_rip) {
                    printf("** hit a breakpoint at 0x%lx.\n", new_rip);
                    break;
                }
            }
            
            disassemble_at_rip(dbg, 1);
            return 0;
        }
    }
    
    // Normal single step (no breakpoint at current position)
    ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL);
    int status;
    waitpid(dbg->child_pid, &status, 0);
    
    if (WIFEXITED(status)) {
        printf("** the target program terminated.\n");
        return 1;
    }
    
    // uint64_t new_rip = get_rip(dbg->child_pid);
    uint64_t new_rip = get_register(dbg->child_pid, 16);
    for (int i = 0; i < dbg->num_breakpoints; i++) {
        if (dbg->breakpoints[i].addr == new_rip) {
            printf("** hit a breakpoint at 0x%lx.\n", new_rip);
            break;
        }
    }
    
    disassemble_at_rip(dbg, 1);
    return 0;
}

// Fixed continue_execution function for dynamic memory breakpoints
// int continue_execution(debugger_t *dbg) {
//     if (!dbg->is_loaded || dbg->child_pid <= 0) {
//         return -1;
//     }
//     
//     uint64_t current_rip = get_register(dbg->child_pid, 16);
//     int breakpoint_at_rip = -1;
//     
//     // Check if current RIP is at a breakpoint
//     for (int i = 0; i < dbg->num_breakpoints; i++) {
//         if (dbg->breakpoints[i].addr == current_rip) {
//             breakpoint_at_rip = i;
//             break;
//         }
//     }
//     
//     if (breakpoint_at_rip >= 0) {
//         printf("Debug: stepping over breakpoint at 0x%lx\n", current_rip);
//         
//         unsigned char orig_byte = dbg->breakpoints[breakpoint_at_rip].original_byte;
//         
//         // Try ptrace first
//         errno = 0;
//         long original_data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, current_rip & ~7, NULL);
//         if (errno == 0) {
//             // Ptrace method
//             int byte_offset = current_rip & 7;
//             long mask = 0xffL << (byte_offset * 8);
//             long restored_data = (original_data & ~mask) | ((long)orig_byte << (byte_offset * 8));
//             ptrace(PTRACE_POKEDATA, dbg->child_pid, current_rip & ~7, restored_data);
//             
//             // Single step
//             ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL);
//             int status;
//             waitpid(dbg->child_pid, &status, 0);
//             
//             if (WIFEXITED(status)) {
//                 printf("** the target program terminated.\n");
//                 return 0;
//             }
//             
//             // Restore breakpoint
//             long new_data = (original_data & ~mask) | (0xccL << (byte_offset * 8));
//             ptrace(PTRACE_POKEDATA, dbg->child_pid, current_rip & ~7, new_data);
//         } else {
//             // /proc/pid/mem method
//             char mem_path[64];
//             snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
//             
//             int mem_fd = open(mem_path, O_WRONLY);
//             if (mem_fd < 0) {
//                 printf("** continue failed: cannot access memory.\n");
//                 return -1;
//             }
//             
//             // Restore original byte
//             lseek(mem_fd, current_rip, SEEK_SET);
//             if (write(mem_fd, &orig_byte, 1) != 1) {
//                 close(mem_fd);
//                 printf("** continue failed: cannot restore instruction.\n");
//                 return -1;
//             }
//             close(mem_fd);
//             
//             // Single step
//             ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL);
//             int status;
//             waitpid(dbg->child_pid, &status, 0);
//             
//             if (WIFEXITED(status)) {
//                 printf("** the target program terminated.\n");
//                 return 0;
//             }
//             
//             // Restore breakpoint
//             mem_fd = open(mem_path, O_WRONLY);
//             if (mem_fd >= 0) {
//                 lseek(mem_fd, current_rip, SEEK_SET);
//                 unsigned char bp_byte = 0xcc;
//                 write(mem_fd, &bp_byte, 1);
//                 close(mem_fd);
//             }
//         }
//     }
// 
//     if (ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL) < 0) {
//         perror("Debug: SINGLESTEP failed");
//         return -1;
//     }
//     
//     int status;
//     if (waitpid(dbg->child_pid, &status, 0) < 0) {
//         perror("Debug: waitpid failed");
//         return -1;
//     }
//     
//     printf("Debug: stepped, status = %d\n", status);
//     
//     // Continue execution
//     ptrace(PTRACE_CONT, dbg->child_pid, NULL, NULL);
//     
//     while (1) {
//         int status;
//         waitpid(dbg->child_pid, &status, 0);
//         
//         if (WIFEXITED(status)) {
//             printf("** the target program terminated.\n");
//             break;
//         }
//         
//         if (WIFSTOPPED(status)) {
//             int sig = WSTOPSIG(status);
//             
//             if (sig == SIGTRAP) {
//                 uint64_t rip = get_register(dbg->child_pid, 16);
//                 uint64_t bp_addr = rip - 1;
//                 int hit_breakpoint = -1;
//                 
//                 // Check if we hit a breakpoint
//                 for (int i = 0; i < dbg->num_breakpoints; i++) {
//                     if (dbg->breakpoints[i].addr == bp_addr) {
//                         hit_breakpoint = i;
//                         break;
//                     }
//                 }
//                 
//                 if (hit_breakpoint >= 0) {
//                     // Move RIP back to the breakpoint instruction
//                     struct user_regs_struct regs;
//                     ptrace(PTRACE_GETREGS, dbg->child_pid, NULL, &regs);
//                     regs.rip = bp_addr;
//                     ptrace(PTRACE_SETREGS, dbg->child_pid, NULL, &regs);
//                     
//                     printf("** hit a breakpoint at 0x%lx.\n", bp_addr);
//                     disassemble_at_rip(dbg, 1);
//                     break;
//                 } else {
//                     // Not our breakpoint, continue
//                     ptrace(PTRACE_CONT, dbg->child_pid, NULL, NULL);
//                 }
//             } else {
//                 // Other signal, continue
//                 ptrace(PTRACE_CONT, dbg->child_pid, NULL, NULL);
//             }
//         }
//     }
//     
//     return 0;
// }

// Continue program execution
int continue_execution(debugger_t *dbg) {
    if (!dbg->is_loaded || dbg->child_pid <= 0) {
        return -1;
    }
    
    // uint64_t current_rip = get_rip(dbg->child_pid);
    uint64_t current_rip = get_register(dbg->child_pid, 16);
    int breakpoint_at_rip = -1;
    
    // Check if current RIP is at a breakpoint
    for (int i = 0; i < dbg->num_breakpoints; i++) {
        if (dbg->breakpoints[i].addr == current_rip) {
            breakpoint_at_rip = i;
            break;
        }
    }
    
    // If at breakpoint, step over it first
    if (breakpoint_at_rip >= 0) {
        // Temporarily restore original instruction
        errno = 0;
        // Correct - word-aligned with byte offset
        uint64_t aligned_addr = current_rip & ~7;
        int byte_offset = current_rip & 7;
        long original_data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, aligned_addr, NULL);
        if (errno != 0) printf("ptrace peek error\n");

        long mask = 0xffL << (byte_offset * 8);
        long restored_data = (original_data & ~mask) | 
                    ((long)dbg->breakpoints[breakpoint_at_rip].original_byte << (byte_offset * 8));
        ptrace(PTRACE_POKEDATA, dbg->child_pid, aligned_addr, restored_data);
        if (errno != 0) printf("ptrace poke error\n");
        
        // Single step over the instruction (1st PTRACE_SINGLE_STEP)
        ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL);
        int status;
        waitpid(dbg->child_pid, &status, 0);
        
        // Check if program terminated during single step
        if (WIFEXITED(status)) {
            printf("** the target program terminated.\n");
            return 0;
        }
        
        // Restore breakpoint
        // After single step, restore breakpoint
        long new_data = (original_data & ~mask) | (0xccL << (byte_offset * 8));
        ptrace(PTRACE_POKEDATA, dbg->child_pid, aligned_addr, new_data);
    }
    
    // Continue execution
    ptrace(PTRACE_CONT, dbg->child_pid, NULL, NULL);
    
    while (1) {
        int status;
        waitpid(dbg->child_pid, &status, 0);

        if (WIFEXITED(status)) {
            printf("** the target program terminated.\n");
            break;
        }
        
        if (WIFSTOPPED(status)) {
            int sig = WSTOPSIG(status);
            
            if (sig == SIGTRAP) {
                // uint64_t rip = get_rip(dbg->child_pid);
                uint64_t rip = get_register(dbg->child_pid, 16);
                
                // Check if we hit a breakpoint (RIP-1 because int3 advances RIP)
                uint64_t bp_addr = rip - 1;     // after executing 0xcc then signal trap
                int hit_breakpoint = -1;
                
                for (int i = 0; i < dbg->num_breakpoints; i++) {
                    if (dbg->breakpoints[i].addr == bp_addr) {
                        hit_breakpoint = i;
                        break;
                    }
                }
                
                if (hit_breakpoint >= 0) {
                    // Move RIP back to the breakpoint instruction
                    struct user_regs_struct regs;
                    ptrace(PTRACE_GETREGS, dbg->child_pid, NULL, &regs);
                    regs.rip = bp_addr;
                    ptrace(PTRACE_SETREGS, dbg->child_pid, NULL, &regs);
                    
                    printf("** hit a breakpoint at 0x%lx.\n", bp_addr);
                    disassemble_at_rip(dbg, 1);
                    break;
                } else {
                    // Not our breakpoint, continue
                    ptrace(PTRACE_CONT, dbg->child_pid, NULL, NULL);
                }
            } else {
                // Other signal, continue
                ptrace(PTRACE_CONT, dbg->child_pid, NULL, NULL);
            }
        }
    }
    
    return 0;
}

// Display register information
void info_registers(debugger_t *dbg) {
    if (!dbg->is_loaded || dbg->child_pid <= 0) {
        printf("** please load a program first.\n");
        return;
    }
    
    struct user_regs_struct regs;
    if (ptrace(PTRACE_GETREGS, dbg->child_pid, NULL, &regs) < 0) {
        perror("ptrace GETREGS");
        return;
    }
    
    // Print registers 3 per line in 64-bit hex format
    printf("$rax 0x%016llx    $rbx 0x%016llx    $rcx 0x%016llx\n", 
           regs.rax, regs.rbx, regs.rcx);
    printf("$rdx 0x%016llx    $rsi 0x%016llx    $rdi 0x%016llx\n", 
           regs.rdx, regs.rsi, regs.rdi);
    printf("$rbp 0x%016llx    $rsp 0x%016llx    $r8  0x%016llx\n", 
           regs.rbp, regs.rsp, regs.r8);
    printf("$r9  0x%016llx    $r10 0x%016llx    $r11 0x%016llx\n", 
           regs.r9, regs.r10, regs.r11);
    printf("$r12 0x%016llx    $r13 0x%016llx    $r14 0x%016llx\n", 
           regs.r12, regs.r13, regs.r14);
    printf("$r15 0x%016llx    $rip 0x%016llx    $eflags 0x%016llx\n", 
           regs.r15, regs.rip, regs.eflags);
}

// Convert hex string to bytes
int hex_string_to_bytes(const char *hex_str, unsigned char *bytes, size_t max_bytes) {
    if (!hex_str || !bytes) return -1;
    
    size_t len = strlen(hex_str);
    if (len > max_bytes * 2 || len % 2 != 0) return -1;
    
    for (size_t i = 0; i < len; i += 2) {
        char hex_byte[3] = {hex_str[i], hex_str[i+1], '\0'};
        char *endptr;
        unsigned long val = strtoul(hex_byte, &endptr, 16);
        
        if (*endptr != '\0') return -1;
        bytes[i/2] = (unsigned char)val;
    }
    
    return len / 2; // Return number of bytes
}

// Parse hex address (supports both 0x prefix and without)
uint64_t parse_hex_address(const char *hex_str) {
    if (!hex_str) return 0;
    
    char *endptr;
    uint64_t addr;
    
    // Handle both "0x123" and "123" formats
    if (strncmp(hex_str, "0x", 2) == 0 || strncmp(hex_str, "0X", 2) == 0) {
        addr = strtoull(hex_str, &endptr, 16);      // string to unsigned long long, 16 means hex
    } else {
        addr = strtoull(hex_str, &endptr, 16);
    }
    
    // Check if conversion was successful
    if (*endptr != '\0' || endptr == hex_str) {
        return 0; // Invalid format
    }
    
    return addr;
}


// Add patch tracking
// int add_patch(debugger_t *dbg, uint64_t addr, unsigned char new_byte) {
//     dbg->patches = realloc(dbg->patches, (dbg->num_patches + 1) * sizeof(patch_t));
//     if (!dbg->patches) return -1;
//     
//     // Read original byte
//     unsigned char orig_byte;
//     read_memory(dbg->child_pid, addr, &orig_byte, 1);
//     
//     dbg->patches[dbg->num_patches].addr = addr;
//     dbg->patches[dbg->num_patches].new_byte = new_byte;
//     dbg->patches[dbg->num_patches].original_byte = orig_byte;
//     dbg->num_patches++;
//     
//     return 0;
// }

// unsigned char get_original_byte(debugger_t *dbg, uint64_t addr) {
//     // Check if we have the original byte stored
//     for (int i = 0; i < dbg->num_original_memory; i++) {
//         if (dbg->original_memory[i].addr == addr) {
//             return dbg->original_memory[i].original_byte;
//         }
//     }
//     
//     // Not found - read current byte and store as original
//     unsigned char byte;
//     read_memory_safe(dbg->child_pid, addr, &byte, 1);
//     
//     // Store for future reference
//     dbg->original_memory = realloc(dbg->original_memory, 
//                                   (dbg->num_original_memory + 1) * sizeof(original_memory_t));
//     dbg->original_memory[dbg->num_original_memory].addr = addr;
//     dbg->original_memory[dbg->num_original_memory].original_byte = byte;
//     dbg->num_original_memory++;
//     
//     return byte;
// }

int patch_memory(debugger_t *dbg, uint64_t addr, const char *hex_str) {
    if (!dbg->is_loaded || dbg->child_pid <= 0) return -1;
    
    // Convert hex string to bytes
    unsigned char patch_bytes[1024];
    int patch_size = hex_string_to_bytes(hex_str, patch_bytes, sizeof(patch_bytes));
    if (patch_size < 0) return -1;
    
    // Load executable regions if needed
    if (dbg->num_exec_regions == 0) {
        load_executable_regions(dbg);
    }
    
    // Validate address range
    for (int i = 0; i < patch_size; i++) {
        if (!is_in_executable_region(dbg, addr + i)) {
            return -1;
        }
    }
    // Handle breakpoints - need to update original bytes for affected breakpoints
    for (int i = 0; i < patch_size; i++) {
        uint64_t current_addr = addr + i;
        
        int found = -1;
        int breakpoint_found = 0;
        for (int j = 0; j < dbg->num_patches; j++) {
            if (dbg->patches[j].addr == current_addr) {
                found = j;
                break;
            }
        }
        
        if (found >= 0) {
            // Update existing patch
            dbg->patches[found].new_byte = patch_bytes[i];
        } else {
            // Add new patch
            unsigned char orig_byte;
            read_memory_safe(dbg->child_pid, current_addr, &orig_byte, 1);
            
            dbg->patches = realloc(dbg->patches, (dbg->num_patches + 1) * sizeof(patch_t));
            dbg->patches[dbg->num_patches].addr = current_addr;
            dbg->patches[dbg->num_patches].new_byte = patch_bytes[i];
            dbg->patches[dbg->num_patches].original_byte = orig_byte;
            dbg->num_patches++;
        }

        // Find if there's a breakpoint at this address
        for (int j = 0; j < dbg->num_breakpoints; j++) {
            if (dbg->breakpoints[j].addr == current_addr) {
                // Update the original byte for this breakpoint
                dbg->breakpoints[j].original_byte = patch_bytes[i];
                breakpoint_found = 1;
                break;
            }
        }

        if (!breakpoint_found) {
            uint64_t current_addr = addr + i;
            // Apply patch to memory first
            errno = 0;
            long data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, current_addr & ~7, NULL);
            if (errno == 0) {
                int byte_offset = current_addr & 7;
                long mask = 0xffL << (byte_offset * 8);
                long new_data = (data & ~mask) | ((long)patch_bytes[i] << (byte_offset * 8));
                ptrace(PTRACE_POKEDATA, dbg->child_pid, current_addr & ~7, new_data);
            } else {
                // Use /proc/pid/mem fallback
                char mem_path[64];
                snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
                int mem_fd = open(mem_path, O_WRONLY);
                if (mem_fd >= 0) {
                    lseek(mem_fd, current_addr, SEEK_SET);
                    write(mem_fd, &patch_bytes[i], 1);
                    close(mem_fd);
                }
            }
        }
    }
    
    // Apply patches to memory
    // for (int i = 0; i < patch_size; i++) {
    //     uint64_t current_addr = addr + i;
    //     
    //     // Track this patch
    //     add_patch(dbg, current_addr, patch_bytes[i]);
    // 
    //     // Read current data
    //     long data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, current_addr & ~7, NULL);
    //     if (errno != 0) return -1;
    //     
    //     // Modify the appropriate byte
    //     int byte_offset = current_addr & 7;
    //     long mask = 0xffL << (byte_offset * 8);
    //     long new_byte = (long)patch_bytes[i] << (byte_offset * 8);
    //     data = (data & ~mask) | new_byte;
    //     
    //     // Write back
    //     if (ptrace(PTRACE_POKEDATA, dbg->child_pid, current_addr & ~7, data) < 0) {
    //         return -1;
    //     }
    //     
    // }
    
    return 0;
}

// Fixed patch_memory function
// int patch_memory(debugger_t *dbg, uint64_t addr, const char *hex_str) {
//     if (!dbg->is_loaded || dbg->child_pid <= 0) return -1;
//     
//     // Convert hex string to bytes
//     unsigned char patch_bytes[1024];
//     int patch_size = hex_string_to_bytes(hex_str, patch_bytes, sizeof(patch_bytes));
//     if (patch_size < 0) return -1;
//     
//     // Load executable regions if needed
//     if (dbg->num_exec_regions == 0) {
//         load_executable_regions(dbg);
//     }
//     
//     // Validate address range
//     for (int i = 0; i < patch_size; i++) {
//         if (!is_memory_accessible(dbg, addr + i)) {
//             return -1;
//         }
//     }
//     
//     // Find the extent of any existing patches at this address range
//     uint64_t start_addr = addr;
//     uint64_t end_addr = addr + patch_size - 1;
//     
//     // Extend range to include any existing patches that overlap
//     for (int i = 0; i < dbg->num_patches; i++) {
//         if (dbg->patches[i].addr >= start_addr && dbg->patches[i].addr <= end_addr) {
//             // Overlapping patch found
//         } else if (dbg->patches[i].addr < start_addr && 
//                    dbg->patches[i].addr >= start_addr - 16) {
//             // Extend backwards to include nearby patches
//             start_addr = dbg->patches[i].addr;
//         } else if (dbg->patches[i].addr > end_addr && 
//                    dbg->patches[i].addr <= end_addr + 16) {
//             // Extend forwards to include nearby patches
//             end_addr = dbg->patches[i].addr;
//         }
//     }
//     
//     // Remove all existing patches in this range
//     for (int i = dbg->num_patches - 1; i >= 0; i--) {
//         if (dbg->patches[i].addr >= start_addr && dbg->patches[i].addr <= end_addr) {
//             // Remove this patch
//             for (int j = i; j < dbg->num_patches - 1; j++) {
//                 dbg->patches[j] = dbg->patches[j + 1];
//             }
//             dbg->num_patches--;
//         }
//     }
//     
//     // First, restore original bytes for the entire range (always restore, handle breakpoints separately)
//     for (uint64_t restore_addr = start_addr; restore_addr <= end_addr; restore_addr++) {
//         unsigned char orig_byte = get_original_byte(dbg, restore_addr);
//         
//         // Always restore to memory, even if there's a breakpoint (we'll re-set breakpoint later)
//         errno = 0;
//         long data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, restore_addr & ~7, NULL);
//         if (errno == 0) {
//             int byte_offset = restore_addr & 7;
//             long mask = 0xffL << (byte_offset * 8);
//             long new_data = (data & ~mask) | ((long)orig_byte << (byte_offset * 8));
//             ptrace(PTRACE_POKEDATA, dbg->child_pid, restore_addr & ~7, new_data);
//         } else {
//             // Use /proc/pid/mem fallback
//             char mem_path[64];
//             snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
//             int mem_fd = open(mem_path, O_WRONLY);
//             if (mem_fd >= 0) {
//                 lseek(mem_fd, restore_addr, SEEK_SET);
//                 write(mem_fd, &orig_byte, 1);
//                 close(mem_fd);
//             }
//         }
//     }
//     
//     // Now apply new patches
//     for (int i = 0; i < patch_size; i++) {
//         uint64_t current_addr = addr + i;
//         unsigned char orig_byte = get_original_byte(dbg, current_addr);
//         
//         // Add new patch
//         dbg->patches = realloc(dbg->patches, (dbg->num_patches + 1) * sizeof(patch_t));
//         dbg->patches[dbg->num_patches].addr = current_addr;
//         dbg->patches[dbg->num_patches].new_byte = patch_bytes[i];
//         dbg->patches[dbg->num_patches].original_byte = orig_byte;
//         dbg->patches[dbg->num_patches].is_active = 1;
//         dbg->num_patches++;
//         
//         // Apply patch to memory first
//         int has_breakpoint = 0;
//         
//         errno = 0;
//         long data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, current_addr & ~7, NULL);
//         if (errno == 0) {
//             int byte_offset = current_addr & 7;
//             long mask = 0xffL << (byte_offset * 8);
//             long new_data = (data & ~mask) | ((long)patch_bytes[i] << (byte_offset * 8));
//             ptrace(PTRACE_POKEDATA, dbg->child_pid, current_addr & ~7, new_data);
//         } else {
//             // Use /proc/pid/mem fallback
//             char mem_path[64];
//             snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
//             int mem_fd = open(mem_path, O_WRONLY);
//             if (mem_fd >= 0) {
//                 lseek(mem_fd, current_addr, SEEK_SET);
//                 write(mem_fd, &patch_bytes[i], 1);
//                 close(mem_fd);
//             }
//         }
//     }
//     
//     // Re-apply any breakpoints that were in the patched range and update their original_byte
//     for (int j = 0; j < dbg->num_breakpoints; j++) {
//         if (dbg->breakpoints[j].addr >= addr && dbg->breakpoints[j].addr < addr + patch_size) {
//             // Update breakpoint's original byte to the NEW patched value
//             int offset = dbg->breakpoints[j].addr - addr;
//             dbg->breakpoints[j].original_byte = patch_bytes[offset];
//             
//             // Re-apply breakpoint (0xcc) over the patch
//             uint64_t bp_addr = dbg->breakpoints[j].addr;
//             errno = 0;
//             long data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, bp_addr & ~7, NULL);
//             if (errno == 0) {
//                 int byte_offset = bp_addr & 7;
//                 long mask = 0xffL << (byte_offset * 8);
//                 long bp_data = (data & ~mask) | (0xccL << (byte_offset * 8));
//                 ptrace(PTRACE_POKEDATA, dbg->child_pid, bp_addr & ~7, bp_data);
//             } else {
//                 char mem_path[64];
//                 snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
//                 int mem_fd = open(mem_path, O_WRONLY);
//                 if (mem_fd >= 0) {
//                     lseek(mem_fd, bp_addr, SEEK_SET);
//                     unsigned char bp_byte = 0xcc;
//                     write(mem_fd, &bp_byte, 1);
//                     close(mem_fd);
//                 }
//             }
//         }
//     }
//     
//     return 0;
// }


// Check if instruction at address is a syscall (0x0f 0x05)
int is_syscall_instruction(debugger_t *dbg, uint64_t addr) {
    unsigned char inst[2];
    read_memory(dbg->child_pid, addr, inst, 2);
    return (inst[0] == 0x0f && inst[1] == 0x05);
}

// Fixed step-over logic for execute_until_syscall_or_breakpoint
int execute_until_syscall_or_breakpoint(debugger_t *dbg) {
    if (!dbg->is_loaded || dbg->child_pid <= 0) {
        return -1;
    }
    
    uint64_t current_rip = get_register(dbg->child_pid, 16);
    // printf("current rip, 0x%016lx\n", current_rip); // Debug output
    
    // Check if we're currently at a breakpoint
    int current_breakpoint = -1;
    for (int i = 0; i < dbg->num_breakpoints; i++) {
        if (dbg->breakpoints[i].addr == current_rip) {
            current_breakpoint = i;
            break;
        }
    }
    
    // If at a breakpoint, we MUST step over it first
    if (current_breakpoint >= 0) {
        uint64_t bp_addr = dbg->breakpoints[current_breakpoint].addr;
        unsigned char orig_byte = dbg->breakpoints[current_breakpoint].original_byte;
        
        // printf("Stepping over breakpoint at 0x%lx\n", bp_addr); // Debug
        
        // Step 1: Restore original byte
        errno = 0;
        long data = ptrace(PTRACE_PEEKDATA, dbg->child_pid, bp_addr & ~7, NULL);
        if (errno == 0) {
            int byte_offset = bp_addr & 7;
            long mask = 0xffL << (byte_offset * 8);
            long orig_data = (data & ~mask) | ((long)orig_byte << (byte_offset * 8));
            
            if (ptrace(PTRACE_POKEDATA, dbg->child_pid, bp_addr & ~7, orig_data) < 0) {
                perror("Failed to restore original byte");
                return -1;
            }
            
            // Step 2: Single step over the instruction
            if (ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL) < 0) {
                perror("ptrace SINGLESTEP");
                return -1;
            }
            
            int status;
            waitpid(dbg->child_pid, &status, 0);
            
            if (WIFEXITED(status)) {
                printf("** the target program terminated.\n");
                return -1;
            }
            
            // Step 3: Restore breakpoint (0xcc)
            long bp_data = (data & ~mask) | (0xccL << (byte_offset * 8));
            if (ptrace(PTRACE_POKEDATA, dbg->child_pid, bp_addr & ~7, bp_data) < 0) {
                perror("Failed to restore breakpoint");
                return -1;
            }
            
        } else {
            // Fallback to /proc/pid/mem
            char mem_path[64];
            snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", dbg->child_pid);
            
            // Restore original byte
            int mem_fd = open(mem_path, O_WRONLY);
            if (mem_fd >= 0) {
                lseek(mem_fd, bp_addr, SEEK_SET);
                write(mem_fd, &orig_byte, 1);
                close(mem_fd);
                
                // Single step
                ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL);
                int status;
                waitpid(dbg->child_pid, &status, 0);
                
                if (WIFEXITED(status)) {
                    printf("** the target program terminated.\n");
                    return -1;
                }
                
                // Restore breakpoint
                mem_fd = open(mem_path, O_WRONLY);
                if (mem_fd >= 0) {
                    lseek(mem_fd, bp_addr, SEEK_SET);
                    unsigned char bp_byte = 0xcc;
                    write(mem_fd, &bp_byte, 1);
                    close(mem_fd);
                }
            }
        }
        
        // Verify we've moved past the breakpoint
        uint64_t new_rip = get_register(dbg->child_pid, 16);
        // printf("After step-over, RIP: 0x%016lx\n", new_rip); // Debug
        
        // Check if we immediately hit another breakpoint
        if (is_breakpoint_at(dbg, new_rip)) {
            printf("** hit a breakpoint at 0x%lx.\n", new_rip);
            disassemble_at_rip(dbg, 1);
            return 0;
        }
    }
    
    // Continue single-stepping until syscall or breakpoint
    while (1) {
        if (ptrace(PTRACE_SINGLESTEP, dbg->child_pid, NULL, NULL) < 0) {
            perror("ptrace SINGLESTEP");
            return -1;
        }
        
        int status;
        waitpid(dbg->child_pid, &status, 0);
        
        if (WIFEXITED(status)) {
            printf("** the target program terminated.\n");
            return -1;
        }
        
        if (!WIFSTOPPED(status)) {
            continue;
        }
        
        uint64_t rip = get_register(dbg->child_pid, 16);
        
        // Check for breakpoint hit
        if (is_breakpoint_at(dbg, rip)) {
            printf("** hit a breakpoint at 0x%lx.\n", rip);
            disassemble_at_rip(dbg, 1);
            return 0;
        }
        
        // Check for syscall
        if (is_syscall_instruction(dbg, rip)) {
            uint64_t rax = get_register(dbg->child_pid, 0);
            
            if (!dbg->in_syscall) {
                dbg->in_syscall = 1;
                dbg->syscall_nr = rax;
                dbg->syscall_addr = rip;
                printf("** enter a syscall(%ld) at 0x%lx.\n", rax, rip);
                disassemble_at_rip(dbg, 1);
                return 0;
            }
        } else if (dbg->in_syscall && rip == dbg->syscall_addr + 2) {
            uint64_t ret_val = get_register(dbg->child_pid, 0);
            printf("** leave a syscall(%ld) = %ld at 0x%lx.\n", 
                   dbg->syscall_nr, ret_val, dbg->syscall_addr);
            disassemble_at_rip(dbg, 1);
            dbg->in_syscall = 0;
            return 0;
        }
    }
}



int is_memory_accessible(debugger_t *dbg, uint64_t addr) {
    // Check traditional executable regions first
    if (is_in_executable_region(dbg, addr)) {
        return 1;
    }
    
    // For high memory addresses (potential JIT code), verify accessibility
    if (addr >= 0x700000000000ULL) {
        unsigned char test_byte;
        return (read_memory_safe(dbg->child_pid, addr, &test_byte, 1) == 0);
    }
    
    return 0;
}

// Improved memory reading function that tries both ptrace and /proc/pid/mem
int read_memory_safe(pid_t pid, uint64_t addr, unsigned char *buffer, size_t size) {
    // First try ptrace method
    int ptrace_success = 1;
    for (size_t i = 0; i < size; i += sizeof(long)) {
        errno = 0;
        long data = ptrace(PTRACE_PEEKDATA, pid, addr + i, NULL);
        if (errno != 0) {
            ptrace_success = 0;
            break;
        }
        
        size_t copy_size = (size - i < sizeof(long)) ? size - i : sizeof(long);
        memcpy(buffer + i, &data, copy_size);
    }
    
    if (ptrace_success) {
        return 0; // Success with ptrace
    }
    
    // Fallback to /proc/pid/mem
    char mem_path[64];
    snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", pid);
    
    int mem_fd = open(mem_path, O_RDONLY);
    if (mem_fd < 0) {
        printf("mem open failed\n");
        return -1; // Both methods failed
    }
    
    // Seek to the target address
    if (lseek(mem_fd, addr, SEEK_SET) == -1) {
        printf("lseek error\n");
        close(mem_fd);
        return -1;
    }
    
    // Read the memory
    read(mem_fd, buffer, size);
    close(mem_fd);
    
    return 0; // Success with /proc/pid/mem
}

// Read memory from child process
void read_memory(pid_t pid, uint64_t addr, unsigned char *buffer, size_t size) {
    if (read_memory_safe(pid, addr, buffer, size) != 0) {
        // Set buffer to zero on failure to avoid undefined behavior
        memset(buffer, 0, size);
        // Note: You might want to add logging here
    }
}

void read_memory_with_patches(debugger_t *dbg, uint64_t addr, unsigned char *buffer, size_t size) {
    // First read raw memory
    read_memory(dbg->child_pid, addr, buffer, size);
    
    // Apply patches and hide breakpoints
    for (size_t i = 0; i < size; i++) {
        uint64_t current_addr = addr + i;
        
        // Check for breakpoints - restore original byte
        for (int j = 0; j < dbg->num_breakpoints; j++) {
            if (dbg->breakpoints[j].addr == current_addr) {
                buffer[i] = dbg->breakpoints[j].original_byte;
                break;
            }
        }
        
        // Check for patches - show patched value
        for (int j = 0; j < dbg->num_patches; j++) {
            if (dbg->patches[j].addr == current_addr) {
                buffer[i] = dbg->patches[j].new_byte;
                break;
            }
        }
    }
}

// Helper function to format hex bytes nicely
char* format_hex_bytes(unsigned char *bytes, size_t size) {
    static char hex_buf[64]; // Static buffer for formatted hex
    hex_buf[0] = '\0';
    
    int bytes_to_show = (size > 8) ? 8 : size; // Show up to 8 bytes
    
    for (int i = 0; i < bytes_to_show; i++) {
        char temp[4];
        snprintf(temp, sizeof(temp), "%02x ", bytes[i]);
        strcat(hex_buf, temp);
    }
    
    if (size > 8) {
        strcat(hex_buf, "...");
    }
    
    return hex_buf;
}

// Disassemble 5 instructions starting from current RIP
void disassemble_at_rip(debugger_t *dbg, int force_disasm) {
    if (!dbg->is_loaded || dbg->child_pid <= 0) {
        return;
    }

    // Only disassemble when explicitly requested or on certain commands
    if (!force_disasm) {
        return;
    }

    // uint64_t rip = get_rip(dbg->child_pid);
    uint64_t rip = get_register(dbg->child_pid, 16);
    if (rip == 0) return;
    
    
    // Load/refresh executable regions
    load_executable_regions(dbg);
    
    // Check if current address is in executable region
    if (!is_in_executable_region(dbg, rip)) {
        printf("** the address is out of the range of the executable region.\n");
        return;
    }

    // Read memory at RIP
    unsigned char code[64]; // Read more bytes to ensure we have enough for 5 instructions
    
    read_memory_with_patches(dbg, rip, code, sizeof(code));

    // Disassemble
    cs_insn *insn;      // information of disassembled instruction
    size_t count = cs_disasm(dbg->capstone_handle, code, sizeof(code), rip, DISASM_COUNT, &insn);
    // args: capstone instance, binary code to disassemble, length, disassemble addr (inital addr), num_inst, result
    
    if (count > 0) {
        int instructions_shown = 0;
        for (size_t j = 0; j < count && instructions_shown < DISASM_COUNT; j++) {

            if (!is_in_executable_region(dbg, insn[j].address)) {
                break; // Stop if we hit non-executable region
            }
            printf("\t%08lx: %-25s %-8s %s\n",
                    insn[j].address,
                    format_hex_bytes(insn[j].bytes, insn[j].size),
                    insn[j].mnemonic, 
                    insn[j].op_str);
            instructions_shown++;
        }
        // If we couldn't show 5 instructions due to boundary
        if (instructions_shown < DISASM_COUNT && count > 0) {
            printf("** the address is out of the range of the executable region.\n");
        }
        cs_free(insn, count);
    } else {
        printf("** the address is out of the range of the executable region.\n");
    }
}

// Parse ELF file to get entry point and check if PIE
int parse_elf(const char *path, debugger_t *dbg) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    Elf64_Ehdr ehdr;
    if (read(fd, &ehdr, sizeof(ehdr)) != sizeof(ehdr)) {    // ELF header is always located at the beginning of the ELF file
        perror("read ELF header");
        close(fd);
        return -1;
    }

    // Check ELF magic
    if (memcmp(ehdr.e_ident, ELFMAG, SELFMAG) != 0) {       // check is ELF file or not
        fprintf(stderr, "Not a valid ELF file\n");
        close(fd);
        return -1;
    }

    // Check if 64-bit
    if (ehdr.e_ident[EI_CLASS] != ELFCLASS64) {
        fprintf(stderr, "Not a 64-bit ELF file\n");
        close(fd);
        return -1;
    }

    dbg->entry_point = ehdr.e_entry;
    
    // Check if PIE (Position Independent Executable), ET_DYN means shared object file or PIE
    dbg->is_pie = (ehdr.e_type == ET_DYN);
    
    close(fd);
    return 0;
}

// Load program into debugger
int load_program(debugger_t *dbg, const char *path) {
    // Clean up previous program if any
    if (dbg->program_path) {
        free(dbg->program_path);
        free(dbg->program_name);
    }

    // Convert to absolute path if needed
    char abs_path[PATH_MAX];
    if (realpath(path, abs_path) == NULL) {
        perror("realpath");
        return -1;
    }

    // Check if file exists and is executable, F_OK means only checks whether exists
    if (access(abs_path, F_OK | X_OK) != 0) {
        perror("access");
        return -1;
    }
    // Parse ELF to get entry point
    if (parse_elf(abs_path, dbg) < 0) {
        return -1;
    }

    // Store program path and extract name
    dbg->program_path = strdup(abs_path);           // returns a pointer to a null-terminated byte string, which is a duplicate of the string pointed to by path
    const char *name_start = strrchr(abs_path, '/');    // locates the last occurrence of a character in a string 
    dbg->program_name = strdup(name_start ? name_start + 1 : abs_path);
    
    if (!dbg || !dbg->program_name) {
        printf("Error: program_name is NULL\n");
        return -1;
    }
    dbg->is_loaded = 1;
    dbg->base_address = 0; // Will be determined when program starts
    
    // if (!dbg->is_pie)
    //     printf("** program '%s' loaded. entry point 0x%lx\n", dbg->program_name, dbg->entry_point);
    
    return 0;
}

// Start the target program under ptrace
int start_program(debugger_t *dbg) {
    if (!dbg->is_loaded) {
        printf("** please load a program first.\n");
        return -1;
    }

    pid_t child = fork();
    if (child == 0) {
        // Child process
        if (ptrace(PTRACE_TRACEME, 0, NULL, NULL) < 0) {
            perror("ptrace TRACEME");
            exit(1);
        }
        
        // Execute the target program
        char *argv[] = {dbg->program_name, NULL};
        execv(dbg->program_path, argv);
        
        perror("execv failed");
        exit(1);
    } else if (child > 0) {
        // Parent process (debugger)
        dbg->child_pid = child;
        
        int status;
        waitpid(child, &status, 0); // Wait for initial stop (in dynamic linker)
        
        if (!WIFSTOPPED(status)) {
            printf("** failed to start program\n");
            return -1;
        }
        
        uint64_t target_entry;
        
        if (dbg->is_pie) {
            // Get base address from /proc/PID/maps
            char maps_path[64];
            snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", child);
            FILE *maps = fopen(maps_path, "r");
            if (!maps) {
                printf("** failed to read memory maps\n");
                return -1;
            }
            
            char line[256];
            if (!fgets(line, sizeof(line), maps)) {
                printf("** failed to parse memory maps\n");
                fclose(maps);
                return -1;
            }
            
            sscanf(line, "%lx-", &dbg->base_address);
            fclose(maps);
            
            // Calculate actual entry point
            target_entry = dbg->base_address + dbg->entry_point;
        } else {
            // Static executable
            target_entry = dbg->entry_point;
        }
        
        // Set breakpoint at the actual entry point BEFORE continuing
        errno = 0;
        long original_data = ptrace(PTRACE_PEEKDATA, child, target_entry, NULL);
        if (errno != 0) {
            perror("ptrace PEEKDATA at entry point");
            return -1;
        }
        
        // Set breakpoint (replace first byte with 0xcc)
        long breakpoint_data = (original_data & ~0xff) | 0xcc;
        if (ptrace(PTRACE_POKEDATA, child, target_entry, breakpoint_data) < 0) {
            perror("ptrace POKEDATA at entry point");
            return -1;
        }
        
        // Now continue execution - the dynamic linker will finish loading and jump to our entry point
        ptrace(PTRACE_CONT, child, NULL, NULL);
        waitpid(child, &status, 0);
        
        if (WIFEXITED(status)) {
            printf("** program terminated before reaching entry point\n");
            return -1;
        }
        
        if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGTRAP) {
            uint64_t current_rip = get_register(child, 16);
            
            // We should be at entry_point + 1 (after the int3 instruction)
            if (current_rip == target_entry + 1) {
                // Restore original instruction
                if (ptrace(PTRACE_POKEDATA, child, target_entry, original_data) < 0) {
                    perror("ptrace POKEDATA restore");
                    return -1;
                }
                
                // Move RIP back to the actual entry point
                struct user_regs_struct regs;
                if (ptrace(PTRACE_GETREGS, child, NULL, &regs) < 0) {
                    perror("ptrace GETREGS");
                    return -1;
                }
                
                regs.rip = target_entry;
                if (ptrace(PTRACE_SETREGS, child, NULL, &regs) < 0) {
                    perror("ptrace SETREGS");
                    return -1;
                }
                
                printf("** program '%s' loaded. entry point 0x%lx\n", 
                       dbg->program_name, target_entry);
                
                // Now we're properly positioned at the entry point with full environment
                disassemble_at_rip(dbg, 1);
                return 0;
            } else {
                printf("** unexpected breakpoint at 0x%lx (expected 0x%lx)\n", 
                       current_rip, target_entry + 1);
                return -1;
            }
        } else {
            printf("** unexpected signal: %d\n", WSTOPSIG(status));
            return -1;
        }
    } else {
        perror("fork");
        return -1;
    }
}


// Main command loop
void command_loop(debugger_t *dbg) {
    char cmd_line[MAX_CMD_LEN];
    char cmd[64], arg[MAX_PATH_LEN];
    
    while (1) {
        printf("(sdb) ");
        fflush(stdout);
        
        if (!fgets(cmd_line, sizeof(cmd_line), stdin)) {
            break; // EOF
        }
        
        // Remove newline
        // strcspn counts how many characters are not \n
        cmd_line[strcspn(cmd_line, "\n")] = 0;
        
        // Skip empty lines
        if (strlen(cmd_line) == 0) {
            continue;
        }
        
        // Parse command
        int argc = sscanf(cmd_line, "%63s %1023s", cmd, arg);
        
        if (strcmp(cmd, "load") == 0) {
            if (argc < 2) {
                printf("Usage: load [path]\n");
                continue;
            }
            
            if (load_program(dbg, arg) == 0) {
                start_program(dbg);
            }
        }
        else if (strcmp(cmd, "break") == 0 || strcmp(cmd, "b") == 0) {
            if (!dbg->is_loaded) {
                printf("** please load a program first.\n");
                continue;
            }
            
            if (argc < 2) {
                printf("Usage: break [hex address]\n");
                continue;
            }
            
            uint64_t addr = parse_hex_address(arg);
            if (addr == 0) {
                printf("parse error.\n");
                printf("** the target address is not valid.\n");
                continue;
            }
            
            if (set_breakpoint(dbg, addr) == 0) {
                printf("** set a breakpoint at 0x%lx.\n", addr);
            } else {
                printf("set breakpoint error.\n");
                printf("** the target address is not valid.\n");
            }
        }else if (strcmp(cmd, "breakrva") == 0) {
            if (!dbg->is_loaded) {
                printf("** please load a program first.\n");
                continue;
            }
            
            if (argc < 2) {
                printf("Usage: breakrva [hex offset]\n");
                continue;
            }
            
            uint64_t offset = parse_hex_address(arg);
            if (offset == 0 && strcmp(arg, "0") != 0) {
                printf("** the target address is not valid.\n");
                continue;
            }
            
            // Calculate actual address for output
            uint64_t base = dbg->base_address;
            if (base == 0) {
                // Get base address for display
                char maps_path[64];
                snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", dbg->child_pid);
                FILE *maps = fopen(maps_path, "r");
                if (maps) {
                    char line[256];
                    if (fgets(line, sizeof(line), maps)) {
                        sscanf(line, "%lx-", &base);
                    }
                    fclose(maps);
                }
            }
            
            if (set_breakpoint_rva(dbg, offset) == 0) {
                printf("** set a breakpoint at 0x%lx.\n", base + offset);
            } else {
                printf("** the target address is not valid.\n");
            }
        }else if (strcmp(cmd, "info") == 0) {
            if (!dbg->is_loaded) {
                printf("** please load a program first.\n");
                continue;
            }
            
            if (argc < 2) {
                printf("Usage: info break\n");
                continue;
            }
            
            if (strcmp(arg, "break") == 0) {
                info_breakpoints(dbg);
            } else if (strcmp(arg, "reg") == 0) {
                info_registers(dbg);
            } else {
                printf("Unknown info command: %s\n", arg);
            }
        }else if (strcmp(cmd, "delete") == 0) {
            if (!dbg->is_loaded) {
                printf("** please load a program first.\n");
                continue;
            }
            
            if (argc < 2) {
                printf("Usage: delete [id]\n");
                continue;
            }
            
            char *endptr;
            int id = strtol(arg, &endptr, 10);
            
            // Check if conversion was successful
            if (*endptr != '\0' || endptr == arg) {
                printf("** breakpoint %s does not exist.\n", arg);
                continue;
            }
            
            if (delete_breakpoint(dbg, id) == 0) {
                printf("** delete breakpoint %d.\n", id);
            } else {
                printf("** breakpoint %d does not exist.\n", id);
            }
        }else if (strcmp(cmd, "si") == 0) {
            if (!dbg->is_loaded) {
                printf("** please load a program first.\n");
                continue;
            }
            
            step_instruction(dbg);
        }
        else if (strcmp(cmd, "cont") == 0 || strcmp(cmd, "c") == 0) {
            if (!dbg->is_loaded) {
                printf("** please load a program first.\n");
                continue;
            }
            
            continue_execution(dbg);
        }else if (strcmp(cmd, "patch") == 0) {
            if (!dbg->is_loaded) {
                printf("** please load a program first.\n");
                continue;
            }
            
            // Parse "patch [addr] [hex_string]" - need to handle two arguments
            char addr_str[64], hex_str[2049];
            if (sscanf(cmd_line, "%63s %63s %2048s", cmd, addr_str, hex_str) != 3) {
                printf("Usage: patch [hex address] [hex string]\n");
                continue;
            }
            
            uint64_t addr = parse_hex_address(addr_str);
            if (addr == 0) {
                printf("** the target address is not valid.\n");
                continue;
            }
            
            if (patch_memory(dbg, addr, hex_str) == 0) {
                printf("** patch memory at 0x%lx.\n", addr);
            } else {
                printf("** the target address is not valid.\n");
            }
        }else if (strcmp(cmd, "syscall") == 0) {
            if (!dbg->is_loaded) {
                printf("** please load a program first.\n");
                continue;
            }
            
            execute_until_syscall_or_breakpoint(dbg);
        }else if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "q") == 0) {
            break;
        }else if (strcmp(cmd, "help") == 0 || strcmp(cmd, "h") == 0) {
            printf("Available commands:\n");
            printf("  load [path] - Load a program\n");
            printf("  quit/q      - Exit debugger\n");
            printf("  help/h      - Show this help\n");
        }else {
            if (!dbg->is_loaded) {
                printf("** please load a program first.\n");
            } else {
                printf("Unknown command: %s\n", cmd);
            }
        }
    }
}

// Add cleanup for original_memory in cleanup_debugger
void cleanup_debugger(debugger_t *dbg) {
    if (dbg->program_path) free(dbg->program_path);
    if (dbg->program_name) free(dbg->program_name);
    if (dbg->exec_regions) free(dbg->exec_regions);
    if (dbg->breakpoints) free(dbg->breakpoints);
    if (dbg->patches) free(dbg->patches);
    if (dbg->original_memory) free(dbg->original_memory);  
    
    if (dbg->child_pid > 0) {
        kill(dbg->child_pid, SIGKILL);
        waitpid(dbg->child_pid, NULL, 0);
    }
    if (dbg->capstone_handle != 0) {
        cs_close(&dbg->capstone_handle);
    }
}

int main(int argc, char *argv[]) {
    debugger_t dbg = {0};
    

    // Initialize Capstone disassembler for x86-64
    if (cs_open(CS_ARCH_X86, CS_MODE_64, &dbg.capstone_handle) != CS_ERR_OK) {
        fprintf(stderr, "Failed to initialize Capstone\n");
        return 1;
    }
    
    // Check if a program was specified as command line argument
    if (argc > 1) {
        printf("Loading program: %s\n", argv[1]);
        if (load_program(&dbg, argv[1]) == 0) {
            start_program(&dbg);
        } else {
            printf("Failed to load program: %s\n", argv[1]);
        }
    }
    
    // Enter command loop
    command_loop(&dbg);
    
    // Cleanup
    cleanup_debugger(&dbg);
    
    return 0;
}