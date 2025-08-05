#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/un.h>   /* For sockaddr_un structure */
#include <fcntl.h>    /* For AT_FDCWD */
#include <unistd.h>

// Define the hook function type
typedef int64_t (*syscall_hook_fn_t)(int64_t, int64_t, int64_t, int64_t,
                                    int64_t, int64_t, int64_t);

// Store the original syscall function
static syscall_hook_fn_t original_syscall = NULL;

// Helper to escape and print a buffer with certain length limit
static void print_escaped_buffer(const char *buffer, int64_t length, int64_t max_length) {
    int64_t print_len = (length < max_length) ? length : max_length;
    
    fprintf(stderr, "\"");
    for (int64_t i = 0; i < print_len; i++) {
        unsigned char c = buffer[i];
        if (c == '\t') {
            fprintf(stderr, "\\t");
        } else if (c == '\n') {
            fprintf(stderr, "\\n");
        } else if (c == '\r') {
            fprintf(stderr, "\\r");
        } else if (c == '"') {
            fprintf(stderr, "\\\"");
        } else if (c == '\\') {
            fprintf(stderr, "\\\\");
        } else if (c >= 32 && c <= 126) {
            // Printable ASCII
            fprintf(stderr, "%c", c);
        } else {
            // Non-printable as hex
            fprintf(stderr, "\\x%02x", c);
        }
    }
    fprintf(stderr, "\"");
    
    // Add ellipsis if truncated
    if (length > max_length) {
        fprintf(stderr, "...");
    }
}

// Handle openat syscall
static void log_openat(int64_t dirfd, const char *pathname, int64_t flags, int64_t mode, int64_t ret) {
    int pid=getpid();
    fprintf(stderr, "[logger] [%d] openat(", pid);
    
    // AT_FDCWD is -100, but it might be passed as an unsigned value
    // Convert to signed for comparison
    int32_t signed_dirfd = (int32_t)dirfd;
    if (signed_dirfd == -100) {
        fprintf(stderr, "AT_FDCWD");
    } else {
        fprintf(stderr, "%d", signed_dirfd);
    }
    
    // Print pathname
    fprintf(stderr, ", \"%s\", 0x%lx, %#o) = %ld\n", pathname, flags, (unsigned int)mode, ret);
}

// Handle read syscall
static void log_read(int64_t fd, const char *buf, int64_t count, int64_t ret) {
    int pid=getpid();
    fprintf(stderr, "[logger] [%d] read(%ld, ", pid, fd);
    
    if (ret >= 0) {
        // Only print content if read was successful
        print_escaped_buffer(buf, ret, 32);
    } else {
        fprintf(stderr, "\"\"");
    }
    
    fprintf(stderr, ", %ld) = %ld\n", count, ret);
}

// Handle write syscall
static void log_write(int64_t fd, const char *buf, int64_t count, int64_t ret) {
    int pid=getpid();
    fprintf(stderr, "[logger] [%d] write(%ld, ", pid, fd);
    
    if (count > 0) {
        print_escaped_buffer(buf, count, 32);
    } else {
        fprintf(stderr, "\"\"");
    }
    
    fprintf(stderr, ", %ld) = %ld\n", count, ret);
}

// Handle connect syscall
static void log_connect(int64_t sockfd, const struct sockaddr *addr, int64_t addrlen, int64_t ret) {
    int pid=getpid();
    fprintf(stderr, "[logger] [%d] connect(%ld, ", pid, sockfd);
    
    if (addr->sa_family == AF_INET) {
        // IPv4 address
        const struct sockaddr_in *ipv4 = (const struct sockaddr_in *)addr;
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(ipv4->sin_addr), ip_str, INET_ADDRSTRLEN);
        uint16_t port = ntohs(ipv4->sin_port);
        
        if (port != 0) {
            fprintf(stderr, "\"%s:%d\"", ip_str, port);
        } else {
            fprintf(stderr, "\"%s:-\"", ip_str);
        }
    } else if (addr->sa_family == AF_INET6) {
        // IPv6 address
        const struct sockaddr_in6 *ipv6 = (const struct sockaddr_in6 *)addr;
        char ip_str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &(ipv6->sin6_addr), ip_str, INET6_ADDRSTRLEN);
        uint16_t port = ntohs(ipv6->sin6_port);
        
        if (port != 0) {
            fprintf(stderr, "\"%s:%d\"", ip_str, port);
        } else {
            fprintf(stderr, "\"%s:0\"", ip_str);
        }
    } else if (addr->sa_family == AF_UNIX) {
        // Unix socket
        fprintf(stderr, "\"UNIX:%s\"", ((struct sockaddr_un *)addr)->sun_path);
    } else {
        // Unknown address family
        fprintf(stderr, "\"UNKNOWN\"");
    }
    
    fprintf(stderr, ", %ld) = %ld\n", addrlen, ret);
}

// Handle execve syscall
static void log_execve(const char *pathname, void *argv, void *envp) {
    int pid=getpid();
    fprintf(stderr, "[logger] [%d] execve(\"%s\", %p, %p)\n", pid, pathname, argv, envp);
}

// Main syscall hook function
static int64_t syscall_hook_fn(int64_t rdi, int64_t rsi, int64_t rdx,
                              int64_t r10, int64_t r8, int64_t r9,
                              int64_t rax) {
    int64_t ret;
    // fprintf(stderr, "syscall number %ld\n", rax);
    // Handle execve specially - log before the call
    if (rax == SYS_execve) {
        log_execve((const char *)rdi, (void *)rsi, (void *)rdx);
    }
    
    // Call the original syscall - using the same parameter order
    ret = original_syscall(rdi, rsi, rdx, r10, r8, r9, rax);
    
    // Log based on syscall number
    switch (rax) {
        case SYS_openat:
            log_openat(rdi, (const char *)rsi, rdx, r10, ret);
            break;
        
        case SYS_read:
            log_read(rdi, (const char *)rsi, rdx, ret);
            break;
        
        case SYS_write:
            log_write(rdi, (const char *)rsi, rdx, ret);
            break;
        
        case SYS_connect:
            log_connect(rdi, (const struct sockaddr *)rsi, rdx, ret);
            break;
    }
    
    return ret;
}

// Hook initialization function
#ifdef __cplusplus
extern "C" 
#endif
void __hook_init(const syscall_hook_fn_t trigger_syscall,
                 syscall_hook_fn_t *hooked_syscall) {
    // Store the original syscall function
    original_syscall = trigger_syscall;
    
    // Set our hook function
    *hooked_syscall = syscall_hook_fn;
}
