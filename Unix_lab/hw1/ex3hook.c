#include <stdint.h>
#include <stdio.h>

typedef int64_t (*syscall_hook_fn_t)(int64_t, int64_t, int64_t, int64_t, 
                                     int64_t, int64_t, int64_t);

static syscall_hook_fn_t original_syscall = NULL;

static int64_t syscall_hook_fn(int64_t rdi, int64_t rsi, int64_t rdx,
                               int64_t r10, int64_t r8, int64_t r9,
                               int64_t rax) {
  fprintf(stderr, "Intercepted syscall: %ld\n", rax);
  return original_syscall(rax, rdi, rsi, rdx, r10, r8, r9);
}

#ifdef __cplusplus
extern "C"
#endif
void __hook_init(const syscall_hook_fn_t trigger_syscall,
                 syscall_hook_fn_t *hooked_syscall) {
  original_syscall = trigger_syscall;
  *hooked_syscall = syscall_hook_fn;
}