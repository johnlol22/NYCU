#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <stdint.h>
#include "libsolver.h"

// Gotoku structure definition
typedef struct gotoku_s {
    int x, y;
    int board[9][9];
} gotoku_t;

// Global board reference
static gotoku_t *game_board = NULL;
static void *main_addr = NULL;
static int solution[9][9];
static int solved = 0;
pthread_mutex_t mutex;

// Function pointers for original functions
static void (*original_gop_up)();
static void (*original_gop_down)();
static void (*original_gop_left)();
static void (*original_gop_right)();
static void (*original_gop_fill_1)();
static void (*original_gop_fill_2)();
static void (*original_gop_fill_3)();
static void (*original_gop_fill_4)();
static void (*original_gop_fill_5)();
static void (*original_gop_fill_6)();
static void (*original_gop_fill_7)();
static void (*original_gop_fill_8)();
static void (*original_gop_fill_9)();

// Function prototypes
bool is_safe(int board[9][9], int row, int col, int num);
bool solve_sudoku(int board[9][9], int row, int col);

// Initialize function pointers 
void init_original_functions() {
    void *handle = dlopen("libgotoku.so", RTLD_LAZY);
    if (!handle) {
        printf("Failed to open libgotoku.so: %s\n", dlerror());
        return;
    }
    
    original_gop_up = dlsym(handle, "gop_up");
    original_gop_down = dlsym(handle, "gop_down");
    original_gop_left = dlsym(handle, "gop_left");
    original_gop_right = dlsym(handle, "gop_right");
    
    original_gop_fill_1 = dlsym(handle, "gop_fill_1");
    original_gop_fill_2 = dlsym(handle, "gop_fill_2");
    original_gop_fill_3 = dlsym(handle, "gop_fill_3");
    original_gop_fill_4 = dlsym(handle, "gop_fill_4");
    original_gop_fill_5 = dlsym(handle, "gop_fill_5");
    original_gop_fill_6 = dlsym(handle, "gop_fill_6");
    original_gop_fill_7 = dlsym(handle, "gop_fill_7");
    original_gop_fill_8 = dlsym(handle, "gop_fill_8");
    original_gop_fill_9 = dlsym(handle, "gop_fill_9");
    
    // Don't close the handle to ensure the pointers remain valid
}

// Sudoku solver functions
bool is_safe(int board[9][9], int row, int col, int num) {
    // Row check
    for (int i = 0; i < 9; i++) 
        if (board[row][i] == num) return false;
    
    // Column check
    for (int i = 0; i < 9; i++) 
        if (board[i][col] == num) return false;
    
    // 3x3 box check
    int startRow = row - row % 3;
    int startCol = col - col % 3;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (board[i + startRow][j + startCol] == num) return false;
    
    return true;
}

bool solve_sudoku(int board[9][9], int row, int col) {
    if (row == 9) return true;
    if (col == 9) return solve_sudoku(board, row + 1, 0);
    if (board[row][col] != 0) return solve_sudoku(board, row, col + 1);
    
    for (int num = 1; num <= 9; num++) {
        if (is_safe(board, row, col, num)) {
            board[row][col] = num;
            if (solve_sudoku(board, row, col + 1)) return true;
            board[row][col] = 0;
        }
    }
    return false;
}

// Plan the sequence of actions needed to solve the puzzle
typedef enum {
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_FILL_1,
    ACTION_FILL_2,
    ACTION_FILL_3,
    ACTION_FILL_4,
    ACTION_FILL_5,
    ACTION_FILL_6,
    ACTION_FILL_7,
    ACTION_FILL_8,
    ACTION_FILL_9
} action_type;

// Structure to hold a planned action
typedef struct {
    action_type type;
} action_t;

// Array to store the planned sequence of actions
#define MAX_ACTIONS 1200
static action_t action_sequence[MAX_ACTIONS];
static int action_count = 0;

// Plan a sequence of movements and fills to solve the puzzle
void plan_solution() {
    if (!game_board || !solved) return;
    
    action_count = 0;
    
    // Find all empty cells that need to be filled
    typedef struct {
        int row;
        int col;
        int value;     // answer
    } empty_cell_t;
    
    empty_cell_t empty_cells[81]; // Maximum possible cells
    int empty_count = 0;
    
    // Find all empty cells and their solution values
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (game_board->board[i][j] == 0) {
                empty_cells[empty_count].row = i;
                empty_cells[empty_count].col = j;
                empty_cells[empty_count].value = solution[i][j];
                
                empty_count++;
            }
        }
    }
    
    printf("SOLVER: Found %d empty cells\n", empty_count);
    
    // Current cursor position
    int current_row = game_board->y;
    int current_col = game_board->x;
    // For each empty cell, plan the movements to get there and the fill action
    for (int i = 0; i < empty_count && action_count < MAX_ACTIONS; i++) {
        int target_row = empty_cells[i].row;
        int target_col = empty_cells[i].col;
        int target_value = empty_cells[i].value;
        
        // Plan movements to get to the target cell
        while (current_row < target_row && action_count < MAX_ACTIONS) {
            action_sequence[action_count++].type = ACTION_DOWN;
            current_row++;
        }
        
        while (current_row > target_row && action_count < MAX_ACTIONS) {
            action_sequence[action_count++].type = ACTION_UP;
            current_row--;
        }
        
        while (current_col < target_col && action_count < MAX_ACTIONS) {
            action_sequence[action_count++].type = ACTION_RIGHT;
            current_col++;
        }
        
        while (current_col > target_col && action_count < MAX_ACTIONS) {
            action_sequence[action_count++].type = ACTION_LEFT;
            current_col--;
        }
        
        // Plan the fill action
        if (action_count < MAX_ACTIONS) {
            switch (target_value) {
                case 1: action_sequence[action_count++].type = ACTION_FILL_1; break;
                case 2: action_sequence[action_count++].type = ACTION_FILL_2; break;
                case 3: action_sequence[action_count++].type = ACTION_FILL_3; break;
                case 4: action_sequence[action_count++].type = ACTION_FILL_4; break;
                case 5: action_sequence[action_count++].type = ACTION_FILL_5; break;
                case 6: action_sequence[action_count++].type = ACTION_FILL_6; break;
                case 7: action_sequence[action_count++].type = ACTION_FILL_7; break;
                case 8: action_sequence[action_count++].type = ACTION_FILL_8; break;
                case 9: action_sequence[action_count++].type = ACTION_FILL_9; break;
            }
        }
    }
    printf("SOLVER: Planned %d actions\n", action_count);
}

// Patch a single GOT entry
bool patch_got_entry(size_t got_offset, void *replacement_func) {
    if (!main_addr) return false;
    
    // Main offset from the binary data
    size_t main_offset = 0x16c89;
    
    // Calculate base address
    uintptr_t base_addr = (uintptr_t)main_addr - main_offset;
    uintptr_t got_addr = base_addr + got_offset;
    
    // Get the page size and calculate page-aligned address
    uintptr_t page_size = sysconf(_SC_PAGESIZE);
    uintptr_t page_mask = ~(page_size - 1);
    uintptr_t got_page = got_addr & page_mask;
    
    // Make the page writable
    if (mprotect((void*)got_page, page_size, PROT_READ | PROT_WRITE) != 0) {
        return false;
    }
    
    // Modify GOT entry
    *(void**)got_addr = replacement_func;
    
    // Restore page protection
    mprotect((void*)got_page, page_size, PROT_READ);
    
    return true;
}

// Map action type to function pointer
void* get_function_for_action(action_type type) {
    switch (type) {
        case ACTION_UP: return (void*)original_gop_up;
        case ACTION_DOWN: return (void*)original_gop_down;
        case ACTION_LEFT: return (void*)original_gop_left;
        case ACTION_RIGHT: return (void*)original_gop_right;
        case ACTION_FILL_1: return (void*)original_gop_fill_1;
        case ACTION_FILL_2: return (void*)original_gop_fill_2;
        case ACTION_FILL_3: return (void*)original_gop_fill_3;
        case ACTION_FILL_4: return (void*)original_gop_fill_4;
        case ACTION_FILL_5: return (void*)original_gop_fill_5;
        case ACTION_FILL_6: return (void*)original_gop_fill_6;
        case ACTION_FILL_7: return (void*)original_gop_fill_7;
        case ACTION_FILL_8: return (void*)original_gop_fill_8;
        case ACTION_FILL_9: return (void*)original_gop_fill_9;
        default: return NULL;
    }
}

// Patch the GOT table with our planned sequence
void patch_planned_got() {
    
    
    int offset_count = 1200;
    
    // Patch each GOT entry with the corresponding action function
    for (int i = 0; i < action_count && i < offset_count; i++) {
        void* func = get_function_for_action(action_sequence[i].type);
        if (func) {
            if (patch_got_entry(gop_offsets[i], func)) {
                printf("SOLVER: Patched gop_%d with %s\n", i+1, 
                       action_sequence[i].type <= ACTION_RIGHT ? "movement" : "fill");
            } else {
                printf("SOLVER: Failed to patch gop_%d\n", i+1);
            }
        }
    }
    
    printf("SOLVER: Patched %d GOT entries\n", 
           action_count < offset_count ? action_count : offset_count);
}

// Apply the solution directly to the board as a fallback
void apply_solution_directly() {
    if (!game_board || !solved) return;
    
    // Count empty cells before
    int empty_before = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (game_board->board[i][j] == 0) {
                empty_before++;
            }
        }
    }
    
    printf("SOLVER: Found %d empty cells before applying solution\n", empty_before);
    
    // Apply solution directly as a fallback
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (game_board->board[i][j] == 0) {
                game_board->board[i][j] = solution[i][j];
            }
        }
    }
    
    // Count empty cells after
    int empty_after = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (game_board->board[i][j] == 0) {
                empty_after++;
            }
        }
    }
    
    printf("SOLVER: Found %d empty cells after applying solution\n", empty_after);
}

// hijacked via LD_PRELOAD
int game_init() {
    printf("UP113_GOT_PUZZLE_CHALLENGE\n");
    
    if (main_addr) {
        printf("SOLVER: _main = %p\n", main_addr);
        
        // Initialize the dynamic linking functions
        init_original_functions();
    } else {
        // Try to find main_addr through dlsym
        void* handle = dlopen("libgotoku.so", RTLD_LAZY);
        if (handle) {
            void* (*get_ptr_func)() = dlsym(handle, "game_get_ptr");
            if (get_ptr_func) {
                main_addr = get_ptr_func();
                printf("SOLVER: _main = %p\n", main_addr);
                
                // Initialize original function pointers
                init_original_functions();
            }
            dlclose(handle);
        }
    }
    
    // Call original game_init
    void* handle = dlopen("libgotoku.so", RTLD_LAZY);
    if (handle) {
        int (*orig_init)() = dlsym(handle, "game_init");
        if (orig_init && orig_init != game_init) {
            int result = orig_init();
            dlclose(handle);
            return result;
        }
        dlclose(handle);
    }
    apply_solution_directly();
    return 0;
}
// hijacked via LD_PRELOAD
gotoku_t* game_load(const char *path) {
    // Call original game_load
    void* handle = dlopen("libgotoku.so", RTLD_LAZY);
    if (handle) {
        gotoku_t* (*orig_load)(const char*) = dlsym(handle, "game_load");
        if (orig_load && orig_load != game_load) {
            game_board = orig_load(path);
            if (game_board) {
                printf("SOLVER: Board loaded successfully, pointer = %p\n", game_board);
            }
        }
        dlclose(handle);
    }
    
    return game_board;
}
// hijacked via LD_PRELOAD
void gop_show() {
    // Call original gop_show
    void* handle = dlopen("libgotoku.so", RTLD_LAZY);
    if (handle) {
        void (*orig_show)() = dlsym(handle, "gop_show");
        if (orig_show && orig_show != gop_show) {
            orig_show();
        }
        dlclose(handle);
    }
    
    // Make sure we have a solution if we haven't solved it yet
    if (!solved && game_board) {
        // Copy the board for solving
        memcpy(solution, game_board->board, sizeof(solution));
        
        // Find a solution for the board
        if (solve_sudoku(solution, 0, 0)) {     // store the answer to solution first
            solved = 1;
            printf("SOLVER: Found a solution!\n");
            
            // Plan the solution actions
            plan_solution();
            
            // Patch the GOT table based on our planned sequence
            patch_planned_got();
        }
    }
}


// Constructor function, called when the library is loaded
__attribute__((constructor))
static void init_solver() {
    // Initialize solved status
    solved = 0;
    action_count = 0;
}