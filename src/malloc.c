/* File: malloc.c
 * --------------
 * This module contains functions to allocate and free memory from the heap. 
 * The heap allocator allows the recycling of used memory blocks. It also automatically merges consequentive free blocks.
 * The module also contains the heap_dump functions, which prints a view of the heap (used for debugging).
 */


 /*
 * The code given below is simple "bump" allocator from lecture.
 * An allocation request is serviced by using sbrk to extend
 * the heap segment.
 * It does not recycle memory (free is a no-op) so when all the
 * space set aside for the heap is consumed, it will not be able
 * to service any further requests.
 *
 * This code is given here just to show the very simplest of
 * approaches to dynamic allocation. You will replace this code
 * with your own heap allocator implementation.
 */

#include "assert.h"
#include "malloc.h"
#include "printf.h"
#include <stddef.h> // for NULL
#include "strings.h"
#include "backtrace.h"

/*
 * Data variables private to this module used to track
 * statistics for debugging/validate heap:
 *    count_allocs, count_frees, total_bytes_requested
 */
static int count_allocs, count_frees, total_bytes_requested;

// Header struct that preceeds each payload. 
struct header {
    int payload_size;
    int status;       // 0 if free, 1 if in use
    frame_t call_stack[3]; // call stack that allocated block, 3 frames deep
};

void report_damaged_redzone (void *ptr);

/*
 * The segment of memory available for the heap runs from &__heap_start
 * to &__heap_max (symbols from memmap.ld establish these boundaries)
 *
 * The variable cur_head_end is initialized to &__heap_start and this
 * address is adjusted upward as in-use portion of heap segment
 * enlarges. Because cur_head_end is qualified as static, this variable
 * is not stored in stack frame, instead variable is located in data segment.
 * The one variable is shared by all and retains its value between calls.
 */

// Call sbrk to enlarge in-use heap area
void *sbrk(size_t nbytes) {
    extern unsigned char __heap_start, __heap_max; // symbols in linker script memmap.ld
    static void *cur_heap_end =  &__heap_start;     // IMPORTANT: static

    void *new_heap_end = (char *)cur_heap_end + nbytes;
    if (new_heap_end > (void *)&__heap_max)    // if request would extend beyond heap max
        return NULL;                // reject
    void *prev_heap_end = cur_heap_end;
    cur_heap_end = new_heap_end;
    return prev_heap_end;
}

// The size of the padding and header in bytes
#define HEADER_SIZE sizeof(struct header)
#define PADDING_SIZE 8
#define TOTAL_OVERHEAD (HEADER_SIZE + 2 * PADDING_SIZE)

// Simple macro to round up x to multiple of n.
// The efficient but tricky bitwise approach it uses
// works only if n is a power of two -- why?
#define roundup(x,n) (((x)+((n)-1))&(~((n)-1)))

// The malloc function allocates the requested bytes in heap memory, returning a pointer to the first byte. 
// Each block starts with an 8-byte header that details the number of usable bytes in the block, as well as its status (0 for free, 1 for in use)
// The malloc function rounds up the requested number of bytes to a multiple of 8 (e.g. 12 bytes rounds up to 16 bytes).
// If the user requests 0 bytes or if the the size of the block (including the header) exceeds the total amount of available memory, returns NULL.
void *malloc (size_t nbytes) {
    if (nbytes == 0)
        return NULL;
    size_t original_bytes_requested = nbytes;
    count_allocs++;
    total_bytes_requested += nbytes; 
    nbytes = roundup(nbytes, 8);

    // First, traverse the heap, and check if there are any freed blocks we can use 
    extern unsigned char __heap_start; 
    char *current_payload = (char *) &__heap_start;
    while (current_payload < (char *)sbrk(0)){
        // Check if there is enough room in this block for payload and header
        struct header* block_header = (struct header *) current_payload;
        if (block_header -> status == 0 && block_header -> payload_size >= nbytes){
            // Write payload to this block
            //Case 1: Payload perfectly fits within block 
            //Case 2: Payload is smaller than block. Assign payload at the back of the block
            //Edge case: If the block just has enough room for the payload+header/redzones, assign full block to payload
            if (block_header -> payload_size > nbytes + TOTAL_OVERHEAD){
                block_header -> payload_size = block_header -> payload_size - nbytes - TOTAL_OVERHEAD;
                current_payload += (block_header -> payload_size) + TOTAL_OVERHEAD;
                // Update header to new block
                block_header = (struct header *) current_payload;
                block_header -> payload_size = nbytes;
                char *padding = current_payload - PADDING_SIZE; // pad before the new block
                // Pad with 7 bytes of % and null terminator
                memset(padding, 9, PADDING_SIZE);
                padding += HEADER_SIZE + PADDING_SIZE; //pad after the new header
                memset(padding, 9, PADDING_SIZE);
            }
            
            //add new stack view
            backtrace_gather_frames(block_header -> call_stack, 3);
            block_header -> status = 1; // mark block as used
            return current_payload + HEADER_SIZE + PADDING_SIZE;
        }

        // If not enough room or occupied, move on to next block
        current_payload += (block_header -> payload_size) + TOTAL_OVERHEAD;
    }

    // If no previously freed blocks could be used, expand heap 
    struct header *prev_heap_end = sbrk(nbytes + TOTAL_OVERHEAD); // including the block header
    if (prev_heap_end == NULL){
        // Request was not successfully processed
        count_allocs--;
        total_bytes_requested -= original_bytes_requested; 
        return NULL;
    }
    prev_heap_end -> payload_size = nbytes;
    prev_heap_end -> status = 1;
    backtrace_gather_frames(prev_heap_end -> call_stack, 3); //include last 3 stack calls
    // pad with 8 bytes of %'s both at the start and end
    char *padding_start = (char *)(prev_heap_end) + HEADER_SIZE;
    memset(padding_start, 9, PADDING_SIZE);
    char *padding_end = padding_start + nbytes + PADDING_SIZE;
    memset(padding_end, 9, PADDING_SIZE);
    // printf("Allocated %ld bytes at %p\n", nbytes, padding_start);
    return padding_start + PADDING_SIZE; // return the address after the header and redzone (start of the payload)
}

static int check_padding(char *padding){
    for (int i = 0; i < PADDING_SIZE; i++) {
        if (padding[i] != 9) {
            return 0;
        }
    }
    return 1;
}
// The free function frees up the assigned section of memory so that it can be later reused. 
// If the input is the NULLptr, does nothing. 
// When freeing a block, the function checks its forward neighbors, and if free, coalesces the blocks into one larger block. 
void free (void *ptr) {
    if (ptr == NULL)
        return;
    count_frees++;
    struct header * block_header =  (struct header *) ((char *)ptr - PADDING_SIZE - HEADER_SIZE); // Actual header starts before the payload
    block_header -> status = 0; // set status to free
    // Check integrity of the red zones
    char *padding = (char *)ptr - PADDING_SIZE;
    if (!check_padding(padding)){
        report_damaged_redzone(ptr);
        // assert(0 == 1);
    }
    padding += block_header -> payload_size + PADDING_SIZE;
    if (!check_padding(padding)){
        report_damaged_redzone(ptr);
        // assert(0 == 1);
    }

    // Check forward neighbor. If free, then coalesce the two blocks
    char *forward_neighbor = (char *)block_header + (block_header -> payload_size) + TOTAL_OVERHEAD;
    struct header *forward_header = (struct header *)forward_neighbor;
    int capacity = block_header -> payload_size;
    while (forward_neighbor < (char *)sbrk(0) && forward_header -> status == 0){
        capacity += forward_header -> payload_size + TOTAL_OVERHEAD;
        forward_neighbor += (forward_header -> payload_size) + TOTAL_OVERHEAD;
        forward_header = (struct header *)forward_neighbor;
    }
    block_header -> payload_size = capacity;
}

// Used for debugging. Prints the current view of the stack with the passed label. 
void heap_dump (const char *label) {
    extern unsigned char __heap_start;
    printf("\n---------- HEAP DUMP (%s) ----------\n", label);
    printf("Heap segment at %p - %p\n", &__heap_start, sbrk(0));

    struct header *current_payload =  (struct header *) &__heap_start;
    while (current_payload < (struct header *) sbrk(0)){
        printf("Block at %p: Size %d bytes, Status: %d\n", current_payload, current_payload->payload_size, current_payload->status);
        char *next_header = (char *)current_payload + (current_payload -> payload_size) + TOTAL_OVERHEAD;
        current_payload = (struct header *)next_header;
    }
    printf("Stopped at %p\n", current_payload);
    printf("----------  END DUMP (%s) ----------\n", label);
    printf("Stats: %d in-use (%d allocs, %d frees), %d total bytes requested\n\n",
        count_allocs - count_frees, count_allocs, count_frees, total_bytes_requested);
}

// Prints a report of the memory allocs and frees after the program finishes running.
// Reports any lost (unfreed) blocks along with their size and the call stack that allocated them.
void memory_report (void) {
    printf("\n=============================================\n");
    printf(  "         Mini-Valgrind Memory Report         \n");
    printf(  "=============================================\n");
    printf("Final stats: %d allocs, %d frees, %d total bytes requested\n", count_allocs, count_frees, total_bytes_requested);
    extern unsigned char __heap_start;
    struct header *current_payload =  (struct header *) &__heap_start;
    int num_leaky_blocks = 0;
    int total_wasted_bytes = 0;
    while (current_payload < (struct header *) sbrk(0)){
        if (current_payload -> status != 0){
            num_leaky_blocks++;
            total_wasted_bytes += current_payload -> payload_size;
            printf("%d bytes are lost, allocated by\n", current_payload -> payload_size);
            backtrace_print_frames(current_payload -> call_stack, 3);
        }
        char *next_header = (char *)current_payload + (current_payload -> payload_size) + TOTAL_OVERHEAD;
        current_payload = (struct header *)next_header;
    }
    printf("Lost %d total bytes in %d blocks.\n", total_wasted_bytes, num_leaky_blocks);
}

// This function is called when redzone damage is detected. 
// Prints the status of this damaged block (including the size and call stack).
void report_damaged_redzone (void *ptr) {
    printf("\n=============================================\n");
    printf(  " **********  Mini-Valgrind Alert  ********** \n");
    printf(  "=============================================\n");
    printf("Attempt to free address %p that has damaged red zone(s):", ptr);
    struct header * block_header =  (struct header *) ((char *)ptr - HEADER_SIZE - PADDING_SIZE);
    printf("Block of size  %d bytes, allocated by\n", block_header -> payload_size);
    backtrace_print_frames(block_header -> call_stack, 3);
}
