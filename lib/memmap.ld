STACK_SIZE = 0x100000; /* 1 MB */

SECTIONS
{
    .text  0x40000000     :  { *(.text.start) *(.text*) }
    .rodata               :  { *(.rodata*) *(.srodata*) }
    __text_end            = .;
    .data  ALIGN(0x20000) :  { *(.sdata*) }
    __bss_start           = .;
    .bss                  :  { *(.bss*)  *(.sbss*) *(COMMON) }
    __bss_end             = .;
    .heap  0x44000000     :  { __heap_start = .; }
    .stack 0x50000000 - STACK_SIZE : {
        __heap_max = .;
        . = . + STACK_SIZE;
        __stack_top = .;
    }
}

/* Force link of _start and verify correct position */
ENTRY(_start_gdb)
ASSERT(_start == ADDR(.text), "_start symbol must be placed first in text section")