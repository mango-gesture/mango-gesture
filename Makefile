NAME =  main
MODULES = inference.c inf.asm.s arducam.c printf.c omni.c i2c.c spi.c
all : $(NAME).bin

ARCH    = -march=rv64imf_zicsr -mabi=lp64
ASFLAGS = $(ARCH)
CFLAGS = $(ARCH) -g -Og -I include $$warn $$freestanding -fno-omit-frame-pointer
LDFLAGS = -nostdlib -L$$CS107E/lib -T memmap.ld
LDLIBS = -lmango -lmango_gcc


OBJECTS = $(addsuffix .o, $(basename $(MODULES))) 

%.bin: %.elf
	riscv64-unknown-elf-objcopy $< -O binary $@

%.elf: $(OBJECTS) %.o 
	riscv64-unknown-elf-gcc $(LDFLAGS) $^ $(LDLIBS) -o $@

%.o: %.c
	riscv64-unknown-elf-gcc $(CFLAGS) -c $< -o $@

%.o: %.s
	riscv64-unknown-elf-as $(ASFLAGS) $< -o $@

%.list: %.o
	riscv64-unknown-elf-objdump $(OBJDUMP_FLAGS) $<

# debug: $(NAME).elf
# 	riscv64-unknown-elf-gdb $(GDB_FLAGS) $<



run: $(NAME).bin
	xfel ddr d1
	xfel write 0x60000000 weights/new_test_256.bin
	xfel write 0x40000000 main.bin
	xfel exec 0x40000000

clean:
	rm -f *.o *.bin *.elf *.list *~

# Access .c and .s source files within shared mylib directory using vpath
# https://www.cmcrossroads.com/article/basics-vpath-and-vpath
vpath %.c .:arducam
vpath %.c .:src
vpath %.c .:inference
vpath %.s .:inference
vpath %.h .:include

.PHONY: all clean run
.PRECIOUS: %.elf %.o

# disable built-in rules (they are not used)
.SUFFIXES:

export warn = -Wall -Wpointer-arith -Wwrite-strings -Werror \
              -Wno-error=unused-function -Wno-error=unused-variable \
              -fno-diagnostics-show-option
export freestanding = -ffreestanding -nostdinc \
                      -isystem $(shell riscv64-unknown-elf-gcc -print-file-name=include)
OBJDUMP_FLAGS = -d --no-show-raw-insn --no-addresses --disassembler-color=terminal --visualize-jumps
# GDB_FLAGS = -q --command=$$CS107E/other/gdbsim.commands
