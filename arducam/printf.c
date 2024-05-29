/* File: printf.c
 * --------------
 * The printf.c module contains functions to write strings and other embedded format types to an output buffer and the uart.
 * The supported format types include integers and longs, pointers, characters, and strings. 
 * It also contains a disassembler that can parse hex commands into their corresponding assembly instructions. 
 */
#include "printf.h"
#include <stdarg.h>
#include <stdint.h>
#include "strings.h"
#include "uart.h"
#include "assert.h" 

/* Prototypes for internal helpers.
 * Typically these would be qualified as static (private to module)
 * but, in order to call them from the test program, we declare them externally
 */
int unsigned_to_base(char *buf,
                     size_t bufsize,
                     unsigned long val,
                     int base, size_t
                     min_width);
int signed_to_base(char *buf,
                   size_t bufsize,
                   long val,
                   int base,
                   size_t min_width);

void disassemble(char *buf, unsigned long *instruction);
#define MAX_OUTPUT_LEN 4096

// The unsigned_to_base method converts the given unsigned long value to a string (in the given base) and writes it to the given buffer.
// The allowed space in the buffer is determined using the input bufsize value, and the written character count (including the null-temrinator) cannot exceed this bufsize.
// Finally, the user can specify the desired minimum width of the output string. We pad the buffer with zeros to achieve this minimum width, as long as we don't exceed the given buffer size.
// The function returns as output the total number of characters that would have been written to the buffer if there was enough room. 
int unsigned_to_base(char *buf, size_t bufsize, unsigned long val, int base, size_t min_width) {
    long base_raised_to_max_power = 1; // raised to 0
    size_t num_digits = 1;
    while (base_raised_to_max_power * base <= val || num_digits < min_width){
       // in rare cases, base raised to power might exceed max range, which would keep us in an infinite loop
        if (base_raised_to_max_power * base < base_raised_to_max_power){
            break;
        }
        base_raised_to_max_power *= base;
        num_digits++;
    }

    int i = 0;
    
    // Note: the last loop condition exists because if bufsize = 0, bufsize-1 returns the max size_t value
    for (; i < num_digits && i < bufsize - 1 && bufsize > 0 ; i++){
        unsigned int next_char = val / base_raised_to_max_power;
        if (next_char <= 9)
            buf[i] = next_char + 48;
        else
            buf[i] = next_char + 97 - 10;

        val %= base_raised_to_max_power;
        base_raised_to_max_power /= base;
    }
    if (bufsize > 0)
        buf[i] = 0; // null-terminate if buffer has room 

    return num_digits;
}

// An extension of unsigned_to_base which accepts both positive and negative long values. 
int signed_to_base(char *buf, size_t bufsize, long val, int base, size_t min_width) {
    // If the value is negative, reserve one spot for the minus sign
    if (val < 0){
        if (min_width >= 1)
            min_width --; // min_width is a size_t so we have to ensure we don't make it negative
        if (bufsize > 1){
            *buf = '-'; // if buffer has room, write - sign
            return (1 + unsigned_to_base(buf + 1, bufsize - 1, -val, base, min_width));
        }
        // Since bufsize is a size_t, if we accidentally decrement a zero, it loops around to the max value of size_t 
        else if (bufsize > 0)
            return (1 + unsigned_to_base(buf + 1, bufsize, -val, base, min_width));
        
        return (1 + unsigned_to_base(buf + 1, bufsize, -val, base, min_width));
    }
    return unsigned_to_base(buf, bufsize, val, base, min_width);
}


// The vsnprintf function receives an input string of characters and embedded format tags, along with a va_list of arguments needed by these tags. 
// It formats and replaces the tags with their corresponding arguments and writes the output to the buffer (space-permitting by the given bufsize).
// It returns the total number of characters that would have been written to the buffer if there was enough room. 
int vsnprintf(char *buf, size_t bufsize, const char *format, va_list args) {
 
    if (bufsize != 0)
        buf[0] = 0; //enables us to use strlcat
    
    // Parse format to figure out how many additional args we need
    int return_val = 0;
    for (int i = 0 ; format[i] != 0 ; i++){
        if (format[i] == '%'){

            // These fields are used to format strings with numbers
            unsigned long width = 0;
            const char *rest = NULL;
            unsigned int is_long = 0; // determines if the input is a long
            switch (format[++i]){
                case 'c':
                    char next_char = (char)va_arg(args, int);
                    size_t current_buf_len = strlen(buf);
                    if (bufsize > 0 && current_buf_len < bufsize - 1){
                       buf[current_buf_len] = next_char;
                       buf[current_buf_len + 1] = 0; //null-terminate
                    }
                    return_val++;                   
                    continue;
                // case 'b':
                //     unsigned char next_byte = (unsigned char)va_arg(args, int);
                //     size_t curr_buf_len = strlen(buf);
                //     if (bufsize > 0 && curr_buf_len < bufsize - 1){
                //        buf[curr_buf_len] = next_byte;
                //     //    buf[current_buf_len + 1] = 0; //null-terminate
                //     }
                //     return_val++;                   
                //     continue;
                case 's':
                    char *src = (char *)va_arg(args, long int);
                    //count the number of characters that would be added 
                    return_val -= strlen(buf);
                    return_val += strlcat(buf, src, bufsize); 
                    continue;
                case '0':
                    width = strtonum(format + i + 1, &rest);
                    i = (int)(rest - format);
                    if (format[i] != 'l')
                        break;
                case 'l':
                    // activate long flag                    
                    is_long = 1;
                    i++;
                    break;
                case 'p':
                    if (format[i + 1] == 'I'){ 
                        unsigned long *instruction_address =(unsigned long *) va_arg(args, long);
                        char formatted_instruction[30] = {0};
                        disassemble(formatted_instruction, instruction_address);
                        return_val -= strlen(buf);
                        return_val += strlcat(buf, formatted_instruction, bufsize);          
                        i++;
                    }
                    else{
                        unsigned long ptr_val = va_arg(args, long);
                        char nums[17] = ""; //stores the digits present in the ptr
                        unsigned_to_base(nums, 17, ptr_val, 16, 8);
                        char formatted_ptr[19] = "0x";
                        strlcat(formatted_ptr, nums, 19);
                        return_val -= strlen(buf);
                        return_val += strlcat(buf, formatted_ptr, bufsize);
                    }
                    continue;
                case '%':
                    // Did not fit into our regular formatting cases, treat as '%' character
                    current_buf_len = strlen(buf);
                    if (bufsize > 0 && current_buf_len < bufsize - 1){
                        buf[current_buf_len] = format[i];
                        buf[current_buf_len + 1] = 0; //null-terminate
                    }
                    return_val++;
                    continue; 
                case 'd': case 'x': break;
                default:
                    uart_putstring("Error! String format incorrect! \n");
                    assert(0 == 1);
                    return -1;
            }

            // If we've made it here, we need to parse number inputs
            char read_number_str[MAX_OUTPUT_LEN] = {0};
            if (is_long){
                long val;
                val = va_arg(args, long);
                if (format[i] == 'd'){
                    signed_to_base(read_number_str, MAX_OUTPUT_LEN, val, 10, width);
                }
                else if (format[i] == 'x'){             
                    unsigned long unsigned_val = (unsigned long) val; 
                    unsigned_to_base(read_number_str, MAX_OUTPUT_LEN, unsigned_val, 16, width);
                }

            }
            else{
                int val;
                val = va_arg(args, int);
                if (format[i] == 'd'){
                    signed_to_base(read_number_str, MAX_OUTPUT_LEN, val, 10, width);
                }
                else if (format[i] == 'x'){            
                    // Note: this casting is necessary to avoid potentially padding by ffffffff if the most significant bit is 1 (In this case the value is interpretted as a signed long)
                    unsigned int unsigned_val = (unsigned int) val;  
                    unsigned_to_base(read_number_str, MAX_OUTPUT_LEN, unsigned_val, 16, width);
                }

            }
            
            return_val -= strlen(buf);
            return_val += strlcat(buf, read_number_str, bufsize);
            continue;
        }
        // Space-permitting, append the next char in format to buf
        size_t current_buf_len = strlen(buf);
        if (bufsize > 0 && current_buf_len < bufsize - 1){
            buf[current_buf_len] = format[i];
            buf[current_buf_len + 1] = 0; //null-terminate
        }
        return_val++;
    }

    return return_val;
}


// The snprintf function recieves a string of characters and embedding formats (called format), followed by any additional arguments needed to parse the input format string. 
// It parses the additional arguments as a va_list and passes the input and arguments to vsnprintf.
// The result is written to a buffer string, space-permitting by the given bufsize. 
// Returns the numbers of characters that would have been written to the buffer if there was enough room.
int snprintf(char *buf, size_t bufsize, const char *format, ...) {

    va_list args; // declare va list   
    va_start(args, format);

    int vsnprintf_return_val = vsnprintf(buf, bufsize, format, args); 
    
    va_end(args); // clean up
    return vsnprintf_return_val;
}


// An extension of snprintf above that writes the resulting buffer string to the uart.
// The return value is the same as snprintf and vsnprintf above. 
int printf(const char *format, ...) {
    char buf[MAX_OUTPUT_LEN] = "";
    va_list args; // declare va list   
    va_start(args, format);
    
    int return_val = vsnprintf(buf, MAX_OUTPUT_LEN, format, args);
    uart_putstring(buf);
 
    va_end(args);
    return return_val;
}


/* From here to end of file is some sample code and suggested approach
 * for those of you doing the disassemble extension. Otherwise, ignore!
 *
 * The struct insn bitfield is declared using exact same layout as bits are organized in
 * the encoded instruction. Accessing struct.field will extract just the bits
 * apportioned to that field. If you look at the assembly the compiler generates
 * to access a bitfield, you will see it simply masks/shifts for you. Neat!
 */

static const char *reg_names[32] = {"zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
                                    "s0/fp", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
                                    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
                                    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6" };

static const char *arithmetic_instructions[8] = {"add", "sll", "slt", "sltu", "xor", "srl", "or", "and"};
static const char *load_instructions[8] = {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "?!"};
static const char *store_instructions[4] = {"sb", "sh", "sw", "sd"};
static const char *branch_instructions[8] = {"beq", "bne", "?!", "?!", "blt", "bge", "bltu", "bgeu"};

struct insn  {
    uint32_t opcode: 7;
    uint32_t reg_d:  5;
    uint32_t funct3: 3;
    uint32_t reg_s1: 5;
    uint32_t reg_s2: 5;
    uint32_t funct7: 7;
};

void sample_use(unsigned int *addr) {
    struct insn in = *(struct insn *)addr;
    printf("opcode is 0x%x, reg_dst is %s\n", in.opcode, reg_names[in.reg_d]);
}

void disassemble(char *operation, unsigned long *instruction){
    struct insn in = *(struct insn *)instruction;
   
    long immediate_val = 0; 
    char instruction_type;
    switch(in.opcode){
        case 0b0110011:
            //R-type instructions
            instruction_type = 'r';
            printf("This is an r-type opcode\n");
            if (in.funct7 == 0)
                strlcat(operation, arithmetic_instructions[in.funct3], MAX_OUTPUT_LEN);
            else if (in.funct3 == 0b000 && in.funct7 == 0b0100000)
                strlcat(operation, "sub", MAX_OUTPUT_LEN);
            else if (in.funct3 == 0b101 && in.funct7 == 0b0100000)
                strlcat(operation, "sra", MAX_OUTPUT_LEN);
            else{
                printf("Error: Invalid operation!\n");
                assert(0 == 1);
            }
            break;
        case 0b0010011:
            //I-type arithmetic or shift 
            instruction_type = 'i';
            //TODO: sltui instead of sltiu
            if (in.funct3 == 0b101 && in.funct7 == 0b0100000)
                strlcat(operation, "sra", MAX_OUTPUT_LEN);
            else
                strlcat(operation, arithmetic_instructions[in.funct3], MAX_OUTPUT_LEN);
            strlcat(operation, "i ", MAX_OUTPUT_LEN);
            
            // find the immediate value to be stored
            // shift-by-immediate instructions use lower 5 buts of the immediate value
            if (in.funct3 == 0b001 || in.funct3 == 0b101)
                immediate_val = in.reg_s2;
            else 
                immediate_val = (in.funct7 << 5) | in.reg_s2;
            break;
        case 0b0000011:
            //L-type (load instructions)
            instruction_type = 'l';
            strlcat(operation, load_instructions[in.funct3], MAX_OUTPUT_LEN);
            immediate_val = (in.funct7 << 5) | in.reg_s2;
            break;
        case 0b0100011:
            //S-type Store 
            instruction_type = 's';
            strlcat(operation, store_instructions[in.funct3], MAX_OUTPUT_LEN);
            immediate_val = (in.funct7 << 5) | in.reg_d;
            break;
        case 0b1100011:
            //b-type branching instructions
            instruction_type = 'b';
            strlcat(operation, branch_instructions[in.funct3], MAX_OUTPUT_LEN);
            immediate_val = ((in.funct7 & 0b1000000) << 6) | ((in.reg_d & 0b1) << 11) | ((in.funct7 & 0b111111) << 5) | (in.reg_d & 0b11110);
            break;
        default:
            //Not implemented error
            printf("Error! Given opcode %d does not match any R-type or I-type instructions.\n", in.opcode);
            assert (0 == 1);
    }
    // Sign-extend the immediate value
    if (immediate_val & (1 << 12)) {  // If bit 12 is set to 1, the value is negative
        immediate_val |= 0xffffffffffffe000;  // Extend the sign through the upper bits
    }
    //Pad with space
    strlcat(operation, " ", MAX_OUTPUT_LEN);
    //Now, add the destination register name to the instructions
    if (instruction_type == 's')
        strlcat(operation, reg_names[in.reg_s2], MAX_OUTPUT_LEN);
    else if (instruction_type == 'b'){
        strlcat(operation, reg_names[in.reg_s1], MAX_OUTPUT_LEN);
        strlcat(operation, ", ", MAX_OUTPUT_LEN);
        strlcat(operation, reg_names[in.reg_s2], MAX_OUTPUT_LEN); 
    }
    else
        strlcat(operation, reg_names[in.reg_d], MAX_OUTPUT_LEN);

    //Pad with space and comma
    strlcat(operation, ", ", MAX_OUTPUT_LEN);


    char immediate_val_string[MAX_OUTPUT_LEN] = "";
    if (instruction_type == 'r'){
        //Add source register1
        strlcat(operation, reg_names[in.reg_s1], MAX_OUTPUT_LEN);
        strlcat(operation, ", ", MAX_OUTPUT_LEN);

        // Source 2
        strlcat(operation, reg_names[in.reg_s2], MAX_OUTPUT_LEN);
    }
    else if (instruction_type == 'i'){
        //Add source register1
        strlcat(operation, reg_names[in.reg_s1], MAX_OUTPUT_LEN);
        strlcat(operation, ", ", MAX_OUTPUT_LEN);


        signed_to_base(immediate_val_string, MAX_OUTPUT_LEN, immediate_val, 10, 0);
        strlcat(operation, immediate_val_string, MAX_OUTPUT_LEN);
    }
    else if (instruction_type == 'l' || instruction_type == 's'){
        signed_to_base(immediate_val_string, MAX_OUTPUT_LEN, immediate_val, 10, 0);
        strlcat(operation, immediate_val_string, MAX_OUTPUT_LEN);       
   
       // put source name in parentheses 
        strlcat(operation, "(", MAX_OUTPUT_LEN);
        strlcat(operation, reg_names[in.reg_s1], MAX_OUTPUT_LEN);
        strlcat(operation, ")", MAX_OUTPUT_LEN); 
    }  
    else if (instruction_type == 'b'){
        signed_to_base(immediate_val_string, MAX_OUTPUT_LEN, immediate_val, 10, 0);
        strlcat(operation, immediate_val_string, MAX_OUTPUT_LEN);
    }
}
