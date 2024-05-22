


.globl get_fs
get_fs:
    csrrs t0, mstatus, x0       # Read the mstatus register into t0
    srli a0, t0, 13             # Shift right by 13 positions to align FS bits at the LSB
    andi a0, a0, 0x03           # Mask with 0x03 to isolate the FS field (two bits)
    ret

.globl set_fs_one
set_fs_one:
    # Read current mstatus
    csrr t0, mstatus        # Read mstatus into t0

    # Clear the FS field (bit 13 and 14)
    li t1, ~(0x3 << 13)     # Load the mask to clear FS bits (inverse of 0x3 shifted left by 13)
    and t0, t0, t1          # Apply mask to clear FS bits

    # Set FS to 01
    li t1, 0x1 << 13        # Load the value to set FS to 01
    or t0, t0, t1           # OR to set FS bits to 01

    # Write back to mstatus
    csrw mstatus, t0        # Write the modified value back to mstatus

    ret  