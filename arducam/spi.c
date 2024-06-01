/*
 * Hardware SPI driver on Mango Pi
 *
 * Author: Yifan Yang <yyang29@stanford.edu>
 *
 * Date: Mar 5, 2024
 * 
 * Modified by Nika Zahedi to support reading in burst mode and reading data
 * of variable length.
 * 
 * Date: May 28, 2024
 */

#include "spi.h"
#include "ccu.h"
#include "gpio.h"
#include "gpio_extra.h"
#include "printf.h"
#include <stdint.h>

typedef union {
    struct {
        uint32_t reserved0;
        struct {
            uint32_t spi_en: 1;
            uint32_t master_mode: 1;
            uint32_t sample_timing_mode: 1;
            uint32_t dbi_mode_sel: 1;
            uint32_t dbi_en: 1;
            uint32_t reserved0: 2;
            uint32_t transmit_pause_en: 1;
            uint32_t reserved2: 23;
            uint32_t soft_reset: 1;
        } gcr;
        struct {
            uint32_t cpha: 1;
            uint32_t cpol: 1;
            uint32_t spol: 1;
            uint32_t ssctl: 1;
            uint32_t chip_sel: 2;
			uint32_t ss_owner: 1;
			uint32_t ss_level: 1;
            uint32_t rest: 2;
            uint32_t rapid_write: 1;
            uint32_t rest2: 20;
            uint32_t start_burst: 1;
        } tcr;
        uint32_t reserved1;
        uint32_t ier;
        struct {
            uint32_t rx_ready: 1;
            uint32_t rx_empty: 1;
            uint32_t rx_full: 1;
            uint32_t reserved0: 1;
            uint32_t tx_ready: 1;
            uint32_t tx_empty: 1;
            uint32_t tx_full: 1;
            uint32_t reserved1: 1;
            uint32_t rx_overflow: 1;
            uint32_t rx_underflow: 1;
            uint32_t tx_overflow: 1;
            uint32_t tx_underflow: 1;
            uint32_t transfer_complete: 1;
            uint32_t ss_invalid: 1;
            uint32_t rest: 18;
        } isr;
        struct {
            uint32_t rest: 31;
            uint32_t fifo_reset: 1;
        } fcr;
        struct {
            uint32_t rx_fifo_cnt: 8;
            uint32_t reserved0: 4;
            uint32_t rb_cnt: 3;
            uint32_t rb_wr: 1;
            uint32_t tx_fifo_cnt: 8;
            uint32_t reserved1: 4;
            uint32_t tb_cnt: 3;
            uint32_t tb_wr: 1;
        } fsr;
        uint32_t wcr;
        uint32_t reserved2;
        uint32_t samp_dl;
        uint32_t reserved3;
        uint32_t mbc;
        uint32_t mtc;
        struct {
            uint32_t stc: 24;
            uint32_t rest: 6;
            uint32_t dual_rx_en: 1;
            uint32_t quad_mode_en: 1;
        } bcc;
        uint32_t reserved4;
        struct {
            uint32_t work_mode: 2;
            uint32_t rest: 23;
            uint32_t tbc: 1;
            uint32_t rest2: 6;
        } batcr;
        uint32_t ba_ccr;
        uint32_t tbr;
        uint32_t rbr;
        uint32_t reserved5[14];
        uint32_t ndma_mode_ctl;
        uint32_t dbi[93];
        unsigned char txd[4];
        uint32_t reserved6[63];
        unsigned char rxd[4];
    } regs;
    unsigned char padding[0x304];
} spi_t;

#define SPI_BASE ((spi_t *)0x04026000)
_Static_assert(&(SPI_BASE->regs.rxd[0]) == (unsigned char *)0x04026300, "SPI1 rxd reg must be at address 0x04026300");
_Static_assert(&(SPI_BASE->regs.txd[0]) == (unsigned char *)0x04026200, "SPI1 txd reg must be at address 0x04026200");
_Static_assert(&(SPI_BASE->regs.ndma_mode_ctl) == (uint32_t *)0x04026088, "SPI1 ndma_mode_ctl reg must be at address 0x04026088");

volatile spi_t *module = ((spi_t *)0x04026000);

void
spi_init (void)
{
    const uint32_t SPI1_CLK_REG = 0x0944;
    const uint32_t SPI_CLK_REG_VAL = (1 << 31) | (0b11 << 8);
    ccu_write (SPI1_CLK_REG, SPI_CLK_REG_VAL);
    ccu_enable_bus_clk (0x096C, (1 << 1), (1 << 17));
    module->regs.gcr.soft_reset = 1;
    while (module->regs.gcr.soft_reset)
        ;
    gpio_set_function (GPIO_PD11, GPIO_FN_ALT4); // SPI1_CLK
    gpio_set_function (GPIO_PD10, GPIO_FN_ALT4); // SPI1_CS0
    gpio_set_function (GPIO_PD12, GPIO_FN_ALT4); // SPI1_MOSI
    gpio_set_function (GPIO_PD13, GPIO_FN_ALT4); // SPI1_MISO

	gpio_set_pullup(GPIO_PD10);

    module->regs.gcr.spi_en = 1;
	module->regs.gcr.dbi_mode_sel = 0;
    module->regs.gcr.master_mode = 1;
	module->regs.tcr.cpol = 0;
	module->regs.tcr.spol = 1;
	module->regs.tcr.cpha = 0;
	module->regs.tcr.ss_level = 0;
    module->regs.tcr.chip_sel = 0;
    // Enable FIFO queue
    module->regs.fsr.rb_wr = 1;
    module->regs.fsr.tb_wr = 1; 
}

void
spi_transfer (unsigned char *tx, unsigned char *rx, int len)
{
    gpio_write(GPIO_PD10, 0); // CS pin goes low to start reading

	module->regs.tcr.ss_level = 1;
    module->regs.mbc = len;
    module->regs.mtc = len;
    module->regs.bcc.stc = len;

    for (int i = 0; i < len; i++) {
        module->regs.txd[0] = tx[i];
        while (!module->regs.isr.tx_ready) 
            ;
    }

    module->regs.tcr.start_burst = 1;
    module->regs.isr.transfer_complete = 1;
    while (!module->regs.isr.transfer_complete)
      ;
    for (int i = 0; i < len; i++) {
        if (module->regs.fsr.rx_fifo_cnt > 0) {
            rx[i] = module->regs.rxd[0];
        } else {
            rx[i] = 0;
        }
    }

	module->regs.tcr.ss_level = 0;
    gpio_write(GPIO_PD10, 1);
}

void
spi_transfer_burst (unsigned char *tx, unsigned char *rx, int len)
{
    module->regs.mbc = len;
    module->regs.mtc = 1;
    module->regs.bcc.stc = len;

    module->regs.txd[0] = tx[0];

    module->regs.tcr.start_burst = 1;
    while (!module->regs.isr.tx_ready) 
            ;

    while (module->regs.fsr.rx_fifo_cnt == 0)
            ;  // Wait for data to be available
    rx[0] = module->regs.rxd[0];
    for (int i = 1; i < len ; i ++) {
        // Check and read the received data as soon as available
        while (module->regs.fsr.rx_fifo_cnt == 0)
            ;  // Wait for data to be available
        rx[i] = module->regs.rxd[0];
    }

    // Ensure all data is transmitted and received
    while (!module->regs.isr.transfer_complete)
        ;

}

// returns the number of read bytes
unsigned int
spi_transfer_jpeg_burst (unsigned char *tx, unsigned char *rx, int max_len)
{
    const unsigned char JPEG_EOF_MARKER[2] = {0xFF, 0xD9};


    gpio_write(GPIO_PD10, 0); // CS pin goes low to start reading

	module->regs.tcr.ss_level = 1;
    module->regs.mbc = max_len;
    module->regs.mtc = 1;
    module->regs.bcc.stc = max_len;

    module->regs.txd[0] = tx[0];
    while (!module->regs.isr.tx_ready)
            ;
    module->regs.tcr.start_burst = 1;

    int i = 0;
    for (; i < max_len; i++) {
        // Check and read the received data as soon as available
        while (module->regs.fsr.rx_fifo_cnt == 0)
            ;  // Wait for data to be available
        rx[i] = module->regs.rxd[0];

        if ((i != 0) && (rx[i-1] == JPEG_EOF_MARKER[0]) && (rx[i] == JPEG_EOF_MARKER[1])) {
            // When JPEG EOF marker found, break
            break;
        }

    }

    while (!module->regs.isr.transfer_complete)
        ;
    module->regs.gcr.soft_reset = 1;
    while (module->regs.gcr.soft_reset)
        ;

	module->regs.tcr.ss_level = 0;
    gpio_write(GPIO_PD10, 1);

    return (i + 1);
}