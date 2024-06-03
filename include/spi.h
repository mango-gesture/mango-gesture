#pragma once

/*
 * Author: Yifan Yang <yyang29@stanford.edu>
 *
 * Date: Mar 5, 2024
 * 
 * Modified by Nika Zahedi to support burst-mode reading
 * Finished May 28, 2024
 */

void spi_init(void);
void spi_transfer(unsigned char *tx, unsigned char *rx, int len);
void spi_transfer_burst (unsigned char *tx, unsigned char *rx, int len);
unsigned int spi_transfer_jpeg_burst (unsigned char *tx, unsigned char *rx, int max_len);