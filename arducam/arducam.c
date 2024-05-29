/* 
 * This code brought to you by the rag tag duo of Arjun and Eric! Visit us at
 * our github accounts: https://github.com/arjunvb, https://github.com/efritz09
 * 
 * Completed on: March 14, 2016
 * 
 * Modified by Nika Zahedi for Arducam: May 28l 2024
 */

#include "i2c.h"
#include "spi.h"
#include "omni.h"
#include "arducam.h"
#include "gl.h"
#include "printf.h"
#include "malloc.h"
#include "timer.h"
#include "strings.h"


// Image defines
#define WRITE_bm  (1 << 7)

// pixel defines (uses RGB565)
#define BIT_5   0x1F
#define BIT_3   0x07

// SPI parameters
#define POL     0
#define PHA     0
#define CLKDIV  40

#define JPEG_MAX_LEN 153600

//Image handling functions
void store_jpeg(void);
void read_jpeg(void);
void stream_bmp(void);
void stream_image(void);

//arducam commands
void arducam_write(unsigned char addr, unsigned char value);
unsigned char arducam_read(unsigned char addr);
int arducam_check_interface();
void arducam_set_mode();
void arducam_clear_fifo();
void arducam_begin_capture();
int arducam_capture_done();
unsigned char arducam_chip_version();
unsigned char arducam_read_fifo_single();


typedef struct {
	unsigned height;
	unsigned width;
	unsigned h;
	unsigned w;	//for pixel locations
	unsigned x;
	unsigned y;
	unsigned* start;
} camera_t;

static volatile camera_t cam;

/* This module not only initializes the camera, it also handles the image
 * dislpaying on the monitor, which is why the struct is defined above
 */

//Initializes all the required peripherals, checks the comms, and sets the camera_t values
void arducam_init(unsigned w, unsigned h, unsigned x, unsigned y) {
	spi_init();
	printf("spi initialized\n");

	i2c_init();
	printf("i2c initialized\n");

	if (arducam_check_interface()) {
		printf("connected to camera!\n");
	} else {
		printf("SPI interface error\n");
		return;
	}

	omni_init(JPG_MODE); 
	arducam_clear_fifo();
	cam.height = h;
	cam.width = w;
	cam.h = y+h;
	cam.w = x+w;
	cam.x = x;
	cam.y = y;
	cam.start = (unsigned*)malloc(h*w);
	printf("--Camera ready--\n");
}

// Reads a BMP image from the Arducam
void read_jpeg(){

	printf("Starting single-frame jpeg capture...\n");
	int start_ticks = timer_get_ticks();

	unsigned char *txd = malloc(JPEG_MAX_LEN + 1);
    unsigned char *rxd = malloc(JPEG_MAX_LEN + 1);

	if (!txd || !rxd) {
        free(txd);
        free(rxd);
        return;  // Handle memory allocation failure
    }

	memset(txd, 0, JPEG_MAX_LEN + 1);
	memset(rxd, 0, JPEG_MAX_LEN + 1);

	txd[0] = ARD_FIFO_BURST_READ;

	int num = spi_transfer_jpeg_burst(txd, rxd, JPEG_MAX_LEN + 1);

	char string_rep [num * 2 + 1]; 
	for (int i = 0 ; i < num ; i++){
		string_rep[2 * i] = rxd[i] % 26 + 'a';
		string_rep[2 * i + 1] = rxd[i] / 26 + 'a';
	}
	string_rep[num * 2 + 1] = 0;
	int elapsed_time_us = (timer_get_ticks() - start_ticks)/TICKS_PER_USEC;
	printf("Time in ms: %d\n", elapsed_time_us/1000);
	printf("%s", string_rep); // Print to minicom to save to file
	free(rxd);  // Free receive buffer after processing
	
	arducam_clear_fifo();
}

// Reads a BMP image from the Arducam and displays to the screen
void stream_bmp(){

	printf("Starting single-frame capture...\n");
	int start_ticks = timer_get_ticks();

	const int length = cam.width * cam.height * 2;
	unsigned char txd[1] = {ARD_FIFO_BURST_READ};
    unsigned char *rxd = malloc(length + 1);

	if (!rxd) {
        free(rxd);
        return;  // Handle memory allocation failure
    }

	memset(rxd, 0, length + 1);

	
	spi_transfer_burst(txd, rxd, length + 1);
	
	
	// Process the received data
    for (int i = cam.y; i < cam.h; i++) {
        for (int j = cam.x; j < cam.w; j++) {
            int index = ((i - cam.y) * cam.width + (j - cam.x)) * 2;
            unsigned char b1 = rxd[index + 1]; // the first recieved rxd bit is a dummy bit
            unsigned char b2 = rxd[index + 2];
			unsigned char r = ((b1 >> 3) & BIT_5) * 8; // 5
			unsigned char b = (b2 & BIT_5) * 8; // 5
			unsigned char g = (((b2 >> 5) & BIT_3) | ((b1 & BIT_3) << 3)) * 4; //6
			int displayX = (cam.width - 1) - (j - cam.x);
            gl_draw_pixel(j, i, gl_color(r, g, b));
        }
    }

	int elapsed_time_us = (timer_get_ticks() - start_ticks)/TICKS_PER_USEC;
	printf("Time in us: %d\n", elapsed_time_us);
	gl_swap_buffer();
	free(rxd);  // Free receive buffer after processing
	arducam_clear_fifo();
}

// Stores the current image using minicom 
// Storage only supported in JPEG mode
void store_jpeg(void) {
	arducam_begin_capture();
	while(!arducam_capture_done());
	read_jpeg();
}

//calls the commands required to stream the images
// Currently only supports BMP mode
void stream_image(void) {
	gl_clear(GL_AMBER);
	gl_swap_buffer();
	gl_clear(GL_AMBER);
	gl_swap_buffer();
	arducam_begin_capture();
	printf("streaming...\n");

	while(!arducam_capture_done());
	printf("capture done!\n");

	stream_bmp();
}


/* ARDUCAM INTERFACE FUNCTIONS */
// writes values to a register via SPI
void arducam_write(unsigned char addr, unsigned char value)
{
	unsigned char txd[2] = {addr | WRITE_bm, value};
	unsigned char rxd[2] = {0, 0};
	spi_transfer(txd, rxd, 2);
}

//reads values from a register via SPI
unsigned char arducam_read(unsigned char addr)
{
	unsigned char txd[2] = {addr, 0};
	unsigned char rxd[2] = {0, 0};
	spi_transfer(txd, rxd, 2);
	return rxd[1];
}

//tests the SPI connection by sending a dummy value
int arducam_check_interface()
{
	arducam_write(ARD_TEST_REGISTER, TEST_CODE);
	return (arducam_read(ARD_TEST_REGISTER) == TEST_CODE);
}

void arducam_set_mode()
{
	arducam_write(ARD_MODE, MCU2LCD_MODE);
}

void arducam_flush_fifo()
{
	arducam_write(ARD_FIFO_CONTROL, FIFO_CLEAR);
}

void arducam_clear_fifo()
{
	arducam_write(ARD_FIFO_CONTROL, FIFO_CLEAR);
}

void arducam_begin_capture()
{
	arducam_write(ARD_SENSE_TIMING, arducam_read(ARD_SENSE_TIMING) | (1 << 4)); // enable FIFO
	arducam_write(ARD_FIFO_CONTROL, FIFO_START);
	arducam_write(ARD_CAPTURE_CTRL, 1);
}

unsigned char arducam_chip_version()
{
	return arducam_read(ARD_CHIP_VERSION);
}

int arducam_capture_done()
{
	int read_val = arducam_read(ARD_CAMERA_STATUS);
	return (read_val & FIFO_WRITE_DONE);
}

unsigned char arducam_read_fifo_single()
{
	return arducam_read(ARD_FIFO_SINGLE_READ);
}
