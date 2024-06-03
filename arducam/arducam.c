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
static int IMAGE_LEN_DIFF_THRESHOLD;
// #define IMAGE_DIFF_THRESHOLD 100

//Image handling functions
void store_jpeg(void);
int read_jpeg(unsigned char *rxd);
void stream_bmp(void);
void stream_image(void);
int image_field_has_changed(void);
int find_field_diff(int *len_diff);


//Arducam initialization functions
void arducam_init(unsigned w, unsigned h, unsigned x, unsigned y);
void arducam_init_bg(void);
void arducam_calibrate(void);

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

typedef struct {
	unsigned char* img;
	int len;
} bg_image;

bg_image REF_BG; // reference background image for diff calculations

static volatile camera_t cam;

/* This module not only initializes the camera, it also handles the image
 * dislpaying on the monitor, which is why the struct is defined above
 */

//Initializes all the required peripherals, checks the comms, and sets the camera_t values
void arducam_init(unsigned w, unsigned h, unsigned x, unsigned y) {
	spi_init();
	// printf("spi initialized\n");

	i2c_init();
	// printf("i2c initialized\n");

	if (arducam_check_interface()) {
		// printf("connected to camera!\n");
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

}

void arducam_init_bg(void){
	unsigned char* img = malloc(JPEG_MAX_LEN + 1);
	if (!img) {
		free(img);
		printf("BG image Memory allocation failed\n");
		return;  // Handle memory allocation failure
	}
	
	int num = read_jpeg(img);
	REF_BG.img = img;
	REF_BG.len = num;
}

int find_field_diff(int *len_diff){
	unsigned char* img = malloc(JPEG_MAX_LEN + 1);
	if (!img) {
		free(img);
		printf("Memory allocation failed\n");
		return -1;  // Handle memory allocation failure
	}
	
	
	int num = read_jpeg(img);

	int count = num > REF_BG.len ? REF_BG.len : num;
	int sum_abs_diff = 0;
	for (size_t i = 0; i < count; i++)
	{
		int diff = img[i] ^ REF_BG.img[i];
		for (int i = 0 ; i < 8 ; i++){
			sum_abs_diff += diff & 0b1;
			diff = diff >> 1;
		}
	}
	// printf("len diff: %d", num2 - num1);
	*len_diff = num - REF_BG.len;
	// printf("Sum of abs differences: %d\n", sum_abs_diff);
	

	free(img);  // Free receive buffers after processing

	return sum_abs_diff;
}

int image_field_has_changed(void){
	int len_diff = 0;
	int abs_dif = find_field_diff(&len_diff);
	if (len_diff > IMAGE_LEN_DIFF_THRESHOLD){
		return 1;
	}
	return 0;
}

// Sets the correct threshold value to be used for image diff calculations.
// Note: arducam_init_bg() must be called before this function.
void arducam_calibrate(void){
	printf("Beginning calibration...\n");
	int num_iter = 100; // Number of iterations to average over
	int len_diff = 0;
	int max_len_dif = -1;
	for (int i = 0 ; i < num_iter ; i++){
		find_field_diff(&len_diff);
		if (len_diff > max_len_dif) max_len_dif = len_diff;
		if (i % 100 == 0)
			printf("Finished calibration iteration %d\n", i);
	}
	printf("Test and hold gesture: \n");
	timer_delay_ms(2000);
	int positive_ctrl = 0;
	int min_len_dif_ctrl = JPEG_MAX_LEN;
	for (int i = 0 ; i < num_iter ; i++){
		find_field_diff(&positive_ctrl);
		if (positive_ctrl < min_len_dif_ctrl) min_len_dif_ctrl = positive_ctrl;
		if (i % 100 == 0)
			printf("Finished calibration iteration %d\n", i);
	}
	printf("Remove hand\n");
	timer_delay_ms(1800);

	IMAGE_LEN_DIFF_THRESHOLD = (max_len_dif + min_len_dif_ctrl) / 2;
	printf("Finished calibrating!\n");

}

// Reads a JPEG image from the Arducam into the rxd buffer and returns the number of bytes read
int read_jpeg(unsigned char *rxd){
	arducam_begin_capture();
	while(!arducam_capture_done());
	// printf("Starting single-frame jpeg capture...\n");
	unsigned char *txd = malloc(JPEG_MAX_LEN + 1);

	if (!txd || !rxd) {
        free(txd);
        free(rxd);
		printf("Memory allocation failed\n");
        return -1;  // Handle memory allocation failure
    }

	memset(txd, 0, JPEG_MAX_LEN + 1);
	memset(rxd, 0, JPEG_MAX_LEN + 1);

	txd[0] = ARD_FIFO_BURST_READ;

	int num = spi_transfer_jpeg_burst(txd, rxd, JPEG_MAX_LEN + 1);

	arducam_clear_fifo();

	free(txd);  // Free transmit buffer after processing
	return num;
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
	unsigned char *image_data = malloc(JPEG_MAX_LEN + 1);

	int num = read_jpeg(image_data);
	char string_rep [num * 2 + 1]; 
	for (int i = 0 ; i < num ; i++){
		string_rep[2 * i] = image_data[i] % 26 + 'a';
		string_rep[2 * i + 1] = image_data[i] / 26 + 'a';
	}
	string_rep[num * 2 + 1] = 0;
	// int elapsed_time_us = (timer_get_ticks() - start_ticks)/TICKS_PER_USEC;
	// printf("Time in ms: %d\n", elapsed_time_us/1000);

	printf("Size %d: ", num); // Add file size separator
	printf("%s\n", string_rep); // Print to minicom to save to file

	free(image_data);  // Free receive buffer after storage
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