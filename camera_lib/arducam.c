/* 
 * This code brought to you by the rag tag duo of Arjun and Eric! Visit us at
 * our github accounts: https://github.com/arjunvb, https://github.com/efritz09
 * 
 * Completed on: March 14, 2016
 */

/*
 * This code brought to you by the rag tag duo of Arjun and Eric! Visit us at
 * our github accounts: https://github.com/arjunvb, https://github.com/efritz09
 *
 * Completed on: March 14, 2016
 */

#include "gpio.h"
#include "gpio_extra.h"
#include "spi.h"

volatile unsigned int *SPI_CS_reg = (unsigned int *) 0x20204000;
#define SPI_TXD_FULL_bm (1 << 18)
#define SPI_RXD_FULL_bm (1 << 17)
#define SPI_DONE_bm     (1 << 16)
#define SPI_REN_bm      (1 << 12)
#define SPI_TA_bm       (1 <<  7)
#define SPI_CSPOL_bm    (1 <<  6)
#define SPI_CLEAR_RX_bm (1 <<  5)
#define SPI_CLEAR_TX_bm (1 <<  4)
#define SPI_CPOL_bm     (1 <<  3)
#define SPI_CPHA_bm     (1 <<  2)
#define SPI_CS_bm       (1 <<  0)

volatile unsigned int *SPI_FIFO_reg = (unsigned int *) 0x20204004;
#define SPI_DATA_bm  (1 << 0)

volatile unsigned int *SPI_CLK_reg = (unsigned int *) 0x20204008;
#define SPI_CDIV_bm  (1 << 0)

// #define CE1  GPIO_PIN7
#define CE0  GPIO_PD10
#define MISO GPIO_PD13
#define MOSI GPIO_PD12
#define SCLK GPIO_PD11

void spi_init_gpio()
{
	gpio_init();

	// set to GPIOs to SPI functionality
	// gpio_set_function(CE1,  GPIO_FUNC_ALT0);
	gpio_set_function(CE0,  GPIO_FN_ALT2); //TODO: is this the correct fn?
	gpio_set_function(MISO, GPIO_FN_ALT2);
	gpio_set_function(MOSI, GPIO_FN_ALT2);
	gpio_set_function(SCLK, GPIO_FN_ALT2);

	gpio_set_pullup(CE0);
	// gpio_set_pullup(CE1);
}

void spi_clear_txrx()
{
	*SPI_CS_reg |= (SPI_CLEAR_TX_bm | SPI_CLEAR_RX_bm);
}

void spi_init(unsigned pol, unsigned pha, unsigned clk_div)
{
	spi_init_gpio();

	*SPI_CS_reg = 0;

	// clear tx/rx channel
	spi_clear_txrx();

	// chip select 0, chip select polarity low

	// clock phase and polarity
	if (pol) *SPI_CS_reg |= SPI_CPOL_bm;
	if (pha) *SPI_CS_reg |= SPI_CPHA_bm;

	// clock divider
	*SPI_CLK_reg = clk_div;
}

void spi_txrx(unsigned char* txbuf, unsigned char* rxbuf, unsigned len)
{
	//clear the tx/rx
	spi_clear_txrx();

	// bring chip select low
	*SPI_CS_reg |= SPI_TA_bm;

	for (int i=0; i < len; i++) {
		// send byte
		*SPI_FIFO_reg = txbuf[i];

		// wait until done
		while(!(*SPI_CS_reg & SPI_DONE_bm));

		while (!(*SPI_CS_reg & SPI_RXD_FULL_bm));
		// read byte
		rxbuf[i] = *SPI_FIFO_reg;
	}

	// bring chip select back up
	*SPI_CS_reg &= ~(SPI_TA_bm);
}


// Image defines
#define WRITE_bm  (1 << 7)
#define WIDTH   320
#define HEIGHT  240

// pixel defines (uses RGB565)
#define BIT_5   0x1F
#define BIT_3   0x07

// SPI parameters
#define POL     0
#define PHA     0
#define CLKDIV  40

//Image handling functions
void get_pixels(unsigned char *rgb);
void print_image(void);
void store_image(void);
void draw_image(void);

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
	spi_init(POL, PHA, CLKDIV);
	printf("spi initialized\n");

	// i2c_init();
	// printf("i2c initialized\n");

	if (arducam_check_interface()) {
		printf("connected to camera!\n");
	} else {
		printf("SPI interface error\n");
		return;
	}

	// omni_init(BMP_MODE);
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

//gets the next pixel value (2 bytes) from the camera and returns a 3 element array
//of RGB values
void get_pixels(unsigned char *rgb)
{
  unsigned char b1 = arducam_read_fifo_single();
  unsigned char b2 = arducam_read_fifo_single();

  unsigned char r = ((b1 >> 3) & BIT_5) * 8; // 5
  unsigned char b = (b2 & BIT_5) * 8; // 5
  unsigned char g = (((b2 >> 5) & BIT_3) | ((b1 & BIT_3) << 3)) * 4; //6
  rgb[0] = r;
  rgb[1] = g;
  rgb[2] = b;
}

//Displays the current camera image to the screen. Does not store the image
void print_image(void)
{
  unsigned char rgb[3];
  for (int i=cam.y; i < cam.h; i++) {
  	if(cam.x == 0) {
	    for (int j=cam.w-1; j >= 0; j--) {
	      get_pixels(rgb);
	      gl_draw_pixel(j, i, gl_color(rgb[0],rgb[1],rgb[2]));
	    }
	} else {
		for (int j=cam.w-1; j >= cam.x; j--) {
	      get_pixels(rgb);
	      gl_draw_pixel(j, i, gl_color(rgb[0],rgb[1],rgb[2]));
	    }
	}
  }
  arducam_clear_fifo();
}

// stores the current image
void store_image(void) {
	color_t (*im)[cam.width] = (unsigned (*)[cam.width])cam.start;
	unsigned char rgb[3];
	for(int i = cam.y; i < cam.h; i++) {
		for (int j=cam.w-1; j >= cam.x; j--) {
			get_pixels(rgb);
			im[j][i] = gl_color(rgb[0],rgb[1],rgb[2]);
		}
	}
	arducam_clear_fifo();
}

// draws the current image to the display
void draw_image(void) {
	color_t (*im)[cam.width] = (unsigned (*)[cam.width])cam.start;
	for(int i = cam.y; i < cam.h; i++) {
		for(int j = cam.x; j < cam.w; j++) {
			gl_draw_pixel(j,i,im[j][i]);
		}
	}
}

//calls the commands required to stream the images
void stream_image(void) {
	arducam_begin_capture();
	printf("streaming...\n");

	while(!arducam_capture_done());
	printf("capture done!\n");

	print_image();
	printf("printing\n");
	gl_swap_buffer();
	printf("done reading!\n");
}

//calls the commands to capture and display an image
void capture_image(void) {
	arducam_begin_capture();
	printf("beginning capture...\n");

	while(!arducam_capture_done());
	printf("capture done!\n");

	store_image();
	printf("image stored, printing...\n");
	draw_image();
	printf("image drawn\n");
	gl_swap_buffer();
	draw_image();
	printf("done reading!\n");
}


/* ARDUCAM INTERFACE FUNCTIONS */
// writes values to a register via SPI
void arducam_write(unsigned char addr, unsigned char value)
{
	unsigned char txd[2] = {addr | WRITE_bm, value};
	unsigned char rxd[2] = {0, 0};
	spi_txrx(txd, rxd, 2);
}

//reads values from a register via SPI
unsigned char arducam_read(unsigned char addr)
{
	unsigned char txd[2] = {addr, 0};
	unsigned char rxd[2] = {0, 0};
	spi_txrx(txd, rxd, 2);
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
	arducam_write(ARD_FIFO_CONTROL, FIFO_START);
}

unsigned char arducam_chip_version()
{
	return arducam_read(ARD_CHIP_VERSION);
}

int arducam_capture_done()
{
	return (arducam_read(ARD_CAMERA_STATUS) & FIFO_WRITE_DONE);
}

unsigned char arducam_read_fifo_single()
{
	return arducam_read(ARD_FIFO_SINGLE_READ);
}
