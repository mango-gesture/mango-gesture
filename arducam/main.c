/* 
 * This code brought to you by the rag tag duo of Arjun and Eric! Visit us at
 * our github accounts: https://github.com/arjunvb, https://github.com/efritz09
 * 
 * Completed on: March 14, 2016
 * Arjun Balasingam, Eric Fritz
 * CS 107e, Final Project
 * Shell to demonstrate camera driver features
 */

#include "timer.h"
#include "gpio.h"
#include "spi.h"
#include "i2c.h"
#include "arducam.h"
#include "omni.h"
#include "uart.h"
#include "gl.h"
#include "printf.h"

// size of the image coming in
#define WIDTH   160
#define HEIGHT  120

//center the image on the screen? 
#define CENTERED	1

#define IMAGE_CAPTURE 0
#define IMAGE_STREAM  1

#define INFRA_SENSOR GPIO_PB2

/* init_peripherals
 * contains all the initialization for the program
 * adjusts the image window and sets the shell parameters
 */
void init_peripherals(void) {
	gpio_set_input(INFRA_SENSOR);
    timer_init();
	timer_delay_ms(100);

	uart_init();

	//set multiplier to 1 for full screen, 2 for half, etc
	unsigned graphics_width = WIDTH*3/2;
	unsigned graphics_height = HEIGHT*3/2;
	unsigned image_start = graphics_width/2 - WIDTH/2;
	
	//initailize graphics
	gl_init(graphics_width, graphics_height, GL_DOUBLEBUFFER);

	//start up the camera, set the start of the image's top left corner in the 
	//graphics display. This starts SPI and I2C inside
	if(CENTERED) arducam_init(WIDTH,HEIGHT,image_start,0);
	else arducam_init(WIDTH,HEIGHT,0,0);
}

/* 
 * continuously capture images
 */
void main(void)
{	
	init_peripherals();

  	timer_delay_ms(100);
	arducam_init_bg();
	arducam_calibrate();

//   printf("Now storing image\n");
  	// store_jpeg();
	// while(1) {
	// 	// while (!gpio_read(INFRA_SENSOR)) {/*spin*/}
    //     store_jpeg();
	// 	store_jpeg();

    //     timer_delay_ms(200);
    // }

	// for (int i = 0 ; i < 5 ; i++){
	// 	store_jpeg();
	// 	store_jpeg();

    //     timer_delay_ms(200);
	// }
	while (!image_field_has_changed()){ /*spin*/}

	store_jpeg();
	printf("\n");
	store_jpeg();

	printf("\nImage pair stored\n");
	// int diff = 0;
	// int len_diff = 0;
	// int avg_len_diff = 0;
	// int max_len_dif = 0;
	// int max_dif = 0;
	// for (int i = 0 ; i < 1000 ; i++){
	// 	diff += find_field_diff(&len_diff);
	// 	avg_len_diff += len_diff;
	// 	if (len_diff > max_len_dif) max_len_dif = len_diff;
	// 	if (diff > max_dif) max_dif = diff;
	// 	if (i % 100 == 0)
	// 		printf("Finished iter %d\n", i);
	// }
	// printf("Average diff: %d\n", diff/1000);
	// printf("Average len diff: %d\n", len_diff/1000);
	// printf("Max diff: %d\n", max_dif);
	// printf("Max len diff: %d\n", max_len_dif);
}
