
#include "timer.h"
#include "gpio.h"
#include "spi.h"
#include "i2c.h"
#include "arducam.h"
#include "omni.h"
#include "uart.h"
#include "printf.h"
#include "inference.h"
#include "malloc.h"
#include "strings.h"
#include "gl.h"

// size of the image coming in
#define WIDTH   160
#define HEIGHT  120

#define JPEG_MAX_LEN 153600
#define IMG_LEN_BYTES 2201

//center the image on the screen? 
#define CENTERED	1
MLP_Model* model;

extern int get_fs(void);

extern void set_fs_one(void);

void init_peripherals(void) {
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

    timer_delay_ms(100);
	arducam_init_bg();
    arducam_calibrate();
}

void normalize_image(float* inputs, unsigned char* img, int len) {
    // Normalize image and copy into inputs
    for (int i = 0; i < len; i++) {
        inputs[i] = (float)(img[i] * 2) / 255.0 - 1.0;
    }
}

void append_null_token(float* inputs, int len) {
    
    for (int i = len; i < IMG_LEN_BYTES; i++) {
        inputs[i] = 2.0;
    }
}

int get_next_action(float* inputs) {
    // Wait for hand to be in front of cam

    unsigned char* img1 = (unsigned char*)malloc(JPEG_MAX_LEN + 1);
    unsigned char* img2 = (unsigned char*)malloc(JPEG_MAX_LEN + 1);
    while (!image_field_has_changed()){ /*spin*/}

    int len1 = read_jpeg(img1);
    char string_rep [len1 * 2 + 1]; 
	for (int i = 0 ; i < len1 ; i++){
		string_rep[2 * i] = img1[i] % 26 + 'a';
		string_rep[2 * i + 1] = img1[i] / 26 + 'a';
	}
	string_rep[len1 * 2 + 1] = 0;
    
	printf("Size %d: ", len1); // Add file size separator
	printf("%s\n", string_rep); // Print to minicom to save to file
    printf("\n");

    int len2 = read_jpeg(img2);

    char string_rep2 [len2 * 2 + 1]; 
	for (int i = 0 ; i < len2 ; i++){
		string_rep2[2 * i] = img2[i] % 26 + 'a';
		string_rep2[2 * i + 1] = img2[i] / 26 + 'a';
	}
	string_rep2[len2 * 2 + 1] = 0;

    printf("Size %d: ", len2); // Add file size separator
	printf("%s\n", string_rep2); // Print to minicom to save to file

    if (len1 > IMG_LEN_BYTES || len2 > IMG_LEN_BYTES) {
        printf("Image too large\n");
        return -1;
    }

    // Normalize images
    normalize_image(inputs, img1 + 1, len1 - 1);
    normalize_image(inputs + IMG_LEN_BYTES, img2 + 1, len2 - 1);

    append_null_token(inputs, len1 - 1);
    append_null_token(inputs + IMG_LEN_BYTES, len2 - 1);
     
    // Wait for hand to be removed from view
    while (image_field_has_changed()){ /*spin*/}

    free(img1);
    free(img2);

    printf("Starting inference\n");
    int forward_results = forward(model, inputs);
    printf("Choice: %d\n", forward_results);
    
    return forward_results;
}

void main(void)
{
    init_peripherals();
    model = load_mlp_model();

    // int fs = get_fs();
    // printf("\nfs: %d\n", fs);
    set_fs_one();
    // fs = get_fs();
    // printf("\nfs: %d\n", fs);

    float* inputs = (float *)malloc(2 * IMG_LEN_BYTES * sizeof(float));
    // inputs = memset(inputs, 2.0, 2 * IMG_LEN_BYTES * sizeof(float));

    while (1){
        get_next_action(inputs);
    }

    free_mlp_model(model);
}
