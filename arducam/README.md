# Interfacing an ArduCAM with the Mango Pi

## Team Members
Nika Zahedi, Aaditya Prasad


### Overview
The arducam subproject aims to interface an ArduCAM with the Mango Pi.The selected camera was an [ArduCAM Mini](https://www.arducam.com/product/arducam-2mp-spi-camera-b0067-arduino/), a 2 MP camera designed to work with the Arduino. The ArduCAM utilizes an Omnivision CMOS Image Sensor [OV2640](https://www.uctronics.com/download/cam_module/OV2640DS.pdf). As a starting point, we adaped code for an ArduCAM-Raspberry Pi interface written by [Eric Fritz](https://github.com/efritz09) and [Arjun Balasingam](https://github.com/arjunvb).

### Communication Drivers
The ArduCAM reqires both SPI and I2C to control the image sensor and recieve images from the camera. SPI was used to send image capture commands as well as transmit the raw pixel or JPEG data. I2C was used to change the Omnivision sensor's registers directly, allowing the user to adjust the camera settings.
The SPI module was adapted from [Yifan Yang's](yyang29@stanford.edu) SPI module. We added code to support reading data in burst mode and data of variable sizes (such as JPEG images).
The I2C module was adapted from [Julie Zelenski's](https://github.com/zelenski) I2C module with minor changes to support compatibility with Omni.

### Wiring the ArduCAM to the Pi
ArduCAM:  Pi
* CS     -> CS0
* MOSI   -> MOSI
* MISO   -> MISO
* SCK    -> SCLK
* GND    -> GND
* VCC    -> 3V
* SDA    -> PG13
* SCL    -> PG12
  
### Initializing the ArduCAM
First, both the SPI and I2C modules have to be initialized. Then, the Omnivision sensor needs to be properly initialized with either the JPEG or BMP modes. 

### Features
* Saving images to local device via uart: Can be used in JPEG mode.
* Streaming images: Can be used in BMP mode.
* Detecting changes in field of view: Can be used in JPEG mode. The arducam needs to first be calibrated to the default background using the `arducam_calibrate` method.
