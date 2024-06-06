# spotify_gesture

This repository contains code for a trainable gesture detection module built for the Mango-Pi. We use detected gestures to control Spotify from a Mac using osascript, but it is incredibly easy to adjust this for other tasks.  

The next few sections detail usage instructions, specifically
1. Setup
2. Collecting data and training 
3. Controlling your computer
4. Putting everything together

We then cover additional design decisions related to the
- Camera system
- Vision model and training
- Inference system
- Miscellaneous

for help with debugging. 

## Usage

### Setup
Our system identifies the start of a gesture via the change in compression ratio when the scene becomes heterogenous. Thus, it is important to pick a background that is static and less "busy" or complex than your hand -- a fully homogenous background is perfect for this. 

We used the surface of a table as our scene and used touchpad gestures on the surface of the table. We also placed a lamp over our setup and suspended our camera so that it faced the table's surface. It is important to note down or otherwise mark your scene area so you can check for distribution shift between train and test times. To run quickly on the Pi, the model has to be kept small and thus not very robust. 

![Setup](media/setup.jpeg){:height="50%" width="50%"}

**Wiring the ArduCAM to the Pi:**
ArduCAM:  Pi
CS     -> CS0
MOSI   -> MOSI
MISO   -> MISO
SCK    -> SCLK
GND    -> GND
VCC    -> 3V
SDA    -> PG13
SCL    -> PG12

**Communication Drivers:**
The ArduCAM reqires both SPI and I2C to control the image sensor and recieve images from the camera. SPI was used to send image capture commands as well as transmit the raw pixel or JPEG data. I2C was used to change the Omnivision sensor's registers directly, allowing the user to adjust the camera settings.
The SPI module was adapted from Yifan Yang's (yyang29@stanford.edu) SPI module. We added code to support reading data in burst mode and data of variable sizes (such as JPG images).
The I2C module was adapted from Julie Zelenski's (https://github.com/zelenski) I2C module with minor changes to support compatibility with Omni.

To collect data written to the Mango Pi (for training or monitoring purposes), we need to send data back to our local device, which we do using uart. 
Our suggested method to do this is to start a minicom window in capture mode before running the program: 

```console
minicom -C capture.txt
```

This saves the output of minicom to the file `capture.txt`. We then convert the data we want to send to text format (e.g. bytes are converted to base-26 text strings) and print to minicom. 
Note that ```minicom -C capture.txt``` appends to the existing content in `capture.txt` and doesn't overwrite it.

You will need Jax, wandb, and some other standard python libraries. We recommend using a conda environment to manage these dependencies. 

After setting up your scene and picking some set of gestures to learn, proceed to the next section.
### Collecting data and training
In `main.c`, use the method `get_training_data` with argument `num_image_pairs` to collect that number of training image pairs. The camera captures an image pair when it detects sufficient change in its field of view compared to the background frame.
When running main, by calling `init_peripherals` with the `calibrate` argument set to one, the program begins by calibrating the camera so that the change detection threshold aligns with the recorded gesture. If this argument is zero, then the calibration step is skipped and some default threshold is used.
The image pairs are written to minicom and then decoded by a python script.

```console
[arducam]$ minicom -C capture.txt
```
In a different terminal window
```console
[spotify_gesture]$ make run
```
When data collection for each class is complete, run `extract_jpeg.py` in the arducam directory, and specify the text file where the images are saved, as well as some output directory. 

```console
[arducam]$ python extract_jpeg.py -c "0" -f "capture.txt"
```
The images will be saved to `arducam/data/[class]` directory in the format `output{i}`, where i is the image number. Only image pairs where both pairs have sufficient difference with a baseline background image (`bg.jpg`) are saved. This feature can be removed by using a completely black background baseline image.
The training data for each gesture should be collected separately and placed in different directories.

After collection, you should have several folders; one for each gesture. Ensure that these folders are numbered 0-N where you have N+1 gestures you wish to learn. These are the class labels for your gestures.  Within the folders, your data should be labeled output0.jpeg ... outputm.jpeg. Each consecutive pair (output0, output1; outputk, output{k+1}) represents a before-after pair that our mlp will learn to predict labels from. 

You also have the max image size that was seen during data collection. While this image size will work for 100% of your training data and likely all of your test data, we recommend you reduce the image size to decrease inference latency and maintain 95-99% of your data. Example.ipynb has many example function calls across our training and inference pipelines. Scroll down to the Image Size Calculator and use the next 4 cells to test out different image sizes. 

Once you have decided on an image size, switch to the inference directory and run the following command to start a training run:

```console
(your_conda_env_here)[inference]$ python train.py --weight_path "your/ckpts/go/here" --data_path "your/data/was/here" -classes N+1 -i chosen_image_size
```

train.py accepts many other cmdline args to customize your training run and enable wandb logging. We recommend utilizing wandb logging (which is off by default) so you can check val accuracy in addition to loss values. 

train.py will save checkpoints as .ckpt files as well as .bin files. The .ckpt files are for additional training and/or inference within Jax; the .bin file is what your Pi will use for inference. 

### Controlling Your Computer
Before you make run (below), start minicom in capture mode (see section "Communication Drivers" above). The application will print predictions to minicom's log file. ```computer_control.py``` watches this log file and calls functions from ```media_ctrl.py``` based on the model's predictions. Start ```computer_control.py``` via

```console
(your_conda_env_here)[inference]$ python computer_control.py path_to_log_file.txt
```

If you wish to use a different number of gestures with seperate functionalities, adapt ```media_ctrl.py``` (or write a new file and point ```computer_control.py``` to it). 

Once ```computer_control.py``` is running, you can start the application. 

### Putting Everything Together
Simply replace the line ```xfel write 0x60000000 demo_weights/256_128_100.bin``` in our Makefile with the path to your desired .bin file. Don't change anything other than the path in that line. 

Ensure minicom is running in log mode and ```computer_control.py``` is watching that log file (see last section).

Then

```console
(your_conda_env_here)[spotify_gesture]$ make_run
```

## Design and Functionality

### Camera
Check out arducam/README.md for details on the ArduCAM.

### Vision Model and Training
Jax is a scientific computing and autodifferentiation framework that simplifies parallelization and just-in-time compiling. We use an mlp for our vision model and train it using Jax. mlp code (```inference/mlp.py```) is thus pretty simple; the user only has to specify the number and sizes of hidden layers as well as the input and output dimensions. 

Training occurrs in ```inference/train.py``` as well as ```inference/dataset.py```. Because we train on JPEGs, which are variable length, null tokens are appended after flattening and normalization to ensure that all inputs to the model have constant length. Training and inferring on JPEG's is enormously beneficial to latency (given fixed accuracy), as the compression format acts like a convolutional layer that would be costly to implement (leaving aside the cost of transmitting a BMP image from the arducam to the Pi in the first place). 

We optimize the cross entropy loss wrt our input classes and use the adamw optimizer. No real tuning was done of the optimizer or hyperparameters, other than layer size. While we have training and dataset code for RGB inputs, these will not work with our inference stack and should not be used as-is. 
### Inference
Our weights are stored in the following format within our .bin files: 
1. The first 4 bytes are the number of non-input layers, n, as an int32
2. The next 2n int32s are layer input/output sizes. 
3. The weights of the first layer follow, then the biases, then the weights of the second layer, etc, all stored as float32s.

As an example, consider an mlp that takes in 2 hundred-byte images, has two hidden layers of size 8 and 4, and outputs 2 class probabilities. 

The first int32 will be 3. 
The next 6 int32s will be 200 8 8 4 4 2. 
The next 1600 float32s will be the first set of weights, the next 8 float32s will be the first set of biases, etc. 

The ```MLP_Model``` and ```MLP_Layer``` structs abstract this system, and ```MLP_Model* load_mlp_model(void)``` initializes the model as per the above format. 

Before computing a forward pass, the same normalization and ```NULL_TOKEN``` padding is performed on the input images. 

After computing a forward pass, the argmax of the class probabilities is taken and returned to the main program. 
