#ifndef MLP_INFERENCE_H
#define MLP_INFERENCE_H

#define MODEL_ADDR 0x60000000
#include <stdint.h>


// Define a structure to hold the MLP layer details
typedef struct {
    int input_neurons;     // Number of neurons in the input layer
    int output_neurons;    // Number of neurons in the output layer
    float* weights;        // Pointer to array of weights
    float* biases;         // Pointer to array of biases
} MLP_Layer;

// Define a structure to hold the entire MLP model
typedef struct {
    int num_layers;        // Number of layers in the model
    MLP_Layer* layers;     // Pointer to array of layers
} MLP_Model;


// Function prototypes for managing MLP model
MLP_Model* load_mlp_model(void); // Loads MLP model from MODEL_ADDR and returns a pointer to the model
void free_mlp_model(MLP_Model* model); // Frees the memory allocated for the MLP model
int forward(MLP_Model* model, float* input); // Perform inference using the MLP model

#endif // MLP_INFERENCE_H
