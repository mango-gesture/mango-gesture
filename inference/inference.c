#include "inference.h"
#include "malloc.h"
#include "printf.h"

MLP_Model* load_mlp_model(void){
    MLP_Model* model = malloc(sizeof(MLP_Model));
    if(model == NULL)   printf("Error: Unable to allocate memory to model in load_mlp_model.");
    

    // Load the number of layers
    model->num_layers = *((int*)MODEL_ADDR);
    model->layers = malloc(model->num_layers*sizeof(MLP_Layer));
    if(model->layers == NULL)   printf("Error: Unable to allocate memory to model->layers in load_mlp_model.");
    

    // Load the layer details
    uintptr_t weights_addr = MODEL_ADDR + 4 + model->num_layers*8;

    for(int i=0; i<model->num_layers; i++){
        model->layers[i].input_neurons = *((int*)(MODEL_ADDR + 4 + i*8)); // 4 bytes for num_layers, 8 bytes for each layer since in/out neurons
        model->layers[i].output_neurons = *((int*)(MODEL_ADDR + 8 + i*8)); // 4 bytes after ^
        model->layers[i].weights = (float*)weights_addr;
        weights_addr += model->layers[i].input_neurons * model->layers[i].output_neurons * sizeof(float); 
        model->layers[i].biases = (float*)weights_addr;
        weights_addr += model->layers[i].output_neurons * sizeof(float);
    }
    
    print("Loaded a model with %d bytes", weights_addr - MODEL_ADDR);

    return model;
}

void free_mlp_model(MLP_Model* model){
    free(model->layers);
    free(model);
}

static float tanh(float x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

static float gelu(float x){
    return 0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)));
}

static float softplus(float x){
    return log(1 + exp(x));
}

static int round(float x){
    return (int)(x + 0.5);
}

static float* forward_layer(const MLP_Layer* layer, const float* input){
    float* output = malloc(layer->output_neurons * sizeof(float));
    if(output == NULL)  printf("Error: Unable to allocate memory to output in forward_layer.");

    for(int i=0; i<layer->output_neurons; i++){
        output[i] = layer->biases[i];
        for(int j=0; j<layer->input_neurons; j++){
            output[i] += input[j] * layer->weights[j*layer->output_neurons + i]; // weights are stored in row major order: 'C'
        }
        output[i] = layer->output_neurons == 1 ? softplus(output[i]) : gelu(output[i]);
    }
    return output;
}

// Runs inference on the model with the given input
// Input must be a flattened pair of images (2 x c x h x w)
// Output is 0 if no gesture, 1 if left swipe, 2 if right swipe
int forward(MLP_Model* model, const float* input){

    float* output;
    for(int i=0; i<model->num_layers; i++){
        output = forward_layer(&model->layers[i], input);
        free(input);
        input = output;
    }

    int out = round(input[0]);
    free(input);
    
    return out;
}