#include "inference.h"
#include "malloc.h"
#include "printf.h"
#include "timer.h"

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
        model->layers[i].input_neurons = *((unsigned int*)(MODEL_ADDR + (unsigned long) (4 + i*8))); // 4 bytes for num_layers, 8 bytes for each layer since in/out neurons
        model->layers[i].output_neurons = *((unsigned int*)(MODEL_ADDR + (unsigned long) (8 + i*8))); // 4 bytes after ^
        model->layers[i].weights = (float*)weights_addr;
        weights_addr += model->layers[i].input_neurons * model->layers[i].output_neurons * sizeof(float); 
        model->layers[i].biases = (float*)weights_addr;
        weights_addr += model->layers[i].output_neurons * sizeof(float);
    }
    
    printf("Loaded a model with %ld bytes", weights_addr - MODEL_ADDR);

    return model;
}

void free_mlp_model(MLP_Model* model){
    free(model->layers);
    free(model);
}

static float exp(float x){
    float next_term = 1;
    float total = 0;
    // TODO: add tolerance
    for (int i = 0 ; i < 20 ; i++){
        total += next_term;
        total *= x / (float)(i + 1);
    }
    return total;
}

static float log(float x){
    // assert (x > 0);
    float y = 0;  // Initial guess
    for (int i = 0; i < 100; i++) { // Limit the iterations to prevent infinite loops
        y = y - 1 + x / exp(y);
    }
    return y;
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

static float relu(float x){
    return (x > 0) ? x : 0;
}

static float* forward_layer(const MLP_Layer* layer, const float* input){
    float* output = malloc(layer->output_neurons * sizeof(float));
    if(output == NULL)  printf("Error: Unable to allocate memory to output in forward_layer.");

    for(int i=0; i<layer->output_neurons; i++){
        output[i] = layer->biases[i];
        for(int j=0; j<layer->input_neurons; j++){

            output[i] += input[j] * layer->weights[j*layer->output_neurons + i]; // weights are stored in row major order: 'C'
        }
        // output[i] = layer->output_neurons == 1 ? softplus(output[i]) : relu(output[i]);
        output[i] = relu(output[i]);
    }
    return output;
}

int argmax(const float* arr, int size){
    int max_index = 0;
    for(int i=1; i<size; i++){
        max_index = (arr[i] > arr[max_index]) ? i : max_index;
    }
    return max_index;
}

union FloatInt {
    float f;
    unsigned int i;
};

// Runs inference on the model with the given input
// Input must be a flattened pair of images (2 x c x h x w)
// Output is 0 if no gesture, 1 if left swipe, 2 if right swipe
int forward(MLP_Model* model, float* input){

    int start_time = timer_get_ticks();
    // int layer_times[model->num_layers];
    float* output;

    for(int i=0; i<model->num_layers; i++){

        output = forward_layer(&model->layers[i], input);

        // layer_times[i] = (int)((timer_get_ticks() - start_time) / (24 * 1000));

        // printf("Freeing input %d\n", i);
        free(input);

        input = output;
    }

    // for(int i = 0; i < model->num_layers; i++){
    //     printf("Layer %d elapsed time %d\n", i, layer_times[i]);
    // }
    // void* void_out = &input[0];
    // printf("\nInt representation of the first output %d\n", *(unsigned int*)void_out);

    // void* void_out2 = &input[1];
    // printf("Int representation of the second output %d\n", *(unsigned int*)void_out2);

    
    int choice = argmax(input, model->layers[model->num_layers-1].output_neurons);
    // printf("Freeing last input\n");
    free(input);
    
    return choice;
}