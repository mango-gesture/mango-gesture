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