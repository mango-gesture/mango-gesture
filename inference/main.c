
#include "uart.h"
#include "printf.h"
#include "inference.h"
#include "malloc.h"

void main(void)
{
  uart_init();
  MLP_Model* model = load_mlp_model();


  float* inputs = malloc(2 * 60 * 80 * sizeof(float));

  int forward_results = forward(model, inputs);
  printf("Results: %d\n", forward_results);

  free_mlp_model(model);
}
