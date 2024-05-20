
#include "uart.h"
#include "printf.h"
#include "inference.h"
#include "malloc.h"
#include "strings.h"

void main(void)
{
  uart_init();
  MLP_Model* model = load_mlp_model();


  float* inputs = malloc(2 * 30 * 40 * sizeof(float));
  inputs = memset(inputs, 0, 2 * 30 * 40 * sizeof(float));

  printf("Starting inference\n");
  int forward_results = forward(model, inputs);

  free_mlp_model(model);
}
