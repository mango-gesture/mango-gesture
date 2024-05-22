
#include "uart.h"
#include "printf.h"
#include "inference.h"
#include "malloc.h"
#include "strings.h"

extern int get_fs(void);

extern void set_fs_one(void);

void main(void)
{
  uart_init();
  MLP_Model* model = load_mlp_model();

  int fs = get_fs();
  printf("\nfs: %d\n", fs);
  set_fs_one();
  fs = get_fs();
  printf("\nfs: %d\n", fs);


  float* inputs = malloc(2 * 30 * 40 * sizeof(float));
  inputs = memset(inputs, 1, 2 * 30 * 40 * sizeof(float));

  int forward_results = forward(model, inputs);

  free_mlp_model(model);
}
