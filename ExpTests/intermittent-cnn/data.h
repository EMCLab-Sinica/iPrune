
#pragma once
#include <stdint.h>

// const is for putting data on NVM
#ifdef __MSP430__
#  define GLOBAL_CONST const
#else
#  define GLOBAL_CONST
#endif

#define SCALE 50
#define NUM_SLOTS 2
#define INTERMEDIATE_VALUES_SIZE 65536

extern GLOBAL_CONST uint8_t *inputs_data;
#define INPUTS_DATA_LEN 36

extern GLOBAL_CONST uint8_t *parameters_data;
#define PARAMETERS_DATA_LEN 90820

extern GLOBAL_CONST uint8_t *model_data;
#define MODEL_DATA_LEN 454

extern GLOBAL_CONST uint8_t *labels_data;
#define LABELS_DATA_LEN 50
