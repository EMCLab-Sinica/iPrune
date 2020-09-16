#pragma once

#include "data.h"
#include <stdint.h>

#define LEA_BUFFER_SIZE 1884 // (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)

#define NEED_DATA_VARS
#define PLAT_LABELS_DATA_LEN LABELS_DATA_LEN
#define PLAT_SAMPLES_DATA_LEN SAMPLES_DATA_LEN

#ifdef __cplusplus
extern "C" {
#endif

void IntermittentCNNTest(void);
void button_pushed(uint16_t button1_status, uint16_t button2_status);

#ifdef __cplusplus
}
#endif
