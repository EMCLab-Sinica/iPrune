#pragma once

#define LEA_BUFFER_SIZE 1884 // (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)
extern int16_t lea_buffer[LEA_BUFFER_SIZE];