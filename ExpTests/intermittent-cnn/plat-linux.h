#pragma once

#include <stdio.h>
#include <inttypes.h>
#include <signal.h>
#include <string.h>
#include "debug.h"

#define ERROR_OCCURRED() do { raise(SIGINT); } while (0);

#define MEMCPY_DELAY_US 0

#define LEA_BUFFER_SIZE 16384

#define NVM_BYTE_ADDRESSABLE 1
// USE_ALL_SAMPLES must be 1 as nvm.bin contains all samples
#define USE_ALL_SAMPLES 1

extern uint32_t *copied_size;

static inline void my_memcpy(void* dest, const void* src, size_t n) {
    *copied_size += n;
#if MEMCPY_DELAY_US
    usleep(MEMCPY_DELAY_US);
#endif
    my_printf_debug(__func__);
    my_printf_debug(" copied %d bytes" NEWLINE, (int)n);
    memcpy(dest, src, n);
}