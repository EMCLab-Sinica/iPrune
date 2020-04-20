#pragma once

#include <stdint.h>

#if defined(CY_TARGET_DEVICE) || defined(CY_PSOC_CREATOR_USED)
#define CYPRESS
#endif

#ifdef CY_PSOC_CREATOR_USED
#define WITH_FAILURE_RESILIENT_OS
#endif

#ifdef __MSP430__
#  include "plat-msp430.h"
#elif defined(CYPRESS)
#  include "plat-psoc6.h"
#else
#  include "plat-linux.h"
#endif

void setOutputValue(uint8_t value);
void registerCheckpointing(uint8_t *addr, size_t len);
