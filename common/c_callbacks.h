#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
struct ParameterInfo;
// FIXME: descript in common/platform.cpp
typedef void (*data_preservation_func)(struct ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n, uint16_t timer_delay);
void record_overflow_handling_overhead(uint32_t cycles);
#ifdef __cplusplus
}
#endif
