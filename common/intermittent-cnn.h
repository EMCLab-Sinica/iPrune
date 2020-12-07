#pragma once

#include <stdint.h>
#include "data.h"
#include "my_debug.h"

extern uint16_t sample_idx;

struct ParameterInfo;
struct Model;
uint8_t run_cnn_tests(uint16_t n_samples);

#if INDIRECT_RECOVERY
uint32_t job_index_to_offset(const ParameterInfo* output, uint32_t job_index);
#endif

uint8_t get_state_bit(struct Model *model, uint8_t slot_id);
#if STATEFUL
static inline uint8_t get_value_state_bit(int16_t val) {
    MY_ASSERT(-0x2000 <= val && val < 0x6000,
        "Unexpected embedded state in value %d" NEWLINE, val);
    return val >= 0x2000;
}
#endif
#if JAPARI
static inline void check_footprint(int16_t val) {
    // -255 and 255 happens when only the first byte of a footprint is written
    MY_ASSERT(val == 0 || val == 1 || val == -1 || val == -255 || val == 255);
}

static inline uint8_t get_value_state_bit(int16_t val) {
    check_footprint(val);
    // 255 (0xff, 0x00 on little-endian systems) happens when the first byte of -1 (0xff, 0xff) is
    // written over 1 (0x01, 0x00), and it should be considered as -1 not completely written. In
    // other words, the state is still 1.
    return (val > 0);
}
#endif
uint8_t param_state_bit(Model *model, const ParameterInfo *param, uint16_t offset);

uint32_t run_recovery(struct Model *model, struct ParameterInfo *output);
void flip_state_bit(struct Model *model, const ParameterInfo *output);
