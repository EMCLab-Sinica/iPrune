#pragma once

#include <stddef.h> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <stdint.h>
#include "data.h"
#include "platform.h"

#define FLAG_SLOTS 0b11
#define FLAG_TEST_SET 0b10
#define NUM_FILTERS 16

/**********************************
 *        Data structures         *
 **********************************/
typedef struct {
    uint16_t inputs_len;
    uint16_t inputs_offset;
    uint16_t op_type;
    uint16_t flags;
    uint16_t scheduled;  /* 16 bits for aligned memory */
} Node;

// _Static_assert in C11 requires the message
// We target C99 - it works, anyway
_Static_assert(sizeof(Node) == 10, "Unexpected size for Node");

/* ParameterInfo may indicate data from the model (parameters) or intermediate values */
typedef struct _ParameterInfo {
    uint32_t params_offset;
    uint32_t params_len;  /* in bytes */
    /* Known bitwidth values:
     * 16: q15
     * 32: iq31
     * 64: INT64 (from ONNX)
     */
    uint8_t bitwidth;
    /* A flag to indicate where the data are. 0b11 indicates data are in
     * parameters; 0b10 indicates data are from the test set; otherwise
     * it's the slot number for one of intermediate_values.
     */
    uint8_t slot;
    uint8_t flags;
    uint8_t dummy;
    // uint8_t is not enough. For example, fully connected layer in MNIST has dims 256x1
    uint16_t dims[4];
} ParameterInfo;

_Static_assert(sizeof(ParameterInfo) == 20, "Unexpected size for ParameterInfo");

typedef struct {
    uint16_t nodes_len;
    uint16_t n_input;
    uint16_t running;
    uint16_t run_counter;
    uint16_t state_bit;
    uint16_t sample_idx;
} Model;

_Static_assert(sizeof(Model) == 12, "Unexpected size for Model");

/**********************************
 *          Global data           *
 **********************************/
extern Model *model;
extern Node *nodes;
extern ParameterInfo *parameter_info;
extern uint16_t *inputs;
extern uint16_t *parameters;
extern uint16_t *samples;
extern uint8_t *labels;
// similar to double buffering
extern uint8_t *intermediate_values;
extern uint16_t *counters;
extern uint16_t *power_counters;
extern uint8_t *counter_idx;
#define COUNTERS_LEN 64


/**********************************
 *          Miscellaneous         *
 **********************************/

/* MSP430 SDK already defines MIN, which means minutes */
#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))
#define MAX_VAL(x, y) ((x) > (y) ? (x) : (y))

/* Better to not use macros
 * https://stackoverflow.com/a/3437484/3786245
 */
static inline int16_t int16_min(int16_t a, int16_t b) {
    return a < b ? a : b;
}

static inline int16_t int16_max(int16_t a, int16_t b) {
    return a > b ? a : b;
}

#define UNUSED(x) (void)(x)

/**********************************
 *       Helpers for nodes        *
 **********************************/
static inline uint16_t get_next_slot(ParameterInfo *param) {
    uint16_t slot_id = param->slot;
    /* Some cases:
     * 1. slot_id == FLAG_SLOTS -> pick the first slot as the current param is input
     * 2. Otherwise, pick the next slot
     */
    uint16_t next_slot_id = (slot_id + 1) & FLAG_SLOTS;
    if (next_slot_id >= NUM_SLOTS) {
        next_slot_id = 0;
    }
    return next_slot_id;
}

static inline uint8_t* get_param_base_pointer(ParameterInfo *param) {
    uint16_t slot_id = param->slot;
    switch (slot_id) {
        case FLAG_SLOTS:
            return (uint8_t*)parameters;
        case FLAG_TEST_SET:
            return (uint8_t*)samples;
        default:
            return intermediate_values + slot_id * INTERMEDIATE_VALUES_SIZE;
    }
}

static inline int16_t* get_q15_param(ParameterInfo *param, size_t i) {
    if (param->bitwidth != 16) {
        // incorrect param passed
        ERROR_OCCURRED();
    }
    return (int16_t*)(get_param_base_pointer(param) + param->params_offset) + i;
}


int32_t* get_iq31_param(ParameterInfo *param, size_t i);
int64_t get_int64_param(ParameterInfo *param, size_t i);
int16_t node_input(Node *node, size_t i);
void node_input_mark(Node *node, size_t i);
void node_input_unmark_all(Node *node);
uint8_t node_input_marked(Node *node, size_t i);
static inline int16_t iq31_to_q15(int32_t val) {
    return (int16_t)(val >> 16);
}

/**********************************
 *       Operation handlers       *
 **********************************/
typedef void (*handler)(ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
extern uint8_t expected_inputs_len[];
extern handler handlers[];