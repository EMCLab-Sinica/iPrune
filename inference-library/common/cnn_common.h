#pragma once

#include <cstddef> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <cstdint>
#include "data.h"

#define ENABLE_COUNTERS 0
#define DEMO 1
#define ENABLE_PER_LAYER_COUNTERS 0
// Some demo codes assume counters are accumulated across layers
static_assert((!ENABLE_PER_LAYER_COUNTERS) || (!DEMO), "ENABLE_PER_LAYER_COUNTERS and DEMO are mutually exclusive");

/**********************************
 *        Data structures         *
 **********************************/

struct ConvNodeFlags {
    uint16_t input_tile_c;
    uint16_t output_tile_c;
    uint16_t output_tile_w;
    uint16_t output_tile_h;
    uint8_t pads[4];
};

struct MaxPoolFlags {
    uint8_t kernel_shape[2];
    uint8_t strides[2];
};

struct GemmNodeFlags {
    uint16_t tile_channel;
};

struct GemmMergeNodeFlags {
    uint16_t tile_length;
};

struct SqueezeNodeFlags {
    uint8_t axes;
};

union ExtraNodeFlags {
    ConvNodeFlags conv;
    MaxPoolFlags maxpool;
    GemmNodeFlags gemm;
    GemmMergeNodeFlags gemmmerge;
    SqueezeNodeFlags squeeze;
};

struct NodeFlags {
    uint8_t generic;
    uint8_t kernel_size;    // used in MaxPool
    uint8_t stride[2];         // used in Conv and MaxPool
    ExtraNodeFlags extra;
};

static_assert(sizeof(NodeFlags) == 16, "Unexpected size for NodeFlags");

typedef struct Node {
    char name[NODE_NAME_LEN];
    char output_name[NODE_NAME_LEN];
    uint16_t inputs_len;
    int16_t inputs[NUM_INPUTS]; // ifm, weight, bias
    uint16_t max_output_id;
    uint16_t op_type;
    NodeFlags flags;
#if HAWAII
    struct Footprint {
        uint16_t dummy;
        uint16_t sub_layer_idx;
        uint16_t value;
        uint8_t version;
    } footprint[2];
#endif
} Node;

static_assert(sizeof(Node) == NODE_NAME_LEN * 2 + 22 + NUM_INPUTS * 2 + HAWAII * 16, "Unexpected size for Node");

struct Scale {
    int16_t fract;
    uint8_t shift;
    uint8_t dummy;

    bool operator>(const Scale& other) const;
    Scale operator*(const Scale& other) const;
    Scale operator/(const Scale& other) const;
    bool operator!=(const Scale& other) const;
    void fromFloat(float scale);
    float toFloat() const;
};

/* ParameterInfo may indicate data from the model (parameters) or intermediate values */
typedef struct ParameterInfo {
    uint32_t params_offset;
    uint32_t params_len;  /* in bytes */
#if SPARSE
    /* Used to store sparse matrix */
    uint32_t params_cols_offset;
    uint32_t params_rows_offset;
    uint32_t first_tile_index_offset; // for bias
#endif
    /* Known bitwidth values:
     * 16: q15
     * 32: iq31
     * 64: INT64 (from ONNX)
     */
    uint8_t bitwidth;
    /* A flag to indicate where the data are. Possible values are SLOT_TEST_SET,
     * SLOT_PARAMETERS and SLOT_INTERMEDIATE_VALUES.
     */
    uint8_t slot;
    // uint8_t is not enough. For example, fully connected layer in MNIST has dims 257x1
    uint16_t dims[N_MAX_DIMS];
    Scale scale;
    uint8_t param_flags;
    uint8_t extra_info[EXTRA_INFO_LEN];
    uint16_t parameter_info_idx; // must be the last member of this struct
} ParameterInfo;

#if SPARSE
    static_assert(sizeof(ParameterInfo) == 32+2*N_MAX_DIMS, "Unexpected size for ParameterInfo");
#else
    static_assert(sizeof(ParameterInfo) == 20+2*N_MAX_DIMS, "Unexpected size for ParameterInfo");
#endif

typedef struct SlotInfo {
#if INDIRECT_RECOVERY
    int8_t state_bit;
    uint8_t n_turning_points;
    uint16_t turning_points[TURNING_POINTS_LEN];
#endif
    int16_t user;
} SlotInfo;

typedef struct Model {
    uint16_t running;
    uint16_t run_counter;
    uint16_t layer_idx;
    // uint16_t sub_layer_idx; // move to footprint
    SlotInfo slots_info[NUM_SLOTS];
    uint8_t dummy;
    uint8_t version; // must be the last field in this struct
} Model;

static_assert(sizeof(Model) == 8 + NUM_SLOTS * (2 + INDIRECT_RECOVERY * (2 + TURNING_POINTS_LEN * 2)), "Unexpected size for Model");

/**********************************
 *          Global data           *
 **********************************/
#if ENABLE_COUNTERS
#define COUNTERS_LEN (MODEL_NODES_LEN+1)
struct Counters {
    uint32_t power_counters;
    uint32_t dma_invocations_r;
    uint32_t dma_invocations_w;
    uint32_t dma_read_filter;
    uint32_t dma_read_input;
    uint32_t dma_write_ofm;
    uint32_t dma_write_fp;
    uint32_t indexing;
    uint32_t dma_bytes_r;
    uint32_t dma_bytes_w;
    uint32_t job_preservation;
    uint32_t footprint_preservation;
    uint32_t macs;

    uint32_t progress_seeking;
};

extern uint32_t total_jobs;
extern Counters *counters_data;
Counters *counters();
void reset_counters();
void report_progress();
#endif

#if ENABLE_COUNTERS && !DEMO
void start_cpu_counter(void);
// pointer to member https://stackoverflow.com/questions/670734/pointer-to-class-data-member
void stop_cpu_counter(uint32_t Counters::* mem_ptr);
#else
#define start_cpu_counter()
#define stop_cpu_counter(mem_ptr)
#endif

extern ParameterInfo intermediate_parameters_info_vm[MODEL_NODES_LEN];


/**********************************
 *          Miscellaneous         *
 **********************************/

/* MSP430 SDK already defines MIN, which means minutes */
#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))
#define MAX_VAL(x, y) ((x) > (y) ? (x) : (y))
// XXX: MSP432 driverlib requires DMA transfer size to be <= 1024. However,
// transfer size < 1024 may be broken as well - copying 1024 items works,
// copying 512 items works, copy a small number of items (e.g., 6, 10, ...)
// works, and copying 626 items (in ConvMerge of conv2 in MNIST) DOES NOT
// WORK (!?).
#define LIMIT_DMA_SIZE(x) MIN_VAL(512, x)

/**********************************
 * Helpers for the model & nodes  *
 **********************************/
const uint8_t* get_param_base_pointer(const ParameterInfo *param, uint32_t *limit_p);
const uint8_t* get_param_row_base_pointer(const ParameterInfo *param, uint32_t *limit_p);
const uint8_t* get_param_col_base_pointer(const ParameterInfo *param, uint32_t *limit_p);
const uint8_t* get_param_first_tile_index_base_pointer(const ParameterInfo *param, uint32_t *limit_p);
int16_t get_q15_param(Model* model, const ParameterInfo *param, uint16_t offset_in_word);
void put_q15_param(ParameterInfo *param, uint16_t offset_in_word, int16_t val);
int64_t get_int64_param(const ParameterInfo *param, size_t i);
uint16_t get_next_slot(Model *model, const ParameterInfo *param);
const ParameterInfo* get_parameter_info(uint16_t i);
const Node* get_node(size_t i);
const Node* get_node(const ParameterInfo* param);
SlotInfo * get_slot_info(Model* model, uint8_t i);
void my_memcpy_from_param(Model* model, void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n);
void my_memcpy_from_param_row(Model* model, void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n);
void my_memcpy_from_param_col(Model* model, void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n);
void my_memcpy_from_param_first_tile_index(Model* model, void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n);
uint16_t get_col_first_tile_index(Model *model, const ParameterInfo *params, uint16_t filter_tile_index);
uint16_t get_row_val(Model *model, const ParameterInfo *params, uint16_t row_index);
uint16_t get_col_val(Model *model, const ParameterInfo *params, uint16_t col_index);

/**********************************
 *       Operation handlers       *
 **********************************/
typedef void (*handler)(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node);
typedef void (*allocator)(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node);
// below are defined in ops.c
extern const handler handlers[];
extern const allocator allocators[];
