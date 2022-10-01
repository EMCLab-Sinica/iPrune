
#pragma once

#include <stdint.h>

struct ParameterInfo;
struct Model;
struct Node;

#define ARM_PSTATE_LEN 8704
#define BATCH_SIZE 1
#define CONFIG "pruned_cifar10"
#define CPU_BUFFER_SIZE 400
#define DEFAULT_TILE_H 32
#define EXTRA_INFO_LEN 3
#define FIRST_SAMPLE_OUTPUTS {4.8955, 4.331344, 4.631835, 11.602396, 4.454658, 10.819544, 5.423588, 6.451203, 5.806091, 5.272837}
#define HAWAII 1
#define INDIRECT_RECOVERY 0
#define INPUTS_DATA_LEN 0
#define INTERMITTENT 1
#define JAPARI 0
#define LEA_BUFFER_SIZE 1884
#define MAX_N_COL_CONV 14
#define MAX_N_COL_FC 0
#define MAX_N_FILTER_GROUP 0
#define MAX_ROW_LEN_CONV 66
#define MAX_ROW_LEN_FC 0
#define METHOD "HAWAII"
#define MODEL_NODES_LEN 30
#define NODE_NAME_LEN 60
#define NUM_INPUTS 3
#define NVM_SIZE 524288
#define N_INPUT 25
#define N_SAMPLES 20
#define SLOT_CONSTANTS_MIN 240
#define SLOT_INTERMEDIATE_VALUES 1
#define SLOT_PARAMETERS 240
#define SLOT_TEST_SET 255
#define SPARSE 1
#define STABLE_POWER 0
#define STATEFUL 0
#define TEMP_FILTER_WIDTH 1
#define TURNING_POINTS_LEN 8
#define USE_ARM_CMSIS 0
#define PARAM_BIN 0
#define SCALE 2
#define INPUT_SCALE 4
#define NUM_SLOTS 3
#define INTERMEDIATE_VALUES_SIZE 65000l
#define N_ALL_SAMPLES 10000
#define OP_FILTERS 4
#define FP32_ACCURACY 0.7704
#define TOTAL_SAMPLE_SIZE 3072
#define GEMM_TILE_LENGTH 0

#define OpConcat 0
#define OpConv 1
#define OpGlobalAveragePool 2
#define OpMaxPool 3
#define OpRelu 4
#define OpSoftmax 5
#define OpSqueeze 6
void alloc_concat(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void handle_concat(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void alloc_conv(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void handle_conv(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void alloc_globalaveragepool(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void handle_globalaveragepool(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void alloc_maxpool(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void handle_maxpool(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void alloc_relu(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void handle_relu(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void alloc_softmax(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void handle_softmax(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void alloc_squeeze(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
void handle_squeeze(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);
#define NHWC2NCHW 1
#define CHANNEL_FIRST 2
#define SEPARATE_TILING 4
#ifdef __MSP430__ 
#define DATA_SECTION_NVM _Pragma("DATA_SECTION(\".nvm2\")")
#else
#define DATA_SECTION_NVM
#endif

extern const uint8_t * const parameters_data;
#define PARAMETERS_DATA_LEN 50324

extern const uint8_t * const samples_data;
#define SAMPLES_DATA_LEN 6144

extern const uint8_t * const model_data;
#define MODEL_DATA_LEN 14

extern const uint8_t * const nodes_data;
#define NODES_DATA_LEN 4920

extern const uint8_t * const model_parameters_info_data;
#define MODEL_PARAMETERS_INFO_DATA_LEN 1000

extern const uint8_t * const intermediate_parameters_info_data;
#define INTERMEDIATE_PARAMETERS_INFO_DATA_LEN 1200

extern const uint8_t * const labels_data;
#define LABELS_DATA_LEN 20

extern const uint8_t * const rows_data;
#define ROWS_DATA_LEN 602

extern const uint8_t * const cols_data;
#define COLS_DATA_LEN 1368

extern const uint8_t * const first_tile_index_data;
#define FIRST_TILE_INDEX_DATA_LEN 26
