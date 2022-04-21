#include <cstdint>
#include <cinttypes> // for PRId32
#include "cnn_common.h"
#include "data.h"
#include "my_debug.h"
#include "op_utils.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "platform.h"

// TODO: make these adjustable on runtime
#if !USE_ARM_CMSIS
#define OUTPUT_LEN 100
#else
#define OUTPUT_LEN 256
#endif

/* Better to not use macros
 * https://stackoverflow.com/a/3437484/3786245
 */
static inline int16_t int16_min(int16_t a, int16_t b) {
    return a < b ? a : b;
}

static inline int16_t int16_max(int16_t a, int16_t b) {
    return a > b ? a : b;
}

#define CONV_TASK_FLAG_PROCESSED_FILTERS_BASE 2
typedef struct ConvTaskParams {
    Model* model;
    const ParameterInfo *conv_input;
    const ParameterInfo *real_conv_input; // for separate channel tiling
    const ParameterInfo *conv_filter;
    const ParameterInfo *conv_bias;
    ParameterInfo *output;
    const NodeFlags* flags;

    /* aux vars remaining constant for a conv layer */
    uint16_t H;
    uint16_t W;
    // OUTPUT_H and OUTPUT_W to handle stride != 1
    uint16_t OUTPUT_H;
    uint16_t OUTPUT_W;
    uint16_t kH;
    uint16_t kW;
    uint16_t CHANNEL; // Cannot use C as a variable name here as C is a macro on MSP430 :(
    uint16_t OUTPUT_CHANNEL;
    uint16_t N_FILTERS;
    uint16_t stride_h;
    uint16_t stride_w;
    uint16_t input_tile_c_offset;
    uint16_t input_tile_c_index;
    int16_t tile_h;
    int16_t tile_w;
    uint8_t cur_input_tile_c;
    uint16_t cur_filter_tile_c;
    uint16_t n_tiles_c;
    int16_t dest_offset;
    int16_t filter_offset;
    // For 1x1 conv
    int16_t kX;
    int16_t kY;
#if SPARSE
    int16_t row_index; // it can also be used to indicate #channel which are computed now (row_index * input_tile_c)
    int16_t cur_row_val;
    int16_t next_row_val;
    int16_t n_cols; // row[row_index] - row[row_index - 1]
    int16_t cur_n_cols; // [0, n_cols)
#endif
    uint8_t truncated;
#if INDIRECT_RECOVERY
    int16_t old_output_offset ;
    uint8_t turning_point_idx;
    uint16_t next_turning_point;
    SlotInfo* cur_slot_info;
#endif
#if JAPARI
    uint8_t conv_input_has_footprints;
    uint16_t input_tile_c_offset_with_footprints;
    uint8_t force_align_footprints;
#endif
#if STATEFUL
    uint8_t output_padding;
#endif

    uint16_t filter_idx;
    uint16_t filter_tile_index;
    // (h, w) for left-top corner of each input window
    int16_t input_h;
    int16_t input_w;
    int16_t input_h_first, input_h_last;
    int16_t input_w_first, input_w_last;
    int16_t *filter_buffer_addr;
    int16_t cached_filter_idx;
    int16_t cached_input_h;
    int16_t cached_input_w;
    uint16_t cached_input_tile_c_offset;
    int16_t cached_kX;
    int16_t cached_kY;

    uint8_t psum_buffer_version;
    int8_t cur_op;
} ConvTaskParams;

static ConvTaskParams conv_params_obj;

int16_t *matrix_mpy_results = lea_buffer + LEA_BUFFER_SIZE;

#if INDIRECT_RECOVERY
static void flip_filter_state_bits(ConvTaskParams *conv_params, uint16_t n_filters, uint16_t len, uint8_t first_round) {
    start_cpu_counter();
    MY_ASSERT(len < OUTPUT_LEN);
#if STATEFUL
    int16_t *to_flip_state_bits = conv_params->filter_buffer_addr + n_filters * conv_params->filter_offset;
    if (first_round) {
        to_flip_state_bits -= len;
    } else {
        to_flip_state_bits -= n_filters;
    }
    // need negating filter value here as it will be multiplied with _Q15(-1.0), or -32768
    int16_t offset = get_value_state_bit(-*(to_flip_state_bits + BATCH_SIZE - 1))*0x4000;
    my_printf_debug("Flipping %d state bits in filters; first_round=%d, offset=%d" NEWLINE, len, first_round, offset);
    my_offset_q15_batched(to_flip_state_bits, offset, to_flip_state_bits, len);
    my_offset_q15_batched(to_flip_state_bits, offset, to_flip_state_bits, len);
#endif
#if JAPARI
    int16_t *to_flip_state_bits = conv_params->filter_buffer_addr + n_filters * (conv_params->filter_offset - 1);
    if (first_round) {
        for (uint16_t idx = BATCH_SIZE; idx < n_filters; idx += BATCH_SIZE + 1) {
            if (idx < n_filters - len) {
                continue;
            }
            to_flip_state_bits[idx] = -to_flip_state_bits[idx];
        }
    } else {
        for (uint16_t idx = BATCH_SIZE; idx < len; idx += BATCH_SIZE + 1) {
            to_flip_state_bits[idx] = -to_flip_state_bits[idx];
        }
    }
#endif
    stop_cpu_counter(&Counters::embedding);
}
#endif

static void convTask(int16_t cur_input_w, int16_t cur_input_h, ConvTaskParams *conv_params) {
    // cur_output_tile_c should be signed, or MAX_VAL below is broken with TI's compiler
    int16_t output_tile_c = conv_params->flags->extra.conv.output_tile_c;
    int16_t cur_output_tile_c = output_tile_c - conv_params->filter_idx % output_tile_c;
    int16_t output_tile_w = conv_params->flags->extra.conv.output_tile_w;
    int16_t output_tile_h = conv_params->flags->extra.conv.output_tile_h;

    my_printf_debug("cur_output_tile_c = %d" NEWLINE, cur_output_tile_c);
    MY_ASSERT(cur_output_tile_c > 0);

    int16_t n_filters = cur_output_tile_c;
    int16_t values_to_preserve = n_filters;

#if !STABLE_POWER
    int16_t channel_offset_c = conv_params->filter_idx % output_tile_c;
#endif // STABLE_POWER
#if JAPARI
    values_to_preserve = extend_for_footprints(n_filters, conv_params->force_align_footprints);
    n_filters = padding_for_lea(values_to_preserve);
    channel_offset_c = extend_for_footprints(channel_offset_c);
#endif
#if STATEFUL
    if (conv_params->output_padding) {
        values_to_preserve += conv_params->output_padding;
        n_filters = padding_for_lea(values_to_preserve);
    }
#endif
    uint16_t output_h = (cur_input_h - conv_params->input_h_first - conv_params->kX) / conv_params->stride_h,
             output_w = (cur_input_w - conv_params->input_w_first - conv_params->kY) / conv_params->stride_w;
    // use NWHC so that output is written continuously on the address space
#if STABLE_POWER
    uint32_t cur_output_data_offset = ((output_w % output_tile_w) * output_tile_h  + (output_h % output_tile_h)) * output_tile_c;
#else // STABLE_POWER
    // 2 * OUTPUT_W * OUTPUT_H * OUTPUT_CHANNEL is the offset of result of psum cmd
    uint32_t cur_output_data_offset =
        2 * conv_params->OUTPUT_W * conv_params->OUTPUT_H * conv_params->OUTPUT_CHANNEL +           // n
        ((output_h % output_tile_h) * output_tile_w  + (output_w % output_tile_w)) * output_tile_c +// hwc
        channel_offset_c;                                                                           // c
    my_printf_debug("channel_offset_c: %d" NEWLINE, channel_offset_c);
#endif // STABLE_POWER
    my_printf_debug("output_tile_w: %d" NEWLINE, output_tile_w);
    my_printf_debug("output_tile_h: %d" NEWLINE, output_tile_h);
    my_printf_debug("cur_output_data_offset: %d" NEWLINE, cur_output_data_offset);
#if INDIRECT_RECOVERY
    SlotInfo *cur_slot_info = conv_params->cur_slot_info;
    int16_t n_keep_state_bits = n_filters;
    if (conv_params->turning_point_idx <= cur_slot_info->n_turning_points && conv_params->next_turning_point != INVALID_TURNING_POINT) {
        my_printf_debug("next_turning_point = %d" NEWLINE, conv_params->next_turning_point);
        uint16_t ending_offset = MAX_VAL(conv_params->next_turning_point, cur_output_data_offset);
        if (ending_offset < cur_output_data_offset + n_filters) {
            n_keep_state_bits -= cur_output_data_offset + n_filters - ending_offset;
        }
    }
    my_printf_debug("n_keep_state_bits = %d" NEWLINE, n_keep_state_bits);
    MY_ASSERT(n_keep_state_bits >= 0);
#endif

    /* copy filter data */
    if (conv_params->cached_filter_idx != conv_params->filter_idx ||
        conv_params->cached_input_tile_c_offset != conv_params->input_tile_c_offset ||
        conv_params->cached_kX != conv_params->kX ||
        conv_params->cached_kY != conv_params->kY) {

        conv_params->filter_buffer_addr = matrix_mpy_results - conv_params->filter_offset * (n_filters + TEMP_FILTER_WIDTH);
        my_fill_q15(0, conv_params->filter_buffer_addr, conv_params->filter_offset * n_filters);

        int16_t *filter_tmp = matrix_mpy_results - conv_params->filter_offset; // before transpose
        uint16_t fill_length = conv_params->filter_offset;
        my_fill_q15(0, filter_tmp, fill_length);
#if SPARSE
        // buffer_size represents the number of data one DMA transfered
        // Load fliter according to index
        uint16_t filter_offset = conv_params->cur_input_tile_c;
        uint16_t col_index = conv_params->cur_row_val + conv_params->cur_n_cols;
        uint16_t block_size = filter_offset * output_tile_c;
        uint16_t buffer_size = sizeof(int16_t) * filter_offset;
        uint16_t filter_src_offset = col_index * block_size;
        int16_t *filter_dest_ptr = filter_tmp;
#else // SPARSE
        uint16_t buffer_size = sizeof(int16_t) * conv_params->cur_filter_tile_c;
        uint16_t filter_len = conv_params->kH * conv_params->kW * conv_params->CHANNEL;
#endif // SPARSE
        for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
            my_printf_debug("Copying filter %d" NEWLINE, conv_params->filter_idx + idx);
#if SPARSE
            // XXX: Need re-checking
            uint16_t cur_filter_src_offset = filter_src_offset + (conv_params->filter_idx % output_tile_c + idx) * filter_offset;
            my_memcpy_from_param(conv_params->model, filter_dest_ptr, conv_params->conv_filter, cur_filter_src_offset, buffer_size);
#else // SPARSE
            uint16_t filter_src_offset = (conv_params->filter_idx + idx) * filter_len;
            // only load one vector
            int16_t *filter_dest_ptr = filter_tmp;
            uint16_t cur_filter_src_offset = filter_src_offset + conv_params->kX * conv_params->kW * conv_params->CHANNEL + conv_params->input_tile_c_offset;
            cur_filter_src_offset += conv_params->kY * conv_params->CHANNEL;
            my_memcpy_from_param(conv_params->model, filter_dest_ptr, conv_params->conv_filter, cur_filter_src_offset, buffer_size);
#endif // SPARSE
#if STATEFUL
            start_cpu_counter();
            if (conv_params->real_conv_input->slot == SLOT_TEST_SET) {
                my_scale_q15(filter_tmp, 0x4000, 0, filter_tmp, conv_params->filter_offset);
            }
            bool has_state = offset_has_state(cur_output_data_offset + idx);
            if (has_state) {
                my_printf_debug("Adding state bit for newly loaded filter idx=%d" NEWLINE, idx);
                filter_tmp[conv_params->filter_offset - 1] = -(idx < n_keep_state_bits ? -conv_params->old_output_offset : conv_params->old_output_offset);
            }
            stop_cpu_counter(&Counters::embedding);
            if (!has_state)
#endif
            {
                // XXX: why is this needed? Should already be zero with my_fill_q15 above
                filter_tmp[conv_params->filter_offset - 1] = 0;
            }
#if SPARSE
            if(conv_params->cur_n_cols == 0) {
#else // SPARSE
            if (conv_params->input_tile_c_index == 0 && conv_params->kX == 0 && conv_params->kY == 0) {
#endif // SPARSE
                my_printf_debug("Append bias!" NEWLINE);
                // convert int16_t to int32_t first as on MSP430, registers are 20 bit while there are only 16 bits when int16_t is converted to uint16_t
                // If the dividend is negative, the quotient is wrong
                int16_t bias_val = 0;
                if (conv_params->conv_bias) {
                    bias_val = -static_cast<int32_t>(get_q15_param(conv_params->model, conv_params->conv_bias, conv_params->filter_idx + idx)) / conv_params->conv_input->scale;
                }
#if !STATEFUL
                filter_tmp[conv_params->filter_offset - 1] = bias_val;
#else
                if (conv_params->real_conv_input->slot == SLOT_TEST_SET) {
                    filter_tmp[conv_params->filter_offset - 1] += bias_val / 2;
                } else {
                    filter_tmp[conv_params->filter_offset - 1] += bias_val;
                }
#endif
            }

            uint16_t channel = idx;
#if JAPARI
            channel += channel / BATCH_SIZE;
#endif
            my_interleave_q15(filter_tmp, channel, n_filters, conv_params->filter_buffer_addr, conv_params->filter_offset);
        }

#if JAPARI
        int16_t* footprint_channels_ptr = conv_params->filter_buffer_addr + n_filters * (conv_params->filter_offset - 1);
        for (int16_t idx = BATCH_SIZE; idx < n_filters; idx += BATCH_SIZE + 1) {
            if (idx < n_keep_state_bits) {
                *(footprint_channels_ptr + idx) = (conv_params->old_output_offset > 0 ? 1 : -1);
            } else {
                *(footprint_channels_ptr + idx) = (conv_params->old_output_offset > 0 ? -1 : 1);
            }
        }
#endif

#if STATEFUL
        if (conv_params->output_padding) {
            conv_params->filter_buffer_addr[n_filters * conv_params->filter_offset - 1] = -((n_filters - 1 < n_keep_state_bits) ? -conv_params->old_output_offset : conv_params->old_output_offset);
        }
#endif

        conv_params->cached_filter_idx = conv_params->filter_idx;
        conv_params->cached_input_tile_c_offset = conv_params->input_tile_c_offset;
        conv_params->cached_kX = conv_params->kX;
        conv_params->cached_kY = conv_params->kY;
    } else {
#if INDIRECT_RECOVERY
        if (n_keep_state_bits != n_filters) {
            int16_t n_flip_state_bits = n_filters - n_keep_state_bits;
            flip_filter_state_bits(conv_params, n_filters, n_flip_state_bits, 1);
        }
#endif
    }

    int16_t *filter_buffer_addr = conv_params->filter_buffer_addr;

    int16_t *input_buffer_addr =
        lea_buffer +
        ((cur_input_w - conv_params->cached_input_w) * conv_params->tile_h + (cur_input_h - conv_params->cached_input_h)) *
        conv_params->dest_offset;
    my_printf_debug("cur_input_w: %d" NEWLINE, cur_input_w);
    my_printf_debug("cur_input_h: %d" NEWLINE, cur_input_h);
    my_printf_debug("cached_input_w: %d" NEWLINE, conv_params->cached_input_w);
    my_printf_debug("cached_input_h: %d" NEWLINE, conv_params->cached_input_h);
    my_printf_debug("input_buffer_addr offset: %ld" NEWLINE, input_buffer_addr - lea_buffer);
    my_printf_debug("filter_buffer_addr offset: %ld" NEWLINE,filter_buffer_addr - lea_buffer);

    uint16_t A_rows, A_cols, B_rows, B_cols;
    A_rows = 1;
    A_cols = B_rows = conv_params->filter_offset;
    B_cols = n_filters;
    my_printf_debug("B_rows: %d /B_cols: %d" NEWLINE, B_rows, B_cols);
    MY_ASSERT(A_rows * B_cols <= OUTPUT_LEN);
    MY_ASSERT(input_buffer_addr + A_rows * A_cols <= filter_buffer_addr);
#if !STATEFUL
#if STABLE_POWER
        my_matrix_mpy_q15_to_vm(A_rows, A_cols, B_rows, B_cols, input_buffer_addr, filter_buffer_addr, matrix_mpy_results,
                      conv_params->output, cur_output_data_offset, values_to_preserve, 0, 0);
#else // STABLE_POWER
        my_matrix_mpy_q15(A_rows, A_cols, B_rows, B_cols, input_buffer_addr, filter_buffer_addr, matrix_mpy_results,
                      conv_params->output, cur_output_data_offset, values_to_preserve, 0, 0);
#endif
#else
    my_matrix_mpy_q15(A_rows, A_cols, B_rows, B_cols, input_buffer_addr, filter_buffer_addr, matrix_mpy_results,
                      conv_params->output, cur_output_data_offset, values_to_preserve,
                      -conv_params->old_output_offset, n_keep_state_bits);
#endif

    /* START dump data */
    my_printf_debug("input_h=%d" NEWLINE, cur_input_h);
    my_printf_debug("filter_idx=");
#if MY_DEBUG >= MY_DEBUG_VERBOSE
    for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
        my_printf_debug("%d ", conv_params->filter_idx + idx);
        MY_ASSERT(conv_params->filter_idx + idx < conv_params->N_FILTERS);
    }
#endif
    my_printf_debug("output_h=%d output_w=%d" NEWLINE, output_h, output_w);

    my_printf_debug("input" NEWLINE);
    dump_matrix_debug(input_buffer_addr, A_rows, A_cols, ValueInfo(conv_params->conv_input, nullptr), false);
    my_printf_debug("filter_buffer_addr = lea_buffer + LEA_BUFFER_SIZE - %d" NEWLINE, static_cast<int>(lea_buffer + LEA_BUFFER_SIZE - filter_buffer_addr));
    my_printf_debug("filter" NEWLINE);
    dump_matrix_debug(filter_buffer_addr, B_rows, B_cols, ValueInfo(conv_params->conv_filter, nullptr), false);

    my_printf_debug("matrix_mpy_results" NEWLINE);
    dump_matrix_debug(matrix_mpy_results, A_rows, B_cols, ValueInfo(conv_params->output));
    my_printf_debug(NEWLINE);

#if STABLE_POWER
    // compare_vm_vm(matrix_mpy_results, conv_params->model, conv_params->output, cur_output_data_offset, values_to_preserve);
#else // STABLE_POWER
    compare_vm_nvm(matrix_mpy_results, conv_params->model, conv_params->output, cur_output_data_offset, values_to_preserve);
#endif // STABLE_POWER
    /* END dump data */

    my_printf_debug("output_data offset = %d" NEWLINE, cur_output_data_offset);

    MY_ASSERT(cur_output_data_offset + n_filters < INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);

#if HAWAII
    hawaii_record_footprints(conv_params->model, values_to_preserve);
#endif

#if INDIRECT_RECOVERY
    if (n_keep_state_bits != n_filters) {
        check_next_turning_point(conv_params->old_output_offset, conv_params->turning_point_idx,
                                 conv_params->next_turning_point, conv_params->cur_slot_info, cur_output_data_offset + conv_params->OUTPUT_CHANNEL);
        my_printf_debug("old_output_offset flipped to %d" NEWLINE, conv_params->old_output_offset);

        flip_filter_state_bits(conv_params, n_filters, n_keep_state_bits, 0);
    }
#endif
}

static inline uint16_t load_input_vector(uint32_t src_addr, int16_t* dest_addr, uint16_t len, const ConvTaskParams* conv_params) {
    my_printf_debug("Load %d IFM values from range [%d, %d) ",
                    len, src_addr, static_cast<int>(src_addr + len));
    int16_t* memcpy_dest_addr = nullptr;
    uint16_t loaded_len = 0;

    MY_ASSERT(len != 0);

#if JAPARI
    if (conv_params->conv_input_has_footprints) {
        memcpy_dest_addr = input_buffer_with_footprints;
    } else
#endif
    {
        memcpy_dest_addr = dest_addr;
        loaded_len = len;
    }
    my_memcpy_from_param(
        conv_params->model, memcpy_dest_addr,
        conv_params->real_conv_input, src_addr,
        len * sizeof(int16_t));
#if JAPARI
    if (conv_params->conv_input_has_footprints) {
        // Use nested loops as skipping footprints by `% (BATCH_SIZE)` is quite slow on boards
        int16_t *dest_ptr = dest_addr,
                *src_ptr = input_buffer_with_footprints;
        for (uint16_t src_idx = 0; src_idx < len; src_idx += (BATCH_SIZE + 1)) {
            for (uint8_t batch_offset = 0; batch_offset < BATCH_SIZE; batch_offset++) {
                *dest_ptr = *src_ptr;
                dest_ptr++;
                src_ptr++;
            }
            src_ptr++; // skipping footprints
        }
        loaded_len = dest_ptr - dest_addr;
    }
#endif

#if MY_DEBUG >= MY_DEBUG_VERBOSE
    for (uint16_t idx = 0; idx < loaded_len; idx++) {
        my_printf_debug("%d ", dest_addr[idx]);
    }
    my_printf_debug(NEWLINE);
#endif
    return loaded_len;
}

static void handle_conv_inner_loop(Model *model, ConvTaskParams *conv_params) {
    /* copy input data, col by col */

    int8_t real_input_index = -1;
    if (conv_params->conv_input->param_flags & SEPARATE_TILING) {
        real_input_index = (2 * conv_params->input_tile_c_index >= conv_params->n_tiles_c) ? 1 : 0;
        conv_params->real_conv_input = get_parameter_info(conv_params->conv_input->extra_info[real_input_index]);
    } else {
        conv_params->real_conv_input = conv_params->conv_input;
    }

    int16_t tile_h_offset =
        ((conv_params->input_h - conv_params->input_h_first - conv_params->kX) / conv_params->stride_h)
        % conv_params->flags->extra.conv.output_tile_h;
    int16_t tile_w_offset =
        ((conv_params->input_w - conv_params->input_w_first - conv_params->kY) / conv_params->stride_w)
        % conv_params->flags->extra.conv.output_tile_w;
    int16_t input_h_tile_begin = conv_params->input_h - conv_params->kX - tile_h_offset * conv_params->stride_h;
    int16_t input_w_tile_begin = conv_params->input_w - conv_params->kY - tile_w_offset * conv_params->stride_w;
    my_printf_debug("tile_h_offset: %d" NEWLINE, tile_h_offset);
    my_printf_debug("tile_w_offset: %d" NEWLINE, tile_w_offset);
    my_printf_debug("input_w_tile_begin: %d" NEWLINE, input_w_tile_begin);

    /*
    const uint8_t* pads = conv_params->flags->extra.conv.pads;
    enum { PAD_H_BEGIN = 0, PAD_W_BEGIN = 1, PAD_H_END = 2, PAD_W_END = 3 };
    // let tile_w be equal to input_tile_w
    conv_params->tile_w += pads[PAD_W_BEGIN] + pads[PAD_W_END];
    */

    /* int32_t instead of int16_t as TI's compiler cannot handle negative
     * offsets correctly. The expression `ptr + (int16_t)(-2)` is
     * compiled as:
     * 1. -2 is represented as 0x00FFFE (general registers are 24-bit long).
     *    Assume this value is stored in R11.
     * 2. RLAM.A #1,R11  # multiply by 2 to transform the offset for int16_t
     *    to the difference of addresses.
     * In step 2, R11 becomes 0x01FFFC, while it should be -4, or 0x00FFFC,
     * and thus the resultant address is offset by 0x10000.
     */
    int32_t w_start = int16_max(0, input_w_tile_begin),
            w_end = int16_min(input_w_tile_begin + conv_params->tile_w, conv_params->W) - 1;
    my_printf_debug("w_start=%" PRId32 " ", w_start);
    my_printf_debug("w_end=%" PRId32 NEWLINE, w_end);
    int16_t *dest;
    int16_t max_n_filters = conv_params->flags->extra.conv.output_tile_c;
#if JAPARI
    max_n_filters *= 2;
#endif
    // TEMP_FILTER_WIDTH additional filters for values before transpose
    // only load one vector
    uint16_t inputs_len = MIN_VAL(
        LEA_BUFFER_SIZE - OUTPUT_LEN - (max_n_filters + TEMP_FILTER_WIDTH) * conv_params->filter_offset,
        conv_params->tile_w * conv_params->tile_h * conv_params->dest_offset);
    MY_ASSERT(inputs_len < LEA_BUFFER_SIZE); // make sure no overflow occurs in the previous line
    my_printf_debug("inputs_len1: %d" NEWLINE, LEA_BUFFER_SIZE - OUTPUT_LEN - (max_n_filters + TEMP_FILTER_WIDTH) * conv_params->filter_offset);
    my_printf_debug("inputs_len2: %d" NEWLINE, conv_params->tile_h * conv_params->tile_w * conv_params->dest_offset);
    my_printf_debug("inputs_len: %d" NEWLINE, inputs_len);

    dest = lea_buffer;
    if(conv_params->cached_input_h == conv_params->input_h_first - 1 ||
            conv_params->cached_input_w == conv_params->input_w_first - 1) {
        my_printf_debug("Start loading input" NEWLINE);
        int32_t h_start = int16_max(0, input_h_tile_begin),
                h_end =   int16_min(input_h_tile_begin + conv_params->tile_h, conv_params->H) - 1;

        my_printf_debug("Reinitialize input buffer" NEWLINE "inputs_len = %d" NEWLINE, inputs_len);

        my_fill_q15(0, lea_buffer, inputs_len);

        // reserve space for padding 0
        dest += (w_start - input_w_tile_begin) * conv_params->dest_offset * conv_params->tile_h;

        my_printf_debug("h_start=%" PRId32 " ", h_start);
        my_printf_debug("h_end=%" PRId32 NEWLINE, h_end);
        my_printf_debug("conv_params->input_tile_c_offset: %d" NEWLINE, conv_params->input_tile_c_offset);

        uint16_t cur_input_tile_c = conv_params->cur_input_tile_c;
        uint8_t im2col_channel_offset = conv_params->dest_offset;
        my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                        static_cast<int>(dest - lea_buffer));
        uint16_t cur_input_channel = conv_params->CHANNEL;
        if (conv_params->conv_input->param_flags & SEPARATE_TILING) {
            cur_input_channel /= 2;
        }
#if JAPARI
        if (conv_params->conv_input_has_footprints) {
            cur_input_tile_c = extend_for_footprints(cur_input_tile_c);
            cur_input_channel = extend_for_footprints(cur_input_channel);
        }
#endif
        int16_t input_src_offset = h_start * conv_params->W * cur_input_channel + w_start * cur_input_channel;
#if JAPARI
        input_src_offset += conv_params->input_tile_c_offset_with_footprints;
#else
        input_src_offset += conv_params->input_tile_c_offset;
#endif
        if (real_input_index == 1) {
            input_src_offset -= cur_input_channel;
        }
        for(int32_t w = w_start; w <= w_end; ++w) {
            // reserve space for padding 0
            int16_t *dest_addr = dest + (h_start - input_h_tile_begin) * im2col_channel_offset;
            uint32_t src_addr = input_src_offset;
            for(int32_t h = h_start; h <= h_end; ++h) {
                load_input_vector(src_addr, dest_addr, cur_input_tile_c, conv_params);
                dest_addr += conv_params->dest_offset;
                src_addr += cur_input_channel * conv_params->W;
            }
            dest += conv_params->tile_h * conv_params->dest_offset;
            input_src_offset += cur_input_channel;
        }
        if (conv_params->real_conv_input->scale != conv_params->conv_input->scale) {
            int16_t scaleFract;
            uint8_t shift;
            float_to_scale_params(&scaleFract, &shift, 1.0f * conv_params->real_conv_input->scale / conv_params->conv_input->scale);
            my_scale_q15(lea_buffer, scaleFract, shift, lea_buffer, inputs_len);
        }
        uint16_t bias_multipler_offset = conv_params->dest_offset - 1;
        while (bias_multipler_offset < inputs_len) {
            lea_buffer[bias_multipler_offset] = -0x8000; // _Q15(-1.0)
            bias_multipler_offset += conv_params->dest_offset;
        }
        conv_params->cached_input_h = input_h_tile_begin;
        conv_params->cached_input_w = input_w_tile_begin;
    }
    my_printf_debug("Loaded inputs" NEWLINE);
    // state = 0 as state bits are already removed by my_offset_q15 above
    dump_matrix_debug(lea_buffer, inputs_len, ValueInfo(conv_params->real_conv_input, nullptr), false);

    int16_t max_input_h =
        MIN_VAL(input_h_tile_begin + conv_params->flags->extra.conv.output_tile_h * conv_params->stride_h + conv_params->kX - 1,
                conv_params->input_h_last + conv_params->kX);
    int16_t max_input_w =
        MIN_VAL(input_w_tile_begin + conv_params->flags->extra.conv.output_tile_w * conv_params->stride_w + conv_params->kY - 1,
                conv_params->input_w_last + conv_params->kY);
    for(int32_t cur_input_w = conv_params->input_w; cur_input_w <= max_input_w; cur_input_w += conv_params->stride_w) {
        for(int32_t cur_input_h = conv_params->input_h; cur_input_h <= max_input_h; cur_input_h += conv_params->stride_h) {
            my_printf_debug("max_input_h: %d, max_input_w: %d" NEWLINE, max_input_h, max_input_w);
            my_printf_debug("cur_input_h: %d, cur_input_w: %d" NEWLINE, cur_input_h, cur_input_w);
            // filter_idx is set to initial_c in handle_conv
            convTask(cur_input_w, cur_input_h, conv_params);
            // reset here for further processing
            conv_params->filter_idx = conv_params->filter_tile_index * conv_params->flags->extra.conv.output_tile_c;
        }
        conv_params->input_h -= tile_h_offset * conv_params->stride_h;
        my_printf_debug("tile_h_offset: %d" NEWLINE, tile_h_offset);
        my_printf_debug("input_h: %d" NEWLINE, conv_params->input_h);
        tile_h_offset = 0;
    }
    conv_params->input_w -= tile_w_offset * conv_params->stride_w;
    my_printf_debug("tile_w_offset: %d" NEWLINE, tile_w_offset);
    my_printf_debug("input_w: %d" NEWLINE, conv_params->input_w);
    tile_w_offset = 0;
}

void alloc_conv(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    const ParameterInfo *conv_input = input[0], *conv_filter = input[1];

    MY_ASSERT(conv_input->bitwidth == 16 && conv_filter->bitwidth == 16);

#if !JAPARI
    // skip the check for JAPARI as it is too complex
    MY_ASSERT(conv_input->dims[1] == conv_filter->dims[1]);
#endif

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t CHANNEL = conv_input->dims[1], H = conv_input->dims[2], W = conv_input->dims[3];
    uint16_t OUTPUT_CHANNEL = conv_filter->dims[0];

    ConvTaskParams *conv_params = &conv_params_obj;

    conv_params->model = model;
    conv_params->flags = &node->flags;

    conv_params->kH = conv_filter->dims[2];
    conv_params->kW = conv_filter->dims[3];

    conv_params->stride_h = conv_params->flags->stride[0];
    conv_params->stride_w = conv_params->flags->stride[1];

    const uint8_t* pads = conv_params->flags->extra.conv.pads;
    enum { PAD_H_BEGIN = 0, PAD_W_BEGIN = 1, PAD_H_END = 2, PAD_W_END = 3 };
    conv_params->input_h_first = -pads[PAD_H_BEGIN];
    conv_params->input_w_first = -pads[PAD_W_BEGIN];
    conv_params->input_h_last = H + pads[PAD_H_END] - conv_params->kH;
    conv_params->input_w_last = W + pads[PAD_W_END] - conv_params->kW;

    conv_params->OUTPUT_H = (conv_params->input_h_last - conv_params->input_h_first) / conv_params->stride_h + 1;
    conv_params->OUTPUT_W = (conv_params->input_w_last - conv_params->input_w_first) / conv_params->stride_w + 1;

#if JAPARI
    conv_params->force_align_footprints = (OUTPUT_CHANNEL % BATCH_SIZE != 0);
    OUTPUT_CHANNEL = extend_for_footprints(OUTPUT_CHANNEL, conv_params->force_align_footprints);
    if (has_footprints(conv_input)) {
        conv_params->n_tiles_c = CHANNEL / (BATCH_SIZE + 1) * BATCH_SIZE / conv_params->flags->extra.conv.input_tile_c;
    } else
#endif
    {
        conv_params->n_tiles_c = CHANNEL / conv_params->flags->extra.conv.input_tile_c;
    }
#if STATEFUL
    if (conv_params->flags->extra.conv.output_tile_c % BATCH_SIZE) {
        conv_params->output_padding = BATCH_SIZE - conv_params->flags->extra.conv.output_tile_c % BATCH_SIZE;
    } else {
        conv_params->output_padding = 0;
    }
    OUTPUT_CHANNEL += conv_params->output_padding;
#endif
    my_printf_debug("input_tile_c=%d, output_tile_c=%d" NEWLINE, conv_params->flags->extra.conv.input_tile_c, conv_params->flags->extra.conv.output_tile_c);

    /* XXX: extend flags; assume dilation=(1, 1) for now */
    output->bitwidth = 16;
    output->slot = get_next_slot(model, conv_input);

#if STABLE_POWER
    output->params_len = conv_params->OUTPUT_H * conv_params->OUTPUT_W * OUTPUT_CHANNEL * sizeof(int16_t);
#else // STABLE_POWER
    /* 3 buffers:
     * 0: the result of accum cmd
     * 1: the result of accum cmd (double buffering)
     * 2: the result of psum cmd
     */
    int16_t output_tile_len =
        conv_params->flags->extra.conv.output_tile_h *
        conv_params->flags->extra.conv.output_tile_w *
        conv_params->flags->extra.conv.output_tile_c;
    my_printf_debug("output_tile_len: %d" NEWLINE, output_tile_len);
    my_printf_debug("output_len: %d" NEWLINE, conv_params->OUTPUT_H * conv_params->OUTPUT_W * OUTPUT_CHANNEL);
    output->params_len = sizeof(int16_t) * (2 * conv_params->OUTPUT_H * conv_params->OUTPUT_W * OUTPUT_CHANNEL + output_tile_len);
#endif // STABLE_POWER
    output->dims[0] = 1;
    output->dims[1] = OUTPUT_CHANNEL;
    output->dims[2] = conv_params->OUTPUT_H;
    output->dims[3] = conv_params->OUTPUT_W;
    output->param_flags &= ~SEPARATE_TILING;
    output->scale = conv_input->scale * conv_filter->scale;
#if STATEFUL
    if (conv_input->slot == SLOT_TEST_SET) {
        output->scale *= 2;
    }
#endif
}

#if SPARSE and STABLE_POWER
static void append_zero_to_pruned_output_channel(Model *model, const Node *node, ParameterInfo *output, uint16_t filter_idx, int16_t output_w, int16_t output_h, int16_t tile_h_offset, int16_t tile_w_offset, int8_t buffer_id) {
    my_printf_debug("==> Start appending zero ..." NEWLINE);
    init_cpu_buffer();
    uint16_t OUTPUT_H = output->dims[2], OUTPUT_W = output->dims[3];
    for(; output_w < OUTPUT_W; output_w += node->flags.extra.conv.output_tile_w) {
        for(; output_h < OUTPUT_H; output_h += node->flags.extra.conv.output_tile_h) {
            preserve_output(model, node, output, filter_idx, output_w, output_h, 0, 0, 0);
        }
    }
}
#endif
#if SPARSE
/* The method find the next nonzero block(or sub-tile)
 * Directly set the config of next nonzero block in conv_params.
 *
 * Following members of conv_params wiil be modified:
 * 1. n_cols
 * 2. cur_n_cols
 * 3. cur_row_val
 * 4. next_row_val
 * 5. row_index
 * 6. filter_tile_index
 * 7. filter_idx
 * 8. input_tile_c_index
 * 9. input_tile_c_offset
 *
 * Note: Some members, such as "kX" and "kY", will not be modified in this method. Therefore,
 * you should change them after calling this method.
 */
static void next_nonzero_value(const Node *node, ConvTaskParams *conv_params, int16_t *col_val) {
    // rows: the number of filter groups
    // cols: the number of input_tile_c
    while(!conv_params->n_cols && conv_params->row_index * conv_params->flags->extra.conv.output_tile_c < conv_params->OUTPUT_CHANNEL) {
        conv_params->cur_row_val = conv_params->next_row_val;
        conv_params->next_row_val = get_row_val(conv_params->model, conv_params->conv_filter, conv_params->row_index + 1);
        conv_params->n_cols = conv_params->next_row_val - conv_params->cur_row_val;
#if STABLE_POWER
        if(!conv_params->n_cols) {
            conv_params->filter_tile_index = conv_params->row_index;
            conv_params->filter_idx = conv_params->filter_tile_index * conv_params->flags->extra.conv.output_tile_c;
            append_zero_to_pruned_output_channel(conv_params->model, node, conv_params->output, conv_params->filter_idx, 0, 0, 0, 0, 0);
        }
#endif // STABLE_POWER
        conv_params->row_index++;
    }
    if(conv_params->n_cols) {
        conv_params->cur_n_cols = 0;
        conv_params->filter_tile_index = conv_params->row_index - 1;
        conv_params->filter_idx = conv_params->filter_tile_index * conv_params->flags->extra.conv.output_tile_c;
        *col_val = get_col_val(conv_params->model, conv_params->conv_filter, conv_params->cur_row_val + conv_params->cur_n_cols);
        conv_params->input_tile_c_index = *col_val / (conv_params->kH * conv_params->kW);
        conv_params->input_tile_c_offset = conv_params->input_tile_c_index * conv_params->flags->extra.conv.input_tile_c;
    }
}
#endif // SPARSE

static void conv_merge(Model *model, ConvTaskParams *conv_params, ParameterInfo *output, int16_t output_w, int16_t output_h, int16_t tile_h_offset, int16_t tile_w_offset) {
    int16_t OUTPUT_C = output->dims[1],
            OUTPUT_H = output->dims[2],
            OUTPUT_W = output->dims[3],
            output_tile_c = conv_params->flags->extra.conv.output_tile_c,
            output_tile_w = MIN_VAL(conv_params->flags->extra.conv.output_tile_w, OUTPUT_W - (output_w - tile_w_offset)),
            output_tile_h = MIN_VAL(conv_params->flags->extra.conv.output_tile_h, OUTPUT_H - (output_h - tile_h_offset));
    MY_ASSERT(output_w + output_tile_w - tile_w_offset <= OUTPUT_W);
    MY_ASSERT(output_h + output_tile_h - tile_h_offset <= OUTPUT_H);
    int16_t default_output_tile_len =
        conv_params->flags->extra.conv.output_tile_c *
        conv_params->flags->extra.conv.output_tile_w *
        conv_params->flags->extra.conv.output_tile_h * 2;
    int16_t *partial_result = lea_buffer + LEA_BUFFER_SIZE - default_output_tile_len;
    uint16_t output_len = OUTPUT_C * OUTPUT_W * OUTPUT_H;
    int16_t psum_offset =
        conv_params->psum_buffer_version * output_len +
        (output_h * OUTPUT_W + output_w) * OUTPUT_C + conv_params->filter_idx; // NHWC
#if !STABLE_POWER
    int16_t input_offset =
        2 * output_len +
        ((output_h % conv_params->flags->extra.conv.output_tile_h) * conv_params->flags->extra.conv.output_tile_w  + (output_w % conv_params->flags->extra.conv.output_tile_w)) * output_tile_c; // hwc
    uint16_t cur_input_offset = input_offset;
#endif // STABLE_POWER
    int16_t vm_offset = 0;
    int16_t chunk_offset = 0;
    uint16_t cur_psum_offset = psum_offset;
    int16_t *to_add, *be_add;
    for(int16_t offset_w = 0; offset_w < output_tile_w - tile_w_offset; ++offset_w) {
        for(int16_t offset_h = 0; offset_h < output_tile_h - tile_h_offset; ++offset_h) {
            uint16_t real_chunk_len = output_tile_c - chunk_offset;
            to_add = partial_result + vm_offset;
            be_add = cpu_buffer + vm_offset;
#if !STABLE_POWER
            cur_input_offset += chunk_offset;
#endif //STABLE_POWER
            cur_psum_offset += chunk_offset;
#if !STABLE_POWER
            my_memcpy_from_param(model, be_add, output, cur_input_offset, real_chunk_len * sizeof(int16_t));
            my_printf_debug(NEWLINE "input offset %d, VM offset %d" NEWLINE, cur_input_offset, vm_offset);
            my_printf_debug("Loaded chunk" NEWLINE);
            dump_matrix_debug(be_add, real_chunk_len, ValueInfo(output));
#endif //STABLE_POWER
            my_memcpy_from_param(model, to_add, output, cur_psum_offset, real_chunk_len * sizeof(int16_t));
            my_printf_debug(NEWLINE "psum offset %d, VM offset %d" NEWLINE, cur_psum_offset, output_tile_h * output_tile_w * output_tile_c + vm_offset);
#if !STABLE_POWER
#if SPARSE
            if(!conv_params->cur_n_cols) {
                memset(to_add, 0, real_chunk_len * sizeof(int16_t));
            }
#else // SPARSE
            if(conv_params->input_tile_c_index == 0 && conv_params->kX == 0 && conv_params->kY == 0) {
                memset(to_add, 0, real_chunk_len * sizeof(int16_t));
            }
#endif // SPARSE
#endif // STABLE_POWER
            my_printf_debug("Loaded chunk" NEWLINE);
            dump_matrix_debug(to_add, real_chunk_len, ValueInfo(output));
            // perform accum
            my_add_q15(be_add, to_add, be_add, real_chunk_len);
            my_printf_debug("Adding Result" NEWLINE);
            dump_matrix_debug(be_add, real_chunk_len, ValueInfo(output));
            chunk_offset = 0;
            cur_psum_offset += OUTPUT_C * OUTPUT_W;
#if !STABLE_POWER
            cur_input_offset += conv_params->flags->extra.conv.output_tile_c * conv_params->flags->extra.conv.output_tile_w;
#endif // STABLE_POWER
            vm_offset += real_chunk_len;
        }
#if !STABLE_POWER
        input_offset -= tile_h_offset * conv_params->flags->extra.conv.output_tile_c * conv_params->flags->extra.conv.output_tile_w;
        cur_input_offset = input_offset + conv_params->flags->extra.conv.output_tile_c * (offset_w + 1);
#endif // STABLE_POWER
        psum_offset -= tile_h_offset * OUTPUT_W * OUTPUT_C;
        cur_psum_offset = psum_offset + OUTPUT_C * (offset_w + 1);
        tile_h_offset = 0;
    }
}

void handle_conv(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    const ParameterInfo *conv_input = input[0], *conv_filter = input[1], *conv_bias = (node->inputs_len == 3) ? input[2] : nullptr;
    my_printf_debug("Conv!" NEWLINE);

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   CHANNEL = conv_filter->dims[1];

    int16_t input_channels = conv_input->dims[1];
    ConvTaskParams *conv_params = &conv_params_obj;
    const uint8_t* pads = conv_params->flags->extra.conv.pads;
    enum { PAD_H_BEGIN = 0, PAD_W_BEGIN = 1, PAD_H_END = 2, PAD_W_END = 3 };

    int16_t output_tile_len =
        conv_params->flags->extra.conv.output_tile_h *
        conv_params->flags->extra.conv.output_tile_w *
        conv_params->flags->extra.conv.output_tile_c * 2;

    // reserve memory for psum
    matrix_mpy_results -= output_tile_len;
    my_printf_debug("matrix_mpy_results offset: %ld" NEWLINE, matrix_mpy_results - lea_buffer);

    // input_tile_w/input_tile_h
    conv_params->tile_w =
        MIN_VAL(W + pads[PAD_W_BEGIN] + pads[PAD_W_END],
                (node->flags.extra.conv.output_tile_w - 1) * conv_params->stride_w + conv_params->kW);
    conv_params->tile_h =
        MIN_VAL(H + pads[PAD_H_BEGIN] + pads[PAD_H_END],
                (node->flags.extra.conv.output_tile_h - 1) * conv_params->stride_h + conv_params->kH);

    my_printf_debug("tile_w = %d" NEWLINE, conv_params->tile_w);
    my_printf_debug("tile_h = %d" NEWLINE, conv_params->tile_h);
    my_printf_debug("n_tiles_c = %d" NEWLINE, conv_params->n_tiles_c);

    conv_params->conv_input = conv_input;
    conv_params->conv_filter = conv_filter;
    conv_params->conv_bias = conv_bias;
    conv_params->output = output;
    conv_params->filter_buffer_addr = NULL;
    conv_params->cached_filter_idx = -1;
    conv_params->cached_input_h = conv_params->input_h_first - 1;
    conv_params->cached_input_w = conv_params->input_w_first - 1;
    conv_params->cached_kX = -1;
    conv_params->cached_kY = -1;
    conv_params->H = H;
    conv_params->W = W;
    conv_params->cur_op = 0; // 0: conv, 1: merge
#if STABLE_POWER
    conv_params->psum_buffer_version = 0;
#else // STABLE_POWER
    conv_params->psum_buffer_version = (conv_params->kH * conv_params->kW * (input_channels / node->flags.extra.conv.input_tile_c)) & 0x1;
#endif // STABLE_POWER

#if JAPARI
    conv_params->conv_input_has_footprints = has_footprints(conv_input);
#endif

    conv_params->CHANNEL = CHANNEL;
    conv_params->OUTPUT_CHANNEL = output->dims[1];
    conv_params->N_FILTERS = conv_filter->dims[0];
#if SPARSE
    int16_t COL_VALS[MAX_N_COL_CONV] = {0};
    int16_t col_val;
    conv_params->row_index = 0;
    conv_params->cur_row_val = 0; // cur_row_val + cur_n_cols => cur_cols_index
    conv_params->next_row_val = 0;
    conv_params->n_cols = 0;
    conv_params->cur_n_cols = 0;
#else // SPARSE
    conv_params->input_tile_c_offset = 0;
    conv_params->input_tile_c_index = 0;
    conv_params->filter_tile_index = 0;
    conv_params->filter_idx = 0;
    conv_params->kX = 0;
    conv_params->kY = 0;
#endif // SPARSE

    conv_params->input_h = conv_params->input_h_first;
    conv_params->input_w = conv_params->input_w_first;
#if INTERMITTENT
    /* Handle sub-layer footprint */
    uint32_t first_unfinished_sub_layer_idx = conv_params->model->sub_layer_idx;
    my_printf_debug("first_unfinished_sub_layer_idx: %d" NEWLINE, first_unfinished_sub_layer_idx);


    uint16_t n_output_tile_w = upper_gauss(conv_params->OUTPUT_W, conv_params->flags->extra.conv.output_tile_w);
    uint16_t n_output_tile_h = upper_gauss(conv_params->OUTPUT_H, conv_params->flags->extra.conv.output_tile_h);
    uint16_t sub_layers_in_a_filter_tile = n_output_tile_w * n_output_tile_h;
    my_printf_debug("sub_layers_in_a_filter_tile: %d" NEWLINE, sub_layers_in_a_filter_tile);
    // clarify the score to avoid compile error (cross initialization)
    {
        MY_ASSERT(n_output_tile_w);
        MY_ASSERT(n_output_tile_h);
        conv_params->filter_tile_index = first_unfinished_sub_layer_idx / sub_layers_in_a_filter_tile;
#if SPARSE
        conv_params->row_index = conv_params->filter_tile_index;
        conv_params->next_row_val = get_row_val(model, conv_filter, conv_params->row_index);
        // TODO: append 0 to the pruned channels
        next_nonzero_value(node, conv_params, &col_val);
        if(conv_params->n_cols) {
            // XXX: add a checker to verify the result
            my_memcpy_from_param_col(model, COL_VALS, conv_filter, conv_params->cur_row_val, conv_params->n_cols * sizeof(int16_t));
            for(int16_t idx = 0; idx < conv_params->n_cols; ++idx) {
                my_printf_debug("%d ", COL_VALS[idx]);
            }
            my_printf_debug(NEWLINE);
            conv_params->psum_buffer_version = conv_params->n_cols & 0x1;
        } else {
            goto EXIT_LAYER;
        }
#endif // SPARSE
        first_unfinished_sub_layer_idx %= sub_layers_in_a_filter_tile;

        uint16_t tile_w_offset = first_unfinished_sub_layer_idx / n_output_tile_h;
        uint16_t tile_h_offset = first_unfinished_sub_layer_idx % n_output_tile_h;
        uint16_t input_w_offset = tile_w_offset * conv_params->flags->extra.conv.output_tile_w * conv_params->stride_w;
        uint16_t input_h_offset = tile_h_offset * conv_params->flags->extra.conv.output_tile_h * conv_params->stride_h;
        my_printf_debug("input_w_offset: %d" NEWLINE, input_w_offset);
        my_printf_debug("input_h_offset: %d" NEWLINE, input_h_offset);
        conv_params->input_w += input_w_offset;
        conv_params->input_h += input_h_offset;

        /* Handle intra-sub-layer footprint */
        uint32_t first_unfinished_job_idx = run_recovery(model, output);
        my_printf_debug("first_unfinished_job_idx: %d\n", first_unfinished_job_idx);
        // XXX: the jobs should be even
        fix_first_unfinished_value_offset(model, &first_unfinished_job_idx);
        my_printf_debug("fixed first_unfinished_job_idx: %d\n", first_unfinished_job_idx);
        MY_ASSERT(~first_unfinished_job_idx & 0x1);

        // XXX: mixing output and input to calculate may be wrong
        // XXX: Handle CHANNEL % output_tile_c != 0
        uint16_t cur_output_tile_h = MIN_VAL(conv_params->flags->extra.conv.output_tile_h, H - input_h_offset);
        uint16_t cur_output_tile_w = MIN_VAL(conv_params->flags->extra.conv.output_tile_w, W - input_w_offset);
        uint16_t cur_output_tile_c = MIN_VAL(conv_params->flags->extra.conv.output_tile_c,
                conv_params->OUTPUT_CHANNEL - conv_params->filter_tile_index * conv_params->flags->extra.conv.output_tile_c);

        if(cur_output_tile_c <= 0) {
            // finished layer
            goto EXIT_LAYER;
        }

        my_printf_debug("cur_output_tile_h: %d" NEWLINE, cur_output_tile_h);
        my_printf_debug("cur_output_tile_w: %d" NEWLINE, cur_output_tile_w);
        my_printf_debug("cur_output_tile_c: %d" NEWLINE, cur_output_tile_c);
        MY_ASSERT(cur_output_tile_h);
        MY_ASSERT(cur_output_tile_w);
        MY_ASSERT(cur_output_tile_c);

        uint16_t jobs_in_a_weight_tile = 2 *
                cur_output_tile_w *
                cur_output_tile_h *
                conv_params->flags->extra.conv.output_tile_c; // psum, accum

        uint16_t n_weight_tiles = conv_params->kH * conv_params->kW;
        my_printf_debug("n_weight_tiles: %d" NEWLINE, n_weight_tiles);
        uint16_t intra_kernel_offset = 0;
#if SPARSE
        int16_t finished_weight_tiles = first_unfinished_job_idx / jobs_in_a_weight_tile;
        my_printf_debug("finished_weight_tiles: %d" NEWLINE, finished_weight_tiles);
        conv_params->input_tile_c_index = COL_VALS[finished_weight_tiles] / n_weight_tiles;
        first_unfinished_job_idx -= finished_weight_tiles * jobs_in_a_weight_tile;
        conv_params->cur_n_cols = finished_weight_tiles;
        intra_kernel_offset = COL_VALS[conv_params->cur_n_cols] % n_weight_tiles;
        jobs_in_a_weight_tile = 2 * cur_output_tile_w * cur_output_tile_h * cur_output_tile_c; // psum, accum
#else // SPARSE
        conv_params->input_tile_c_index = first_unfinished_job_idx / (jobs_in_a_weight_tile * n_weight_tiles);
        first_unfinished_job_idx %= (jobs_in_a_weight_tile * n_weight_tiles);
        jobs_in_a_weight_tile = 2 * cur_output_tile_w * cur_output_tile_h * cur_output_tile_c; // psum, accum
        intra_kernel_offset = first_unfinished_job_idx / jobs_in_a_weight_tile;
        first_unfinished_job_idx %= jobs_in_a_weight_tile;
#endif // SPARSE

        my_printf_debug("first_unfinished_job_idx: %d\n", first_unfinished_job_idx);
        my_printf_debug("jobs_in_a_weight_tile: %d" NEWLINE, jobs_in_a_weight_tile);
        my_printf_debug("intra_kernel_offset: %d" NEWLINE, intra_kernel_offset);
#if SPARSE
        my_printf_debug("cur_n_cols: %d" NEWLINE, conv_params->cur_n_cols);
        my_printf_debug("cur_n_cols: %d" NEWLINE, conv_params->cur_n_cols);
        my_printf_debug("col val: %d" NEWLINE, COL_VALS[conv_params->cur_n_cols]);
        my_printf_debug("real intra_kernel_offset: %d" NEWLINE, intra_kernel_offset);
#endif // SPARSE
        conv_params->kY = intra_kernel_offset / conv_params->kH;
        conv_params->kX = intra_kernel_offset % conv_params->kH;

        uint16_t jobs_in_a_set_psum_cmd = cur_output_tile_w * cur_output_tile_h * cur_output_tile_c;
        conv_params->cur_op = first_unfinished_job_idx / jobs_in_a_set_psum_cmd;

        my_printf_debug("cur_op: %d"  NEWLINE, conv_params->cur_op);
        MY_ASSERT(!(conv_params->cur_op & ~1)); // should be 0 or 1

        first_unfinished_job_idx %= jobs_in_a_set_psum_cmd;
        uint16_t input_tile_w_offset = first_unfinished_job_idx / (cur_output_tile_h * cur_output_tile_c) * conv_params->stride_w;
        first_unfinished_job_idx %= (cur_output_tile_h * cur_output_tile_c);
        my_printf_debug("remain: %d" NEWLINE, first_unfinished_job_idx);

        uint16_t input_tile_h_offset = first_unfinished_job_idx / cur_output_tile_c * conv_params->stride_h;
        my_printf_debug("input_tile_w_offset: %d" NEWLINE, input_tile_w_offset);
        my_printf_debug("input_tile_h_offset: %d" NEWLINE, input_tile_h_offset);
        my_printf_debug("jobs_in_a_set_psum_cmd: %d" NEWLINE, jobs_in_a_set_psum_cmd);
        conv_params->input_w += input_tile_w_offset;
        conv_params->input_h += input_tile_h_offset;

        // XXX: do not recover conv_merge
        uint16_t filter_offset_in_tile = 0;
        if(conv_params->cur_op) {
            filter_offset_in_tile = 0;
#if HAWAII
            write_hawaii_layer_footprint(model->layer_idx, -(first_unfinished_job_idx % cur_output_tile_c)); // discard jobs
#endif // HAWAII
        } else {
            filter_offset_in_tile = first_unfinished_job_idx % cur_output_tile_c;
        }
        my_printf_debug("filter_offset_in_tile: %d" NEWLINE, filter_offset_in_tile);
        conv_params->filter_idx =
            conv_params->filter_tile_index * conv_params->flags->extra.conv.output_tile_c +
            filter_offset_in_tile;
        conv_params->input_tile_c_offset =
            conv_params->input_tile_c_index * conv_params->flags->extra.conv.input_tile_c;
        if(conv_params->input_tile_c_offset >= CHANNEL) {
            conv_params->input_tile_c_offset = 0;
            conv_params->input_tile_c_index = 0;
            conv_params->model->sub_layer_idx++;
            commit_model();
        }

#if SPARSE
        uint16_t tile_1x1xTn_offset = conv_params->cur_n_cols;
#else // SPARSE
        uint16_t tile_1x1xTn_offset =
            conv_params->input_tile_c_index * conv_params->kH * conv_params->kW +
            conv_params->kH * conv_params->kY +
            conv_params->kX;
#endif // SPARSE
        conv_params->psum_buffer_version ^= tile_1x1xTn_offset & 0x1;
    }
    my_printf_debug("initial output N = %d" NEWLINE, conv_params->input_tile_c_index);
    my_printf_debug("initial output H = %d" NEWLINE, (conv_params->input_h - conv_params->input_h_first) / conv_params->stride_h);
    my_printf_debug("initial output W = %d" NEWLINE, (conv_params->input_w - conv_params->input_w_first) / conv_params->stride_w);
    my_printf_debug("initial output C = %d" NEWLINE, conv_params->filter_idx);
    my_printf_debug("conv_params->input_tile_c_offset: %d\n", conv_params->input_tile_c_offset);
    my_printf_debug("conv_params->input_tile_c_index: %d\n", conv_params->input_tile_c_index);
    my_printf_debug("conv_params->filter_tile_index: %d\n", conv_params->filter_tile_index);
    my_printf_debug("conv_params->filter_idx: %d\n", conv_params->filter_idx);
    my_printf_debug("conv_params->kX: %d\n", conv_params->kX);
    my_printf_debug("conv_params->kY: %d\n", conv_params->kY);
    my_printf_debug("conv_params->psum_buffer_version: %d\n", conv_params->psum_buffer_version);
    my_printf_debug("\n");
    // = happens when all values are finished
    MY_ASSERT(conv_params->input_tile_c_index <= conv_params->n_tiles_c);
#else // INTERMITTENT
#if SPARSE
    next_nonzero_value(node, conv_params, &col_val);
    if(conv_params->n_cols) {
        // XXX: add a checker to verify the result
        my_memcpy_from_param_col(model, COL_VALS, conv_filter, conv_params->cur_row_val, conv_params->n_cols * sizeof(int16_t));
    }
    conv_params->kY = (col_val % (conv_params->kW * conv_params->kH)) / conv_params->kH;
    conv_params->kX = (col_val % (conv_params->kW * conv_params->kH)) % conv_params->kH;
#if STABLE_POWER
    conv_params->psum_buffer_version = 0;
#else // STABLE_POWER
    conv_params->psum_buffer_version = (conv_params->n_cols - conv_params->cur_n_cols) & 0x1;
#endif // STABLE_POWER
#endif // SPARSE
#endif // INTERMITTENT
#if SPARSE
    my_printf_debug("conv_params->row_index: %d\n", conv_params->row_index);
    my_printf_debug("conv_params->cur_row_val: %d\n", conv_params->cur_row_val);
    my_printf_debug("conv_params->next_row_val: %d\n", conv_params->next_row_val);
    my_printf_debug("conv_params->n_cols: %d\n", conv_params->n_cols);
    my_printf_debug("conv_params->cur_n_cols: %d\n", conv_params->cur_n_cols);
    my_printf_debug("conv_params->input_tile_c_offset: %d\n", conv_params->input_tile_c_offset);
    my_printf_debug("conv_params->input_tile_c_index: %d\n", conv_params->input_tile_c_index);
    my_printf_debug("conv_params->filter_tile_index: %d\n", conv_params->filter_tile_index);
    my_printf_debug("conv_params->filter_idx: %d\n", conv_params->filter_idx);
    my_printf_debug("conv_params->kX: %d\n", conv_params->kX);
    my_printf_debug("conv_params->kY: %d\n", conv_params->kY);
    my_printf_debug("conv_params->psum_buffer_version: %d\n", conv_params->psum_buffer_version);
    my_printf_debug("\n");
#endif // SPARSE
#if JAPARI
    if (conv_params->conv_input_has_footprints) {
        input_channels = input_channels / (BATCH_SIZE + 1) * BATCH_SIZE;
    }
#endif

    for (; conv_params->filter_idx < conv_params->OUTPUT_CHANNEL;) {
        my_printf_debug("filter_idx: %d\n", conv_params->filter_idx);
#if STABLE_POWER
        init_cpu_buffer();
#endif // STABLE_POWER
        conv_params->cur_input_tile_c = MIN_VAL(conv_params->flags->extra.conv.input_tile_c, input_channels - conv_params->input_tile_c_offset);
        conv_params->cur_filter_tile_c = conv_params->cur_input_tile_c;
        my_printf_debug("cur_input_tile_c = %d" NEWLINE, conv_params->cur_input_tile_c);
        conv_params->dest_offset = 1 * conv_params->cur_input_tile_c; // only process one position
        // +1 for bias
        conv_params->dest_offset++;
        /* MSP430 LEA requires length to be even */
        conv_params->truncated = conv_params->dest_offset & 1;
        if (conv_params->truncated) {
            // when CHANNEL * kH * kW is odd, CHANNEL * kW (dest_offset) is
            // also odd, so dummy values are needed between slices to make
            // addresses even.
            // a dummy value for each slice (kW * CHANNEL q15 values)
            conv_params->dest_offset++;
        }
        conv_params->filter_offset = 1 * conv_params->dest_offset;
        while (true) {
            my_printf_debug("input_h: %d/input_w: %d" NEWLINE, conv_params->input_h, conv_params->input_w);
            for (; conv_params->input_w <= conv_params->input_w_last; conv_params->input_w += conv_params->flags->extra.conv.output_tile_w * conv_params->stride_w) {
                for (; conv_params->input_h <= conv_params->input_h_last; conv_params->input_h += conv_params->flags->extra.conv.output_tile_h * conv_params->stride_h) {
                    for(; conv_params->kY < conv_params->kW;) {
                        for(; conv_params->kX < conv_params->kH;) {
                            conv_params->input_w += conv_params->kY;
                            conv_params->input_h += conv_params->kX;
                            my_printf_debug("(%d, %d) (%d, %d)" NEWLINE, conv_params->input_h, conv_params->input_w, conv_params->kX, conv_params->kY);
                            my_printf_debug("current psum buffer version: %d" NEWLINE, conv_params->psum_buffer_version);
#if !STABLE_POWER
                            int16_t tile_h_offset =
                                ((conv_params->input_h - conv_params->input_h_first - conv_params->kX) / conv_params->stride_h) %
                                conv_params->flags->extra.conv.output_tile_h;
                            int16_t tile_w_offset =
                                ((conv_params->input_w - conv_params->input_w_first - conv_params->kY) / conv_params->stride_w) %
                                conv_params->flags->extra.conv.output_tile_w;
                            my_printf_debug("tile_h_offset: %d, tile_w_offset: %d" NEWLINE, tile_h_offset, tile_w_offset);
                            int16_t output_h = (conv_params->input_h - conv_params->input_h_first - conv_params->kX) /
                                                conv_params->stride_h,
                                    output_w = (conv_params->input_w - conv_params->input_w_first - conv_params->kY) /
                                                conv_params->stride_w;
#endif // STABLE_POWER
                            if(conv_params->cur_op == 0) {
                                // perform psum
                                handle_conv_inner_loop(model, conv_params);
#if !STABLE_POWER
                                conv_params->cur_op ^= 1;
                                output_h = (conv_params->input_h - conv_params->input_h_first - conv_params->kX) /
                                            conv_params->stride_h;
                                output_w = (conv_params->input_w - conv_params->input_w_first - conv_params->kY) /
                                            conv_params->stride_w;
                                tile_h_offset = tile_w_offset = 0;
                            }
                            if(conv_params->cur_op == 1) {
                                my_printf_debug("tile_h_offset: %d, tile_w_offset: %d" NEWLINE, tile_h_offset, tile_w_offset);
                                my_printf_debug("output_h: %d, output_w: %d" NEWLINE, output_h, output_w);
                                // perform accum
                                conv_merge(model, conv_params, output, output_w, output_h, tile_h_offset, tile_w_offset);
                                preserve_output(model, node, output, conv_params->filter_idx, output_w, output_h, tile_h_offset, tile_w_offset, conv_params->psum_buffer_version ^ 0x1);
                                conv_params->cur_op ^= 1;
                                conv_params->input_h -= tile_h_offset * conv_params->stride_h;
                                conv_params->input_w -= tile_w_offset * conv_params->stride_w;
                            }
                            conv_params->psum_buffer_version ^= 0x1;
                            my_printf_debug("Reseted input_h: %d/input_w: %d" NEWLINE, conv_params->input_h, conv_params->input_w);
#else // STABLE_POWER
                            }
#endif // STABLE_POWER
                            conv_params->input_w -= conv_params->kY;
                            conv_params->input_h -= conv_params->kX;
#if SPARSE
                            if(++conv_params->cur_n_cols >= conv_params->n_cols) {
                                goto EXIT_TILE;
                            }
                            col_val = COL_VALS[conv_params->cur_n_cols];
                            my_printf_debug("col_val: %d" NEWLINE, col_val);
                            if(col_val / (conv_params->kH * conv_params->kW) != conv_params->input_tile_c_index) {
                                conv_params->input_tile_c_index = col_val / (conv_params->kH * conv_params->kW);
                                conv_params->cached_input_h = conv_params->input_h_first - 1;
                                conv_params->cached_input_w = conv_params->input_w_first - 1;
                                conv_params->input_tile_c_offset = conv_params->input_tile_c_index * conv_params->flags->extra.conv.input_tile_c;
                                my_printf_debug("Swap tile_c !" NEWLINE);
                            }
                            conv_params->kX = (col_val % (conv_params->kW * conv_params->kH)) % conv_params->kH;
                            conv_params->kY = (col_val % (conv_params->kW * conv_params->kH)) / conv_params->kH;
#else // SPARSE
                            conv_params->kX++;
#endif // SPARSE
                        }
#if !SPARSE
                        conv_params->kX = 0;
                        conv_params->kY++;
                        // check if the all tile_c have finished
                        // If not. Do not leave the loop!
                        // reset cached_input_h and cached_input_w
                        if(conv_params->kY >= conv_params->kW) {
                            conv_params->input_tile_c_index++;
                            if (conv_params->input_tile_c_index * conv_params->flags->extra.conv.input_tile_c >= input_channels) {
                                break;
                            }
                            conv_params->kX = conv_params->kY = 0;
                            // reset for loading new IFM tile
                            conv_params->cached_input_h = conv_params->input_h_first - 1;
                            conv_params->cached_input_w = conv_params->input_w_first - 1;
                            conv_params->input_tile_c_offset = conv_params->input_tile_c_index * conv_params->flags->extra.conv.input_tile_c;
                            my_printf_debug("Swap tile_c !" NEWLINE);
                        }
#endif // SPARSE
                    }
#if SPARSE
                    EXIT_TILE:
                    conv_params->cached_input_h = conv_params->input_h_first - 1;
                    conv_params->cached_input_w = conv_params->input_w_first - 1;
#else // SPARSE
                    conv_params->kX = conv_params->kY = 0;
#endif // SPARSE
#if STABLE_POWER
                    uint16_t output_h = (conv_params->input_h - conv_params->input_h_first) / conv_params->stride_h,
                             output_w = (conv_params->input_w - conv_params->input_w_first) / conv_params->stride_w;
                    preserve_output(model, node, output, conv_params->filter_idx, output_w, output_h, 0, 0, 0);
                    init_cpu_buffer();
#else // STABLE_POWER
                    /* If (conv_params->kH * conv_params->kW * (input_channels / conv_params->flags->extra.conv.input_tile_c))
                     * is odd, the first buffer read and
                     * the last buffer wrote are different. Flip the version to let the final results
                     * be stored in the same buffer.
                     */
#if SPARSE
                    conv_params->psum_buffer_version = conv_params->n_cols & 0x1 ;
#else // SPARSE
                    conv_params->psum_buffer_version = (conv_params->kH * conv_params->kW * (input_channels / conv_params->flags->extra.conv.input_tile_c)) & 0x1 ;
#endif // SPARSE
#if HAWAII
                    reset_hawaii_sub_layer_footprint(conv_params->model->layer_idx);
#endif // HAWAII
                    // FIXME: Handle the issue if power fails at this line
                    // commit model for sub_layer
                    conv_params->model->sub_layer_idx++;
                    commit_model();
                    my_printf_debug("sub_layer_idx: %d" NEWLINE, conv_params->model->sub_layer_idx);
#endif // STABLE_POWER
#if SPARSE
                    int16_t next_input_h = conv_params->input_h + conv_params->flags->extra.conv.output_tile_h * conv_params->stride_h;
                    int16_t next_input_w = conv_params->input_w + conv_params->flags->extra.conv.output_tile_w * conv_params->stride_w;
                    if(next_input_h > conv_params->input_h_last && next_input_w > conv_params->input_w_last) {
                        break;
                    }
                    conv_params->cur_n_cols = 0;
                    col_val = COL_VALS[conv_params->cur_n_cols];
                    conv_params->kY = (col_val % (conv_params->kW * conv_params->kH)) / conv_params->kH;
                    conv_params->kX = (col_val % (conv_params->kW * conv_params->kH)) % conv_params->kH;
#endif // SPARSE
                    conv_params->input_tile_c_index = 0;
                    conv_params->input_tile_c_offset = conv_params->input_tile_c_index * conv_params->flags->extra.conv.input_tile_c;
                    conv_params->cached_input_h = conv_params->input_h_first - 1;
                    my_printf_debug("Swap tile_h !" NEWLINE);
                }
                conv_params->input_h = conv_params->input_h_first;
                conv_params->cached_input_w = conv_params->input_w_first - 1;
                my_printf_debug("Swap tile_w !" NEWLINE);
            }
            conv_params->cached_input_h = conv_params->input_h_first - 1;
            // finish computing a weight tile
#if SPARSE
            if(conv_params->cur_n_cols >= conv_params->n_cols) {
                // break when the weight tiles in the same filters are finished.
                break;
            }
#endif // SPARSE
            if(conv_params->input_w > conv_params->input_w_last) break;
        }
        my_printf_debug("Finish output channel [%d, %d)" NEWLINE, conv_params->filter_idx,
                    conv_params->filter_idx + conv_params->flags->extra.conv.output_tile_c);
#if SPARSE
        // FIXME: handle the issues if the entire row is pruned
        conv_params->n_cols = conv_params->cur_n_cols = 0;
        next_nonzero_value(node, conv_params, &col_val);
        if(conv_params->n_cols) {
            // XXX: add a checker to verify the result
            my_memcpy_from_param_col(model, COL_VALS, conv_filter, conv_params->cur_row_val, conv_params->n_cols * sizeof(int16_t));
            conv_params->kY = (col_val % (conv_params->kW * conv_params->kH)) / conv_params->kH;
            conv_params->kX = (col_val % (conv_params->kW * conv_params->kH)) % conv_params->kH;
        } else {
            goto EXIT_LAYER;
        }
#if HAWAII
        // commit model for sub_layer
        conv_params->model->sub_layer_idx = conv_params->filter_tile_index * sub_layers_in_a_filter_tile;
        commit_model();
        my_printf_debug("sub_layer_idx: %d" NEWLINE, conv_params->model->sub_layer_idx);
#endif // HAWAII
#else // SPARSE
        conv_params->input_tile_c_index = conv_params->input_tile_c_offset = 0;
        conv_params->filter_tile_index++;
        conv_params->filter_idx = conv_params->filter_tile_index * conv_params->flags->extra.conv.output_tile_c;
        conv_params->kX = conv_params->kY = 0;
        conv_params->cached_filter_idx = conv_params->cached_input_tile_c_offset = -1;
        conv_params->cached_kX = conv_params->cached_kY = -1;
#endif // SPARSE
        conv_params->input_h = conv_params->input_h_first;
        conv_params->input_w = conv_params->input_w_first;
        /* init version */
#if STABLE_POWER
        conv_params->psum_buffer_version = 0;
#else // STABLE_POWER
#if SPARSE
        conv_params->psum_buffer_version = conv_params->n_cols & 0x1;
#else // SPARSE
        conv_params->psum_buffer_version = (conv_params->kH * conv_params->kW * input_channels / node->flags.extra.conv.input_tile_c) & 0x1;
#endif // SPARSE
#endif // STABLE_POWER
    }
#if !STABLE_POWER or SPARSE
EXIT_LAYER:
#endif
    matrix_mpy_results += output_tile_len;
    flip_state_bit(model, output);

    my_printf_debug("handle_conv output" NEWLINE);
    dump_params_nhwc_debug(model, output, node->output_name);
}

#if STATEFUL
struct ConvMergeInputChunkHandlerParams {
    int16_t *to_add;
    uint16_t input_offset;
};

void ConvMergeInputChunkHandler(uint32_t range_offset, uint16_t range_len, int8_t state_bit, void* _params) {
    ConvMergeInputChunkHandlerParams* params = reinterpret_cast<ConvMergeInputChunkHandlerParams*>(_params);
    my_printf_debug("input range_offset=%d range_len=%d state_bit=%d" NEWLINE, range_offset, range_len, state_bit);
    int16_t *to_offset = params->to_add + range_offset - params->input_offset;
    my_offset_q15_batched(to_offset, -state_bit*0x4000, to_offset, range_len);
}
#endif

#if JAPARI
struct ConvMergeOutputChunkHandlerParams {
    uint32_t tiling_results_offset;
};

void ConvMergeOutputChunkHandler(uint32_t range_offset, uint16_t range_len, int8_t state_bit, void* _params) {
    ConvMergeOutputChunkHandlerParams* params = reinterpret_cast<ConvMergeOutputChunkHandlerParams*>(_params);
    my_printf_debug("output range_offset=%d range_len=%d state_bit=%d" NEWLINE, range_offset, range_len, state_bit);
    int16_t *to_offset = lea_buffer + range_offset - params->tiling_results_offset;
    uint16_t n_footprints = (range_len + BATCH_SIZE) / (BATCH_SIZE + 1);
    int16_t* footprint_buffer = lea_buffer + (LEA_BUFFER_SIZE - n_footprints) / 2 * 2;
    my_fill_q15(-state_bit, footprint_buffer, n_footprints);
    my_interleave_q15(footprint_buffer, BATCH_SIZE - (range_offset % (BATCH_SIZE + 1)), BATCH_SIZE + 1, to_offset, n_footprints);
}
#endif

#if SPARSE
void set_index(Model *model, const ParameterInfo *conv_filter, uint16_t n_output_tile_c, uint16_t n_tiles_c, uint16_t *cols, uint16_t *rows) {
#if STABLE_POWER
    uint16_t n_rows = n_output_tile_c  + 1;
    my_memcpy_from_param_row(model, rows, conv_filter, 0, (n_rows) * sizeof(int16_t));
    uint16_t n_cols = rows[n_rows - 1]; // calculate from row values
    my_memcpy_from_param_col(model, cols, conv_filter, 0, (n_cols) * sizeof(int16_t));
#else // STABLE_POWER
    uint16_t n_rows = n_tiles_c + 1;
    my_memcpy_from_param_row(model, rows, conv_filter, 0, (n_rows) * sizeof(int16_t));
    uint16_t n_cols = rows[n_rows - 1]; // calculate from row values
    my_memcpy_from_param_col(model, cols, conv_filter, 0, (n_cols) * sizeof(int16_t));
#endif // STABLE_POWER
}
#endif // SPARSE
