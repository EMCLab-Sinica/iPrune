#include <cstdint>
#include "data.h"
#include "cnn_common.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "op_utils.h"
#include "my_dsplib.h"
#include "platform.h"

struct MaxPoolParams {
    uint16_t output_h;
    uint16_t output_w;
    uint16_t start_channel;
    uint8_t n_channels;
    uint8_t need_nhwc2nchw;
    uint16_t new_H;
    uint16_t new_W;
    const MaxPoolFlags* flags;
    const ParameterInfo *data;
    const ParameterInfo *output;
    Model *model;
};
static MaxPoolParams maxpool_params_obj;

enum {
    KERNEL_SHAPE_H = 0,
    KERNEL_SHAPE_W = 1,
    STRIDE_H = 0,
    STRIDE_W = 1,
};

void alloc_maxpool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {

    const ParameterInfo *data = input[0];

    const uint16_t H = data->dims[2], W = data->dims[3];
    uint16_t CHANNEL = data->dims[1];

    MaxPoolParams* maxpool_params = &maxpool_params_obj;
    maxpool_params->flags = &(node->flags.extra.maxpool);

    maxpool_params->new_H = H / maxpool_params->flags->strides[STRIDE_H];
    maxpool_params->new_W = W / maxpool_params->flags->strides[STRIDE_W];
    maxpool_params->need_nhwc2nchw = (node->flags.generic == NHWC2NCHW);

    output->params_len = maxpool_params->new_H * maxpool_params->new_W * CHANNEL * sizeof(int16_t);
    output->slot = get_next_slot(model, data);
    output->dims[0] = 1;
    output->dims[1] = CHANNEL;
    output->dims[2] = maxpool_params->new_H;
    output->dims[3] = maxpool_params->new_W;
    if (maxpool_params->need_nhwc2nchw) {
        output->param_flags |= CHANNEL_FIRST;
    }
}

static uint8_t maxpool_patch(MaxPoolParams *maxpool_params) {
    const uint16_t CHANNEL = maxpool_params->data->dims[1], W = maxpool_params->data->dims[3];

    int16_t offset_h, offset_w;
    offset_h = W * CHANNEL;
    offset_w = CHANNEL;

    my_printf_debug("output_h=% 3d ", maxpool_params->output_h);
    my_printf_debug("output_w=% 3d ", maxpool_params->output_w);
    my_printf_debug("c=[% 3d, % 3d) ", maxpool_params->start_channel, maxpool_params->start_channel + maxpool_params->n_channels);

    int16_t* const input_buffer = lea_buffer + maxpool_params->n_channels;
    int16_t* const output_buffer = lea_buffer;
    my_fill_q15(INT16_MIN, output_buffer, maxpool_params->n_channels);

    // explicitly initialize this as -Wmaybe-uninitialized may be triggered with -O3
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60165
    uint8_t output_channel_offset = 0;

    for (uint16_t sH = 0; sH < maxpool_params->flags->kernel_shape[KERNEL_SHAPE_H]; sH++) {
        for (uint16_t sW = 0; sW < maxpool_params->flags->kernel_shape[KERNEL_SHAPE_W]; sW++) {
            uint16_t val_offset = (maxpool_params->output_h*maxpool_params->flags->strides[STRIDE_H]+sH) * offset_h +
                                  (maxpool_params->output_w*maxpool_params->flags->strides[STRIDE_W]+sW) * offset_w + maxpool_params->start_channel;
            my_memcpy_from_param(maxpool_params->model, input_buffer, maxpool_params->data, val_offset, maxpool_params->n_channels * sizeof(int16_t));
            output_channel_offset = 0;
            for (uint8_t input_channel_offset = 0; input_channel_offset < maxpool_params->n_channels; input_channel_offset++) {
                int16_t val = input_buffer[input_channel_offset];
                // dump_value_debug(model, maxpool_params->data, val_offset);
                my_printf_debug("% 6d ", val);
                if (val > output_buffer[output_channel_offset]) {
                    output_buffer[output_channel_offset] = val;
                }
                output_channel_offset++;
            }
            my_printf_debug("; ");
        }
    }
    return output_channel_offset;
}

#if STATEFUL
static inline void offset_vector(int16_t* const buffer, int16_t offset, uint8_t len, const uint16_t output_offset, const uint16_t next_output_turning_point) {
    int16_t cur_offset = offset;
    for (uint8_t idx = BATCH_SIZE - 1; idx < len; idx += BATCH_SIZE) {
        if (output_offset + idx == next_output_turning_point + BATCH_SIZE - 1) {
            cur_offset = -cur_offset;
        }
        buffer[idx] += cur_offset;
    }
}
#endif
#if JAPARI
static inline void offset_vector(int16_t* const buffer, int16_t offset, uint8_t len, const uint16_t output_offset, const uint16_t next_output_turning_point) {
    int16_t cur_footprint = (offset == 0x4000 ? 1 : -1);
    uint8_t reverted = 0;
    for (uint8_t idx = BATCH_SIZE; idx < len; idx += BATCH_SIZE + 1) {
        if (output_offset + idx >= next_output_turning_point && !reverted) {
            cur_footprint = -cur_footprint;
            reverted = 1;
        }
        buffer[idx] = cur_footprint;
    }
}
#endif

void handle_maxpool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("MaxPool!" NEWLINE);

    /* XXX: add flags; assume no padding for now */
    const ParameterInfo *data = input[0];

    MaxPoolParams* maxpool_params = &maxpool_params_obj;
    maxpool_params->data = data;
    maxpool_params->output = output;
    maxpool_params->model = model;

    const uint16_t CHANNEL = data->dims[1], OUTPUT_CHANNEL = output->dims[1];

    uint16_t output_h = 0, output_w = 0, c = 0;
    uint16_t output_offset = 0;

#if INTERMITTENT
    uint32_t first_unfinished_value_offset = batch_start(job_index_to_offset(output, run_recovery(model, output)));
    if (first_unfinished_value_offset * sizeof(int16_t) == output->params_len) {
        // give up early, or initial_real_tile_c may be zero and results in SIGFPE
        goto finished;
    }

    uint16_t initial_c, initial_h, initial_w;

    output_offset = first_unfinished_value_offset;
    if (!maxpool_params->need_nhwc2nchw) {
        initial_c = first_unfinished_value_offset % OUTPUT_CHANNEL;
        first_unfinished_value_offset /= OUTPUT_CHANNEL;
        initial_w = first_unfinished_value_offset % maxpool_params->new_W;
        first_unfinished_value_offset /= maxpool_params->new_W;
        initial_h = first_unfinished_value_offset % maxpool_params->new_H;
    } else {
        initial_w = first_unfinished_value_offset % maxpool_params->new_W;
        first_unfinished_value_offset /= maxpool_params->new_W;
        initial_h = first_unfinished_value_offset % maxpool_params->new_H;
        first_unfinished_value_offset /= maxpool_params->new_H;
        initial_c = first_unfinished_value_offset % OUTPUT_CHANNEL;
    }
    output_h = initial_h;
    output_w = initial_w;
    c = initial_c;
    my_printf_debug("initial_h = %d" NEWLINE, initial_h);
    my_printf_debug("initial_w = %d" NEWLINE, initial_w);
    my_printf_debug("initial_c = %d" NEWLINE, initial_c);
#endif

    {
        if (!maxpool_params->need_nhwc2nchw) {
            // NHWC
            for (; output_h < maxpool_params->new_H; output_h++) {
                maxpool_params->output_h = output_h;
                for (; output_w < maxpool_params->new_W; output_w++) {
                    uint8_t len = OUTPUT_CHANNEL - c;
                    maxpool_params->output_w = output_w;
                    maxpool_params->n_channels = len;
                    maxpool_params->start_channel = c;
                    len = maxpool_patch(maxpool_params);
                    my_printf_debug("output_offset=[% 5d, % 5d) ", output_offset, output_offset + len);
#if MY_DEBUG >= MY_DEBUG_VERBOSE
                    // need a space as dump_value does not append spaces when DUMP_INTEGERS is not defined
                    my_printf_debug(" max=");
                    for (uint8_t idx = 0; idx < len; idx++) {
                        my_printf_debug("% 6d ", lea_buffer[idx]);
                    }
                    my_printf_debug(NEWLINE);
#endif
                    my_memcpy_to_param(output, output_offset, lea_buffer, len * sizeof(int16_t), 0);
#if HAWAII
                    hawaii_record_footprints(model, len);
#endif
                    output_offset += len;
                    c = 0;
                }
                output_w = 0;
            }
            output_h = 0;
        } else {
            // NCHW
            uint8_t channel_stride = 1;
            for (; c < CHANNEL; c += channel_stride) {
                for (; output_h < maxpool_params->new_H; output_h++) {
                    maxpool_params->output_h = output_h;
                    maxpool_params->output_w = output_w;
                    for (; output_w < maxpool_params->new_W; output_w++) {
                        maxpool_params->start_channel = c;
                        maxpool_params->n_channels = 1;
                        uint8_t len = maxpool_patch(maxpool_params);
                        if (!len) {
                            my_printf_debug(NEWLINE);
                            continue;
                        }
                        my_printf_debug("output_offset=% 5d ", output_offset);
                        my_printf_debug("max=% 6d " NEWLINE, lea_buffer[0]);
                        put_q15_param(output, output_offset, lea_buffer[0]);
#if HAWAII
                        if (offset_has_state(output_offset)) {
                            write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
                        }
#endif
                        output_offset++;
                        maxpool_params->output_w++;
                    }
                    output_w = 0;
                }
                output_h = 0;
            }
            c = 0;
        }
    }

    MY_ASSERT(output_offset == output->params_len / sizeof(int16_t),
              "Expect output offset %d, got %d" NEWLINE, output->params_len / sizeof(int16_t), output_offset);

#if INTERMITTENT
finished:
    flip_state_bit(model, output);
#endif

    my_printf_debug("handle_maxpool output" NEWLINE);
    if (!maxpool_params->need_nhwc2nchw) {
        dump_params_nhwc_debug(model, output, node->output_name);
    } else {
        dump_params_debug(model, output, node->output_name);
    }
}

void alloc_globalaveragepool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    const ParameterInfo *data = input[0];

    MY_ASSERT(data->dims[0] == 1);
    uint16_t output_len = data->dims[1];

    output->dims[0] = output->dims[2] = output->dims[3] = 1;
    output->dims[1] = output_len;
    output->params_len = output_len * sizeof(int16_t);
    output->bitwidth = 16;
    output->slot = get_next_slot(model, data);
}

void handle_globalaveragepool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("GlobalAveragePool!" NEWLINE);

    const ParameterInfo *data = input[0];

#if STATEFUL
    int16_t offset;
    uint16_t next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, 0 /*TODO: first_unfinished_value_offset*/, model, output);
    offset = -offset;
#endif

    uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t len = H * W;
    uint16_t output_channel = 0;
    for (uint16_t input_channel = 0; input_channel < CHANNEL; input_channel++) {
        int16_t output_val;
#if JAPARI
        if (offset_has_state(input_channel)) {
            output_val = -param_state_bit(model, output, output_channel);
        } else
#endif
        {
            uint32_t total = 0;
            for (uint16_t h = 0; h < H; h++) {
                for (uint16_t w = 0; w < W; w++) {
                    // Input is from Conv, which uses NHWC
                    int16_t val = get_q15_param(model, data, h * W * CHANNEL + w * CHANNEL + input_channel);
#if STATEFUL
                    start_cpu_counter();
                    if (offset_has_state(input_channel)) {
                        strip_state(&val);
                    }
                    val *= 2;
                    stop_cpu_counter(&Counters::stripping);
#endif
                    total += val;
                }
            }
            output_val = total / len;
#if STATEFUL
            output_val /= 2;
            if (offset_has_state(output_channel)) {
                check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_channel);
                output_val += offset;
            }
#endif
        }
        put_q15_param(output, output_channel, output_val);
        output_channel++;
    }

    flip_state_bit(model, output);

    dump_params_debug(model, output, node->output_name);
}
