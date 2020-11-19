#include <stdint.h>
#include "data.h"
#include "cnn_common.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "op_utils.h"

void alloc_maxpool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    uint16_t stride = flags->stride;

    const ParameterInfo *data = input[0];

    const uint16_t H = data->dims[2], W = data->dims[3];
    uint16_t CHANNEL = data->dims[1];
    uint16_t new_H = H / stride;
    uint16_t new_W = W / stride;
#if JAPARI
    CHANNEL = CHANNEL / TILE_C_WITH_FOOTPRINTS * DEFAULT_TILE_C;
#endif

    output->params_len = new_H * new_W * CHANNEL * sizeof(int16_t);
    output->slot = get_next_slot(model, data);
    output->dims[0] = 1;
    output->dims[1] = CHANNEL;
    output->dims[2] = new_H;
    output->dims[3] = new_W;
    output->flags |= NO_FOOTPRINTS;
}

static int16_t maxpool_patch(uint16_t output_h, uint16_t output_w, uint16_t c, const NodeFlags* flags, const ParameterInfo *data, Model *model) {
    const uint16_t CHANNEL = data->dims[1], W = data->dims[3];
    uint16_t stride = flags->stride;
    uint16_t kernel_size = flags->kernel_size;

    int16_t offset_h, offset_w;
    offset_h = W * CHANNEL;
    offset_w = CHANNEL;

    my_printf_debug("output_h=% 3d ", output_h);
    my_printf_debug("output_w=% 3d ", output_w);
    my_printf_debug("c=% 3d ", c);

    int16_t max_val = INT16_MIN;
    for (uint16_t sH = 0; sH < kernel_size; sH++) {
        for (uint16_t sW = 0; sW < kernel_size; sW++) {
            uint16_t val_offset = (output_h*stride+sH) * offset_h + (output_w*stride+sW) * offset_w + c;
            int16_t val = get_q15_param(model, data, val_offset);
#if STATEFUL
            if (get_value_state_bit(val)) {
                // assuming input state bits are correct...
                val -= 0x4000;
            }
#endif
            // dump_value_debug(model, data, val_offset);
            my_printf_debug("% 5d ", val);
            // XXX: use LEA?
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    // need a space as dump_value does not append spaces when DUMP_INTEGERS is not defined
    my_printf_debug(" max=% 5d ", max_val);
    return max_val;
}

void handle_maxpool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    my_printf_debug("MaxPool!" NEWLINE);

    uint16_t stride = flags->stride;
    uint8_t need_nhwc2nchw = (flags->generic == NHWC2NCHW);

    /* XXX: add flags; assume no padding for now */
    const ParameterInfo *data = input[0];

    my_printf_debug("handle_maxpool input" NEWLINE);
    dump_params_nhwc_debug(model, data, 0);

    const uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t new_H = H / stride;
    uint16_t new_W = W / stride;

    determine_tile_c(output);
    uint16_t tile_c = output->tile_c;
#if JAPARI
    tile_c = extend_for_footprints(tile_c);
#endif
    my_printf_debug("tile_c = %d" NEWLINE, tile_c);

    uint16_t tile_c_offset = 0;

    uint16_t output_h = 0, output_w = 0, c = 0;
    uint16_t output_offset = 0;

#if INTERMITTENT
    uint16_t initial_real_tile_c;

    uint32_t first_unfinished_value_offset = run_recovery(model, output);
    if (first_unfinished_value_offset * sizeof(int16_t) == output->params_len) {
        // give up early, or initial_real_tile_c may be zero and results in SIGFPE
        goto finished;
    }

    uint16_t initial_n, initial_c, initial_h, initial_w;
    initial_n = first_unfinished_value_offset / (new_H * new_W * tile_c);

    tile_c_offset = initial_n * tile_c;

#if STATEFUL
    int16_t offset, next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, first_unfinished_value_offset, model, output);
    offset ^= 0x4000;
#endif

    initial_real_tile_c = MIN_VAL(tile_c, CHANNEL - tile_c_offset);
    output_offset = first_unfinished_value_offset;
    if (!need_nhwc2nchw) {
        initial_c = first_unfinished_value_offset % initial_real_tile_c;
        first_unfinished_value_offset /= initial_real_tile_c;
        initial_w = first_unfinished_value_offset % new_W;
        first_unfinished_value_offset /= new_W;
        initial_h = first_unfinished_value_offset % new_H;
    } else {
        initial_w = first_unfinished_value_offset % new_W;
        first_unfinished_value_offset /= new_W;
        initial_h = first_unfinished_value_offset % new_H;
        first_unfinished_value_offset /= new_H;
        initial_c = first_unfinished_value_offset % initial_real_tile_c;
    }
    output_h = initial_h;
    output_w = initial_w;
    c = initial_c;
    my_printf_debug("initial_n = %d" NEWLINE, initial_n);
    my_printf_debug("initial_h = %d" NEWLINE, initial_h);
    my_printf_debug("initial_w = %d" NEWLINE, initial_w);
    my_printf_debug("initial_c = %d" NEWLINE, initial_c);
#endif

    for (; tile_c_offset < CHANNEL; tile_c_offset += tile_c) {
        uint16_t cur_tile_c = MIN_VAL(tile_c, CHANNEL - tile_c_offset);
        if (!need_nhwc2nchw) {
            // NHWC
            for (; output_h < new_H; output_h++) {
                for (; output_w < new_W; output_w++) {
                    for (; c < cur_tile_c; c++) {
#if JAPARI
                        if (is_footprint_channel(c) || is_footprint_padding_channel(c)) {
                            continue;
                        }
#endif
                        int16_t max_val = maxpool_patch(output_h, output_w, c + tile_c_offset, flags, data, model);
                        my_printf_debug("output_offset=%d" NEWLINE, output_offset);
#if STATEFUL
                        check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
                        max_val += offset;
#endif
                        put_q15_param(output, output_offset, max_val);
#if HAWAII
                        write_hawaii_layer_footprint(model->layer_idx, 1);
#endif
                        output_offset++;
                    }
                    c = 0;
                }
                output_w = 0;
            }
            output_h = 0;
        } else {
            // NCHW
            uint8_t channel_stride = 1;
            for (; c < cur_tile_c; c += channel_stride) {
#if JAPARI
                if (is_footprint_channel(c) || is_footprint_padding_channel(c)) {
                    continue;
                }
#endif
                for (; output_h < new_H; output_h++) {
                    for (; output_w < new_W; output_w++) {
                        int16_t max_val = maxpool_patch(output_h, output_w, c + tile_c_offset, flags, data, model);
                        my_printf_debug("output_offset=%d" NEWLINE, output_offset);
#if STATEFUL
                        check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
                        max_val += offset;
#endif
                        put_q15_param(output, output_offset, max_val);
#if HAWAII
                        write_hawaii_layer_footprint(model->layer_idx, 1);
#endif
                        output_offset++;
                    }
                    output_w = 0;
                }
                output_h = 0;
            }
            c = 0;
        }
    }

    MY_ASSERT(output_offset == output->params_len / sizeof(int16_t));

#if INTERMITTENT
finished:
#if STATEFUL
    flip_state_bit(model, output);
#endif
#endif

    my_printf_debug("handle_maxpool output" NEWLINE);
    if (!need_nhwc2nchw) {
        dump_params_nhwc_debug(model, output, 0);
    } else if (tile_c == CHANNEL) {
        dump_params_debug(model, output);
    }
}

void alloc_globalaveragepool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *data = input[0];

    MY_ASSERT(data->dims[0] == 1);
    uint16_t output_len = data->dims[1];

    output->dims[0] = output->dims[2] = output->dims[3] = 1;
    output->dims[1] = data->dims[1];
    output->params_len = output_len * sizeof(int16_t);
    output->bitwidth = 16;
    output->slot = get_next_slot(model, data);
}

void handle_globalaveragepool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("GlobalAveragePool!" NEWLINE);

    const ParameterInfo *data = input[0];

#if STATEFUL
    int16_t offset, next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, 0 /*TODO: first_unfinished_value_offset*/, model, output);
    offset ^= 0x4000;
#endif

    uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t len = H * W;
    for (uint16_t c = 0; c < CHANNEL; c++) {
        uint32_t total = 0;
        for (uint16_t h = 0; h < H; h++) {
            for (uint16_t w = 0; w < W; w++) {
                // Input is from Conv, which uses NHWC
                int16_t val = get_q15_param(model, data, h * W * CHANNEL + w * CHANNEL + c);
#if STATEFUL
                if (get_value_state_bit(val)) {
                    val -= 0x4000;
                }
#endif
                total += val;
            }
        }
        int16_t avg = total / len;
#if STATEFUL
        check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, c);
        avg += offset;
#endif
        put_q15_param(output, c, avg);
    }

#if STATEFUL
    flip_state_bit(model, output);
#endif

    dump_params_debug(model, output);
}
