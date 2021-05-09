#include <cstdint>
#include <cmath>
#include "cnn_common.h"
#include "data.h"
#include "op_utils.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "platform.h"

#define RESHAPE_AUTO_DIM static_cast<uint16_t>(-1)

void alloc_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *data = input[0];
    output->slot = get_next_slot(model, data);
    output->flags &= ~TRANSPOSED;
}

void handle_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("ReLu!" NEWLINE);

    const ParameterInfo *X = input[0];

    uint16_t CHANNEL = X->dims[1];
    uint16_t OUTPUT_CHANNEL = output->dims[1];

    /* XXX: use LEA? */
    uint16_t bitwidth = X->bitwidth;
    MY_ASSERT(bitwidth == 16);
    int16_t data_len = X->params_len / (bitwidth / 8);

    uint16_t data_offset = 0;
    uint16_t output_offset = 0;
#if INTERMITTENT

    uint32_t first_unfinished_value_offset = job_index_to_offset(output, run_recovery(model, output));

#if JAPARI
    first_unfinished_value_offset -= BATCH_SIZE;
#else
    first_unfinished_value_offset -= (BATCH_SIZE - 1);
#endif
    data_offset += first_unfinished_value_offset;
    output_offset += first_unfinished_value_offset;

#if INDIRECT_RECOVERY
    uint16_t next_output_turning_point;
    int16_t offset;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           first_unfinished_value_offset, model, output);
    offset ^= 0x4000;
#endif

#endif

    if (X->flags & TRANSPOSED) {
        // input is in NWHC
        // TODO: state-aware recovery
        uint16_t H = X->dims[2], W = X->dims[3];
        uint16_t output_h = 0, output_w = 0, c = 0;
#if INTERMITTENT
        output_h = first_unfinished_value_offset / (W * CHANNEL);
        first_unfinished_value_offset %= (W * CHANNEL);
        output_w = first_unfinished_value_offset / CHANNEL;
        c = first_unfinished_value_offset % CHANNEL;
        my_printf_debug("initial output_h = %d, ", output_h);
        my_printf_debug("initial output_w = %d, ", output_w);
        my_printf_debug("initial c = %d" NEWLINE, c);
#endif

        for (; output_h < H; output_h++) {
            for (; output_w < W; output_w++) {
                    int16_t val_offset = output_w * H * CHANNEL + output_h * CHANNEL + c;
                    output_offset = output_h * W * OUTPUT_CHANNEL + output_w * OUTPUT_CHANNEL + c;
#if INDIRECT_RECOVERY
                    check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
#endif
                    uint16_t len = CHANNEL - c;
                    my_memcpy_from_param(model, lea_buffer, X, val_offset, len * sizeof(int16_t));

                    my_printf_debug("output_h=% 3d, output_w=% 3d, c=[% 3d, % 3d), val_offset=[% 6d, % 6d), input val=",
                                    output_h, output_w, c, c + len, val_offset, val_offset + len);
                    for (uint16_t idx = 0; idx < len; idx++) {
                        my_printf_debug("% 6d ", lea_buffer[idx]);
                    }

                    uint16_t output_idx = 0;
                    for (uint16_t idx = 0; idx < len; idx++) {
                        int16_t input_val = 0, output_val;
#if JAPARI
                        if ((c + idx) % (BATCH_SIZE + 1) == BATCH_SIZE) {
                            output_val = (offset ? 1 : -1);
                            if (next_output_turning_point != INVALID_TURNING_POINT && (output_offset + idx >= next_output_turning_point)) {
                                output_val = -output_val;
                            }
                        } else
#endif
                        {
                            input_val = lea_buffer[idx];
#if STATEFUL
                            // assuming input state bits are correct...
                            if (get_value_state_bit(input_val)) {
                                input_val -= 0x4000;
                            }
#endif
                            output_val = MAX_VAL(input_val, 0);
                        }
                        lea_buffer[output_idx] = output_val;
                        output_idx++;
                    }
#if STATEFUL
                    if (offset) {
                        uint8_t block_size;
                        if (next_output_turning_point == INVALID_TURNING_POINT) {
                            block_size = len;
                        } else {
                            block_size = MIN_VAL(len, next_output_turning_point - output_offset);
                        }
                        my_offset_q15_batched(lea_buffer, offset, lea_buffer, block_size);
                    } else if (next_output_turning_point < output_offset + len) {
                        int16_t* to_offset = lea_buffer + next_output_turning_point - output_offset;
                        my_offset_q15_batched(to_offset, 0x4000, to_offset, output_offset + len - next_output_turning_point);
                    }
#endif
                    my_memcpy_to_param(output, output_offset, lea_buffer, output_idx * sizeof(int16_t), 0);
#if HAWAII
                    hawaii_record_footprints(model, len);
#endif

                    my_printf_debug("output_offset=[% 6d, % 6d), output val=", output_offset, output_offset + output_idx);
#if MY_DEBUG >= 1
                    for (uint16_t idx = 0; idx < output_idx; idx++) {
                        my_printf_debug("% 6d ", lea_buffer[idx]);
                    }
#endif
                    my_printf_debug(NEWLINE);
                c = 0;
            }
            output_w = 0;
        }
    } else {
        uint16_t i = output_offset;
#if JAPARI
        uint8_t cur_batch_offset = i % (BATCH_SIZE + 1);
#else
        uint8_t cur_batch_offset = i % BATCH_SIZE;
#endif
        for (; i < data_len; i++) {
            int16_t output_val;
#if JAPARI
            if (cur_batch_offset == BATCH_SIZE) {
                cur_batch_offset -= BATCH_SIZE + 1;
                output_val = (offset ? 1 : -1);
            } else
#endif
            {
                int16_t input_val = get_q15_param(model, X, data_offset);
#if INDIRECT_RECOVERY
#if STATEFUL
                if (get_value_state_bit(input_val)) {
                    input_val -= 0x4000;
                }
#endif
                check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
#endif
                output_val = MAX_VAL(input_val, 0);
            }
#if STATEFUL
            if (cur_batch_offset == BATCH_SIZE - 1) {
                cur_batch_offset -= BATCH_SIZE;
                output_val += offset;
            }
#endif
            my_printf_debug("output_offset=%d output_val=%d" NEWLINE, output_offset, output_val);
            put_q15_param(output, output_offset, output_val);
#if HAWAII
            if (cur_batch_offset == BATCH_SIZE - 1) {
                write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
                cur_batch_offset -= BATCH_SIZE;
            }
#endif
            data_offset++;
            output_offset++;
            cur_batch_offset++;
        }
    }

    flip_state_bit(model, output);

    my_printf_debug("handle_relu output" NEWLINE);
    if (X->flags & TRANSPOSED) {
        dump_params_nhwc_debug(model, output);
    } else {
        dump_params_debug(model, output);
    }
}

void handle_backward_relu(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    ERROR_OCCURRED();
}

void handle_reshape(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("Reshape!" NEWLINE);

    const ParameterInfo *data = input[0], *shape = input[1];
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    if (cur_slot_info) {
        cur_slot_info->user = model->layer_idx;
    }
    MY_ASSERT(shape->bitwidth == 64);
    /*
     * At most one dimension of the new shape can be -1. In this case, the
     * value is inferred from the size of the tensor and the remaining
     * dimensions.
     *
     * A dimension could also be 0, in which case the actual dimension value
     * is unchanged (i.e. taken from the input tensor).
     * */
    uint32_t new_len = 1;
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = get_int64_param(shape, i);
        if (!output->dims[i]) {
            output->dims[i] = data->dims[i];
        }
        if (output->dims[i] != RESHAPE_AUTO_DIM) {
            new_len *= output->dims[i];
        }
    }
    for (uint8_t i = shape->dims[0]; i < 4; i++) {
        output->dims[i] = 0;
    }
    uint16_t inferred_dim = output->params_len / sizeof(int16_t);
    int8_t auto_idx = -1;
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i] != RESHAPE_AUTO_DIM && output->dims[i] != 0) {
            inferred_dim /= output->dims[i];
        } else if (output->dims[i] == RESHAPE_AUTO_DIM) {
            auto_idx = i;
        }
    }
    if (auto_idx != -1) {
        output->dims[auto_idx] = inferred_dim;
        new_len *= inferred_dim;
    }
#if JAPARI
    else {
        new_len = extend_for_footprints(new_len);
    }
#endif
    MY_ASSERT(new_len * sizeof(int16_t) == output->params_len);
}

void handle_backward_reshape(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    ERROR_OCCURRED();
}

void handle_squeeze(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    my_printf_debug("Squeeze!" NEWLINE);

    const ParameterInfo *data = input[0];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth = data->bitwidth;
    output->slot = data->slot;
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    if (cur_slot_info) {
        cur_slot_info->user = model->layer_idx;
    }
    uint8_t axes = flags->extra.squeeze.axes;
    // If axes is not provided, all the single dimensions will be removed from the shape.
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#squeeze
    uint8_t j = 0;
    if (axes == 0) {
        for (uint8_t i = 0; i < 4; i++) {
            if (input[0]->dims[i] != 1) {
                output->dims[j] = input[0]->dims[i];
                j++;
            }
        }
    } else {
        for (uint8_t i = 0; i < 4; i++) {
            if (axes & (1 << i)) {
                MY_ASSERT(input[0]->dims[i] == 1);
            } else {
                output->dims[j] = input[0]->dims[i];
                j++;
            }
        }
    }
    for (; j < 4; j++) {
        output->dims[j] = 0;
    }
}

void handle_backward_squeeze(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    ERROR_OCCURRED();
}

void alloc_concat(Model *, const ParameterInfo *[], ParameterInfo*, const NodeFlags*) {
}

void handle_concat(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("Concat!" NEWLINE);

    const ParameterInfo *A = input[0], *B = input[1];
    // XXX: assume concatenating 2 tensors at the CHANNEL dimension and they
    // have the same number of channels.
    MY_ASSERT(A->dims[1] == B->dims[1]);
    output->dims[1] *= 2;
    output->flags |= SEPARATE_TILING;

    // The one with smaller `scale` (with larger values) is scaled down
    output->scale = MAX_VAL(A->scale, B->scale);

    // saving slots here as it might be changed during the downscaling loop above
    output->extra_info[0] = A->parameter_info_idx;
    output->extra_info[1] = B->parameter_info_idx;
    output->slot = A->slot;

    dump_params_nhwc_debug(model, A);
    dump_params_nhwc_debug(model, B);
}

void handle_backward_concat(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    ERROR_OCCURRED();
}

void alloc_softmax(Model *model, const ParameterInfo **input, ParameterInfo *output, const struct NodeFlags*) {
    const ParameterInfo* X = input[0];
    output->slot = get_next_slot(model, X);
}

void handle_softmax(Model* model, const ParameterInfo* input[], ParameterInfo* output, const NodeFlags*) {
    my_printf_debug("Softmax!" NEWLINE);

    const ParameterInfo* X = input[0];

    const int16_t data_len = X->params_len / sizeof(int16_t);
    int16_t *buffer_input = lea_buffer;
    float *buffer_input_exp = reinterpret_cast<float*>(buffer_input + data_len);
    MY_ASSERT(reinterpret_cast<int16_t*>(buffer_input_exp) + data_len < lea_buffer + LEA_BUFFER_SIZE);
    my_memcpy_from_param(model, buffer_input, X, 0, X->params_len);

    float denominator = 0;

    for (uint8_t idx = 0; idx < data_len; idx++) {
        buffer_input_exp[idx] = exp(q15_to_float(buffer_input[idx], ValueInfo(X)));
        MY_ASSERT(!std::isnan(buffer_input_exp[idx]));
        denominator += buffer_input_exp[idx];
    }

    for (uint8_t idx = 0; idx < data_len; idx++) {
        buffer_input[idx] = static_cast<int16_t>(buffer_input_exp[idx] / denominator * 0x8000);
    }

    output->scale = 1;

    my_memcpy_to_param(output, 0, buffer_input, data_len * sizeof(int16_t), 0);

    dump_matrix_debug(buffer_input, data_len, ValueInfo(output));
}

void handle_backward_softmax(Model* model, const ParameterInfo* input[], ParameterInfo* output, const NodeFlags*) {
    const uint8_t *labels = labels_data;
    put_q15_param(output, labels[0], get_q15_param(model, output, labels[0]) - 0x8000);
    dump_params(model, output);
}

void handle_transpose(Model*, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("Transpose!" NEWLINE);

    const ParameterInfo *X = input[0];
    // not actually transpose data as we happen to need NHWC
    // XXX: assume NHWC -> NCHW
    output->dims[1] = X->dims[3];
    output->dims[2] = X->dims[1];
    output->dims[3] = X->dims[2];
}

void handle_backward_transpose(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    ERROR_OCCURRED();
}

void alloc_batchnormalization(Model* model, const ParameterInfo* input[], ParameterInfo* output, const NodeFlags* flags) {
    const ParameterInfo* X = input[0];
    output->slot = get_next_slot(model, X);
}

void handle_batchnormalization(Model* model, const ParameterInfo* input[], ParameterInfo* output, const NodeFlags* flags) {
    my_printf_debug("BatchNormalization!" NEWLINE);

    const ParameterInfo *X = input[0], *scale = input[1], *B = input[2], *mean = input[3], *var = input[4];
    const uint16_t CHANNEL = X->dims[1], H = X->dims[2], W= X->dims[3];
    int16_t *buffer_x = lea_buffer,
            *buffer_scale = buffer_x + CHANNEL,
            *buffer_b = buffer_scale + CHANNEL,
            *buffer_mean = buffer_b + CHANNEL,
            *buffer_var = buffer_mean + CHANNEL;
    MY_ASSERT(buffer_var + CHANNEL < lea_buffer + LEA_BUFFER_SIZE);

    my_memcpy_from_param(model, buffer_scale, scale, 0, CHANNEL * sizeof(int16_t));
    my_memcpy_from_param(model, buffer_b, B, 0, CHANNEL * sizeof(int16_t));
    my_memcpy_from_param(model, buffer_mean, mean, 0, CHANNEL * sizeof(int16_t));
    my_memcpy_from_param(model, buffer_var, var, 0, CHANNEL * sizeof(int16_t));

    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, 1.0f * mean->scale / (X->scale * 2));
    my_scale_q15(buffer_mean, scaleFract, shift, buffer_mean, CHANNEL);

    int16_t var_scale_sqrt = static_cast<int16_t>(sqrtf(1.0f * var->scale));
    MY_ASSERT(var_scale_sqrt * var_scale_sqrt == var->scale);
    uint16_t max_multiplier = find_max_multiplier(model, var, nullptr, 0, var_scale_sqrt);
    var_scale_sqrt /= max_multiplier;

    float_to_scale_params(&scaleFract, &shift, 1.0f * var_scale_sqrt / (X->scale * 2));
    my_scale_q15(buffer_b, scaleFract, shift, buffer_b, CHANNEL);

    output->scale = scale->scale * (X->scale * 2) / var_scale_sqrt;

    // assume conventional epsilon
    my_offset_q15(buffer_var, static_cast<int16_t>(0.00001 * 0x8000 / var->scale), buffer_var, CHANNEL);
    my_vsqrt_q15(buffer_var, buffer_var, CHANNEL);

    float_to_scale_params(&scaleFract, &shift, max_multiplier);
    my_scale_q15(buffer_var, scaleFract, shift, buffer_var, CHANNEL);

    uint32_t offset = 0;
    for (uint16_t idx = 0; idx < H * W; idx++) {
        my_memcpy_from_param(model, buffer_x, X, offset, CHANNEL * sizeof(int16_t));

        // FIXME: work around overflows. Needs a better approach
        float_to_scale_params(&scaleFract, &shift, 0.5);
        my_scale_q15(buffer_x, scaleFract, shift, buffer_x, CHANNEL);

        my_printf_debug("(h, w) = (%d, %d)" NEWLINE, idx / W, idx % W);

        my_sub_q15(buffer_x, buffer_mean, buffer_x, CHANNEL);
        my_printf_debug("x - mean" NEWLINE);
        dump_matrix_debug(buffer_x, CHANNEL, ValueInfo(X->scale * 2));

        my_div_q15(buffer_x, buffer_var, buffer_x, CHANNEL);
        my_printf_debug("(x - mean)/sqrt(var+epsilon)" NEWLINE);
        dump_matrix_debug(buffer_x, CHANNEL, ValueInfo((X->scale * 2) / var_scale_sqrt));

        my_mpy_q15(buffer_x, buffer_scale, buffer_x, CHANNEL);
        my_printf_debug("(x - mean)/sqrt(var+epsilon)*scale" NEWLINE);
        dump_matrix_debug(buffer_x, CHANNEL, ValueInfo(output, model));

        my_add_q15(buffer_x, buffer_b, buffer_x, CHANNEL);
        my_printf_debug("(x - mean)/sqrt(var+epsilon)*scale+B" NEWLINE);
        dump_matrix_debug(buffer_x, CHANNEL, ValueInfo(output, model));

        my_memcpy_to_param(output, offset, buffer_x, CHANNEL * sizeof(int16_t), 0);
        offset += CHANNEL;
    }

    my_printf_debug("handle_batchnormalization output" NEWLINE);
    dump_params_nhwc_debug(model, output);
}

void handle_backward_batchnormalization(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    ERROR_OCCURRED();
}
