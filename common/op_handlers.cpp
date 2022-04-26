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

void alloc_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    const ParameterInfo *data = input[0];
    output->slot = get_next_slot(model, data);
}

void handle_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("ReLu!" NEWLINE);

    const ParameterInfo *X = input[0];

    /* XXX: use LEA? */
    uint16_t bitwidth = X->bitwidth;
    MY_ASSERT(bitwidth == 16);
    int16_t data_len = X->params_len / (bitwidth / 8);

#if !STABLE_POWER
    // FIXME: After removing FC merge, this branch can be removed
    if(X->dims[2] != 0) // conv
        data_len = X->dims[0] * X->dims[1] * X->dims[2] * X->dims[3];
#endif // STABLE_POWER
    my_printf_debug("data_len: %d" NEWLINE, data_len);
    uint16_t data_offset = 0;
    uint16_t output_offset = 0;
#if INTERMITTENT

    uint32_t first_unfinished_value_offset = batch_start(job_index_to_offset(output, run_recovery(model, output)));
    data_offset += first_unfinished_value_offset;
    output_offset += first_unfinished_value_offset;

#if INDIRECT_RECOVERY
    uint16_t next_output_turning_point;
    int16_t offset;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           first_unfinished_value_offset, model, output);
    offset = -offset;
#endif

#endif

    {
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
                output_val = (offset > 0? 1 : -1);
            } else
#endif
            {
                int16_t input_val = get_q15_param(model, X, data_offset);
#if INDIRECT_RECOVERY
#if STATEFUL
                start_cpu_counter();
                if (offset_has_state(data_offset)) {
                    strip_state(&input_val);
                }
                input_val *= 2;
                stop_cpu_counter(&Counters::stripping);
#endif
                check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
#endif
                output_val = MAX_VAL(input_val, 0);
            }
#if STATEFUL
            start_cpu_counter();
            output_val /= 2;
            if (cur_batch_offset == BATCH_SIZE - 1) {
                cur_batch_offset -= BATCH_SIZE;
                output_val += offset;
            }
            stop_cpu_counter(&Counters::embedding);
#endif
            my_printf_debug("input_offset=%d output_offset=%d output_val=%d" NEWLINE, data_offset, output_offset, output_val);
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
    dump_params_nhwc_debug(model, output, node->output_name);
}

void handle_reshape(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    my_printf_debug("Reshape!" NEWLINE);

    const ParameterInfo *data = input[0], *shape = input[1];
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
            {
                inferred_dim /= output->dims[i];
            }
        } else if (output->dims[i] == RESHAPE_AUTO_DIM) {
            auto_idx = i;
        }
    }
    if (auto_idx != -1) {
        output->dims[auto_idx] = inferred_dim;
        new_len *= inferred_dim;
    }
    my_printf_debug("new_len: %d" NEWLINE, new_len);
    my_printf_debug("output->params_len: %d" NEWLINE, output->params_len);
    // MY_ASSERT(new_len * sizeof(int16_t) == output->params_len);
}

void handle_squeeze(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("Squeeze!" NEWLINE);

    uint8_t axes = node->flags.extra.squeeze.axes;
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

void handle_unsqueeze(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node) {
    my_printf_debug("Unsqueeze!" NEWLINE);
    uint8_t axes = node->flags.extra.squeeze.axes;
    uint8_t input_dim_offset = 0, output_dim_offset = 0;
    for (uint8_t i = 0; i < 4; i++) {
        if (axes & (1 << i)) {
            output->dims[output_dim_offset] = 1;
            output_dim_offset++;
        } else {
            output->dims[output_dim_offset] = input[0]->dims[input_dim_offset];
            input_dim_offset++;
            output_dim_offset++;
        }
    }
}

void alloc_concat(Model *, const ParameterInfo *[], ParameterInfo*, const Node*) {
}

void handle_concat(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    my_printf_debug("Concat!" NEWLINE);

    const ParameterInfo *A = input[0], *B = input[1];
    // XXX: assume concatenating 2 tensors at the CHANNEL dimension and they
    // have the same number of channels.
    MY_ASSERT(A->dims[1] == B->dims[1]);
    output->dims[1] *= 2;
    output->param_flags |= SEPARATE_TILING;

    // The one with smaller `scale` (with larger values) is scaled down
    output->scale = MAX_VAL(A->scale, B->scale);

    // saving slots here as it might be changed during the downscaling loop above
    output->extra_info[0] = A->parameter_info_idx;
    output->extra_info[1] = B->parameter_info_idx;
    output->slot = A->slot;

    dump_params_nhwc_debug(model, A);
    dump_params_nhwc_debug(model, B);
}

void handle_softmax(Model*, const ParameterInfo*[], ParameterInfo*, const Node*) {
    // Do nothing - softmax does not change the relative order of values.
    // Just let run_model determine the max value
}

void handle_transpose(Model*, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    my_printf_debug("Transpose!" NEWLINE);

    const ParameterInfo *X = input[0];
    // not actually transpose data as we happen to need NHWC
    // XXX: assume NHWC -> NCHW
    output->dims[1] = X->dims[3];
    output->dims[2] = X->dims[1];
    output->dims[3] = X->dims[2];
}

void alloc_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node) {
    output->slot = get_next_slot(model, input[0]);
}

void handle_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node) {
    my_printf_debug("Add!" NEWLINE);

    const ParameterInfo *X = input[0], *Y = input[1];
    uint16_t buffer_size = X->dims[1];
    int16_t *buffer_a = lea_buffer,
            *buffer_b = buffer_a + buffer_size;
    my_memcpy_from_param(model, buffer_b, Y, 0, buffer_size * sizeof(int16_t));

    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, 1.0f*Y->scale/X->scale);
    my_scale_q15(buffer_b, scaleFract, shift, buffer_b, buffer_size);

    for (uint16_t idx = 0; idx < X->dims[2]; idx++) {
        my_memcpy_from_param(model, buffer_a, X, idx*buffer_size, buffer_size * sizeof(int16_t));
        my_add_q15(buffer_a, buffer_b, buffer_a, buffer_size);
        my_memcpy_to_param(output, idx*buffer_size, buffer_a, buffer_size * sizeof(int16_t), 0);
    }
    dump_params_nhwc_debug(model, output, node->output_name);
}

void alloc_batchnormalization(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node*) {
    const ParameterInfo* X = input[0];
    output->slot = get_next_slot(model, X);
}

void handle_batchnormalization(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node*) {
    my_printf_debug("BatchNormalization!" NEWLINE);

    const ParameterInfo *X = input[0], *scale = input[1], *B = input[2], *mean = input[3], *var = input[4];
    const uint16_t CHANNEL = X->dims[1], H = X->dims[2], W= X->dims[3];
    int16_t *buffer_x = lea_buffer,
            *buffer_scale = buffer_x + CHANNEL,
            *buffer_b = buffer_scale + CHANNEL,
            *buffer_mean = buffer_b + CHANNEL,
            *buffer_var = buffer_mean + CHANNEL;
    MY_ASSERT(buffer_var + CHANNEL < lea_buffer + LEA_BUFFER_SIZE);

    uint32_t offset = 0;
    uint16_t idx = 0;
#if INTERMITTENT
    uint32_t first_unfinished_value_offset = batch_start(job_index_to_offset(output, run_recovery(model, output)));
    // Re-execute from the begin of CHANNEL
    offset = first_unfinished_value_offset / CHANNEL * CHANNEL;
    idx = first_unfinished_value_offset / CHANNEL;
#if HAWAII
    // Resume footprint cnt
    write_hawaii_layer_footprint(model->layer_idx, offset - first_unfinished_value_offset);
#endif // HAWAII
#endif

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

    float_to_scale_params(&scaleFract, &shift, 1.0f * var_scale_sqrt / (X->scale * 2));
    my_scale_q15(buffer_b, scaleFract, shift, buffer_b, CHANNEL);

    output->scale = scale->scale * (X->scale * 2) / var_scale_sqrt;

    // assume conventional epsilon
    my_offset_q15(buffer_var, static_cast<int16_t>(0.00001 * 0x8000 / var->scale), buffer_var, CHANNEL);
    my_vsqrt_q15(buffer_var, buffer_var, CHANNEL);

    for (; idx < H * W; idx++) {
        my_memcpy_from_param(model, buffer_x, X, offset, CHANNEL * sizeof(int16_t));

        my_printf_debug("(h, w) = (%d, %d)" NEWLINE, idx / W, idx % W);

        my_sub_q15(buffer_x, buffer_mean, buffer_x, CHANNEL);
        my_printf_debug("x - mean" NEWLINE);
        // dump_matrix_debug(buffer_x, CHANNEL, ValueInfo(X->scale * 2));

        my_div_q15(buffer_x, buffer_var, buffer_x, CHANNEL);
        my_printf_debug("(x - mean)/sqrt(var+epsilon)" NEWLINE);
        // dump_matrix_debug(buffer_x, CHANNEL, ValueInfo((X->scale * 2) / var_scale_sqrt));

        my_mpy_q15(buffer_x, buffer_scale, buffer_x, CHANNEL);
        my_printf_debug("(x - mean)/sqrt(var+epsilon)*scale" NEWLINE);
        dump_matrix_debug(buffer_x, CHANNEL, ValueInfo(output, model));

        my_add_q15(buffer_x, buffer_b, buffer_x, CHANNEL);
        my_printf_debug("(x - mean)/sqrt(var+epsilon)*scale+B" NEWLINE);
        dump_matrix_debug(buffer_x, CHANNEL, ValueInfo(output, model));

        my_memcpy_to_param(output, offset, buffer_x, CHANNEL * sizeof(int16_t), 0);
        offset += CHANNEL;
#if HAWAII
        hawaii_record_footprints(model, CHANNEL);
#endif
    }

    my_printf_debug("handle_batchnormalization output" NEWLINE);
    dump_params_nhwc_debug(model, output);
}
