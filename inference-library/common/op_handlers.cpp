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

void alloc_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    const ParameterInfo *data = input[0];
    output->slot = get_next_slot(model, data);
}

void handle_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("ReLu!" NEWLINE);

    const ParameterInfo *X = input[0];
    uint16_t N = X->dims[0], CHANNEL = X->dims[1], H = X->dims[2], W = X->dims[3];

    /* XXX: use LEA? */
    uint16_t bitwidth = X->bitwidth;
    MY_ASSERT(bitwidth == 16);
    int16_t data_len = X->params_len / (bitwidth / 8);

#if !STABLE_POWER
    // FIXME: After removing FC merge, this branch can be removed
    if(H != 0) // conv
        data_len = N * CHANNEL * H * W;
#endif // STABLE_POWER
    my_printf_debug("data_len: %d" NEWLINE, data_len);
    uint16_t data_offset = 0;
    uint16_t output_offset = 0;
#if INTERMITTENT
    uint32_t first_unfinished_value_offset = batch_start(job_index_to_offset(output, run_recovery(model, output)));
    my_printf_debug("first_unfinished_value_offset: %d" NEWLINE, first_unfinished_value_offset);
    data_offset += first_unfinished_value_offset;
    if(H != 0) {
        if(node->flags.generic == NHWC2NCHW) {
            output_offset += (data_offset / CHANNEL) + (data_offset % CHANNEL) * H * W;
        } else {
            output_offset += first_unfinished_value_offset;
        }
    } else {
        output_offset += first_unfinished_value_offset;
    }
#endif

    if(node->flags.generic == NHWC2NCHW) {
        // NHWC -> NCHW
        uint16_t i = data_offset;
        uint16_t cur_batch_offset = i % BATCH_SIZE;
        for (; i < data_len; i++) {
            int16_t output_val;
            {
                int16_t input_val = get_q15_param(model, X, data_offset);
                output_val = MAX_VAL(input_val, 0);
            }
            my_printf_debug("input_offset=%d output_offset=%d output_val=%d" NEWLINE, data_offset, output_offset, output_val);
            put_q15_param(output, output_offset, output_val);
#if HAWAII
            if (cur_batch_offset == BATCH_SIZE - 1) {
                write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
                cur_batch_offset -= BATCH_SIZE;
            }
#endif
            data_offset++;
            if(H != 0) // conv
                output_offset = (data_offset / CHANNEL) + (data_offset % CHANNEL) * H * W;
            else // fc
                output_offset++;
            cur_batch_offset++;
        }
    } else {
        // NHWC -> NHWC
        uint16_t i = data_offset;
        uint16_t cur_batch_offset = i % BATCH_SIZE;
        for (; i < data_len; i++) {
            int16_t output_val;
            {
                int16_t input_val = get_q15_param(model, X, data_offset);
                output_val = MAX_VAL(input_val, 0);
            }
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

    // my_printf_debug("handle_relu output" NEWLINE);
    // if(node->flags.generic == NHWC2NCHW) {
    //    dump_params_debug(model, output, node->output_name);
    // } else {
    //    dump_params_nhwc_debug(model, output, node->output_name);
    // }
}

void handle_sigmoid(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node) {
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
    float_to_scale_params(&scaleFract, &shift, Y->scale / X->scale);
    my_scale_q15(buffer_b, scaleFract, shift, buffer_b, buffer_size);

    for (uint16_t idx = 0; idx < X->dims[2]; idx++) {
        my_memcpy_from_param(model, buffer_a, X, idx*buffer_size, buffer_size * sizeof(int16_t));
        my_add_q15(buffer_a, buffer_b, buffer_a, buffer_size);
        my_memcpy_to_param(output, idx*buffer_size, buffer_a, buffer_size * sizeof(int16_t), 0);
    }
    dump_params_nhwc_debug(model, output, node->output_name);
}

void handle_mul(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node) {
}

void alloc_batchnormalization(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node*) {
    const ParameterInfo* X = input[0];
    output->slot = get_next_slot(model, X);
}

void handle_batchnormalization(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node*) {
    my_printf_debug("BatchNormalization!" NEWLINE);

    const ParameterInfo *X = input[0], *scale = input[1], *B = input[2], *mean = input[3], *var = input[4];
    const uint16_t CHANNEL = X->dims[1], H = X->dims[2], W= X->dims[3];
    my_printf_debug("H: %d, W: %d, CHANNEL: %d" NEWLINE, H, W, CHANNEL);
    uint16_t area = 1;
    if(H * W != 0) area = H * W;
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
    // re-execute from the begin of CHANNEL
    offset = first_unfinished_value_offset & ~0x1;
    MY_ASSERT(!(offset & 0x1));
    idx = first_unfinished_value_offset / CHANNEL;
    if (idx == area) {
        goto EXIT_LAYER;
    }
#if HAWAII
    // reset footprint cnt
    write_hawaii_layer_footprint(model->layer_idx, offset - first_unfinished_value_offset);
#endif // HAWAII
#endif

    my_memcpy_from_param(model, buffer_scale, scale, 0, CHANNEL * sizeof(int16_t));
    my_memcpy_from_param(model, buffer_b, B, 0, CHANNEL * sizeof(int16_t));
    my_memcpy_from_param(model, buffer_mean, mean, 0, CHANNEL * sizeof(int16_t));
    my_memcpy_from_param(model, buffer_var, var, 0, CHANNEL * sizeof(int16_t));

    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, mean->scale / X->scale);
    my_scale_q15(buffer_mean, scaleFract, shift, buffer_mean, CHANNEL);

    Scale var_scale_sqrt;
    var_scale_sqrt.fromFloat(sqrtf(var->scale.toFloat()));
    // MY_ASSERT((var_scale_sqrt * var_scale_sqrt).toFloat() == var->scale.toFloat());

    float_to_scale_params(&scaleFract, &shift, (var_scale_sqrt.toFloat() * B->scale.toFloat()) / (X->scale.toFloat() * scale->scale.toFloat()));
    my_scale_q15(buffer_b, scaleFract, shift, buffer_b, CHANNEL);

    output->scale = scale->scale * X->scale / var_scale_sqrt;
    my_printf_debug("scale: %f" NEWLINE, scale->scale.toFloat());
    my_printf_debug("X: %f" NEWLINE, X->scale.toFloat());
    my_printf_debug("output: %f" NEWLINE, output->scale.toFloat());
    // assume conventional epsilon
    my_printf_debug("var" NEWLINE);
    dump_matrix_debug(buffer_var, CHANNEL, ValueInfo(output, model));
    my_offset_q15(buffer_var, static_cast<int16_t>(0.00001 * 0x8000 / var->scale.toFloat()), buffer_var, CHANNEL);
    my_printf_debug("var + epsilon" NEWLINE);
    dump_matrix_debug(buffer_var, CHANNEL, ValueInfo(output, model));
    my_vsqrt_q15(buffer_var, buffer_var, CHANNEL);
    my_printf_debug("sqrt(var + epsilon)" NEWLINE);
    dump_matrix_debug(buffer_var, CHANNEL, ValueInfo(output, model));

    uint16_t channel_offset, cur_channel;
    channel_offset = offset % CHANNEL;
    cur_channel = CHANNEL - channel_offset;
    buffer_x += channel_offset;
    buffer_mean += channel_offset;
    buffer_scale += channel_offset;
    buffer_var += channel_offset;
    buffer_b += channel_offset;
    for (; idx < area; idx++) {
        my_memcpy_from_param(model, buffer_x, X, offset, cur_channel * sizeof(int16_t));

        my_sub_q15(buffer_x, buffer_mean, buffer_x, cur_channel);
        my_printf_debug("x - mean" NEWLINE);
        dump_matrix_debug(buffer_x, cur_channel, ValueInfo(output, model));

        my_mpy_q15(buffer_x, buffer_scale, buffer_x, cur_channel);
        my_printf_debug("(x - mean)*scale" NEWLINE);
        dump_matrix_debug(buffer_x, cur_channel, ValueInfo(output, model));

        my_div_q15(buffer_x, buffer_var, buffer_x, cur_channel);
        my_printf_debug("(x - mean)*scale/sqrt(var+epsilon)" NEWLINE);
        dump_matrix_debug(buffer_x, cur_channel, ValueInfo(output, model));

        my_add_q15(buffer_x, buffer_b, buffer_x, cur_channel);
        my_printf_debug("(x - mean)/sqrt(var+epsilon)*scale+B" NEWLINE);
        dump_matrix_debug(buffer_x, cur_channel, ValueInfo(output, model));

        my_memcpy_to_param(output, offset, buffer_x, cur_channel * sizeof(int16_t), 0);
        offset += cur_channel;
#if HAWAII
        hawaii_record_footprints(model, cur_channel);
#endif
        cur_channel += channel_offset;
        buffer_x -= channel_offset;
        buffer_mean -= channel_offset;
        buffer_scale -= channel_offset;
        buffer_var -= channel_offset;
        buffer_b -= channel_offset;
        channel_offset = 0;
    }
#if INTERMITTENT
EXIT_LAYER:
#endif // INTERMITTENT
    my_printf_debug("handle_batchnormalization output" NEWLINE);
    // dump_params_nhwc_debug(model, output);
}
