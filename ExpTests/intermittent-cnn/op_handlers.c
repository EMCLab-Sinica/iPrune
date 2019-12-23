#include <string.h>

#include <DSPLib.h>

#ifdef __MSP430__
#include <driverlib.h>
#include <FreeRTOS.h>
#include <task.h>
#include <croutine.h>
#define USE_DMA 1
#define USE_CONCURRENT_CONV 1
#else
#define USE_DMA 0
#define USE_CONCURRENT_CONV 0
#endif

#include "ops.h"
#include "op_handlers.h"

#define configCONV_STACK_SIZE 100

#ifdef __MSP430__
#pragma DATA_SECTION(lea_buffer_input, ".leaRAM")
#pragma DATA_SECTION(lea_buffer_filter, ".leaRAM")
#pragma DATA_SECTION(lea_buffer_another, ".leaRAM")
#pragma DATA_SECTION(lea_buffer_temp, ".leaRAM")
#pragma DATA_SECTION(iq31_mac_result, ".leaRAM")
#endif
int16_t lea_buffer_input[256], lea_buffer_filter[256], lea_buffer_another[256], lea_buffer_temp[64];
int32_t iq31_mac_result;

uint16_t counters[10];
uint8_t counter_idx = 0;

static struct ConvTaskParams {
    ParameterInfo *conv_input;
    ParameterInfo *conv_filter;
    ParameterInfo *bias;
    ParameterInfo *output;
    uint16_t conv_idx;
    uint16_t output_h;
    uint16_t output_w;
} conv_params;

#if USE_DMA
#define MY_DMA_CHANNEL DMA_CHANNEL_0
static DMA_initParam dma_params = {
    .channelSelect = MY_DMA_CHANNEL,
    .transferModeSelect = DMA_TRANSFER_BLOCK,
};
#endif

static void my_memcpy(void* dest, const void* src, size_t n) {
#if !USE_DMA
    memcpy(dest, src, n);
#else
    DMA_init(&dma_params);
    DMA_setSrcAddress(MY_DMA_CHANNEL, (uint32_t)(src), DMA_DIRECTION_INCREMENT);
    DMA_setDstAddress(MY_DMA_CHANNEL, (uint32_t)(dest), DMA_DIRECTION_INCREMENT);
    /* transfer size is in words (2 bytes) */
    DMA_setTransferSize(MY_DMA_CHANNEL, (n) >> 1);
    DMA_enableInterrupt(MY_DMA_CHANNEL);
    DMA_enableTransfers(MY_DMA_CHANNEL);
    DMA_startTransfer(MY_DMA_CHANNEL);
#endif
}


#if USE_CONCURRENT_CONV
static void convTask(CoRoutineHandle_t xHandle, UBaseType_t uxIndex) {
    crSTART(xHandle);

    for (;;) {
#else
static void convTask(void) {
#endif
    /* Cannot use C as a variable name here as C is a macro on MSP430 :( */
    uint16_t H = conv_params.conv_input->dims[1], W = conv_params.conv_input->dims[2],
             kH = conv_params.conv_filter->dims[1], kW = conv_params.conv_filter->dims[2],
             CHANNEL = conv_params.conv_filter->dims[3];

    /* MSP430 LEA requires length to be even */
    msp_mac_q15_params mac_params = { .length = (uint16_t)(CHANNEL * kH * kW / 2 * 2) };
    uint8_t truncated = (mac_params.length != CHANNEL * kH * kW);
    uint16_t buffer_size = (uint16_t)(sizeof(uint16_t) * mac_params.length);
    if (buffer_size > sizeof(lea_buffer_filter)) {
        my_printf("Error: buffer too small." NEWLINE);
        ERROR_OCCURRED();
    }

    /* copy filter data */
    /* TODO: cache it */
    my_memcpy(lea_buffer_filter,
              get_q15_param(conv_params.conv_filter, (size_t)(conv_params.conv_idx * CHANNEL * kH * kW)),
              buffer_size);

    /* copy input data, row by row */
    int16_t *input_addr = get_q15_param(conv_params.conv_input, (size_t)((conv_params.output_h * W + conv_params.output_w) * CHANNEL));
    for (uint16_t h = 0; h < kH; h++) {
        size_t size = (size_t)(kW * CHANNEL);
        if (truncated && h == kH - 1) {
            size--;
        }
        /* TODO: handle padding */
        my_memcpy(lea_buffer_input + h * kW * CHANNEL,  // dest
                  input_addr + h * W * CHANNEL,  // src
                  size * sizeof(uint16_t));  // size
    }

#if USE_CONCURRENT_CONV
    /* TODO: do context switch after msp_lea_doInvokeCommand */
    msp_status status = msp_do_mac_q15(&mac_params, lea_buffer_input, lea_buffer_filter, &iq31_mac_result, 1);
#else
    msp_status status = msp_mac_q15(&mac_params, lea_buffer_input, lea_buffer_filter, &iq31_mac_result);
#endif
    msp_checkStatus(status);
    if (truncated) {
#ifndef MY_NDEBUG
        // my_printf("Adding truncated product back" NEWLINE);
#endif
        uint16_t last_idx = (uint16_t)(kH * kW - 1);
        iq31_mac_result += (*get_q15_param(conv_params.conv_input, last_idx)) * (*get_q15_param(conv_params.conv_filter, last_idx)) * 2;
    }

#if defined(DUMP_PARAMS) && !defined(__MSP430__)
    my_printf("%f ", (float)iq31_mac_result / 2147483648.0f);
#endif
    int16_t q15_mac_result = iq31_to_q15(&iq31_mac_result);
    q15_mac_result = (int16_t)(q15_mac_result + *get_q15_param(conv_params.bias, conv_params.conv_idx));

    int16_t *output_data = get_q15_param(conv_params.output, 0);
    output_data[conv_params.conv_idx * H * W + conv_params.output_h * W + conv_params.output_w] = q15_mac_result;

#if USE_CONCURRENT_CONV
    crDELAY(xHandle, 0);

    }

    crEND();
#endif
}

uint8_t handle_conv(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("Conv!" NEWLINE);
#endif

#if USE_CONCURRENT_CONV
    static bool task_created = false;

    if (!task_created) {
        if (xCoRoutineCreate(convTask, 0, 0) != pdPASS) {
            my_printf("Failed to create co-routines." NEWLINE);
            ERROR_OCCURRED();
        }
        task_created = true;
    }
#endif

    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *bias = input[2];
    if (conv_input->bitwidth_and_flags >> 1 != 16 || conv_filter->bitwidth_and_flags >> 1 != 16) {
        my_printf("Error: incorrect bitwidth." NEWLINE);
        return 1;
    }
    /* original: input: N x C x H x W, filter: M x C x kW x kW
     * remapped: input: N x H x W x C, filter: M x kH x kW x C */
    /* TODO: really use remapped dimensions */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   input_N = conv_filter->dims[0];
    /* TODO: add flags; assume auto_pad=SAME_UPPER, stride=(1, 1), dilation=(1, 1) for now */
    output->params_len = (uint16_t)(input_N * H * W * 2);
    output->bitwidth_and_flags = 16 << 1 | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = input_N;
    output->dims[2] = H;
    output->dims[3] = W;

    uint8_t ret = 0;

#ifdef __MSP430__
    TickType_t start, end;
    start = xTaskGetTickCount();
#endif
    for (uint16_t conv_idx = 0; conv_idx < input_N; conv_idx++) {
        //my_printf("conv_idx = %d" NEWLINE, conv_idx);
        for (uint16_t output_h = 0; output_h < H; output_h++) {
            for (uint16_t output_w = 0; output_w < W; output_w++) {
                conv_params.conv_input = conv_input;
                conv_params.conv_filter = conv_filter;
                conv_params.bias = bias;
                conv_params.output = output;
                conv_params.conv_idx = conv_idx;
                conv_params.output_h = output_h;
                conv_params.output_w = output_w;
#if USE_CONCURRENT_CONV
                vCoRoutineSchedule();
#else
                convTask();
#endif
            }
        }
    }
#ifdef __MSP430__
    end = xTaskGetTickCount();
    counters[counter_idx] = end - start;
    counter_idx++;
#endif

#ifndef MY_NDEBUG
    my_printf("handle_conv output" NEWLINE);
#endif

    return ret;
}

uint8_t handle_maxpool(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("MaxPool!" NEWLINE);
#endif
    /* TODO: add flags; assume stripe=2, no padding for now */
    const uint16_t stride = 2; // for less type conversions
    ParameterInfo *data = input[0];
    output->params_len = data->params_len / (uint16_t)(stride * stride);
    output->bitwidth_and_flags = data->bitwidth_and_flags | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = data->dims[1];
    output->dims[2] = data->dims[2] / stride;
    output->dims[3] = data->dims[3] / stride;
    const uint16_t channel = data->dims[1], H = data->dims[2], W = data->dims[3];
    msp_max_q15_params params = { .length = 4 };
    int16_t max_val;
    uint16_t index;
    int16_t *lea_buffer_maxpool = lea_buffer_input;
    for (uint16_t i = 0; i < channel; i++) {
        for (uint16_t j = 0; j < H; j = (uint16_t)(j + stride)) {
            for (uint16_t k = 0; k < W; k = (uint16_t)(k + stride)) {
                lea_buffer_maxpool[0] = *get_q15_param(data, (size_t)(i * H * W + j     * W + k    ));
                lea_buffer_maxpool[1] = *get_q15_param(data, (size_t)(i * H * W + j     * W + (k+1)));
                lea_buffer_maxpool[2] = *get_q15_param(data, (size_t)(i * H * W + (j+1) * W + k    ));
                lea_buffer_maxpool[3] = *get_q15_param(data, (size_t)(i * H * W + (j+1) * W + (k+1)));
                msp_status status = msp_max_q15(&params, lea_buffer_maxpool, &max_val, &index);
                msp_checkStatus(status);
                *get_q15_param(output, (size_t)(i * H * W + j * W + k)) = max_val;
            }
        }
    }

#ifndef MY_NDEBUG
    my_printf("handle_maxpool output" NEWLINE);
    dump_params(output);
#endif

    return 0;
}

uint8_t handle_add(ParameterInfo *input[], ParameterInfo *output) {
    /* Add: Y = X + W */
#ifndef MY_NDEBUG
    my_printf("Add!" NEWLINE);
#endif
    if (input[0]->bitwidth_and_flags >> 1 != 16 || input[1]->bitwidth_and_flags >> 1 != 16) {
        my_printf("Error: unsupported bitwidth" NEWLINE);
        return 1;
    }
    ParameterInfo *A = input[0], *B = input[1];
    output->params_len = input[0]->params_len;
    output->bitwidth_and_flags = input[0]->bitwidth_and_flags | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = A->dims[1];

    msp_add_q15_params params = { .length = A->dims[1] };

    int16_t *lea_buffer_A = lea_buffer_input,
            *lea_buffer_B = lea_buffer_another;
    my_memcpy(lea_buffer_A, get_q15_param(A, 0), output->params_len);
    my_memcpy(lea_buffer_B, get_q15_param(B, 0), output->params_len);
    msp_status status = msp_add_q15(&params, lea_buffer_A, lea_buffer_B, lea_buffer_A);
    msp_checkStatus(status);

    my_memcpy(get_q15_param(output, 0), lea_buffer_A, output->params_len);

    return 0;
}


#ifdef DUMP_PARAMS
static void dump_matrix(int16_t *mat, size_t len) {
    for (size_t j = 0; j < len; j++) {
        my_printf("%d ", mat[j]);
        if (j && (j % 16 == 0)) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE);
}
#else
#define dump_matrix(mat, len)
#endif

uint8_t handle_matmul(ParameterInfo *input[], ParameterInfo *output) {
    ParameterInfo *A = input[0], *B = input[1];

#ifndef MY_NDEBUG
    my_printf("handle_matmul inputs" NEWLINE);
    dump_params(A);
    dump_params(B);

    my_printf("MatMul! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);
#endif

    uint16_t output_len = (uint16_t)(A->dims[0] * B->dims[1]);
    output->dims[0] = A->dims[0];
    output->dims[1] = B->dims[1];
    output->params_len = (uint16_t)(output_len * 2);
    output->bitwidth_and_flags = 16 << 1 | FLAG_INTERMEDIATE_VALUES;

    if (A->dims[0] * A->dims[1] > 256) {
        my_printf("Matrix A too large!" NEWLINE);
        return 1;
    }

    /* Seems TI's debugger does not like alias of pointers :/ */
#define lea_buffer_A lea_buffer_filter
#define lea_buffer_B lea_buffer_another
#define lea_buffer_matmul lea_buffer_input

    msp_fill_q15_params fill_params = {
        .length = 256,
        .value = 0,
    };
    msp_status status = msp_fill_q15(&fill_params, lea_buffer_matmul);
    msp_checkStatus(status);

    my_memcpy(lea_buffer_A, get_q15_param(A, 0), (uint16_t)(A->dims[0] * A->dims[1]));

    /* LEA wants addresses to be 4-aligned */
    uint16_t step = (uint16_t)((256 / B->dims[1]) / 4 * 4);
    for (uint16_t i = 0; i < B->dims[0]; i = (uint16_t)(i + step)) {
        msp_matrix_mpy_q15_params params;
        uint16_t current_width = (uint16_t)MIN_VAL(step, B->dims[0] - i);
        params.srcARows = A->dims[0];
        params.srcACols = current_width;
        params.srcBRows = current_width;
        params.srcBCols = B->dims[1];

        my_memcpy(lea_buffer_B, get_q15_param(B, (uint16_t)(i * B->dims[1])), (uint16_t)(current_width * B->dims[1]));

#ifdef DUMP_PARAMS
        my_printf("strip for A" NEWLINE);
        dump_matrix(lea_buffer_A + A->dims[0] * i, (size_t)(A->dims[0] * current_width));
        my_printf("B" NEWLINE);
        dump_matrix(lea_buffer_B, (size_t)(current_width * B->dims[1]));
#endif

        status = msp_matrix_mpy_q15(
            &params,
            lea_buffer_A + A->dims[0] * i,
            lea_buffer_B,
            lea_buffer_temp);
        msp_checkStatus(status);

#ifdef DUMP_PARAMS
        my_printf("temp" NEWLINE);
        dump_matrix(lea_buffer_temp, (size_t)(A->dims[0] * B->dims[1]));
#endif

        msp_add_q15_params params2 = { .length = output_len };
        status = msp_add_q15(&params2, lea_buffer_matmul, lea_buffer_temp, lea_buffer_matmul);
        msp_checkStatus(status);
    }
    my_memcpy(get_q15_param(output, 0), lea_buffer_matmul, output->params_len);

#undef lea_buffer_A
#undef lea_buffer_B
#undef lea_buffer_matmul

#ifndef MY_NDEBUG
    my_printf("handle_matmul output" NEWLINE);
    dump_params(output);
#endif

    return 0;
}

uint8_t handle_relu(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("ReLu!" NEWLINE);
#endif
    ParameterInfo *X = input[0];
    memcpy(output, X, sizeof(ParameterInfo));
    /* TODO: use LEA? */
    uint16_t bitwidth = X->bitwidth_and_flags >> 1;
    for (uint32_t i = 0; i < X->params_len / (bitwidth / 8); i++) {
        if (bitwidth == 16) {
            int16_t *ptr = get_q15_param(X, i);
            if (*ptr < 0) {
                *ptr = 0;
            }
        } else {
            my_printf("Error: unsupported bitwidth for ReLu." NEWLINE);
        }
    }
#ifndef MY_NDEBUG
    dump_params(output);
#endif
    return 0;
}

uint8_t handle_reshape(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("Reshape!" NEWLINE);
#endif
    ParameterInfo *data = input[0], *shape = input[1];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    if (shape->bitwidth_and_flags >> 1 != 64) {
        my_printf("Error: unsupported shape format." NEWLINE);
        return 1;
    }
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = (uint16_t)get_int64_param(shape, i);
    }
    return 0;
}

uint8_t handle_squeeze(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("Squeeze!" NEWLINE);
#endif
    ParameterInfo *data = input[0];
    /* TODO: add flags; assume squeeze all one-size axes */
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    for (uint8_t i = 0, j = 0; i < 4; i++) {
        if (input[0]->dims[i] != 1) {
            output->dims[j] = input[0]->dims[i];
            j++;
        }
    }
    return 0;
}

#ifdef __MSP430__

#pragma vector=DMA_VECTOR
__interrupt void DMA_ISR(void)
{
    switch(__even_in_range(DMAIV,16))
    {
        case 0: break;
        case 2: break; // DMA0IFG = DMA Channel 0
        case 4: break; // DMA1IFG = DMA Channel 1
        case 6: break; // DMA2IFG = DMA Channel 2
        case 8: break; // DMA3IFG = DMA Channel 3
        case 10: break; // DMA4IFG = DMA Channel 4
        case 12: break; // DMA5IFG = DMA Channel 5
        case 14: break; // DMA6IFG = DMA Channel 6
        case 16: break; // DMA7IFG = DMA Channel 7
        default: break;
    }
}

#endif