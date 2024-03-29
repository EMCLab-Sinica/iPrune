#include <cstdint>
#include <cstring>
#include <cinttypes> // for PRId32
#include <cmath>

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "data.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

uint16_t sample_idx;

static void handle_node(Model *model, uint16_t node_idx) {
    const Node *cur_node = get_node(node_idx);
#if MY_DEBUG >= MY_DEBUG_LAYERS
    my_printf("Current node: %d, ", node_idx);
    my_printf("name = %.*s, ", NODE_NAME_LEN, cur_node->name);
    my_printf("op_type = %d" NEWLINE, cur_node->op_type);
#endif

    int16_t input_id[5];
    const ParameterInfo *input[5];
    for (uint16_t j = 0; j < cur_node->inputs_len; j++) {
        input_id[j] = cur_node->inputs[j];
        my_printf_debug("input_id[%d] = %d" NEWLINE, j, input_id[j]);
        input[j] = get_parameter_info(input_id[j]);
        // dump_params(model, input[j], cur_node->name);
    }
#if SPARSE
#ifdef OpGemmMerge
    // FIXME: remove this after removing op_merge
    if(cur_node->op_type == OpGemmMerge) {
        // cur node is ConvMerge or GemmMerge
        input_id[1] = get_node(node_idx - 1)->inputs[1];
        input[1] = get_parameter_info(input_id[1]);
    }
#endif // OpGemmMerge
#endif
    my_printf_debug(NEWLINE);

    /* Allocate an ParameterInfo for output. Details are filled by
     * individual operation handlers */
    ParameterInfo *output = get_intermediate_parameter_info(node_idx);
    my_memcpy(output, input[0], sizeof(ParameterInfo) - sizeof(uint16_t)); // don't overwrite parameter_info_idx
    output->params_offset = 0;
    allocators[cur_node->op_type](model, input, output, cur_node);
    my_printf_debug("Needed mem = %u" NEWLINE, output->params_len);
    MY_ASSERT(output->params_len < INTERMEDIATE_VALUES_SIZE);
    if (output->slot == SLOT_INTERMEDIATE_VALUES) {
        my_printf_debug("New params_offset = %d" NEWLINE, output->params_offset);
    }

#if STATEFUL
    my_printf_debug("Old output state bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif
    handlers[cur_node->op_type](model, input, output, cur_node);
    // For some operations (e.g., ConvMerge), scale is determined in the handlers
    my_printf_debug("Ouput scale = %f" NEWLINE, output->scale.toFloat());
#if STATEFUL
    my_printf_debug("New output state bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif

    MY_ASSERT(output->bitwidth);

    commit_intermediate_parameter_info(node_idx);

    if (node_idx == MODEL_NODES_LEN - 1) {
        model->running = 0;
        model->run_counter++;
#if ENABLE_COUNTERS
        if (!total_jobs) {
            total_jobs = counters()->job_preservation / 2;
        }
#endif
    }
}

#if MY_DEBUG >= MY_DEBUG_NORMAL
const float first_sample_outputs[] = FIRST_SAMPLE_OUTPUTS;
#endif

static void run_model(int8_t *ansptr, const ParameterInfo **output_node_ptr) {
    my_printf_debug("N_INPUT = %d" NEWLINE, N_INPUT);

    Model *model = get_model();
    if (!model->running) {
        // reset model
        model->layer_idx = 0;
        for (uint8_t idx = 0; idx < NUM_SLOTS; idx++) {
            SlotInfo *cur_slot_info = get_slot_info(model, idx);
            cur_slot_info->user = -1;
        }
#if HAWAII
        for (uint16_t node_idx = 0; node_idx < MODEL_NODES_LEN; node_idx++) {
            reset_hawaii_layer_footprint(node_idx);
        }
#endif
        model->running = 1;
        commit_model();
#if ENABLE_COUNTERS
        memset(counters_data, 0, sizeof(Counters) * COUNTERS_LEN);
#endif
    }

#if ENABLE_COUNTERS
    counters()->power_counters++;
#endif

    dump_model_debug(model);

    for (uint16_t node_idx = model->layer_idx; node_idx < MODEL_NODES_LEN; node_idx++) {
        handle_node(model, node_idx);
        model->layer_idx++;

        commit_model();

#if 0
        notify_layer_finished();
#endif

        dump_model_debug(model);
    }

    // the parameter info for the last node should also be refreshed when MY_DEBUG == 0
    // Otherwise, the model is not correctly re-initialized in some cases
    const ParameterInfo *output_node = get_parameter_info(MODEL_NODES_LEN + N_INPUT - 1);
    if (output_node_ptr) {
        *output_node_ptr = output_node;
    }
#if MY_DEBUG >= MY_DEBUG_NORMAL
    int16_t max = INT16_MIN;
    uint16_t u_ans;
    uint8_t ans_len = sizeof(first_sample_outputs) / sizeof(float);
    uint8_t buffer_len = MIN_VAL(output_node->dims[1], ans_len);
    if(!output_node->dims[1])
        buffer_len = MIN_VAL(output_node->dims[0], ans_len);
    my_memcpy_from_param(model, lea_buffer, output_node, 0, buffer_len * sizeof(int16_t));

    if (sample_idx == 0) {
        for (uint8_t buffer_idx = 0, ofm_idx = 0; buffer_idx < buffer_len; buffer_idx++) {
            // int16_t got_q15 = lea_buffer[buffer_idx];
            {
                // float got_real = q15_to_float(got_q15, ValueInfo(output_node), nullptr, false);
                // float expected = first_sample_outputs[ofm_idx];
                // float error = fabs((got_real - expected) / expected);
                // Errors in CIFAR-10/Stateful are quite large...
                // MY_ASSERT(error <= 0.15,
                //          "Value error too large at index %d: got=%f, expected=%f" NEWLINE, buffer_idx, got_real, expected);
                ofm_idx++;
            }
        }
    }

    my_max_q15(lea_buffer, buffer_len, &max, &u_ans);
    *ansptr = u_ans;
#endif
}

#if ENABLE_COUNTERS
template<uint32_t Counters::* MemPtr>
static uint32_t print_counters() {
    uint32_t total = 0;
    for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
        total += (counters_data + i)->*MemPtr;
#if ENABLE_PER_LAYER_COUNTERS
        my_printf("%8" PRIu32, counters_data[i].*MemPtr);
#else
        break;
#endif
        if (i % 16 == 15) {
            my_printf(NEWLINE);
        }
    }
    my_printf(" total=%8" PRIu32, total);
    return total;
}
#endif

#if (MY_DEBUG >= MY_DEBUG_NORMAL) || (ENABLE_COUNTERS && !DEMO)
static void print_results(const ParameterInfo *output_node) {
    Model *model = get_model();

    dump_params(model, output_node);

    my_printf("op types:            ");
    for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
        my_printf("% 8d", get_node(i)->op_type);
        if (i % 16 == 15) {
            my_printf(NEWLINE);
        }
    }

#if ENABLE_COUNTERS
    uint32_t total_dma_bytes = 0, total_overhead = 0;
    my_printf(NEWLINE "Power counters:      "); print_counters<&Counters::power_counters>();
    my_printf(NEWLINE "DMA invocations (R): "); print_counters<&Counters::dma_invocations_r>();
    my_printf(NEWLINE "DMA invocations (W): "); print_counters<&Counters::dma_invocations_w>();
    my_printf(NEWLINE "DMA read of filter:  "); print_counters<&Counters::dma_read_filter>();
    my_printf(NEWLINE "DMA read of input:   "); print_counters<&Counters::dma_read_input>();
    my_printf(NEWLINE "DMA write of ofm:    "); print_counters<&Counters::dma_write_ofm>();
    my_printf(NEWLINE "DMA write of fp:     "); print_counters<&Counters::dma_write_fp>();
    my_printf(NEWLINE "Indexing:            "); print_counters<&Counters::indexing>();
    my_printf(NEWLINE "DMA bytes (R):       "); total_dma_bytes = print_counters<&Counters::dma_bytes_r>();
    my_printf(NEWLINE "DMA bytes (W):       "); total_dma_bytes = print_counters<&Counters::dma_bytes_w>();
    my_printf(NEWLINE "Job preservation:    "); print_counters<&Counters::job_preservation>();
    my_printf(NEWLINE "FP preservation:     "); print_counters<&Counters::footprint_preservation>();
    my_printf(NEWLINE "MACs:                "); print_counters<&Counters::macs>();
    // recovery overheads
    my_printf(NEWLINE "Progress seeking:    "); total_overhead += print_counters<&Counters::progress_seeking>();

    my_printf(NEWLINE "Total DMA bytes: %d", total_dma_bytes);
    my_printf(NEWLINE "Total overhead: %" PRIu32, total_overhead);
    my_printf(NEWLINE "run_counter: %d" NEWLINE, model->run_counter);
#endif

    my_printf("NVM writes: %ld" NEWLINE, get_nvm_writes());
}
#endif

uint8_t run_cnn_tests(uint16_t n_samples) {
    int8_t predicted = -1;
    const ParameterInfo *output_node;
#if (MY_DEBUG >= MY_DEBUG_NORMAL) || ENABLE_COUNTERS
    int8_t label = -1;
    uint32_t correct = 0, total = 0;
    if (!n_samples) {
        n_samples = PLAT_LABELS_DATA_LEN;
    }
    const uint8_t *labels = labels_data;
#endif
    for (uint16_t i = 0; i < n_samples; i++) {
        sample_idx = i;
        run_model(&predicted, &output_node);
#if (MY_DEBUG >= MY_DEBUG_NORMAL) || ENABLE_COUNTERS
        label = labels[i];
        total++;
        if (label == predicted) {
            correct++;
        }
        if (i % 100 == 99) {
            my_printf("Sample %d finished" NEWLINE, sample_idx);
            // stdout is not flushed at \n if it is not a terminal
            my_flush();
        }
        my_printf_debug("idx=%d label=%d predicted=%d correct=%d" NEWLINE, i, label, predicted, label == predicted);
#endif
    }
#if (MY_DEBUG >= MY_DEBUG_NORMAL) || (ENABLE_COUNTERS && !DEMO)
    if (n_samples == 1) {
        print_results(output_node);
    }
    my_printf("correct=%" PRId32 " ", correct);
    my_printf("total=%" PRId32 " ", total);
    my_printf("rate=%f" NEWLINE, 1.0*correct/total);

    // Allow only 1% of accuracy drop
    if (N_SAMPLES == N_ALL_SAMPLES && correct < (FP32_ACCURACY - 0.01) * total) {
        return 1;
    }
#endif
    return 0;
}


#if INDIRECT_RECOVERY
static void check_feature_map_states(Model *model, const ParameterInfo* output, uint32_t first_unfinished_job_index, uint32_t len, const char* func) {
#if MY_DEBUG >= MY_DEBUG_NORMAL
    my_printf_debug("Running check_feature_map_states..." NEWLINE);
#if 0
    for (uint32_t idx = 0; idx < len; idx++) {
        my_printf_debug("% 6d ", get_q15_param(model, output, idx));
        if (idx % 16 == 15) {
            my_printf_debug(NEWLINE);
        }
    }
#endif
    for (uint32_t idx = 0; ; idx++) {
        uint32_t offset = job_index_to_offset(output, idx);
        if (offset >= len) {
            break;
        }
        int16_t val = get_q15_param(model, output, offset);
        int8_t cur_state_bit = param_state_bit(model, output, offset);
        if (idx < first_unfinished_job_index) {
            cur_state_bit = -cur_state_bit;
        }
        MY_ASSERT(get_value_state_bit(val) == cur_state_bit,
            "Value %d at job index %d (offset %" PRIu32 ") does not have expected state bit %d" NEWLINE, val, idx, offset, cur_state_bit);
    }
#endif
}
#endif

#if STATEFUL
static uint8_t value_finished(Model* model, const ParameterInfo* output, uint32_t job_index) {
    uint32_t offset = job_index_to_offset(output, job_index);
    int16_t val = get_q15_param(model, output, offset);
    uint8_t ret = (get_value_state_bit(val) != param_state_bit(model, output, offset));
    my_printf_debug("Value %d at job index %d (offset %" PRIu32 ") indicates %s" NEWLINE, val, job_index, offset, ret ? "finished" : "unfinished");
    return ret;
}

#endif

void flip_state_bit(Model *model, const ParameterInfo *output) {
#if INDIRECT_RECOVERY
    start_cpu_counter();

#if JAPARI
    MY_ASSERT(has_footprints(output));
#endif
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    // XXX: better way than copying the array?
#if JAPARI
    // abandon output features smaller than a batch
    uint16_t new_turning_point = (output->params_len / 2) / (BATCH_SIZE + 1) * (BATCH_SIZE + 1);
#else
    uint16_t new_turning_point = (output->params_len / 2) / BATCH_SIZE * BATCH_SIZE;
#endif
    my_printf_debug("New turning point=%d" NEWLINE, new_turning_point);
    uint8_t new_turning_point_inserted = 0;
    for (uint8_t idx = 0; idx < cur_slot_info->n_turning_points; idx++) {
        if (new_turning_point < cur_slot_info->turning_points[idx]) {
            uint8_t new_turning_point_idx = idx;
            cur_slot_info->n_turning_points++;
            MY_ASSERT(cur_slot_info->n_turning_points <= TURNING_POINTS_LEN);
            for (uint8_t idx2 = cur_slot_info->n_turning_points - 1; idx2 > new_turning_point_idx; idx2--) {
                cur_slot_info->turning_points[idx2] = cur_slot_info->turning_points[idx2 - 1];
            }
            cur_slot_info->turning_points[new_turning_point_idx] = new_turning_point;
            new_turning_point_inserted = 1;
            break;
        } else if (new_turning_point == cur_slot_info->turning_points[idx]) {
            cur_slot_info->n_turning_points--;
            for (uint8_t idx2 = idx; idx2 < cur_slot_info->n_turning_points; idx2++) {
                cur_slot_info->turning_points[idx2] = cur_slot_info->turning_points[idx2 + 1];
            }
            new_turning_point_inserted = 1;
            break;
        }
    }
    if (!new_turning_point_inserted) {
        cur_slot_info->n_turning_points++;
        cur_slot_info->turning_points[cur_slot_info->n_turning_points - 1] = new_turning_point;
    }

    dump_turning_points_debug(model, output);

    cur_slot_info->state_bit = -cur_slot_info->state_bit;

    // Use first_unfinished_job_index = 0 here as all values finished and the initial state bit is flipped above
    check_feature_map_states(model, output, 0, output->params_len / sizeof(int16_t), __func__);

    stop_cpu_counter(&Counters::table_updates);
#endif // INDIRECT_RECOVERY
}

#if INDIRECT_RECOVERY

int8_t get_state_bit(Model *model, uint8_t slot_id) {
    switch (slot_id) {
        case SLOT_PARAMETERS:
        case SLOT_TEST_SET:
            return 0;
        default:
            return get_slot_info(model, slot_id)->state_bit;
    }
}

int8_t param_state_bit(Model *model, const ParameterInfo *param, uint16_t offset) {
    int8_t ret = get_state_bit(model, param->slot);
    SlotInfo *cur_slot_info = get_slot_info(model, param->slot);
    if (!cur_slot_info) {
        return 0;
    }
    for (uint8_t idx = 0; idx < cur_slot_info->n_turning_points; idx++) {
        if (offset >= cur_slot_info->turning_points[idx]) {
            ret = -ret;
        } else {
            break;
        }
    }
    return ret;
}

#endif

#if HAWAII
uint32_t run_recovery(Model* model, ParameterInfo*) {
    uint32_t footprint = read_hawaii_layer_footprint(model->layer_idx);
    return footprint / BATCH_SIZE;
}
#endif

#if JAPARI
static uint8_t value_finished(Model* model, const ParameterInfo* output, uint32_t job_index) {
    uint32_t offset = job_index_to_offset(output, job_index);
    int16_t val = get_q15_param(model, output, offset);
    int16_t expected_footprint = -param_state_bit(model, output, offset);
    check_footprint(val);
    uint8_t ret = (val == expected_footprint);
    my_printf_debug("Footprint %d (expected %d) at job index %d (offset %" PRIu32 ") indicates %s" NEWLINE, val, expected_footprint, job_index, offset, ret ? "finished" : "unfinished");
    return ret;
}
#endif

#if SPARSE
uint16_t find_row_index(Model *model, const ParameterInfo *filter_params, const ParameterInfo *output, const Node *node, uint16_t col_index, int16_t *cur_row_val) {
    uint16_t n_tiles = 0;
#ifdef OpConv
    if(node->op_type == OpConv) {
        n_tiles = filter_params->dims[1] * filter_params->dims[2] * filter_params->dims[3] / node->flags.extra.conv.input_tile_c;
    }
#endif
#ifdef OpGemm
    if(node->op_type == OpGemm) {
        n_tiles = output->params_len / sizeof(int16_t) / (output->dims[0] * output->dims[1]);
    }
#endif
    my_printf_debug("n_tiles: %d" NEWLINE, n_tiles);
    MY_ASSERT(n_tiles != 0);
    uint16_t l = 0, r = n_tiles + 1;
    uint16_t tmp_row_val = 0;
    my_printf_debug("col index: %d" NEWLINE, col_index);
    while(l < r) {
        uint16_t m = l + ((r - l) >> 1);
        tmp_row_val = get_row_val(model, filter_params, m);
        my_printf_debug("tmp_row_val: %d" NEWLINE, tmp_row_val);
        if(tmp_row_val > col_index) {
            r = m;
        } else {
            *cur_row_val = tmp_row_val;
            my_printf_debug("cur_row_val: %d" NEWLINE, *cur_row_val);
            l = m + 1;
            my_printf_debug("l: %d r: %d" NEWLINE, l, r);
        }
    }
    return l - 1;
}

// XXX: support FC x hawaii only
uint32_t job_index_to_offset_sparse(Model *model, const ParameterInfo *params_filter, const ParameterInfo* output, uint16_t job_index) {
    // Handle FC recovery via binary search
    const Node* node = get_node(output);
    uint16_t output_len = output->dims[0] * output->dims[1]; // 256
    uint16_t output_jobs = output_len / BATCH_SIZE; // 256
    uint16_t jobs_in_an_op = OP_FILTERS / BATCH_SIZE; // 2
    uint16_t cur_col_index = job_index / jobs_in_an_op; // 165
    uint16_t col_val = get_col_val(model, params_filter, cur_col_index); // [0 "2"] // 107
    int16_t cur_row_val = 0; // 143
    uint16_t row_index = find_row_index(model, params_filter, output, node, cur_col_index, &cur_row_val); // 15
    uint16_t filter_tile_c = col_val; // 107
    uint16_t fixed_jobs_index_in_tile_c = row_index * output_jobs + filter_tile_c * (OP_FILTERS / BATCH_SIZE) + (job_index % jobs_in_an_op);
    my_printf_debug("fixed_jobs_index_in_tile_c: %d\n", fixed_jobs_index_in_tile_c);
    return (fixed_jobs_index_in_tile_c + 1) * BATCH_SIZE - 1;
}
#endif // SPARSE
uint32_t job_index_to_offset(const ParameterInfo* output, uint16_t job_index) {
    start_cpu_counter();
#if STATEFUL
    if (job_index >= output->params_len / sizeof(int16_t)) {
        return job_index;
    }
#endif
#if JAPARI
    if (job_index >= output->params_len / sizeof(int16_t) / (BATCH_SIZE + 1)) {
        return job_index * (BATCH_SIZE + 1) + BATCH_SIZE;
    }
#endif

    const Node* node = get_node(output);
#ifdef OpConv
    uint8_t is_conv = (node->op_type == OpConv);
#else
    uint8_t is_conv = 0;
#endif

#if !JAPARI
    if (!is_conv) {
        return (job_index + 1) * BATCH_SIZE - 1;
    }
#else
    if (!is_conv) {
        if (node->op_type == OpRelu) {
            uint16_t OUTPUT_CHANNEL = output->dims[1];
            if (OUTPUT_CHANNEL % (BATCH_SIZE + 1) != 0) {
                uint8_t jobs_in_a_tile = OUTPUT_CHANNEL / (BATCH_SIZE + 1);
                return job_index / jobs_in_a_tile * OUTPUT_CHANNEL + job_index % jobs_in_a_tile * (BATCH_SIZE + 1) + BATCH_SIZE;
            }
        }
        return (job_index + 1) * (BATCH_SIZE + 1) - 1;
    }
#endif

    /* BEGIN constants */
    uint16_t input_tile_len, input_tile_jobs, jobs_in_a_filter_tile, jobs_in_an_op, output_tile_c, OUTPUT_CHANNEL;
    output_tile_c = node->flags.extra.conv.output_tile_c;
    OUTPUT_CHANNEL = output->dims[1];

#if !INDIRECT_RECOVERY
    // not taking this shortcut for approaches that use indirect recovery as
    // output padding is used in those approaches
    if (output_tile_c == OUTPUT_CHANNEL) {
        return job_index * BATCH_SIZE + BATCH_SIZE - 1;
    }
#endif

    uint16_t OUTPUT_H = output->dims[2], OUTPUT_W = output->dims[3];
    input_tile_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W;
#if JAPARI
    input_tile_jobs = input_tile_len / (BATCH_SIZE + 1);
#else
    input_tile_jobs = input_tile_len / BATCH_SIZE;
#endif
    output_tile_c = upper_gauss(output_tile_c, BATCH_SIZE) * BATCH_SIZE;
    jobs_in_a_filter_tile = OUTPUT_H * OUTPUT_W * output_tile_c / BATCH_SIZE;
    jobs_in_an_op = output_tile_c / BATCH_SIZE;
    // TODO: handle cases where the following condition is not met
    MY_ASSERT(output_tile_c % BATCH_SIZE == 0);
#if JAPARI
    output_tile_c = extend_for_footprints(output_tile_c);
#endif
    /* END constants */

    uint8_t input_tile_c_index = job_index / input_tile_jobs;
    job_index = job_index % input_tile_jobs;
    uint16_t channel_offset = job_index / jobs_in_a_filter_tile * output_tile_c;
    job_index %= jobs_in_a_filter_tile;
    uint32_t offset = input_tile_c_index * input_tile_len +
                      channel_offset;

    if (jobs_in_an_op) {
        // an op contains at least a batch
        offset += OUTPUT_CHANNEL * (job_index / jobs_in_an_op);
#if !JAPARI
        offset += (job_index % jobs_in_an_op + 1) * BATCH_SIZE - 1;
#else
        offset += (job_index % jobs_in_an_op + 1) * (BATCH_SIZE + 1) - 1;
#endif
    } else {
        // TODO
        ERROR_OCCURRED();
    }
    stop_cpu_counter(&Counters::progress_seeking);
    return offset;
}

uint32_t batch_start(uint32_t batch_end_offset) {
#if JAPARI
    return batch_end_offset - BATCH_SIZE;
#else
    return batch_end_offset - (BATCH_SIZE - 1);
#endif
}

#if INDIRECT_RECOVERY

static uint8_t after_recovery = 1;

uint32_t run_recovery(Model *model, ParameterInfo *output) {
    if (!after_recovery) {
        return 0;
    }

    start_cpu_counter();

    // recovery from state bits
    uint32_t end_job_index = output->params_len / 2;
#if JAPARI
    end_job_index /= (BATCH_SIZE + 1);
#endif
    my_printf_debug("end_job_index = %d" NEWLINE, end_job_index);
    uint32_t cur_begin_job_index = 0;
    uint32_t cur_end_job_index = end_job_index;
    uint32_t first_unfinished_job_index = 0;

    my_printf_debug("new_output_state_bit for first value = %d" NEWLINE, -param_state_bit(model, output, 0));
    dump_turning_points_debug(model, output);

    while (1) {
        if (cur_end_job_index - cur_begin_job_index <= 1) {
            if (!value_finished(model, output, cur_begin_job_index)) {
                first_unfinished_job_index = 0;
            } else if (!value_finished(model, output, cur_end_job_index)) {
                first_unfinished_job_index = cur_end_job_index;
            } else if (cur_end_job_index == end_job_index) {
                // all values finished - power failure just before the state
                // bit for the output is flipped
                first_unfinished_job_index = end_job_index;
            } else {
                MY_ASSERT(false);
            }
            break;
        }
        uint32_t middle_job_index = cur_begin_job_index + (cur_end_job_index - cur_begin_job_index) / 2;
        if (value_finished(model, output, middle_job_index)) {
            cur_begin_job_index = middle_job_index;
        } else {
            cur_end_job_index = middle_job_index;
        }
        my_printf_debug(
            "job_index of begin = %" PRId32 ", job_index of end = %" PRId32 NEWLINE,
            cur_begin_job_index, cur_end_job_index
        );
    }

    my_printf_debug("first_unfinished_job_index = %d" NEWLINE, first_unfinished_job_index);

    if (!after_recovery) {
        MY_ASSERT(first_unfinished_job_index == 0);
    } else {
        after_recovery = 0;
    }

    check_feature_map_states(model, output, first_unfinished_job_index, output->params_len / 2, __func__);

    stop_cpu_counter(&Counters::progress_seeking);

    return first_unfinished_job_index;
}
#endif
