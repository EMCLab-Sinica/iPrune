#include <driverlib.h>
#include <stdint.h>
#include "FreeRTOSConfig.h"
#include "common.h"
#include "platform.h"

/* on FRAM */

#pragma NOINIT(_intermediate_values)
static uint8_t _intermediate_values[NUM_SLOTS * INTERMEDIATE_VALUES_SIZE];
uint8_t *intermediate_values = _intermediate_values;

#pragma NOINIT(_counters)
static uint16_t _counters[COUNTERS_LEN];
uint16_t *counters = _counters;

#pragma NOINIT(_power_counters)
static uint16_t _power_counters[COUNTERS_LEN];
uint16_t *power_counters = _power_counters;

#pragma NOINIT(_counter_idx)
static uint8_t _counter_idx;
uint8_t *counter_idx = &_counter_idx;

#define MY_DMA_CHANNEL DMA_CHANNEL_0
static DMA_initParam dma_params = {
    .channelSelect = MY_DMA_CHANNEL,
    .transferModeSelect = DMA_TRANSFER_BLOCK,
};

void my_memcpy(void* dest, const void* src, size_t n) {
    DMA_init(&dma_params);
    DMA_setSrcAddress(MY_DMA_CHANNEL, (uint32_t)(src), DMA_DIRECTION_INCREMENT);
    DMA_setDstAddress(MY_DMA_CHANNEL, (uint32_t)(dest), DMA_DIRECTION_INCREMENT);
    /* transfer size is in words (2 bytes) */
    DMA_setTransferSize(MY_DMA_CHANNEL, (n) >> 1);
    // DMA_enableInterrupt(MY_DMA_CHANNEL);
    DMA_enableTransfers(MY_DMA_CHANNEL);
    DMA_startTransfer(MY_DMA_CHANNEL);
}

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

#pragma vector=configTICK_VECTOR
__interrupt void vTimerHandler( void )
{
    // one tick is configured as roughly 1 millisecond
    // See vApplicationSetupTimerInterrupt() in main.h and FreeRTOSConfig.h
    counters[*counter_idx]++;
}
