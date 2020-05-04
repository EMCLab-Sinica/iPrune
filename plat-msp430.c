#include <driverlib.h>
#include <stdint.h>
#include "FreeRTOSConfig.h"
#include "cnn_common.h"
#include "platform.h"

/* on FRAM */

#pragma DATA_SECTION(_intermediate_values, ".nvm")
static uint8_t _intermediate_values[NUM_SLOTS * INTERMEDIATE_VALUES_SIZE];
uint8_t *intermediate_values(uint8_t slot_id, uint8_t will_write) {
    return _intermediate_values + slot_id * INTERMEDIATE_VALUES_SIZE;
}

#pragma DATA_SECTION(_counters, ".nvm")
static Counters _counters;
Counters *counters() {
    return &_counters;
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
    counters()->time_counters[counters()->counter_idx]++;
}

void setOutputValue(uint8_t value)
{
    if (value) {
        GPIO_setOutputHighOnPin(GPIO_PORT_P1, GPIO_PIN3);
    } else {
        GPIO_setOutputLowOnPin(GPIO_PORT_P1, GPIO_PIN3);
    }
}