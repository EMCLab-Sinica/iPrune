import glob
import logging
import platform
import os
import subprocess
from subprocess import PIPE

current_os = platform.system()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('minicom-launcher')

def find_msp430_usb_interfaces():
    from ctypes import CDLL, POINTER, c_char_p, c_int32

    try:
        libmsp430 = CDLL('libmsp430.so')
    except OSError:
        logger.info('libmsp430.so is not found, skipping detection of MSP430 debugging interfaces')
        return []

    # Adopted from the example in https://www.ti.com/lit/ug/slau656b/slau656b.pdf
    STATUS_OK = 0
    MSP430_GetNumberOfUsbIfs = libmsp430.MSP430_GetNumberOfUsbIfs
    MSP430_GetNumberOfUsbIfs.argtypes = (
        POINTER(c_int32),   # int32_t* Number
    )
    MSP430_GetNumberOfUsbIfs.restype = c_int32
    MSP430_GetNameOfUsbIf = libmsp430.MSP430_GetNameOfUsbIf
    MSP430_GetNameOfUsbIf.argtypes = (
        c_int32,            # int32_t Idx
        POINTER(c_char_p),  # char** Name
        POINTER(c_int32),   # int32_t* Status
    )
    MSP430_Error_Number = libmsp430.MSP430_Error_Number
    MSP430_Error_Number.argtypes = ()
    MSP430_Error_Number.restype = c_int32

    interface_names = []

    number = c_int32()
    if MSP430_GetNumberOfUsbIfs(number) != STATUS_OK:
        logger.error('Could not determine number of USB interfaces. Error = ', MSP430_Error_Number().value)
        return

    for idx in range(number.value):
        name = c_char_p()
        status = c_int32()
        if MSP430_GetNameOfUsbIf(idx, name, status) != STATUS_OK:
            logger.error('Could not obtain port name of USB interface. Error = ', MSP430_Error_Number().value)
            continue
        interface_names.append(name.value.decode('ascii'))

    logger.debug('Found %s USB debugging interfaces: %s', number.value, ', '.join(interface_names))

    return interface_names

def find_430_macOS():
    """
    Find MSP430 device path
        args: None
        return: [path_to_dev]
    """
    DEVICE_PATH = "/dev"
    devices = os.listdir(DEVICE_PATH)
    usb_devices = [dev for dev in devices if dev.startswith("cu.usbmodem")]
    usb_devices.sort()
    msp430_uart_terminals = [os.path.join(DEVICE_PATH, dev)
                             for dev in usb_devices if dev.endswith("03")]

    return msp430_uart_terminals

def find_430_Linux():
    msp430_usb_interfaces = find_msp430_usb_interfaces()
    ret = []
    for serial_interface in glob.glob("/dev/serial/by-id/*"):
        serial_interface_basename = os.path.basename(os.readlink(serial_interface))
        if serial_interface_basename in msp430_usb_interfaces:
            logger.info('%s (%s) is detected as a debugging interface, skipping...',
                        serial_interface, serial_interface_basename)
            continue
        ret.append(serial_interface)
    return sorted(ret)

def check_minicom():
    try:
        minicom_version_raw = subprocess.run(["minicom", "-v"], check=False, stdout=PIPE).stdout
    except FileNotFoundError:
        print("Minicom is not installed")
        exit(1)

    minicom_version = minicom_version_raw.split()[2].decode()
    return minicom_version

def open_minicom(device, baudrate, current_os):
    cmd = ["minicom", f"--device={device}", "--baudrate", f"{baudrate}"]
    if current_os == "Darwin":
        cmd.append("-m")
    subprocess.run(cmd)

def baudrate(target_dev):
    if 'Cypress' in target_dev:
        return 115200
    else:
        return 9600

def shell(device_list: list, current_os: str):

    minicom_version = check_minicom()

    print(f"EMCLab MSP430 Minicom Connector (minicom version {minicom_version})\n")
    print("Found {} MSP430 UART Terminal(s)".format(len(device_list)))
    for index, device in enumerate(device_list):
        print(f"{index}\t{device}")

    if len(device_list) > 1:
        dev_num = input("Device to connect {}: ".format([n for n in range(len(device_list))]))

        try:
            dev_num = int(dev_num)
        except ValueError:
            print("Exit")
            exit(0)
    elif len(device_list) == 1:
        dev_num = 0
    else:
        print("No devices found")
        exit(0)

    try:
        target_dev = device_list[dev_num]
    except IndexError as e:
        print(e)
        exit(1)

    open_minicom(target_dev, baudrate(target_dev), current_os)

if __name__ == "__main__":
    msp430_devices = []
    os.environ['LC_MESSAGES'] = 'en_US.UTF-8'
    if current_os == "Darwin": # macOS
        msp430_devices = find_430_macOS()
    elif current_os == "Linux":
        msp430_devices = find_430_Linux()

    shell(msp430_devices, current_os)
