import usb1
import numpy as np


class PyhubIO:
    def __init__(self):
        context = usb1.USBContext()
        self.device = context.openByVendorIDAndProductID(0x0403, 0x6014)
        if self.device is None:
            raise Exception("unable to access USB device")
        self.timeout = 1000
        # reset mode
        self.device.controlWrite(0x40, 0x0B, 0x00FF, 0x01, bytes(), self.timeout)
        # sync fifo mode
        self.device.controlWrite(0x40, 0x0B, 0x40FF, 0x01, bytes(), self.timeout)
        # latency timer
        self.device.controlWrite(0x40, 0x09, 0x00FF, 0x01, bytes(), self.timeout)

    def flush(self):
        # read possible residual data
        while len(self.device.bulkRead(0x81, 512, self.timeout)) > 2:
            continue
        # purge buffers
        self.device.controlWrite(0x40, 0x00, 0x0001, 0x01, bytes(), self.timeout)
        self.device.controlWrite(0x40, 0x00, 0x0002, 0x01, bytes(), self.timeout)

    def write(self, data, port=0, addr=0, incr=False):
        view = data.view(np.uint8)
        for part in np.split(view, np.arange(16384, view.size, 16384)):
            size = part.size // 4
            command = np.uint32((1 << 31) + ((size - 1) << 19) + (port << 16) + addr)
            if incr:
                addr += 4096
            self.device.bulkWrite(0x02, command.tobytes(), self.timeout)
            self.device.bulkWrite(0x02, part.tobytes(), self.timeout)

    def read(self, data, port=1, addr=0, incr=False):
        view = data.view(np.uint8)
        size = view.size // 4
        div, mod = divmod(size, 4096)
        command = np.zeros(div + (mod > 0), np.uint32)
        for i in range(div):
            command[i] = (4095 << 19) + (port << 16) + addr
            if incr:
                addr += 4096
        if mod > 0:
            command[-1] = ((mod - 1) << 19) + (port << 16) + addr
        self.device.bulkWrite(0x02, command.tobytes(), self.timeout)
        offset = 0
        limit = size * 4
        while offset < limit:
            buffer = self.device.bulkRead(0x81, 16384, self.timeout)
            buffer = np.frombuffer(buffer, np.uint8)
            buffer = buffer[np.mod(np.arange(buffer.size), 512) > 1]
            size = buffer.size
            view[offset : offset + size] = buffer
            offset += size
