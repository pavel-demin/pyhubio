import usb1
import numpy as np
import time


class PyhubIO:
    def __init__(self):
        self.context = None
        self.device = None
        self.timeout = 1000

    def start(self):
        if self.device:
            return
        self.context = usb1.USBContext()
        self.device = self.context.openByVendorIDAndProductID(0x0403, 0x6014)
        if self.device is None:
            raise Exception("unable to access USB device")
        # reset mode
        self.device.controlWrite(0x40, 0x0B, 0x0000, 0x01, bytes(), self.timeout)
        # sync fifo mode
        self.device.controlWrite(0x40, 0x0B, 0x4000, 0x01, bytes(), self.timeout)
        # latency timer
        self.device.controlWrite(0x40, 0x09, 0x00FF, 0x01, bytes(), self.timeout)

    def stop(self):
        if self.device:
            self.context.close()
            self.context = None
            self.device = None

    def flush(self):
        if self.device:
            # read possible residual data
            while len(self.device.bulkRead(0x81, 512, self.timeout)) > 2:
                continue
            # purge buffers
            self.device.controlWrite(0x40, 0x00, 0x0001, 0x01, bytes(), self.timeout)
            self.device.controlWrite(0x40, 0x00, 0x0002, 0x01, bytes(), self.timeout)

    def write(self, data, port=0, addr=0):
        if self.device:
            view = data.view(np.uint8)
            for part in np.split(view, np.arange(16384, view.size, 16384)):
                size = part.size // 4
                command = np.uint32((1 << 31) + ((size - 1) << 19) + (port << 16) + addr)
                addr = (addr + 4096) & 0xFFFF
                self.device.bulkWrite(0x02, command.tobytes(), self.timeout)
                self.device.bulkWrite(0x02, part.tobytes(), self.timeout)

    def read(self, data, port=1, addr=0):
        if self.device:
            view = data.view(np.uint8)
            size = view.size // 4
            div, mod = divmod(size, 4096)
            command = np.zeros(div + (mod > 0), np.uint32)
            for i in range(div):
                command[i] = (4095 << 19) + (port << 16) + addr
                addr = (addr + 4096) & 0xFFFF
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

    def edge(self, data, bit, positive=True, addr=0):
        result = data
        if self.device:
            command = (1 << 31) + addr
            mask = 1 << bit
            lo = data & ~mask
            hi = data | mask
            if positive:
                sequence = np.uint32([command, lo, command, hi])
                result = hi
            else:
                sequence = np.uint32([command, hi, command, lo])
                result = lo
            self.device.bulkWrite(0x02, sequence.tobytes(), self.timeout)
        return result


class PyhubJTAG:
    def __init__(self):
        self.context = None
        self.device = None
        self.timeout = 1000

    def start(self):
        if self.device:
            return
        self.context = usb1.USBContext()
        self.device = self.context.openByVendorIDAndProductID(0x0403, 0x6010)
        if self.device is None:
            raise Exception("unable to access USB device")
        # reset mode
        self.device.controlWrite(0x40, 0x0B, 0x0000, 0x01, bytes(), self.timeout)
        # mpsse mode
        self.device.controlWrite(0x40, 0x0B, 0x028B, 0x01, bytes(), self.timeout)
        # latency timer
        self.device.controlWrite(0x40, 0x09, 0x0001, 0x01, bytes(), self.timeout)

    def stop(self):
        if self.device:
            self.context.close()
            self.context = None
            self.device = None

    def flush(self):
        if self.device:
            # read possible residual command
            while len(self.device.bulkRead(0x81, 512, self.timeout)) > 2:
                continue
            # purge buffers
            self.device.controlWrite(0x40, 0x00, 0x0001, 0x01, bytes(), self.timeout)
            self.device.controlWrite(0x40, 0x00, 0x0002, 0x01, bytes(), self.timeout)

    def setup(self):
        if self.device:
            buffer = np.zeros(2, np.uint8)
            # loopback
            command = [0x85, 0xAB]
            self.write(np.uint8(command))
            self.read(buffer)
            # clock
            command = [0x8A, 0x86, 0x00, 0x00, 0xAB]
            self.write(np.uint8(command))
            self.read(buffer)
            # gpio
            command = [0x80, 0x80, 0x8B, 0xAB]
            self.write(np.uint8(command))
            self.read(buffer)

    def write(self, data):
        if self.device:
            self.device.bulkWrite(0x02, data.tobytes(), self.timeout)

    def read(self, data):
        if self.device:
            view = data.view(np.uint8)
            offset = 0
            limit = view.size
            while offset < limit:
                buffer = self.device.bulkRead(0x81, 16384, self.timeout)
                buffer = np.frombuffer(buffer, np.uint8)
                buffer = buffer[np.mod(np.arange(buffer.size), 512) > 1]
                size = buffer.size
                view[offset : offset + size] = buffer
                offset += size

    def tms(self, data, bits):
        command = [0x4B, bits - 1, data]
        self.write(np.uint8(command))

    def idle(self):
        self.tms(0x1F, 6)

    def shift_dr(self):
        self.tms(0x01, 3)

    def shift_ir(self):
        self.tms(0x03, 4)

    def shift_bits(self, data, bits):
        bits -= 1
        command = []
        if bits > 0:
            command += [0x1B, bits - 1, data]
        command += [0x4B, 2, ((data >> bits) & 1) << 7 | 0x03]
        self.write(np.uint8(command))

    def shift_bytes(self, data):
        view = data.view(np.uint8)
        for part in np.split(view[:-1], np.arange(65536, view.size - 1, 65536)):
            size = part.size - 1
            command = [0x19, size & 0xFF, size >> 8]
            self.write(np.uint8(command))
            self.write(part)
        self.shift_bits(view[-1], 8)

    def idcode(self):
        if self.device:
            buffer = np.zeros(1, np.uint32)
            self.shift_ir()
            self.shift_bits(0x09, 6)
            self.shift_dr()
            command = [0x2C, 0x03, 0x00]
            self.write(np.uint8(command))
            self.read(buffer)
            self.idle()
            return buffer[0] & 0x0FFFFFFF
        else:
            return 0x0FFFFFFF

    def program(self, path):
        if self.device:
            data = np.fromfile(path, np.uint8)
            for i in range(data.size - 4):
                if np.array_equal(data[i : i + 4], [0xAA, 0x99, 0x55, 0x66]):
                    break
            data = data[i:]
            data = ((data >> 1) & 0x55) | ((data & 0x55) << 1)
            data = ((data >> 2) & 0x33) | ((data & 0x33) << 2)
            data = ((data >> 4) & 0x0F) | ((data & 0x0F) << 4)
            # jprogram
            self.shift_ir()
            self.shift_bits(0x0B, 6)
            self.idle()
            time.sleep(0.01)
            # cfg_in
            self.shift_ir()
            self.shift_bits(0x05, 6)
            self.shift_dr()
            self.shift_bytes(data)
