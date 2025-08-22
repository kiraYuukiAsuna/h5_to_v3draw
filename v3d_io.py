import sys
import os
import struct
import numpy as np


def read_v3d_header(filename):
    """
    读取v3d文件头信息，不加载图像数据

    Returns:
    --------
    dict: 包含文件头信息的字典，或None如果出错
    """
    try:
        with open(filename, 'rb') as f_obj:
            # 读取格式键
            len_formatkey = len('raw_image_stack_by_hpeng')
            formatkey = f_obj.read(len_formatkey)
            formatkey = struct.unpack(str(len_formatkey) + 's', formatkey)
            if formatkey[0] != b'raw_image_stack_by_hpeng':
                print("ERROR: File unrecognized (not raw, v3draw) or corrupted.")
                return None

            # 读取字节序
            endiancode = f_obj.read(1)
            endiancode = struct.unpack('c', endiancode)[0]
            if endiancode != b'B' and endiancode != b'L':
                print("ERROR: Only supports big- or little- endian.")
                return None

            # 读取数据类型
            datatype = f_obj.read(2)
            if endiancode == b'L':
                datatype = struct.unpack('<h', datatype)[0]
            else:
                datatype = struct.unpack('>h', datatype)[0]
            
            if datatype < 1 or datatype > 4:
                print(f"ERROR: Unrecognized data type code [{datatype}].")
                return None

            # 读取图像尺寸 (X Y Z C)
            size = f_obj.read(4 * 4)
            if endiancode == b'L':
                size = struct.unpack('<4l', size)
            else:
                size = struct.unpack('>4l', size)

            return {
                'endian': endiancode,
                'datatype': datatype,
                'size': size,
                'header_size': f_obj.tell()
            }
    except Exception as e:
        print(f"Error reading header: {e}")
        return None
    

def write_v3d_header(filename, size, datatype=2):
    """
    写入v3d文件头

    Args:
        filename: 输出文件路径
        size: 图像尺寸 (X, Y, Z, C)
        datatype: 数据类型 (1=uint8, 2=uint16, 4=uint32)
    """
    try:
        f_obj = open(filename, 'wb')
    except:
        print(f"ERROR: Failed to create file [{filename}]")
        return False

    try:
        # write image header - formatkey(24bytes)
        formatkey = b'raw_image_stack_by_hpeng'
        formatkey = struct.pack('<24s', formatkey)
        f_obj.write(formatkey)

        # write image header - endianCode(1byte)
        endiancode = b'L'
        endiancode = struct.pack('<s', endiancode)
        f_obj.write(endiancode)

        # write image header - datatype(2bytes)
        datatype = struct.pack('<h', datatype)  # 'h' = short
        f_obj.write(datatype)

        # write image header - size(4*4bytes)
        size_packed = struct.pack('<4l', size[0], size[1], size[2], size[3])  # 'l' = long
        f_obj.write(size_packed)

        f_obj.close()
        return True
    except Exception as e:
        print(f"ERROR: Failed to write header to [{filename}]: {e}")
        f_obj.close()
        return False


def append_v3d_chunk(filename, chunk_data):
    """
    向v3d文件追加数据块

    Args:
        filename: v3d文件路径
        chunk_data: 数据块，应该是 (C, Z, Y, X) 格式的numpy数组
    """
    try:
        f_obj = open(filename, 'ab')  # 追加模式
    except:
        print(f"ERROR: Failed to open file [{filename}] for appending")
        return False

    try:
        # 确保数据顺序为 (C, Z, Y, X)
        if chunk_data.ndim != 4:
            print(f"ERROR: chunk_data should be 4D (C,Z,Y,X), got {chunk_data.ndim}D")
            f_obj.close()
            return False

        # 写入数据块
        # v3d格式存储顺序为 (C, Z, Y, X)，直接写入
        chunk_bytes = chunk_data.tobytes()
        f_obj.write(chunk_bytes)

        f_obj.close()
        return True
    except Exception as e:
        print(f"ERROR: Failed to append chunk to [{filename}]: {e}")
        f_obj.close()
        return False


def load_v3d_chunk(filename, z_start, z_end):
    """
    从v3d文件加载指定Z范围的数据块

    Args:
        filename: v3d文件路径
        z_start: Z起始位置
        z_end: Z结束位置
    Returns:
        chunk_data: (C, Z, Y, X) 格式的numpy数组  # 修改注释
    """
    try:
        f_obj = open(filename, 'rb')
    except FileNotFoundError:
        print(f"ERROR: Failed in reading [{filename}]")
        return None

    try:
        # 读取文件头
        len_formatkey = len('raw_image_stack_by_hpeng')
        formatkey = f_obj.read(len_formatkey)
        formatkey = struct.unpack(str(len_formatkey) + 's', formatkey)
        if formatkey[0] != b'raw_image_stack_by_hpeng':
            print("ERROR: File unrecognized (not raw, v3draw) or corrupted.")
            f_obj.close()
            return None

        # 读取端序
        endiancode = f_obj.read(1)
        endiancode = struct.unpack('c', endiancode)[0]

        # 读取数据类型
        datatype = f_obj.read(2)
        if endiancode == b'L':
            datatype = struct.unpack('<h', datatype)[0]
        else:
            datatype = struct.unpack('>h', datatype)[0]

        # 读取尺寸
        size = f_obj.read(16)
        if endiancode == b'L':
            size = struct.unpack('<4l', size)
        else:
            size = struct.unpack('>4l', size)

        X, Y, Z, C = size[0], size[1], size[2], size[3]

        # 确定数据类型
        if datatype == 1:
            dtype = np.uint8
            bytes_per_pixel = 1
        elif datatype == 2:
            dtype = np.uint16
            bytes_per_pixel = 2
        else:
            dtype = np.uint32
            bytes_per_pixel = 4

        # 计算Z范围
        z_start = max(0, z_start)
        z_end = min(Z, z_end)
        chunk_z = z_end - z_start

        if chunk_z <= 0:
            f_obj.close()
            return None

        # 计算数据偏移
        pixels_per_z_per_channel = X * Y
        bytes_per_z_per_channel = pixels_per_z_per_channel * bytes_per_pixel

        # 跳过前面的Z层
        skip_bytes = z_start * bytes_per_z_per_channel * C
        f_obj.seek(f_obj.tell() + skip_bytes)

        # 读取目标Z范围的数据
        read_bytes = chunk_z * bytes_per_z_per_channel * C
        data_bytes = f_obj.read(read_bytes)

        if len(data_bytes) != read_bytes:
            print(f"ERROR: Expected {read_bytes} bytes, got {len(data_bytes)}")
            f_obj.close()
            return None

        # 重构数据
        chunk_data = np.frombuffer(data_bytes, dtype=dtype)
        chunk_data = chunk_data.reshape((C, chunk_z, Y, X))

        f_obj.close()
        return chunk_data
    except Exception as e:
        print(f"ERROR: Failed to load chunk from [{filename}]: {e}")
        f_obj.close()
        return None
