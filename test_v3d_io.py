#!/usr/bin/env python3
"""
测试v3d_io.py的接口规范
"""

import numpy as np
from v3d_io import read_v3d_header, write_v3d_header, load_v3d_chunk, append_v3d_chunk

def test_v3d_io():
    """测试v3d_io模块的接口"""
    
    # 测试文件
    test_file = "test.v3draw"
    
    # 创建测试数据
    X, Y, Z, C = 100, 200, 50, 1
    test_data = np.random.randint(0, 65535, size=(C, Z, Y, X), dtype=np.uint16)
    
    print(f"测试数据维度: {test_data.shape} (C, Z, Y, X)")
    
    # 1. 测试写入文件头
    print("\n1. 测试写入文件头...")
    success = write_v3d_header(test_file, (X, Y, Z, C), datatype=2)
    print(f"写入文件头: {'成功' if success else '失败'}")
    
    # 2. 测试写入数据
    print("\n2. 测试写入数据...")
    success = append_v3d_chunk(test_file, test_data)
    print(f"写入数据: {'成功' if success else '失败'}")
    
    # 3. 测试读取文件头
    print("\n3. 测试读取文件头...")
    header_info = read_v3d_header(test_file)
    if header_info:
        print(f"文件头信息: {header_info}")
        print(f"尺寸: {header_info['size']} (X, Y, Z, C)")
    else:
        print("读取文件头失败")
    
    # 4. 测试读取数据
    print("\n4. 测试读取数据...")
    chunk_data = load_v3d_chunk(test_file, 0, Z)
    if chunk_data is not None:
        print(f"读取数据维度: {chunk_data.shape} (C, Z, Y, X)")
        print(f"数据类型: {chunk_data.dtype}")
        
        # 验证数据一致性
        if np.array_equal(test_data, chunk_data):
            print("数据一致性验证: 通过")
        else:
            print("数据一致性验证: 失败")
    else:
        print("读取数据失败")
    
    # 5. 测试部分读取
    print("\n5. 测试部分读取...")
    partial_chunk = load_v3d_chunk(test_file, 10, 30)
    if partial_chunk is not None:
        print(f"部分读取维度: {partial_chunk.shape} (C, Z, Y, X)")
        expected_z = 30 - 10
        if partial_chunk.shape[1] == expected_z:
            print(f"部分读取Z维度验证: 通过 (期望{expected_z}, 实际{partial_chunk.shape[1]})")
        else:
            print(f"部分读取Z维度验证: 失败 (期望{expected_z}, 实际{partial_chunk.shape[1]})")
    else:
        print("部分读取失败")
    
    print("\n测试完成")

if __name__ == "__main__":
    test_v3d_io()
