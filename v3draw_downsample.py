import numpy as np
import os
import struct
import gc
import psutil
from v3d_io import read_v3d_header, write_v3d_header, load_v3d_chunk, append_v3d_chunk


def get_memory_usage():
    """获取当前内存使用情况(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def downsample_chunk(chunk_data, factors, method='mean'):
    """
    对数据块进行下采样
    
    Args:
        chunk_data: (C, Z, Y, X) 格式的numpy数组（来自v3d_io.load_v3d_chunk）
        factors: 下采样因子 (x, y, z)
        method: 下采样方法
    
    Returns:
        downsampled_data: (C, Z, Y, X) 格式的下采样数组
    """
    # chunk_data格式: (C, Z, Y, X)
    original_shape = chunk_data.shape
    C, Z, Y, X = original_shape
    
    # factors格式: (x, y, z)
    x_factor, y_factor, z_factor = factors
    
    # 计算新的形状
    new_shape = [
        C,                      # C维度不变
        Z // z_factor,          # Z
        Y // y_factor,          # Y  
        X // x_factor           # X
    ]
    
    # 修剪到可被因子整除的大小
    trimmed_shape = [
        C,
        new_shape[1] * z_factor,  # Z
        new_shape[2] * y_factor,  # Y
        new_shape[3] * x_factor   # X
    ]
    
    if list(original_shape) != trimmed_shape:
        chunk_data = chunk_data[:, :trimmed_shape[1], :trimmed_shape[2], :trimmed_shape[3]]
    
    # 为每个通道进行下采样
    downsampled_data = np.zeros(new_shape, dtype=chunk_data.dtype)
    
    for c in range(C):
        channel_data = chunk_data[c, :, :, :]  # (Z, Y, X)
        
        # 重塑数据以便下采样
        reshaped = channel_data.reshape(
            new_shape[1], z_factor,  # Z维度
            new_shape[2], y_factor,  # Y维度
            new_shape[3], x_factor   # X维度
        )
        
        # 应用下采样方法
        if method == 'mean':
            downsampled_data[c, :, :, :] = reshaped.mean(axis=(1, 3, 5))
        elif method == 'max':
            downsampled_data[c, :, :, :] = reshaped.max(axis=(1, 3, 5))
        elif method == 'min':
            downsampled_data[c, :, :, :] = reshaped.min(axis=(1, 3, 5))
        elif method == 'nearest':
            downsampled_data[c, :, :, :] = reshaped[:, 0, :, 0, :, 0]
        else:
            print(f"Unknown method: {method}. Using 'mean'.")
            downsampled_data[c, :, :, :] = reshaped.mean(axis=(1, 3, 5))
    
    return downsampled_data


def downsample_v3d_image_chunked(input_filename, output_filename=None, 
                                factors=(2, 2, 2), method='mean', 
                                chunk_size_z=50, use_chunked=True):
    """
    使用分片模式进行v3d图像下采样，降低内存占用
    
    修复内容：
    1. 修正了v3d格式的轴顺序处理
    2. 优化了分片写入逻辑，避免条纹问题
    3. 确保分片边界与下采样因子对齐
    
    Parameters:
    -----------
    input_filename : str
        输入v3d raw图像文件路径
    output_filename : str, optional
        输出下采样图像文件路径
    factors : tuple, optional
        各空间维度的下采样因子 (x, y, z)，默认(2, 2, 2)
    method : str, optional
        下采样方法: 'mean', 'max', 'min', 'nearest'
    chunk_size_z : int, optional
        Z轴分片大小，默认50
    use_chunked : bool, optional
        是否使用分片模式
    
    Returns:
    --------
    bool: 处理是否成功
    """
    print(f"开始处理文件: {input_filename}")
    print(f"分片模式: {'启用' if use_chunked else '禁用'}")
    print(f"下采样因子: {factors}")
    print(f"下采样方法: {method}")
    
    if use_chunked:
        print(f"Z轴分片大小: {chunk_size_z}")
    
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.0f}MB")
    
    try:
        # 读取文件头信息
        header_info = read_v3d_header(input_filename)
        if not header_info:
            print("Error: 无法读取文件头")
            return False
        
        original_size = header_info['size']  # (X, Y, Z, C)
        print(f"原始图像尺寸: {original_size} (X, Y, Z, C)")
        
        # 转换因子格式 (x, y, z) -> 传递给downsample_chunk
        if len(factors) == 3:
            # factors是(x, y, z)格式，直接传递
            ds_factors_xyz = factors
        else:
            ds_factors_xyz = factors[:3]
        
        # 计算下采样后的尺寸 (按size的顺序: X, Y, Z, C)
        new_size = [
            original_size[0] // ds_factors_xyz[0],  # X
            original_size[1] // ds_factors_xyz[1],  # Y
            original_size[2] // ds_factors_xyz[2],  # Z
            original_size[3]                        # C
        ]
        
        header_info['new_size'] = new_size
        print(f"下采样后尺寸: {new_size} (X, Y, Z, C)")
        print(f"下采样因子 (X, Y, Z): {ds_factors_xyz}")
        
        # 设置输出文件名
        if output_filename is None:
            base, ext = os.path.splitext(input_filename)
            output_filename = f"{base}_downsampled{ext}"
        
        # 写入文件头 - write_v3d_header需要的是(X, Y, Z, C)格式
        if not write_v3d_header(output_filename, new_size, header_info['datatype']):
            print("错误: 无法写入文件头")
            return False
        
        # 关键修复：确保分片大小与下采样因子对齐
        z_factor = ds_factors_xyz[2]  # 使用Z因子
        aligned_chunk_size = (chunk_size_z // z_factor) * z_factor
        if aligned_chunk_size == 0:
            aligned_chunk_size = z_factor
        
        # 计算有效的Z轴范围（确保能被下采样因子整除）
        effective_z_size = (original_size[2] // z_factor) * z_factor
        print(f"有效Z轴大小: {effective_z_size} (原始: {original_size[2]})")
        if aligned_chunk_size != chunk_size_z:
            print(f"分片大小调整: {chunk_size_z} -> {aligned_chunk_size} (对齐到下采样因子)")
        
        # 分片处理
        total_chunks = 0
        processed_z = 0
        
        for z_start in range(0, effective_z_size, aligned_chunk_size):
            z_end = min(z_start + aligned_chunk_size, effective_z_size)
            current_chunk_size = z_end - z_start
            
            print(f"\r处理Z切片 {z_start}-{z_end-1} ({current_chunk_size}层)", end='', flush=True)
            
            # 加载当前分片 - v3d_io.load_v3d_chunk返回(C, Z, Y, X)格式
            chunk_data = load_v3d_chunk(input_filename, z_start, z_end)
            if chunk_data is None:
                print(f"\n错误: 无法加载Z切片 {z_start}-{z_end}")
                return False
            
            # 下采样当前分片 - chunk_data格式为(C, Z, Y, X)
            downsampled_chunk = downsample_chunk(chunk_data, ds_factors_xyz, method)
            
            # 释放原始分片数据
            del chunk_data
            gc.collect()
            
            # 追加下采样后的分片到输出文件
            append_v3d_chunk(output_filename, downsampled_chunk)
            
            # 释放下采样分片数据
            del downsampled_chunk
            gc.collect()
            
            processed_z += current_chunk_size
            total_chunks += 1
            
            # 显示内存使用情况
            current_memory = get_memory_usage()
            if total_chunks % 5 == 0:  # 每5个分片显示一次
                print(f" [内存: {current_memory:.0f}MB]", end='')
        
        final_memory = get_memory_usage()
        print(f"\n处理完成！")
        print(f"输出文件: {output_filename}")
        print(f"处理的分片数: {total_chunks}")
        print(f"最终内存使用: {final_memory:.0f}MB")
        print(f"内存增长: {final_memory - initial_memory:.0f}MB")
        
        return True
        
    except Exception as e:
        error_memory = get_memory_usage()
        print(f"\n处理失败: {e}")
        print(f"错误时内存使用: {error_memory:.0f}MB")
        return False


def downsample_v3d_image_efficient(input_filename, output_filename, 
                                  factors=(2, 2, 2), method='mean', 
                                  chunk_size_z=50):
    """
    高效下采样v3d图像的包装函数
    
    Parameters:
    -----------
    input_filename : str
        输入v3d raw图像文件路径
    output_filename : str, optional
        输出下采样图像文件路径
    factors : tuple, optional
        各空间维度的下采样因子 (x, y, z)，默认(2, 2, 2)
    method : str, optional
        下采样方法: 'mean', 'max', 'min', 'nearest'
    chunk_size_z : int, optional
        Z轴分片大小，仅在use_chunked=True时有效
    
    Returns:
    --------
    dict or bool
        如果use_chunked=False，返回图像字典；如果use_chunked=True，返回成功状态
    """
    success = downsample_v3d_image_chunked(input_filename, output_filename, factors, method, chunk_size_z, True)
    return success


def Batch_DownsampleV3D(input_dir, factors=(4, 4, 2), method='mean', use_chunked=True, chunk_size_z=50):
    """
    批量下采样指定目录下的所有v3d图像文件
    
    Parameters:
    -----------
    input_dir : str
        输入目录，包含待处理的v3d图像文件
    factors : tuple, optional
        各空间维度的下采样因子 (x, y, z)，默认(2, 2, 2)
    method : str, optional
        下采样方法: 'mean', 'max', 'min', 'nearest'
    use_chunked : bool, optional
        是否使用分片模式（推荐用于大文件）
    chunk_size_z : int, optional
        Z轴分片大小，仅在use_chunked=True时有效
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('_8bit.v3draw'):
                input_file = os.path.join(root, file)
                base_name = file.split('.')[0]
                output_file = os.path.join(root, f"{base_name}_downsampled.v3draw")

                if os.path.exists(output_file):
                    print(f"{output_file} exist, skip")
                    continue
                
                # 使用分片模式（推荐，内存占用低）
                success = downsample_v3d_image_efficient(
                    input_file, 
                    output_file,
                    factors=factors,
                    method=method,
                    use_chunked=use_chunked,
                    chunk_size_z=chunk_size_z
                )
                
                if success:
                    print("下采样成功完成！")
                else:
                    print("下采样失败！")


def Single_Downsample(input_file, output_file=None, 
                      factors=(4, 4, 2), method='mean', 
                      use_chunked=True, chunk_size_z=50):
    """
    单个文件下采样入口函数

    Parameters:
    -----------
    input_file : str
        输入v3d raw图像文件路径
    output_file : str, optional
        输出下采样图像文件路径
    factors : tuple, optional
        各空间维度的下采样因子 (x, y, z)，默认(2, 2, 2)
    method : str, optional
        下采样方法: 'mean', 'max', 'min', 'nearest'
    use_chunked : bool, optional
        是否使用分片模式（推荐用于大文件）
    chunk_size_z : int, optional
        Z轴分片大小，仅在use_chunked=True时有效
    """

    # 如果没有指定输出文件名，则生成默认输出文件名
    if output_file is None:
        directory = os.path.dirname(input_file)
        filename = os.path.basename(input_file)
        base_name = filename.split('.')[0]
        output_file = os.path.join(directory, f"{base_name}_downsampled.v3draw")

    return downsample_v3d_image_efficient(
        input_file,
        output_file,
        factors=factors,
        method=method,
        chunk_size_z=chunk_size_z
    )

if __name__ == "__main__":
    # python3 {script_path} --image-path "{image_path}"
    # 解析参数 --image-path
    # import argparse
    # parser = argparse.ArgumentParser(description="Downsample v3d images.")
    # parser.add_argument('--image-path', type=str, required=True, help='Path to the v3d image file or directory.')
    # args = parser.parse_args()

    # input_path = args.image_path.strip()

    input_path = R"D:\Workspace\h5_to_v3draw\Data\H5\P095_T01_R01_S004\P095_T01_R01_S004_8bit.v3draw"
    if os.path.isdir(input_path):
        # 批量处理目录下的所有v3d文件
        Batch_DownsampleV3D(input_path)
    else:
        # 单个文件处理
        Single_Downsample(input_path)
