import transfer_h5_to_v3draw as h5_v3draw
import os


if __name__ == "__main__":
    # H5图像所在文件夹
    H5_Image_Input_Path = "/mnt/d/Workspace/h5_to_v3draw/Data/H5"
    # 结果输出文件夹
    Output_Path = "/mnt/d/Workspace/h5_to_v3draw/Data/Output"

    if os.path.exists(Output_Path) is False:
        os.makedirs(Output_Path)
    h5files = os.listdir(H5_Image_Input_Path)
    print(f"Found H5 files: {len(h5files)}")
    for h5file in h5files:
        print("Processing: " + h5file)

        full_h5file_path = os.path.join(H5_Image_Input_Path, h5file)

        v3draw_output_path = os.path.join(Output_Path, "V3draw")
        if os.path.exists(v3draw_output_path) is False:
            os.makedirs(v3draw_output_path)

        terafly_output_path = os.path.join(Output_Path, "Terafly")
        if os.path.exists(terafly_output_path) is False:
            os.makedirs(terafly_output_path)

        print(f"开始处理: {h5file}, h5到v3draw转换")
        h5_v3draw.Single_H5ToV3draw(full_h5file_path, v3draw_output_path)
        print(f"完成处理: {h5file}, h5到v3draw转换")

        print(f"开始处理: {h5file}, h5到Terafly转换")
        filename = os.path.basename(full_h5file_path)
        spRes = filename.split(".")
        if len(spRes) > 0:
            filename = spRes[0]
        input_v3draw_file = os.path.join(v3draw_output_path, filename, filename + ".v3draw")
        terafly_output_file_path = os.path.join(terafly_output_path, h5file+"_Terafly")
        if os.path.exists(terafly_output_file_path) is False:
            os.makedirs(terafly_output_file_path)
        command = f"./teraconverter64 -s=\"{input_v3draw_file}\" -d=\"{terafly_output_file_path}\" --resolutions=012 --width=256 --height=256 --depth=256 --sfmt=\"Vaa3D raw\" --dfmt=\"TIFF (tiled, 3D)\" --libtiff_rowsperstrip=-1 --halve=max"
        # 执行命令
        os.system(command)
        print(f"完成处理: {h5file}, h5到Terafly转换")
    print(f"全部处理完成：{len(h5files)}")
