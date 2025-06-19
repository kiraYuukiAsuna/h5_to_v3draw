# Terafly 数据格式转换：
1. 桌面目录下（/home/seuallen/Desktop/transfer_h5_to_v3draw_New）transfer_h5_to_v3draw_New文件夹有脚本transfer_h5_to_v3draw.py，先修改H5数据所在路径和terafly保存路径（ input_folder output_folder，两个路径不要相同），再运行这个脚本。

最终output_folder文件夹中每个文件夹下有v3draw格式和terafly格式的图像，根据需要拷贝。

注意，在拷贝后，一定要弹出硬盘再拔出，文件资源管理器中看到的拷贝完成不一定真的完成了，可能后台还在拷贝，所以最好点击弹出后等待弹窗提示说"已与文件系统断开连接"后再拔出硬盘，不然有些文件可能还未拷贝完成。

# 如何运行脚本：
1. 可以使用VSCode打开transfer_h5_to_v3draw_New文件夹，然后检查右下角Python环境是否选中的是3.12.3(".venv")的环境，
如果是，则打开transfer_h5_to_v3draw.py直接在右上角三角图标点击运行即可。

2. 或者在transfer_h5_to_v3draw_New文件夹中右键在终端打开，输入下面的命令
./.venv/bin/python "transfer_h5_to_v3draw.py"
运行即可

