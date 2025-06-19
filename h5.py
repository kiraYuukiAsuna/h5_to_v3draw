import h5py
import numpy as np


def get_shape_from_h5file(input_file:str):
    """
    mode==0: return shape
    mode==1: return arrange method of images (rows, cols)
    """
    with h5py.File(input_file, 'r') as f:
        base_shape = f['0']['0']['0'].shape
        base_score = np.log(base_shape[1]/base_shape[2])
        print('base_score:',base_score)
        shapes = {}
        arrangements = {}
        for level in f['0']:
            size = len(f['0'][level])
            # 乘法分解
            arranges = factor_pairs(size)
            min_score = 1e9
            arrange = (0,0)
            target_shape = (0,0,0)
            for row,col in arranges:
                size0 = f['0'][level]['0'].shape[0]
                # print([f['0'][level][str(i)].shape[1] for i in range(col)])
                size1 = sum(f['0'][level][str(i)].shape[1] for i in range(col))
                # print('size1:',size1)
                # print([f['0'][level][str(i)].shape[2] for i in range(0,size,col)])
                size2 = sum(f['0'][level][str(i)].shape[2] for i in range(0,size,col))
                # print('size2',size2)
                score = abs(np.log(size1/size2)-base_score)
                print('score for',row,col,'is',score)
                if score < min_score:
                    min_score = score
                    shape = (size0,size1,size2)
                    arrange = (row,col)
            shapes[level] = shape
            arrangements[level] = arrange
        return shapes,arrangements

def factor_pairs(n):
    if n < 1:
        raise ValueError("Input must be a positive integer.")

    pairs = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            pairs.append((i, n // i))
            if i != n // i:
                pairs.append((n // i, i))
    return pairs

#filepath = r'E:\gqb_workspace\cell_matching\test\P00095-T001-R001-S031-4-OMZ.pyramid.h5'
while True:
    filepath = input('输入图像路径:').strip()
    if not filepath:
        break

    with h5py.File(filepath, 'r') as f:
        print(type(f['0']['0']['0']),f['0']['0']['0'].shape)
        image = f['0']['0']['0'][:]
        print(type(image),image.shape)
        if len(image.shape) == 4:
            print('图像有%d个通道'%image.shape[-1])
        print('图像有%d层分辨率'%len(f['0']))
        for key1 in f['0']:
            print('第%d层：'%(int(key1)+1),end = ' ')
            print('有%d张图像'%len(f['0'][key1]),'每一张大小为')
            shapes = []
            for key2 in f['0'][key1]:
                if int(key1)<3:
                    print(key2,end=',')
                if f['0'][key1][key2].shape not in shapes:
                    shapes.append(f['0'][key1][key2].shape)
            if int(key1)<3: print()
            print(shapes)
        print('---------------')
        """
        for key2 in range(256):
            key2 = str(key2)
            print(f['0'][key1][key2][:].shape)
        method = 2
        image = []
        for i in range(method):
            image.append(np.concatenate((f['0']['1'][str(j)][:] for j in range(i * method, i * method + method)),
                                        axis=2))
        image = np.concatenate(image, axis=1)
        """
    s,a = get_shape_from_h5file(filepath)
    print(s)
    print(a)
    print('END\n')
