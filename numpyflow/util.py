import numpy as np
import copy


def im2col(input, filter_h, filter_w, stride=1, padding=0):
    N, C, H, W = input.shape
    FH, FW, S, P = filter_h, filter_w, stride, padding
    OH = int(1+(H+2*P-FH)/S)
    OW = int(1+(W+2*P-FW)/S)
    temp = np.zeros((N, C, H+2*P, W+2*P))
    temp[:,:,P:P+H,P:P+W] = copy.deepcopy(input)
    output = np.zeros((N,OH*OW,C*FH*FW))
    for i in range(OH*OW):
        output[:, i] = temp[:, :,(i//OW)*S:FH+(i//OW)*S,(i%OW)*S:FW+(i%OW)*S].reshape(N, -1)
    
    return output


def col2im(input, out_height, out_width, filter_h, filter_w, stride = 1, padding = 0):
    N, C, H, W = input.shape[0], input.shape[2]/(filter_h*filter_w), out_height, out_width
    FH, FW, S, P = filter_h, filter_w, stride, padding
    OH = int(1+(H+2*P-FH)/S)
    OW = int(1+(W+2*P-FW)/S)

    pass



def im2col(input_data, filter_h, filter_w, stride=1, padding=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    padding : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*padding - filter_h)//stride + 1
    out_w = (W + 2*padding - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, padding=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    padding

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*padding - filter_h)//stride + 1
    out_w = (W + 2*padding - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*padding + stride - 1, W + 2*padding + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding:H + padding, padding:W + padding]


def label_to_onehotkey(x,num_feature):
    output = np.zeros((x.size,num_feature))
    output[np.arange(x.size),x[np.arange(x.size)]]=1
    return output

def onehotkey_to_label(x):
    output = np.argmax(x, axis=1)
    return output

if __name__=="__main__":
    input = np.random.rand(100,3,280,280)
    print(im2col(input,3,3,1,1).shape)