#!/usr/bin/env python
# coding=utf-8
'''
@Author: xieydd
@since: 2020-02-20 09:51:17
@lastTime: 2020-02-22 23:10:07
@LastAuthor: Do not edit
@message: GEMM CLASS
'''
import numpy as np
import time


class GEMM():
    def __init__(self, type, *args, **kwargs):
        self.type = type

    '''
    @description: GEMM 3D for CONV
    @param
        A: feature [IH,IH,IC]
        B: kernel [KH,KW,KC]
        C: bias [OH,OW,OC]
    @return:
        D: output [OH, OW, OC]
    '''

    def gemm_3d(self, A, B, C, stride_h=1, stride_w=1):

        if len(A.shape) != 3 and len(B.shape) != 3:
            raise NotImplementedError("Only support 3 dims matrix")

        # Simple
        IH = A.shape[0]
        IW = A.shape[1]
        IC = A.shape[2]
        KH = B.shape[0]
        KW = B.shape[1]
        KC = B.shape[2]
        OH = C.shape[0]
        OW = C.shape[1]
        OC = C.shape[2]
        D = np.zeros((OH, OW, OC))
        for oh in range(OH):
            for ow in range(OW):
                for oc in range(OC):
                    D[oh][ow][oc] += C[oh][ow][oc]
                    for kh in range(0, KH):
                        for kw in range(0, KW):
                            for ic in range(IC):
                                D[oh][ow][oc] += A[oh*stride_h+kh][ow*stride_w+kw][ic] * \
                                    B[kh][kw][ic]
        return D

    '''
    @description: GEMM 2D for CONV
    @param
        A: feature [IH,IW]
        B: kernel [KH,KW]
        C: bias [OH,OW]
    @return:
        D: output [OH, OW]
    '''

    def gemm_2d(self, A, B, C):
        IH = A.shape[0]
        IW = A.shape[1]
        KH = B.shape[0]
        KW = B.shape[1]
        OH = IH
        OW = KH

        D = np.zeros((OH, OW))
        for ih in range(IH):
            for kh in range(KH):
                D[ih][kh] += C[0][0][0]
                for iw in range(IW):
                    D[ih][kh] += A[ih][iw]*B[kh][iw]
        return D


'''
@description: 
@param 
    A: feature [IH,IW,IC]
    B: kernel [KH,KW,IC]
    OC: output channel
    stride_h: height stride
    stride_w: width  stride
@return: 
    TA: Tensor Activation [OH*OW,KH*KW*IC]
    TK: Tensor Kernel [OC,KH*KW*IC]

site:
    - https://jackwish.net/2019/convolution-neural-networks-optimization.html
    - https://blog.csdn.net/dwyane12138/article/details/78449898
'''


def im2col(A, B, OC, stride_h, stride_w):
    IH = A.shape[0]
    IW = A.shape[1]
    IC = A.shape[2]

    KH = B.shape[0]
    KW = B.shape[1]
    KC = B.shape[2]

    assert KC == IC

    OH = (IH - KH)/stride_h + 1
    OW = (IW - KW)/stride_w + 1

    # TWH = Tensor activate height
    TAH = IH * IW
    TAW = KH * KW * IC

    TKH = OC
    TKW = KH * KW * IC

    TA = np.zeros((TAH, TAW))
    TK = np.zeros((TKH, TKW))
    for oh in range(OH):
        for ow in range(OW):
            for ic in range(IC):
                for kh in range(KH):
                    for kw in range(KW):
                        col = kw + ow * stride_w
                        row = kh + oh * stride_h
                        TA[oh*OW + ow][ic * (KH*KW) + kh * KW
                                       + kh] = A[row][col][ic]
                        for oc in range(OC):
                            TK[oc, ic * (KH*KW) + kh * KW
                               + kh] = B[kh][kw][ic]
    return TA, TK


def main():

    g = GEMM('Simple')
    # 3x3x5
    A = np.array(
        [
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15]
            ],
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15]
            ],
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15]
            ]
        ]
    )

    # 1x1x5
    B = np.array(
        [
            [
                [1, 1, 1, 1, 1]
            ]
        ]
    )

    # 2x2x5
    # B = np.array(
    #     [
    #         [
    #             [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #         ],
    #         [
    #             [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #         ]
    #     ]
    # )

    bias = 1

    stride_h = 2
    stride_w = 2
    IH = A.shape[0]
    IW = A.shape[1]
    KH = B.shape[0]
    KW = B.shape[1]
    # OH = (IH + 2 * pad - dilation_h * (KH - 1) - 1) / stride_h + 1
    OH = (IH - KH)/stride_h + 1
    OW = (IW - KW)/stride_w + 1
    C = np.ones((OH, OW, 1)) * bias
    print(C.shape)
    start = time.time()
    D = g.gemm_3d(A, B, C, stride_h=stride_h, stride_w=stride_w)
    print(D)
    for i in range(100):
        D = g.gemm_3d(A, B, C)
    end = time.time()
    print((end - start) * 1000, " ms")

    TA, TK = im2col(A, B, C.shape[0], stride_h, stride_w)
    print(TA)
    print(TK)
    E = g.gemm_2d(TA, TK, C)
    print(E[:OH][:OW])

    for i in range(100):
        D = g.gemm_2d(TA, TK, C)
    end = time.time()
    print((end - start) * 1000, " ms")


if __name__ == "__main__":
    main()
