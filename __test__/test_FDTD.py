import matplotlib.pyplot as plt
import numpy as np

import FDTD
import Objects

FDTD.basicConfig()


if __name__ == "__main__":

    # 物体3: 直方体エンクロージャー
    obj3xl = 0.4
    obj3yl = 0.3
    obj3 = Objects.Rectangle(
        0.5 - obj3xl / 2,
        0.3 - obj3yl / 2,
        obj3xl,
        obj3yl,
        angle=45,
    )

    fig, ax = plt.subplots(1, 1)
    ax.grid(False)
    ax.set_aspect(1)
    ax.set_xlabel("x[m]")
    ax.set_ylabel("y[m]")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1)
    pcx = np.arange(0, 1.5, 0.02)
    pcy = np.arange(0, 1, 0.01)
    mgx, mgy = np.meshgrid(pcx, pcy, indexing="ij")
    plotx, ploty = obj3.get_geometry(mgx, mgy)
    ax.plot(plotx, ploty, "o", color="gray", markersize=1)
    cx, cy = obj3.centerIndex(mgx, mgy)
    print(cx, cy)
    ax.plot(cx, cy, "o", color="red", markersize=3)

    fig.show()

    input()

    # Nx = 10
    # Ny = 5
    # X = np.arange(0, Nx)
    # Y = np.arange(0, Ny)
    # Z = np.arange(0, Nx * Ny)

    # # 縦(y)軸が第0軸、横(x)軸が第1軸になるようにリシェイプ
    # Z = np.reshape(Z, (Ny, Nx))

    # fig, ax = plt.subplots(1, 1)
    # ax.grid(False)
    # ax.set_aspect(1)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")

    # # 音場
    # im = ax.pcolormesh(X, Y, Z, cmap="bwr", rasterized=True)

    # fig.show()

    # input()

    # # gp = FDTD.GaussianPulse()
    # w = 0.25
    # h = 0.05
    # rect = FDTD.RectangleObject(
    #     0.5 - h / 2,
    #     0.5 - w / 2,
    #     h,
    #     w,
    #     0.0,
    #     15,
    # )

    # print(FDTD.dh)

    # print(int(1 / FDTD.dh))

    # mask = rect.get_mask((int(1 / FDTD.dh), int(1 / FDTD.dh)), 32)

    # print(mask.shape)

    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         print(f"{int(mask[i, j])}", end="")
    #     print("")
