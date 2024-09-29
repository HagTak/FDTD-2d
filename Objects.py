"""
FDTD Objects

mgx, mgyについて
pcx = np.arange(xl, xh)
pcy = np.arange(yl, yh)
# indexing='ij'とすること
# ijを指定しないと、xyが逆転する
mgx, mgy = meshgrid(pcx, pcy, indexing='ij')
"""

import matplotlib.pyplot as plt
import numpy as np


class Object:
    """
    物体
    """

    def __init__(self, alpn: float = 0):
        self.alpn = alpn

    def get_mask(self, mgx: np.array, mgy: np.array, logic: bool = True):
        """ """
        return None

    def get_geometry(self, mgx: np.array, mgy: np.array, index=False):
        mask = self.get_mask(mgx, mgy, logic=True)
        plot = np.where(mask == 1)
        dx = mgx[1, 1] - mgx[0, 0] if not index else 1
        dy = mgy[1, 1] - mgy[0, 0] if not index else 1
        return plot[0] * dx + mgx[0, 0], plot[1] * dy + mgy[0, 0]


class Rectangle(Object):
    """
    長方形
    """

    def __init__(
        self,
        x: float,
        y: float,
        xl: float,
        yl: float,
        angle: float = 0.0,
        alpn: float = 0.0,
    ):
        """
        x: 下端
        y: 左端
        h: 高さ(x方向長さ)
        w: 幅(y方向長さ)
        angle: 角度(deg)
        """
        super().__init__(alpn)
        self.x = x
        self.y = y
        self.xl = xl
        self.yl = yl
        self.angle = angle

    def centerIndex(self, mgx: np.array, mgy: np.array):
        """
        メッシュグリッドでの長方形中心のインデクス
        """
        xc = self.x + self.xl / 2
        yc = self.y + self.yl / 2
        ax = np.absolute(mgx[:, 0] - xc).argmin()
        ay = np.absolute(mgy[0, :] - yc).argmin()
        cx = mgx[ax, 0]
        cy = mgy[0, ay]
        return cx, cy

    def get_mask(self, mgx: np.array, mgy: np.array, logic: bool = True) -> np.array:
        """
        マスク行列を求める

            zyh
            -----
        zxl |   |zxh
            -----
            zyl

        Returns:
            マスク行列np.array<bool>
        """

        th = self.angle / 180.0 * np.pi  # θ
        sin2 = np.sin(th) / 2.0  # sinθ / 2
        cos2 = np.cos(th) / 2.0  # cosθ / 2
        tan = np.tan(th)  # tanθ
        xl = self.xl
        yl = self.yl
        xc = self.x + xl / 2
        yc = self.y + yl / 2

        def zyl(x, y):
            if th >= -np.pi / 2 and th <= np.pi / 2:
                return tan * (x - yl * sin2) - yl * cos2 <= y
            else:
                return tan * (x - yl * sin2) - yl * cos2 > y

        def zxl(x, y):
            if th >= 0 and th <= np.pi:
                return -1 / tan * (x + xl * cos2) - xl * sin2 <= y
            else:
                return -1 / tan * (x + xl * cos2) - xl * sin2 > y

        def zyh(x, y):
            if th >= -np.pi / 2 and th <= np.pi / 2:
                return tan * (x + yl * sin2) + yl * cos2 > y
            else:
                return tan * (x + yl * sin2) + yl * cos2 <= y

        def zxh(x, y):
            if th >= 0 and th <= np.pi:
                return -1 / tan * (x - xl * cos2) + xl * sin2 > y
            else:
                return -1 / tan * (x - xl * cos2) + xl * sin2 <= y

        mask = (
            zxl(mgx - xc, mgy - yc)
            * zyl(mgx - xc, mgy - yc)
            * zxh(mgx - xc, mgy - yc)
            * zyh(mgx - xc, mgy - yc)
        ) * 1

        if not logic:
            mask = (mask - 1) * -1

        return mask


class Circle(Object):
    """
    円
    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        r: float = 1,
        alpn: float = 0.0,
    ):
        super().__init__(alpn)
        self.x = x
        self.y = y
        self.r = r

    def centerIndex(self, mgx: np.array, mgy: np.array):
        """
        メッシュグリッドでの長方形中心のインデクス
        """
        xc = self.x
        yc = self.y
        ax = np.absolute(mgx[:, 0] - xc).argmin()
        ay = np.absolute(mgy[0, :] - yc).argmin()
        cx = mgx[ax, 0]
        cy = mgy[0, ay]
        return cx, cy

    def get_mask(self, mgx: np.array, mgy: np.array, logic: bool = True) -> np.array:
        """

        Returns:
            マスク行列(np.array<1 or 0>)
        """
        mask = (
            (mgx - self.x) * (mgx - self.x) + (mgy - self.y) * (mgy - self.y)
            < self.r * self.r
        ) * 1

        if not logic:
            mask = (mask - 1) * -1

        return mask


if __name__ == "__main__":

    class Enclosure:
        """
        エンクロージャー
        """

        def __init__(self, x, y):
            """
            x: x基準座標
            y: y基準座標
            """
            self.c1 = Circle(x - 0.011, y - 0.156, 0.252)
            self.c2 = Circle(x - 0.011, y + 0.156, 0.252)
            self.c3 = Circle(x, y, 0.11775)
            self.r1 = Rectangle(x, y - 0.1, 0.125, 0.2)

        def get_mask(self, mgx, mgy, logic: bool = True):
            """
            マスク行列
            logic: True=存在領域が1
            """
            m1 = self.c1.get_mask(mgx, mgy, logic=True)
            m2 = self.c2.get_mask(mgx, mgy, logic=True)
            m3 = self.c3.get_mask(mgx, mgy, logic=True)
            m4 = self.r1.get_mask(mgx, mgy, logic=True)
            mask = m1 * m2 * (((m3 + m4) > 0) * 1)

            if not logic:
                mask = (mask - 1) * -1

            return mask

        def get_geometry(self, mgx, mgy, index=False):
            mask = self.get_mask(mgx, mgy, logic=True)
            plot = np.where(mask == 1)
            dx = mgx[1, 1] - mgx[0, 0] if not index else 1
            dy = mgy[1, 1] - mgy[0, 0] if not index else 1
            return plot[0] * dx + mgx[0, 0], plot[1] * dy + mgy[0, 0]

    obj3 = Enclosure(0.5, 0.5)

    pcx = np.arange(0, 1, 0.01)
    pcy = np.arange(0, 1, 0.01)
    mgx, mgy = np.meshgrid(
        pcx,
        pcy,
        indexing="xy",
        sparse=False,
        copy=True,
    )

    # z1 = c1.mask(x, y)
    # z2 = c2.mask(x, y)
    # z3 = c3.mask(x, y)
    # z4 = r1.mask(x, y)

    # z = z1
    # z5 = z1 * z2
    # z6 = z5 * z3
    # z7 = z3 + z4
    # z8 = z5 * z7

    # z8plot = np.where(z8 == True)
    # dx = x[1, 1] - x[0, 0]
    # dy = y[1, 1] - y[0, 0]
    # plotx, ploty = z8plot[1] * dx, z8plot[0] * dy

    plotx, ploty = obj3.get_geometry(mgx, mgy)

    # for i in range(z.shape[0]):
    #     for j in range(z.shape[1]):
    #         print("*", end="") if z[i, j] else print("-", end="")
    #     print("")

    # fig, axes = plt.subplots(4, 2)

    # heatmap = axes[0, 0].pcolor(z1, cmap=plt.cm.Blues)
    # heatmap = axes[0, 1].pcolor(z2, cmap=plt.cm.Blues)
    # heatmap = axes[1, 0].pcolor(z3, cmap=plt.cm.Blues)
    # heatmap = axes[1, 1].pcolor(z4, cmap=plt.cm.Blues)
    # heatmap = axes[2, 0].pcolor(z5, cmap=plt.cm.Blues)
    # heatmap = axes[2, 1].pcolor(z6, cmap=plt.cm.Blues)
    # heatmap = axes[3, 0].pcolor(z7, cmap=plt.cm.Blues)
    # heatmap = axes[3, 1].pcolor(z8, cmap=plt.cm.Blues)

    # axes[0, 0].set_aspect("equal")
    # axes[0, 1].set_aspect("equal")
    # axes[1, 0].set_aspect("equal")
    # axes[1, 1].set_aspect("equal")
    # axes[2, 0].set_aspect("equal")
    # axes[2, 1].set_aspect("equal")
    # axes[3, 0].set_aspect("equal")
    # axes[3, 1].set_aspect("equal")

    # fig.show()

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    ax.plot(plotx, ploty, "o", color="red")
    # cx, cy = r1.centerIndex(x, y)
    # ax.plot(cx, cy, "o", color="Red")
    fig.show()

    input()
    # for i in range(180):
    #     r1.angle = i
    #     z4 = r1.mask(x, y)

    #     heatmap = ax.pcolor(z4, cmap=plt.cm.Blues)
    #     cx, cy = r1.centerIndex(x, y)
    #     ax.plot(cx, cy, "o", color="Red")
    #     plt.pause(0.01)
    #     ax.clear()
