"""
FDTD Objects

mgx, mgyについて
pcx = np.arange(xl, xh)
pcy = np.arange(yl, yh)
# indexing='ij'とすること
# ijを指定しないと、xyが逆転する
mgx, mgy = meshgrid(pcx, pcy, indexing='ij')
"""

import numpy as np


class Object:
    """
    物体
    """

    def __init__(self, alpn: float = 0):
        self.alpn = alpn
        self.name = "no name"

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
                return tan * (x - yl * sin2) - yl * cos2 >= y

        def zxl(x, y):
            if th >= 0 and th <= np.pi:
                return -1 / tan * (x + xl * cos2) - xl * sin2 <= y
            else:
                return -1 / tan * (x + xl * cos2) - xl * sin2 >= y

        def zyh(x, y):
            if th >= -np.pi / 2 and th <= np.pi / 2:
                return tan * (x + yl * sin2) + yl * cos2 >= y
            else:
                return tan * (x + yl * sin2) + yl * cos2 <= y

        def zxh(x, y):
            if th >= 0 and th <= np.pi:
                return -1 / tan * (x - xl * cos2) + xl * sin2 >= y
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
            <= self.r * self.r
        ) * 1

        if not logic:
            mask = (mask - 1) * -1

        return mask
