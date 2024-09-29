import time

import numpy as np

# 基本物理量 -----------------------------
# 変更する場合は、import直後にbasicConfigを介して実施すること
c0 = 3.435e2  # 空気の音速 [m/s]
row0 = 1.205e0  # 空気の密度 [kg/m^3]
# --------------------------------------

# 解析基本設定 ----------------------------
# 変更する場合は、import直後にbasicConfigを介して実施すること
fmax = 48 * 1e3  # 最大解析周波数[Hz]
clf = 1 / np.sqrt(3)  # クーラン数(1より小さくするのが原則)
# --------------------------------------


# 基本定数 -------------------------------
# basicConfigで自動算出される
def basicConfig(
    newC0: float = None,
    newRow0: float = None,
    newFmax: float = None,
    newClf: float = None,
):
    """
    基本定数算出用
    """
    if newC0 is not None:
        global c0
        c0 = newC0
    if newRow0 is not None:
        global row0
        row0 = newRow0
    if newFmax is not None:
        global fmax
        fmax = newFmax
    if newClf is not None:
        global clf
        clf = newClf

    global dt
    dt = 1 / 2.0 / fmax  # 時間離散化幅[s]

    global dh
    dh = c0 * dt / clf  # 時間離散化幅[s]

    global kp0
    kp0 = row0 * c0 * c0  # 体積弾性率

    global z0
    z0 = row0 * c0  # 特性インピーダンス

    global vc
    vc = clf / z0  # 粒子速度用更新係数

    global pc
    pc = clf * z0  # 音圧用更新係数


basicConfig()
# --------------------------------------


class PML:
    """
    PML層
    """

    def __init__(
        self,
        pl: int,  # PML層数
        pm: int = 4,  # PML減衰係数テーパー乗数
        emax: float = 1.200e0,  # PML減衰係数最大値
        ndim: int = 1,
    ):
        self.pl = pl
        self.pm = pm
        self.emax = emax
        self.ex = np.zeros(pl + 1)  # PML用更新係数
        self.a = np.zeros((pl + 1, ndim))
        self.b = np.zeros((pl + 1, ndim))
        self.c = np.zeros((pl + 1, ndim))
        self._set_exabc()

    def _set_exabc(self):
        for i in range(self.pl):
            i += 1
            self.ex[i] = self.emax * np.power(
                float(self.pl - i + 1) / float(self.pl), float(self.pm)
            )
        for i in range(self.pl):
            i += 1
            self.a[i, :] = (1.0 - self.ex[i]) / (1.0 + self.ex[i])
            self.b[i, :] = clf / z0 / (1.0 + self.ex[i])
            self.c[i, :] = clf * z0 / (1.0 + self.ex[i])

        self.at = self.a.T
        self.bt = self.b.T
        self.ct = self.c.T


class ListeningPoint:
    """
    受聴点
    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        tmax: float = 1,
    ):
        self.x = x  # 音源位置x
        self.y = y  # 音源位置y
        self.tmax = tmax  # 解析時間
        self.u = np.zeros(int(self.tmax / dt) + 1)  # 入力波形

    @property
    def timepoints(self):
        """
        時刻[msec]
        """
        return dt * np.arange(int(self.tmax / dt) + 1) * 1000

    @property
    def freqpoints(self):
        """
        周波数
        """
        return 1 / dt / (int(self.tmax / dt) + 1) * np.arange(int(self.tmax / dt) + 1)

    def get_geometryIndex(self, mgx: np.array, mgy: np.array) -> tuple:
        """
        座標を返す
        Returns:
            座標([x1, ...], [y1, ...])
        """
        idx = np.abs(mgx[:, 0] - self.x).argmin()
        idy = np.abs(mgy[0, :] - self.y).argmin()

        return [idx], [idy]


class SoundSource:
    """
    音源
    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
    ):
        self.x = x  # 音源位置x
        self.y = y  # 音源位置y
        self.q = None  # 音源波形

    def get_geometryIndex(self, mgx: np.array, mgy: np.array) -> tuple:
        """
        座標を返す
        Returns:
            座標([x1, ...], [y1, ...])
        """
        idx = np.abs(mgx[:, 0] - self.x).argmin()
        idy = np.abs(mgy[0, :] - self.y).argmin()

        return [idx], [idy]


class GaussianPulse(SoundSource):
    """
    ガウシアンパルス
    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        m: float = 5.00 * 1e-3,  # ガウシアンパルス最大値 [m^3/s]
        t0: float = 1.000 * 1e-3,  # ガウシアンパルス中心時間 [s] 低周波信号
        a: float = 20.000e5 / (0.001 * 0.001 * 400 * 400),  # ガウシアンパルス係数 [-]
    ):
        super().__init__(x, y)
        self._m = m
        self._t0 = t0
        self._a = a
        self._create_waveform()

    @property
    def tdr(self):
        """
        加振時間
        """
        return int((2.0 * self._t0) / dt)

    @property
    def timepoints(self):
        """
        時刻[msec]
        """
        return dt * np.arange(self.tdr + 1) * 1000

    def _create_waveform(self):
        """
        音源波形の生成
        """
        self.q = np.zeros(self.tdr + 1)
        for t in range(self.tdr):
            t += 1
            self.q[t] = self._m * np.exp(-self._a * pow(float(t * dt - self._t0), 2.0))


class SoundField:
    """
    解析対象音場
    """

    def __init__(
        self,
        xmax: float = 1,
        ymax: float = 1,
        tmax: float = 0.001,
        pml: PML = None,
        soundSources: list = [],
        objects: list = [],
        listeningPoints: list = [],
    ):
        self.xmax = xmax
        self.ymax = ymax
        self.tmax = tmax
        self.pml = pml
        self.soundSources = soundSources if len(soundSources) > 0 else []
        self.objects = objects if len(objects) > 0 else []
        self.listeningPoints = listeningPoints if len(listeningPoints) > 0 else []
        self.p = None
        self.px = None
        self.py = None
        self.vx = None
        self.vy = None
        self.ix = None
        self.jx = None
        self.tx = None

    def create(self):
        """
        作成
        """
        # 解析範囲
        self.ix, self.jx = self.get_size_index(self.xmax, self.ymax, self.pml.pl)
        self.tx = int(self.tmax / dt)

        pcx, pcy, mgx, mgy = self.get_grid()

        self.p = np.zeros((self.ix + 1, self.jx + 1))
        self.px = np.zeros((self.ix + 1, self.jx + 1))
        self.py = np.zeros((self.ix + 1, self.jx + 1))
        self.vx = np.zeros((self.ix + 1, self.jx + 1))
        self.vy = np.zeros((self.ix + 1, self.jx + 1))

        # 音源の座標インデクス
        self.ssi, self.ssj = self.soundSources[0].get_geometryIndex(mgx, mgy)

        # 物体のマスク
        self.masks = []
        for obj in self.objects:
            self.masks.append(obj.get_mask(mgx, mgy, logic=False))

        self.ii = list(range(1, self.ix + 1))
        self.jj = list(range(1, self.jx + 1))

        # 受聴点の座標インデクス
        self.lpIndexes = []
        for lp in self.listeningPoints:
            lpi, lpj = lp.get_geometryIndex(mgx, mgy)
            self.lpIndexes.append([lpi[0], lpj[0]])

    def update(self, t, callback: (callable, any) = None):
        """
        シミュレーション
        """
        t0 = time.time()

        # 更新対象でない
        pmla = self.pml.a[:, 0]
        # pmlb = self.pml.b[:, 0]
        pmlc = self.pml.c[:, 0]
        ix = self.ix
        jx = self.jx
        pl = self.pml.pl
        q = self.soundSources[0].q
        tdr = self.soundSources[0].tdr

        # 更新対象
        vx = self.vx
        vy = self.vy
        px = self.px
        py = self.py
        p = self.p

        ii = self.ii
        jj = self.jj

        callback[1][0] += time.time() - t0

        # ------------------------------------------
        # 粒子速度の更新
        # ------------------------------------------
        # vx ----------------------------------------
        # 左側のPML
        # for i in range(pl + 1):
        #     vx[i, 1 : jx + 1] = pmla[i] * vx[i, 1 : jx + 1] - pmlb[i] * (
        #         p[i + 1, 1 : jx + 1] - p[i, 1 : jx + 1]
        #     )
        vx[0 : pl + 1, jj] = self.pml.a[0 : pl + 1, jj] * vx[
            0 : pl + 1, jj
        ] - self.pml.b[0 : pl + 1, jj] * (p[1 : pl + 2, jj] - p[0 : pl + 1, jj])
        callback[1][1] += time.time() - t0

        # 音響領域
        # for i in range(pl + 1, ix - pl - 1 + 1):
        #     vx[i, : jx + 1] = vx[i, : jx + 1] - vc * (
        #         p[i + 1, : jx + 1] - p[i, : jx + 1]
        #     )
        # vx[pl + 1 : ix - pl, : jx + 1] = vx[pl + 1 : ix - pl, : jx + 1] - vc * (
        #     p[pl + 1 + 1 : ix - pl + 1, : jx + 1] - p[pl + 1 : ix - pl, : jx + 1]
        # )
        vx[pl + 1 : ix - pl, :] = vx[pl + 1 : ix - pl, :] - vc * (
            p[pl + 1 + 1 : ix - pl + 1, :] - p[pl + 1 : ix - pl, :]
        )
        callback[1][2] += time.time() - t0

        # 右側PML
        # for i in range(ix - pl, ix - 1 + 1):
        #     vx[i, 1 : jx + 1] = pmla[ix - i] * vx[i, 1 : jx + 1] - pmlb[ix - i] * (
        #         p[i + 1, 1 : jx + 1] - p[i, 1 : jx + 1]
        #     )
        vx[ix - pl : ix, jj] = self.pml.a[pl:0:-1, jj] * vx[
            ix - pl : ix, jj
        ] - self.pml.b[pl:0:-1, jj] * (
            p[ix - pl + 1 : ix + 1, jj] - p[ix - pl : ix, jj]
        )
        callback[1][3] += time.time() - t0

        # vy
        # for j in range(1, pl + 1):
        #     vy[1 : ix + 1, j] = pmla[j] * vy[1 : ix + 1, j] - pmlb[j] * (
        #         p[1 : ix + 1, j + 1] - p[1 : ix + 1, j]
        #     )
        vy[ii, 1 : pl + 1] = self.pml.at[ii, 1 : pl + 1] * vy[
            ii, 1 : pl + 1
        ] - self.pml.bt[ii, 1 : pl + 1] * (p[ii, 2 : pl + 1 + 1] - p[ii, 1 : pl + 1])
        callback[1][4] += time.time() - t0

        # for j in range(pl + 1, jx - pl - 1 + 1):
        #     vy[: ix + 1, j] = vy[: ix + 1, j] - vc * (
        #         p[: ix + 1, j + 1] - p[: ix + 1, j]
        #     )
        vy[:, pl + 1 : jx - pl] = vy[:, pl + 1 : jx - pl] - vc * (
            p[:, pl + 1 + 1 : jx - pl + 1] - p[:, pl + 1 : jx - pl]
        )
        callback[1][5] += time.time() - t0

        # for j in range(jx - pl, jx - 1 + 1):
        #     vy[1 : ix + 1, j] = pmla[jx - j] * vy[1 : ix + 1, j] - pmlb[jx - j] * (
        #         p[1 : ix + 1, j + 1] - p[1 : ix + 1, j]
        #     )
        vy[ii, jx - pl : jx] = self.pml.at[ii, pl:0:-1] * vy[
            ii, jx - pl : jx
        ] - self.pml.bt[ii, pl:0:-1] * (
            p[ii, jx - pl + 1 : jx + 1] - p[ii, jx - pl : jx]
        )
        callback[1][6] += time.time() - t0

        # ------------------------------------------
        # 境界条件
        # ------------------------------------------
        # 音響領域とPMLの境界
        vx[0, :] = 0.0
        vx[ix, :] = 0.0
        vy[:, 0] = 0.0
        vy[:, jx] = 0.0

        # 音響領域中の障害物表面
        for mask in self.masks:
            vx *= mask
            vy *= mask

        callback[1][7] += time.time() - t0

        # ------------------------------------------
        # 音圧の更新
        # ------------------------------------------
        if t <= tdr:
            K = dt * kp0 * q[t] / 3.0 / (dh * dh * dh)

        # px
        for i in range(1, pl + 1):
            px[i, 1 : jx + 1] = pmla[i] * px[i, 1 : jx + 1] - pmlc[i] * (
                vx[i, 1 : jx + 1] - vx[i - 1, 1 : jx + 1]
            )
        callback[1][8] += time.time() - t0

        # for i in range(pl + 1, ix - pl + 1):
        #     for j in range(jx + 1):
        #         px[i, j] = px[i, j] - pc * (vx[i, j] - vx[i - 1, j])
        # if (i == idr) and (j == jdr) and (t <= tdr):
        #     px[i, j] = px[i, j] + dt * kp0 * q[t] / 3.0 / (dh * dh * dh)
        px[pl + 1 : ix - pl + 1, :] = px[pl + 1 : ix - pl + 1, :] - pc * (
            vx[pl + 1 : ix - pl + 1, :] - vx[pl + 1 - 1 : ix - pl + 1 - 1, :]
        )
        if t <= tdr:
            px[self.ssi, self.ssj] += K

        callback[1][9] += time.time() - t0

        for i in range(ix - pl + 1, ix + 1):
            px[i, 1 : jx + 1] = pmla[ix - i + 1] * px[i, 1 : jx + 1] - pmlc[
                ix - i + 1
            ] * (vx[i, 1 : jx + 1] - vx[i - 1, 1 : jx + 1])

        callback[1][10] += time.time() - t0

        # py
        for j in range(1, pl + 1):
            py[1 : ix + 1, j] = pmla[j] * py[1 : ix + 1, j] - pmlc[j] * (
                vy[1 : ix + 1, j] - vy[1 : ix + 1, j - 1]
            )

        callback[1][11] += time.time() - t0

        # for j in range(pl + 1, jx - pl + 1):
        #     for i in range(ix + 1):
        #         py[i, j] = py[i, j] - pc * (vy[i, j] - vy[i, j - 1])
        # if (i == idr) and (j == jdr) and (t <= tdr):
        #     py[i, j] = py[i, j] + dt * kp0 * q[t] / 3.0 / (dh * dh * dh)
        py[:, pl + 1 : jx - pl + 1] = py[:, pl + 1 : jx - pl + 1] - pc * (
            vy[:, pl + 1 : jx - pl + 1] - vy[:, pl + 1 - 1 : jx - pl + 1 - 1]
        )
        if t <= tdr:
            py[self.ssi, self.ssj] += K

        callback[1][12] += time.time() - t0

        for j in range(jx - pl + 1, jx + 1):
            py[1 : ix + 1, j] = pmla[jx - j + 1] * py[1 : ix + 1, j] - pmlc[
                jx - j + 1
            ] * (vy[1 : ix + 1, j] - vy[1 : ix + 1, j - 1])

        callback[1][13] += time.time() - t0

        # 音圧の合成
        p = px + py
        self.p = p

        # 受聴点信号取り出し
        for lpk, lp in enumerate(self.listeningPoints):
            lpi, lpj = self.lpIndexes[lpk]
            lp.u[t] = self.p[lpi, lpj]

        # ユーザーコールバック
        if callback is not None:
            callback[0](self, t, callback[1])

    def get_grid(self) -> tuple:
        """
        pcolor(mesh)のXYグリッドを取得する

        pcx, pcy: pcolormeshの引数
        mgx, mgy:　meshgrid
        """
        pcx = np.arange(-dh * self.pml.pl, self.xmax + dh * self.pml.pl, dh)
        pcy = np.arange(-dh * self.pml.pl, self.ymax + dh * self.pml.pl, dh)

        mgx, mgy = np.meshgrid(
            pcx,
            pcy,
            indexing="ij",
            sparse=False,
            copy=True,
        )

        return pcx, pcy, mgx, mgy

    @staticmethod
    def get_size_index(xmax, ymax, pl):
        ix = int(xmax / dh) + pl * 2
        jx = int(ymax / dh) + pl * 2
        return ix, jx
