# FDTD-2d

PythonによるFDTD法プログラム(2次元)

## 音速

$$
c_0
$$

## 最大解析周波数

$$
f_{max}
$$


## 時間離散化幅

$$
\Delta t = \frac{1}{2f_{max}}
$$


## クーラン条件(2次元)

x方向、y方向それぞれの空間離散化幅を $\Delta l_x, \Delta l_y$ とすると

$$
c_0 \Delta t > \frac{1}{\sqrt{(\frac{1}{\Delta l})^2 + (\frac{1}{\Delta l})^2}}
$$

$\Delta l_x = \Delta l_y = \Delta l$ とすると、

$$
C = \frac{c_0 \Delta t}{\Delta l} < \frac{1}{\sqrt{2}}
$$

## 空間離散化幅

$$
\Delta l = \frac{c_0 \Delta t}{C}
$$