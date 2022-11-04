# QuTrunk发布记录

---
<p id="0"></p >

## 内容

$\qquad$[关于QuTrunk](#1)
$\qquad$[最新版本](#2)
$\qquad$[历史版本](#3)
$\qquad$[更多资讯](#4)

---

<p id="1"></p >

## 关于QuTrunk
QuTrunk 是启科量子自主研发的一款免费、开源、跨平台的量子计算编程框架，包括量子编程API、量子命令转译、量子计算后端接口等, 所有支持Python编程的IDE均可安装使用.
[Back to Top](#0)

---
<p id="2"></p >

## 最新版本 v0.1.12
本次发布包括以下新功能和改善的内容:
* 新增量子相位估计功能.
* 新增AMP算符--通过增幅编码制备任意量子态.
* 新增AWS Braket量子计算后端模块.
* 提供Matrix类，通过自定义矩阵实现量子门操作.
[Back to Top](#0)

---
<p id="3"></p >

## 历史版本

#### v0.1.11:
* 新增振幅放大功能--QAA.
* 提供@Gate修饰符自定义量子门操作.
* 新增量子线路期望值计算以及期望值之和计算.
* 增加量子线路反转功能--circuit.inverse
* 新增量子线路宽度和深度计算

#### v0.1.10:
*  新增量子计算本地模拟(Python版本).
* 完善OpenQASM 2.0的解析功能.
* 优化QuTrunk项目结构.

#### v0.1.7:
* 新增量子电路的可视化.
* 增加crx, cry, crz, barrier, u1,u2,u3量子门实现.
* 量子门增加获取对应矩阵接口.
* 量子线路增加Classical类型初始化，可以通过Classical指定线路初始值.
* 通过量子线路实现数值自增/自减算法例子.increment_decrement.py.
* 增加量子线路多次运行/统计功能.
* 增加量子傅立叶变换QFT.
* 后端模块支持IBM量子计算平台.

#### v0.1.6:
* 支持本地量子模拟计算(利用个人PC进行量子计算).
* 与QuIDE配合，支持可视化量子编程.
* 优化了QuBox设置入口(提供box配置读取/设置).
* 优化了量子线路打印功能(默认打印quqasm指令格式).

#### v0.1.0:
* 支持基本量子编程, 实现量子编程基本模块(Qcircuit, Qubit, Qureg, Gate, Backend).
* 支持基础量子门操作, 包括：H, Measure, CNOT, Toffoli, P, R, Rx, Ry, Rz, S, Sdg, T, Tdg, X, Y, Z, NOT, Swap, SqrtSwap, SqrtX, All, C, Rxx, Ryy, Rzz；
* 支持量子线路打印(OpenQASM 2.0标准).
* 支持量子线路运行统计功能，统计：量子比特，量子门，运行耗时等.
* 支持对QuBox连接配置，IP+ port.
* 支持单节点全振幅量子模拟.

[Back to Top](#0)

---
<p id="4"></p >

## 更多资讯
为了找到最新的产品信息, 有用的资源, 请访问: http://www.developer.queco.cn/.
[Back to Top](#0)
