## QuTrunk

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](http://developer.queco.cn/qutrunk_api/)
[![Release](https://img.shields.io/badge/release-v0.1.9-blue.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Download Code](https://img.shields.io/badge/download-zip-green.svg)](https://github.com/Angel-ML/angel/archive/refs/heads/branch-3.2.0.zip)

[English](./README.md) | 简体中文

### **概述**
---
* QuTrunk 是启科量子自主研发的一款免费、开源、跨平台的量子计算编程框架，包括量子编程API、量子命令转译、量子计算后端接口等。
* QuTrunk 使用 Python 作为宿主语言，利用 Python 的语法特性实现针对量子程序的 DSL (领域专用语言)，所有使用 Python 编程的 IDE 均可使用安装。
* QuTrunk 基于量子逻辑门、量子线路等概念提供量子编程所需各类 API，这些 API 由相应的模块实现。例如 QCircuit 实现量子线路，Qubit 实现量子比特，Qureg 实现量子寄存器，Command 实现每个量子门操作的指令，Backend 实现运行量子线路的后端模块，gate 模块实现各类基础量子门操作。
* QuTrunk 还可以作为其他上层量子计算应用的基础，例如：量子算法、量子可视化编程、量子机器学习等。


### 核心特点
---
* 基于量子逻辑门、量子线路实现量子编程。
* 提供 QuSL 量子汇编指令标准，QuSL 量子汇编与 Python 代码完全兼容。
* 设备独立，在不同的量子后端（例如：BackendLocalCpp, BackendLocalPy, BackendQuSprout, BackendIBM等）上运行同一个量子电路。
* 本地量子计算后端提供全振幅量子模拟计算，量子云服务提供：OMP 多线程、多点并行 MPI、GPU 等计算加速，同时预留了接口对接离子阱量子计算机。
* 兼容 OpenQASM 2.0 标准。
* 支持量子可视化编程（需要配合启科量子研发的量子集成开发环境 QuBranch）。
* 在经典计算机上模拟量子程序，提供全振幅计算。


### **核心模块**
---
* QCircuit
  量子线路，维护对所有量子比特的各种门操作及操作时序，代表了整个量子算法的实现。
* Qubit
  代表单个量子比特，是量子门操作的对象。
* Qureg
 维护若干个量子比特，用于实现一个具体的量子算法。
* Command
  每个量子门操作其背后都会转换成一个基础指令，这些指令按照时间顺序存放在QCircuit中，当整个算法结束或者需要计算当前量子线路的某种状态取值时，这些指令会被发送到指定的后端去执行。
* Backend
  量子计算后端模块，用于执行量子线路，支持Python和C++两种本地后端，QuSprout后端以及第三方后端(目前支持IBM)等。
* Gate
  量子算法基本组成单元，提供各类量子门操作，包括:*H*, *X*, *Y*, *Z*，*P*, *R*, *Rx*, *Ry*, *Rz*, *S*, *Sdg*, *T*, *Tdg*, *CNOT*, *Toffoli*, *Swap*等。

>**注意**:
>  QuTrunk默认只提供Python版本的量子计算模拟器，如果用户需要更高性能的模拟器，可以尝试从: https://pypi.org/project/qutrunk/ 获取QuTrunk源码包进行安装，具体安装步骤可以参考后续的安装章节。


### **下载和安装**
---
#### QuTrunk最新版本: [v0.1.9]()
* #### **pip安装** 

   QuTrunk 已发布于 PyPI 官网，可以通过 pip 命令进行安装。
注意在正式使用 QuTurnk 之前，您需要先安装 Python（版本 3.7+）。

  ```python
  pip install qutrunk
  ```

* #### **源码安装**  

  #### **Windows**

  安装 C++ 编译器：

  ```
  下载visual studio community 2019, 在安装界面勾选`使用C++的桌面开发`进行安装
  ```

  然后再安装 cmake，使用命令行执行：

  ```python
  pip install cmake
  ```

  最后编译安装，解压下载的源码安装包，进入到解压目录下，执行:

  ```python
  python setup.py install
  ```

  #### **MacOS**

  安装之前，确认已下载 QuTrunk（可于PyPI官网下载）

  确认安装 C/C++ 编译器（一般 MacOS 已默认安装）

  ```
  执行 gcc –version 命令查询是否已经安装
  ```
  
  安装 cmake，在终端执行：

  ```python
  pip install cmake
  ```
  
  编译依赖安装完成后，再切换到下载目录，解压并开始编译安装QuTrunk：

  ```python
  tar -zxvf qutrunk-0.1.9.tar.gz
  cd qutrunk-0.1.9
  python setup.py install
  ```

  #### **Ubuntu**

  首先安装 C/C++ 编译器

  ```python
  sudo apt install build-essential
  ```
  
  然后安装cmake编译工具

  ```python
  sudo apt install cmake
  ```
  
  最后编译安装 QuTrunk

  ```python
  tar -zxvf qutrunk-0.1.9.tar.gz
  cd qutrunk-0.1.9
  python setup.py install
  ```


### **示例代码**
---
以下示例展示了利用 QuTrunk 运行 bell-pair 量子算法：

  ```python
  # import package
  from qutrunk.circuit import QCircuit
  from qutrunk.circuit.gates import H, CNOT, Measure, All

  # allocate resource
  qc = QCircuit()
  qr = qc.allocate(2) 

  # apply quantum gates
  H * qr[0]   
  CNOT * (qr[0], qr[1])
  All(Measure) * qr

  # print circuit
  qc.print(qc)   
  # run circuit
  res = qc.run(shots=1024) 
  # print result
  print(res.get_counts()) 
  # draw circuit
  qc.draw()
  ```

运行结果：

  ```
  qreg q[2]
  creg c[2]
  H * q[0]
  MCX * (q[0], q[1])
  Measure * q[0]
  Measure * q[1]
  [{"00":519},{"11":505}]
        ┌───┐      ┌─┐   
  q[0]: ┤ H ├──■───┤M├───
        └───┘┌─┴──┐└╥┘┌─┐
  q[1]: ─────┤ CX ├─╫─┤M├
             └────┘ ║ └╥┘
   c: 2/════════════╩══╩═
                    0  1 
  ```

### **文档**
---
* [QuTrunk 快速上手教程](http://developer.queco.cn/learn/doc/detail?id=12)
* [QuTrunk API](http://developer.queco.cn/learn/doc/detail?id=12&childrenid=14)


## 如何参与开发

1. 阅读源代码，了解我们当前的开发方向
2. 找到自己感兴趣的功能或模块
3. 进行开发，开发完成后自测功能是否正确
4. Fork代码库，将修复代码提交到fork的代码库
5. 发起pull request
6. 更多详情请参见[链接](http://192.168.170.196/qudoor/qubox/-/blob/dev_ll/CONTRIBUTING.md)


### **许可证**
---
QuTrunk是自由和开源的，在Apache 2.0许可证版本下发布。
