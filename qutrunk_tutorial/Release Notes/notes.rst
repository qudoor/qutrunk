Contents
==========
About QuTrunk + QuSprout
------------------------------------
QuTrunk: 一个基于Python开发的量子编程语言框架, 包括量子编程API, 量子命令转译, 量子计算后端接口等, 所有支持Python编程的IDE均可安装使用. 

QuSprout: 基于经典计算资源的量子计算模拟软件, 支持多线程、多节点、GPU加速, 可预安装在QuBox中(启科自主研发量子计算设备).

New Features
-----------------------
本次发布包括以下新功能和改善的内容：

1. 新增量子计算本地模拟(Python版本).
2. 完善OpenQASM 2.0的解析功能.
3. 优化QuTrunk项目结构.

Previous Version
--------------------------
v0.1.8支持的内容:
>>>>>>>>>>>>>>>>
1. 新增量子电路的可视化. 
2. 后端支持分布式量子任务计算和管理.
3. 以MPI方式实现单任务多节点并行计算.
4. 量子计算任务以及量子计算资源统计和展示.

v0.1.7支持的内容:
>>>>>>>>>>>>>>>>
1. 增加crx, cry, crz, barrier, u1,u2,u3量子门实现. 
2. 量子门增加获取对应矩阵接口. 
3. 量子线路增加Classical类型初始化，可以通过Classical指定线路初始值.
4. 通过量子线路实现数值自增/自减算法例子.increment_decrement.py. 
5. 增加量子线路多次运行/统计功能. 
6. 增加量子傅立叶变换QFT. 
7. 后端模块支持IBM量子计算平台.

v0.1.6支持的内容:
>>>>>>>>>>>>>>>>
1. 支持本地量子模拟计算(利用个人PC进行量子计算). 
2. 支持噪声模型.
3. 与QuBranch配合，支持可视化量子编程. 
4. QuSim增加任务管理，支持多任务并行. 
5. 优化了QuBox设置入口(提供box配置读取/设置). 
6. 优化了量子线路打印功能(默认打印QuSL指令格式).

v0.1.0支持的内容:
>>>>>>>>>>>>>>>>
1. 支持基本量子编程, 实现量子编程基本模块(Qcircuit, Qubit, Qureg, Gate, Backend).
2. 支持基础量子门操作, 包括：H, Measure, CNOT, Toffoli, P, R, Rx, Ry, Rz, S, Sdg, T, Tdg, X, Y, Z, NOT, Swap, SqrtSwap, SqrtX, All, C, Rxx, Ryy, Rzz.
3. 支持量子线路打印(OpenQASM 2.0标准).
4. 支持量子线路运行统计功能，统计：量子比特，量子门，运行耗时等.
5. 支持在QuPy上对QuBox连接配置，IP+ port. 
6. 支持单节点全振幅量子模拟.

Materials to Deliver
----------------------------
QuTrunk
>>>>>>>>>>>>>>>>
.. list-table::
	:widths: 20 80

	*
		-     

		- Software

	*
		- whl 安装包

		- qutrunk-0.1.9-py3-none-any.whl

	*
		- 源码包

		- qutrunk-0.1.9.tar.gz

	*
		- 测试脚本

		- test_qutrunk.py, test_qutrunk_qusprout.py


QuSprout
>>>>>>>>>>>>>>>>

.. list-table::
	:widths: 20 80

	*
		-     

		- Software

	*
		- kylin 安装包

		- qusprout-ky10-arm-64-v0.1.9.tar.gz
	*
		- macso 安装包

		- qusprout-macosx_12_0_x86_64-v0.1.9.tar.gz

	*
		- ubuntu 安装包

		- qusprout-ubuntu-x86-64-v0.1.9.tar.gz


For More Information
-------------------------------
为了找到最新的产品信息, 有用的资源, 请访问: http://www.developer.queco.cn/. 