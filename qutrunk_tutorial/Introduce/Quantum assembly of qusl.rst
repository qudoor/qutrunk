QuSL量子汇编
============

qutrunk使用Python作为宿主语言，利用Python的语法特性实现针对量子程序的DSL(领域专用语言)，
我们把用于量子编程的专用语言称为：QuSL（一套类似OpenQASM的量子汇编语言），
QuSL主要特点是最左边是一个量子门操作，中间加入( * )号链接符，最右边是操作的量子比特，形式如下：

.. code-block:: 

    gate * qubits

几个例子:

.. code-block:: 

    H * q[0];               # 对q[0]做Hadamard门操作
    CNOT * (q[0], q[1]);    # q[0]为控制位，q[1]为目标位
    All(Measure) * q        # 对q中的所有量子比特做测量操作

使用该标准是充分利用了Python语法对( * )运算符的重载特性，该表形式更接近量子物理计算公式，同时
( * )在计算机编程语言上表示乘法的意思，借此表示左边的量子门操作实际上是对量子比特做矩阵乘法运算，
使用该标准编写的量子汇编是可以直接被qutrunk解析运行的，不需要做词法/语法方面的解析处理工作,
基于该特性，qutrunk可以无缝衔接qubranch通过可视化量子编程生成的量子线路，即qutrunk可以直接运行
qubranch生成的量子线路（只需做一些简单的初始化工作），而无需做语法上的编译/转译处理，下面是QuSL
每个量子门操作介绍：

QuSL量子汇编说明
>>>>>>>>>>>>>>>>>>>>>>>

QuSL量子汇编提供三个层面的量子操作语句实现，分别是：基础量子门，量子算符和量子元操作。


1.Basic Gate: 基础量子门（提供对量子比特的基础操作，是实现量子算法的基本构件）
-------------------
.. code-block:: 

    //H(hadamard): 哈达马门，对a做H门操作，常用于使量子比特处于叠加态
    H * a

    //X(NOT): 非门（Pauli-X）对a进行取反操作, 量子比特绕布洛赫球的x轴旋转pi角度
    X * a

    //Y: Pauli-Y, 量子比特绕布洛赫球的y轴旋转pi角度
    Y * a

    //Z: Pauli-Z, 量子比特绕布洛赫球的z轴旋转pi角度
    Z * a

    //CNOT(CX): 受控非门，a作为控制位，b为目标位，如果a为1则对b进行取反，如果a为0则不做任何操作
    CNOT * (a, b)

    //Toffoli: 托佛利门，a, b作为控制位，c为目标位, 如果a,b均为1则对b进行取反，否则不做任何操作
    Toffoli * (a, b, c)

    //Measure: 测量门，对a进行测量，结果要么是0，要么是1，测量结果受概率振幅影响
    Measure * a

    //P: 相移门，将量子比特0>态和1>态的相位根据给定的角度进行移动
    P(theta) * a

    //Rx: 量子比特绕布洛赫球的x轴旋转theta角度
    Rx(theta) * a

    //Ry: 量子比特绕布洛赫球的y轴旋转theta角度
    Ry(theta) * a

    //Rz: 量子比特绕布洛赫球的z轴旋转theta角度
    Rz(theta) * a

    //S: 量子比特绕布洛赫球的z轴旋转pi/2角度
    S * a

    //Sdg: 对S门的反向操作, 绕布洛赫球的z轴反方向旋转pi/2角度
    Sdg * a

    //T: 量子比特绕布洛赫球的z轴旋转pi/4角度
    T * a

    //Tdg: 对T门的反向操作, 绕布洛赫球的z轴反方向旋转pi/4角度
    Tdg * a

    //Swap: 交换两个量子比特的状态
    Swap * (a, b)

    //SqrtSwap: 对两个量子比特做sqrt交换
    SqrtSwap * (a, b)

    //SqrtX: 平方根X门
    SqrtX * a

    //Rxx: 两个量子比特绕x^x旋转，旋转角度为theta
    Rxx(theta) * (a, b)

    //Ryy: 两个量子比特绕y^y旋转，旋转角度为theta
    Ryy(theta) * (a, b)

    //Rzz: 两个量子比特绕z^z旋转，旋转角度为theta
    Rzz(theta) * (a, b)

    //Barrier: 分隔量子比特，阻止量子线路对相应量子比特做优化等处理
    Barrier * a
    Barrier * (a, b)

    //U1: 对单个量子比特绕z轴旋转
    U1(lambda) * a

    //U2: 对单个量子比特绕x+z轴旋转
    U1(phi, lambda) * a

    //U3: 通用单量子比特旋转门
    U1(theta, phi, lambda) * a
    
    //CH: 阿达玛门控制
    CH * (a, b)

    //CP: 控制相位门
    CP(theta) * (a, b)

    //CR: 控制旋转门
    CR(theta) * (a, b)

    //CRx: 控制Rx门
    CRx(theta) * (a, b)

    //CRy: 控制Ry门
    CRz(theta) * (a, b)

    //CRz: 控制Rz门
    CRz(theta) * (a, b)
	
    //CSx: 控制√X门
    CSx * (a, b)

    //CU: 控制U门
    CU(theta, phi, lambda, beta) * (a, b)

    //CU1: 控制U1门
    CU1(theta) * (a, b)

    //CU3: 控制U3门
    CU3(theta, phi, lambda) * (a, b)
	
    //CY: 控制Y门
    CY * (a, b)

    //CZ: 多控制Z型门
    CZ * (a, b)

    //I: 对单量子比特应用单位矩阵
    I * a
	
    //ISwap: 在量子比特a和b之间执行iSWAP门
    ISwap(theta) * (a, b)

    //R: 绕cos(theta) + sin(theta)轴旋转角度phi
    R(theta, phi) * a

    //X1: 应用单量子比特X1门
    X1 * a

    //Y1: 应用单量子比特Y1门
    Y1 * a
	
    //Z1: 应用单量子比特Z1门
    Z1 * a

    //Sxdg: Sqrt(X)门逆操作
    Sxdg * a

    //MCX: 多控制X(非)门，前两个量子比特为控制位
    MCX(2) * (a, b, c)

    //MCZ: 多控制Z门，前两个量子比特为控制位
    MCZ(2) * (a, b, c)

    //CSwap: 受控交换门，第一个量子比特为控制位
    CSwap * (a, b, c)

    //CSqrtX: 控制√X门
    CSqrtX * (a, b)

    //SqrtXdg: Sqrt(X)门逆操作
    SqrtXdg * a
	
    
2.Operator: 量子算符（将若干基础量子门封装成一些通用量子操作）
-------------------
.. code-block:: 	
    
    //QAA: 量子振幅放大
    QAA(3, 7) * qreg (对qreg中的量子选取状态值7作为标记值进行三次QAA迭代计算)
	
    //QFT: 量子傅里叶转移算子
    QFT * qreg (对qreg中的所有量子比特进行QFT操作)
    qubits = list(qreg)[::-2]
    QFT * qubits
	
    //QSP: 量子态制备算子
    QSP("+") * qreg (对qreg中的所有量子比特进行QSP操作)


3.Meta: 量子元操作（需要配合其他基础量子门实现特定运算，一般不单独使用）
-------------------
.. code-block:: 
		
    //All: 对所给量子门进行封装，提供对多量子比特便捷操作

    //对qreg中的所有量子比特进行测量
    All(Measure) * qreg
    //对qreg中的所有量子比特进行H门操作
    All(H) * qreg
	
    //Inv: 对所给量子门进行反转

    //对H门做反转操作
    Inv(H) * a

    //Power: 对给定量子门做指数运算

    //对量子比特a连续做两次H门操作
    Power(2, H) * a

	
4.示例
-------------------
circuit导出成QuSL
>>>>>>>>>>>>>>>>

.. code-block:: 

    from qutrunk.circuit import QCircuit
    from qutrunk.circuit.gates import H, CNOT, Measure
    qc = QCircuit()

    qr = qc.allocate(2)

    # apply gate
    H * qr[0]
    CNOT * (qr[0], qr[1])
    Measure * qr[0]
    Measure * qr[1]

    # print circuit
    qc.print("bell_pair.qusl")

导出QuSL格式文件，内容如下

.. code-block::

    {"target": "QuSL", "version": "1.0", "meta": {"circuit_name": "circuit-23694", "qubits": "2"}, "code": ["H * q[0]\n", "MCX * (q[0], q[1])\n", "Measure * q[0]\n", "Measure * q[1]\n"]}


解析并运行QuSL量子线路
>>>>>>>>>>>>>>>>>>>>>>>

.. code-block:: 

	import os

	from qutrunk.circuit import QCircuit

	BASE_DIR = os.getcwd()
	file_path = BASE_DIR + '/qutrunk/example/bell_pair.qusl'

    circuit = QCircuit.load(file_path, "qusl")
	circuit.print()
	result = circuit.run(shots=100)
	print(result.get_counts())   
	
运行结果如下

.. code-block::

	qreg q[2]
	creg c[2]
	H * q[0]
	MCX(1) * (q[0], q[1])
	Measure * q[0]
	Measure * q[1]
	[{"00": 50}, {"11": 50}]
