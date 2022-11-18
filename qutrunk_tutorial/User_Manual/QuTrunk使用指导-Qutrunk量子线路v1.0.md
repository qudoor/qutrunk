# QuTrunk使用指导-量子电路（QuTrunk-0.1.13版本）



 现在开始介绍QuTrunk的使用方法，QuTrunk提供了量子编程所有需要使用到的基本构建模块。核心的模块有如下几个:

- QCircuit 量子电路，维护对所有量子比特的各种门操作及操作时序，代表了整个量子算法的实现。
- Qubit 代表单个量子比特，是量子门操作的对象。
- Qureg 维护若干个量子比特，用于实现一个具体的量子算法。
- Command 每个量子门操作其背后都会转换成一个基础指令，这些指令按照时间顺序存放在QCircuit中，当整个算法结束或者需要计算当前量子电路的某种状态取值时，这些指令会被发送到指定的后端去执行。
- Backend 量子计算后端模块，用于执行量子电路，支持Python和C++两种本地后端，QuSprout后端以及第三方后端(目前支持IBM)等。
- Gate 量子算法基本组成单元，提供各类量子门操作，包括:*H*, *X*, *Y*, *Z*，*P*, *R*, *Rx*, *Ry*, *Rz*, *S*, *Sdg*, *T*, *Tdg*, *CNOT*, *Toffoli*, *Swap*等

使用QuTrunk进行量子程序编程流程主要就是2个步骤，首先是构建量子电路，然后就是运行量子电路。构建量子电路允许用户根据实际的问题的解决方案使用QuTrunk设计对应的量子电路，运行则允许用户选择不同的backend作为后端量子运算输出量子结果。

## 1、构建量子电路

我们开始一个3比特的量子电路的编写，这个程序使用到的元素有circuit和gates两个基本元素，输出程序首先需要先import QuTrunk的circut模块和量子门gates模块，如下：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure
```

然后构建量子电路，通过qc分配3个量子比特，即完成一个3量子比特的量子电路的构建：

```python
qc = QCircuit()
qr = qc.allocate(3)
```

## 2、执行量子门操作

构建电路后，然后可以添加门（“操作”）来操作寄存器。本示例中我们需要得到一个GHZ状态的线路,其数学表达式如下：
$$
\left|\varphi\right \rangle=\left (\left | 000 \right \rangle+\left|111 \right \rangle \right )/\sqrt{2}
$$
为了创建这样的状态，我们从一个三量子位量子寄存器开始。默认情况下，寄存器中的每个量子位都初始化为 $\left|0\right\rangle$。为了获得GHZ状态，我们先后对3个量子执行不同的量子门操作来完成，首先对量子比特0执行H门使其置于叠加态，然后在量子比特qr[0]和qr[1]之间执行CNOT门，最后在量子比特qr[0]和量子比特qr[2]之间执行CNOT,在理想的量子计算机上，运行改电路产生的状态是上面的GHZ状态。通过QuTrunk将逐个操作添加到电路中，程序如下：

```python
H * qr[0]   
CNOT * (qr[0], qr[1])
CNOT * (qr[0], qr[2])
```

## 3、可视化输出量子电路

QuTrunk可以将量子电路以可视化的形式输出，通过调用draw方法来实现，程序如下：

```python
qc.draw()
```

量子电路图打印输出结果如下图：

<img src="image\image-20221103165323970.png" alt="image-20221103165323970" style="zoom:80%;" />

该电路图中，量子比特是从上到下排列，q[0]第一个量子比特在最上方，q[2]第三个量子比特在最下方，门的排列顺序是从左到右，即最早操作的门在最左边，最后操作的门在最右边。

## 4、打印量子电路

如果需要将程序在终端上输出，可以使用QuTrunk的print方法来输出，默认情况下不指定格式输出的是QuTrunk自己的QuSL格式

```python
qc.print()
```

打印的输出的结果格式如下：

```python
qreg q[3]
creg c[3]
H * q[0]
MCX(1) * (q[0], q[1])
MCX(1) * (q[0], q[2])
```

如果指定打印格式为openQASM的格式，在参数里面指定format为openqasm，如下：

```python
qc.print(format=openqasm)
```

输出打印格式结果如下

```python
OPENQASM 2.0;
include "qulib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[0],q[2];
```

##  5、测量    

实际应用中，构建电路图后如果不实施测量，将无法获得量子比特的状态，所以线路中需要增加测量。实施测量后，量子系统将坍塌为经典量子位|0>和|1>。本示例中，我们对GHZ状态的3个量子分别实施测量。可以对每个实施过门操作的qubit逐个施加measure操作来测量

```python
Measure * qr[0]
Measure * qr[1]
Measure * qr[2]
```

也可以通过All操作，对所有的量子比特线路实施测量，这个时候需要import gates里面的All操作

```python
All(Measure) * qr
```

添加测量后输出电路打印如下，此线路中增加了3个经典寄存器c[0]~c[3]，用于将量子比特测量结果映射到经典位。

<img src="image\image-20221103165240268.png" alt="image-20221103165240268" style="zoom:80%;" />

## 6、 运行量子电路并获取结果

为了得到量子比特状态的分布统计数据，需要对线路运行多次并测量，运行次数可以通过参数shots来指定，例如本例子中执行次数设定为1024次。获得结果对象后，可以通过函数访问获取统计结果并通过print打印输出。

```python
result = qc.run(shots=1024)
print(result.get_counts())
```

获取结果对象后就可以通过get_counts方法获取统计数据，本示例的完整程序如下

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure,All

qc= QCircuit()
qr = qc.allocate(3)

H * qr[0]   
CNOT * (qr[0], qr[1])
CNOT * (qr[0], qr[2])
All(Measure) * qr

qc.print()

result = qc.run(shots=1024)
print(result.get_counts())

qc.draw()
```

运行后打印结果如下：

<img src="image\image-20221103165200409.png" alt="image-20221103165200409" style="zoom:80%;" />

从结果可以看到000状态的次数是537次，获得111状态的次数为487次，每个状态的次数各占大约50%。

## 7、获取线路所有状态的对应概率

QuTrunk提供了获取各状态概率的实现函数，通过调用QCircuit的get_probs方法就可以获取。调用如下程序语句即可：

```python
print(qc.get_probs())
```

完整程序程序如下：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT

qc= QCircuit()
qr = qc.allocate(3)

H * qr[0]  
CNOT * (qr[0], qr[1])
CNOT * (qr[0], qr[2])

print(qc.get_probs())
qc.draw()
```

执行后终端显示的结果如下，总共7个状态，其中状态是000和111的概率各占约50%

<img src="image\image-20221103165120012.png" alt="image-20221103165120012" style="zoom:80%;" />

## 8、获取状态向量

如果需要知道线路的状态向量，QuTrunk也提供了获取线路状态向量的函数，通过调用QCircuit的get_statevector可获得

```python
print(qc.get_statevector())
```

完整的示例代码如下：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure,All

qc = QCircuit()
# allocate 2 qureg: q1, q2
q = qc.allocate(3)

H * q[0]
CNOT * (q[0], q[1])
CNOT * (q[1], q[2])

qc.draw()

print(qc.get_statevector())
```

运行后输出如下结果如下：

<img src="image\image-20221103165026039.png" alt="image-20221103165026039" style="zoom:80%;" />

## 9、导出量子电路

通过QuTrunk构建的量子电路支持以openQASM或者QuSL格式导出，以方便提供给其他量子应用程序直接使用。通过调用QuTrunk.circuit.QCircuit的dump方法可以完成电路的导出操作，下面以openQASM格式的文件导出为例说明其实现方法：

```Python
#导入需要使用到的QuTrunk的模块
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure

# 分配2个量子寄存器构建2比特的量子电路
qc = QCircuit()
qr = qc.allocate(2)

# 执行门操作和测量
H * qr[0]
CNOT * (qr[0], qr[1])
Measure * qr[0]
Measure * qr[1]

# 将量子电路通过openqasm格式导出，保存到当前目录下，dump的两个参数，第一个为文件导出名，第二个为导出格式指定）
qc.dump(file="bell_pair.qasm", format="openqasm")

#根据导出的openqasm文件打印出量子程序
with open(file="bell_pair.qasm") as f:
    for line in f:
        print(line, end="")
```

输出结果如下，可以看到在示例程序的目录下生成了bell_pair.qasm文件，打印输出的内容与原始量子电路一致。

<img src="image\image-20221104115440864.png" alt="image-20221104115440864" style="zoom:80%;" />

## 10、反序列化OpenQASM或者QuSL文件对象，并运行量子电路

QuTrunk支持将开发设计的量子电路以openasm和qusl格式导出，同样也支持这两种类型的文件对象反向导入并运行。通过QuTrunk.circuit.QCircuit.load方法导入实现，示例如下：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure

circuit = QCircuit.load(file="bell_pair.qasm", format="openqasm")#format可以是openqasm也可以是qusl
# run circuit
res = circuit.run(shots=100)
# print result
circuit.print()
circuit.draw()
print(res.get_measure())
print(res.get_counts())
```

输出结果如下：

<img src="image\image-20221104113954558.png" alt="image-20221104113954558" style="zoom:80%;" />

## 11、申请多个量子寄存器

前面第一节讲到构建量子电路的时候需要先申请量子寄存器，可以申请一个量子寄存器保存量子比特，也可以同时申请多个量子寄存器，每个量子比特保存到一个独立的量子寄存器。实现方法如下：

```Python
#导入需要使用到的QuTrunk的模块
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure

#构建量子电路
qc = QCircuit()

# 分配2个量子寄存器: q1, q2，即总共申请2个量子比特
q1, q2 = qc.allocate([1, 1])

#对量子比特执行门操作和执行测量
H * q1[0]
CNOT * (q1[0], q2[0])
Measure * q1[0]
Measure * q2[0]
#运行量子电路
res = qc.run(shots=100)
#打印量子电路，
qc.draw()

#获取量子寄存器测量结果
print(q1.to_cl())
print(q2.to_cl())

#运行线路并获取统计数据
res = qc.run(shots=100)
print(res.get_counts())
```

运行后结果如下：

<img src="image\image-20221103164752232.png" alt="image-20221103164752232" style="zoom:80%;" />

## 12、量子电路追加

量子电路构建过程中，构建好一个量子电路后，如果需要再追加线路并且把追加线路的门操作添加到之前的线路上，QuTrunk针对此场景也提供了解决办法，通过append方法可以实现，如下示例就是先构建了一个2比特的量子电路circ1，然后追加线路circuit执行measure操作，此种场景可以用于后续机器学习，一个用于编码，另追加一个线路用于神经网络训练，示例如下：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure, X, All

circ1 = QCircuit()
qr = circ1.allocate(2)

# apply gate
H * qr[0]
CNOT * (qr[0], qr[1])

circuit = QCircuit()
q = circuit.allocate(2)

circuit.append_circuit(circ1)

All(Measure) * q

# run circuit
circuit.print()
res = circuit.run(shots=100)
```

运行结果如下：

<img src="image\image-20221104114525168.png" alt="image-20221104114525168" style="zoom:80%;" />

## 13、参数化量子电路

参数化量子电路（Parameterized quantum circuit, PQC）是进行量子机器学习的一种途径，QuTrunk提供处理带参数的量子电路的方法，并通过与AI深度学习框架结合利用量子神经网络对其进行训练获取最优值。

QuTrunk中参数化对应创建是通过circuit模块下如下几个函数来实现：

- create_parameter(name)：分配一个参数化对象，参数name是str类型，为参数的名字，函数返回值为参数对象
- create_parameters(names)：分配一组参数化对象，参数names是list类型，是一组参数的名字。函数返回值为元组（tuple）
- get_parameters（)：列出线路所有参数
- bind_parameters(params)：将参数绑定特定值，参数params为dict类型{parameter：value，...}

如下示例中展示QuTrunk中如何使用参数化函数构建参数化量子电路：

```python
#1、导入所需QuTrunk的包
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Ry

#2、构建量子电路2个量子比特，给线路创建2个角度参数theta和phi
circuit = QCircuit()
q = circuit.allocate(2)
theta, phi = circuit.create_parameters(["theta", "phi"])

#3、执行带参数角度的Ry门操作
Ry(theta) * q[0]
Ry(phi) * q[1]

#4、将参数绑定到量子电路c1，c2，并打印状态向量
c1 = circuit.bind_parameters({"theta": 1, "phi": 1})
print(c1.get_statevector())
c2 = circuit.bind_parameters({"theta": 2, "phi": 2})
print(c2.get_statevector())
```

输出的结果如下：

<img src="image\image-20221115175923733.png" alt="image-20221115175923733" style="zoom:67%;" />

## 14、量子电路的深度和宽度

对于构建的量子电路，如何评估一个量子电路的”大小“？量子电路也提供了几种重要特性以帮助量化电路的“大小”以及它们在嘈杂的量子设备上运行的能力。其中一些特性，如量子比特的数量很容易理解，而另一些，如深度和张量分量的数量，则需要更多的解释。本节中我们将讨论量子电路的深度和宽度，看看在QuTrunk中如何获取线路的深度和宽度，并为理解电路在实际设备上运行时如何变化做准备，强调它们变化的条件。我们可以从如下构建的一个量子电路的示例来讨论：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import *

qc=QCircuit()
qr=qc.allocate(12)

for idx in range(5):
    H * qr[idx]  
    CX * (qr[idx],qr[idx+5])

CX * (qr[1],qr[7])
X * qr[8]
CX * (qr[1],qr[9])
X * qr[7]
CX * (qr[1],qr[11])

Swap * (qr[6],qr[11])
Swap * (qr[6],qr[9])
Swap * (qr[6],qr[10])
X * qr[6]

qc.draw()
```

获取的电路如下：

<img src="image\image-20221116095358139.png" alt="image-20221116095358139" style="zoom:67%;" />

从图上我们可以直观的看到总共12个qubits和12个cbits，也可以方便数出$H,CX,X,SWAP$这些门的数量，但是从程序中如何获取?QuTrunk中提供了这些属性的获取方法。

QuTunk中的circuit模块提供了num_qubits这个属性来获取qbits数量，通过print可以打印出有多少个qubits，调用方法如下：

```python
print(qc.num_qubits)
```

QuTrunk中量子电路中qubits和cbits是同时一起创建的，所以width就是qubits和cbits的数量之和，通过线路的属性width可以获取，调用方法如下：

```python
print(qc.width())
```

circuit的模块也提供了num_gates这个属性，用于统计量子电路中所有门的总数，调用方法：

```python
print(qc.num_gates)
```

获取的数量分别是width=24，num_qubits=12，num_gates=19执行结果如下：

![image-20221116104639303](image\image-20221116104639303.png)

量子电路还有一个重要的特性称为电路深度（depth）。量子电路的深度是衡量并行执行的量子门有多少“层”，完成电路定义的计算所需的时间。因为量子门实现需要时间，所以电路的深度大致对应于量子计算机执行电路所需的时间。因此，电路的深度是衡量量子电路能否在器件上运行的一个重要量。

量子电路的深度在数学上定义为有向无环图（DAG）中最长的路径。通过定义直观看是难以理解的也不容易获取，我们可以按IBM介绍的使用旋转法，将电路图逆时针向上翻转，让门从上往下掉落，掉落得越远越好，类似于俄罗斯方块游戏，每掉落到一层就累积一层深度。QuTrunk中定义的量子线路的深度为circuit模块的一个属性depth，可以通过如下方法调用：

```python
cir_dep=qc.depth()
print(f'the circuit depth is {cir_dep}')
```

![image-20221116140217940](image\image-20221116140217940.png)

## 15、计算量子电路期望值

在量子力学里，重复地做同样实验，通常会得到不同的测量结果，期望值（expectation value）是理论平均值，可以用来预测测量结果的统计平均值。量子力学显露出一种内禀统计行为。同样的一个实验重复地做很多次，每次实验的测量结果通常不会一样，只有从很多次的实验结果计算出来的统计平均值，才是可克隆的数值。量子理论不能预测单次实验的测量结果，量子理论可以用期望值来预测多次实验得到的统计平均值。

采用狄拉克符号标记，假设量子系统的量子态为$\ket{\psi}$,则对这个量子态，可观察量$O$的期望值$\left \langle O \right \rangle$可以定义为：
$$
\langle O\rangle \ {\stackrel  {def}{=}}\ \langle \psi |{\hat  {O}}|\psi \rangle
$$
假设算符 ${\displaystyle {\hat {O}}}$ 的一组本征态 ${\displaystyle |e_{i}\rangle ,\quad i=1,\,2,\,3,\,\dots \ }$ 形成了一个具有正交归一性的基底：$\langle e_{i}|e_{j}\rangle =\delta _{{ij}} ;$

其中，${\displaystyle \delta _{ij}}$ 是克罗内克函数。本征态 ${\displaystyle |e_{i}\rangle }$ 的本征值为$ {\displaystyle O_{i}}:\hat{O} \ket{e_i}=O_i\ket{e_i} $ ,量子态$\ket{\psi}$可以展开为这些本征态的线性组合：
$$
\ket{\psi}=\sum_{i}c_i\ket{e_i}
$$
其中$c_i=\bra{e_i}\ket{\psi}$是复系数，是在量子态$\ket{e_i}$里找到的量子态$\ket{\psi}$的概率幅，应用全等式：$\sum_{i}\ket{e_i}\bra{e_i}=1$可观测量$O$的期望值可以写为：
$$
{\begin{aligned}\langle O\rangle &=\langle \psi |{\hat  {O}}|\psi \rangle \\&=\sum _{{i,j}}\langle \psi |e_{i}\rangle \langle e_{i}|{\hat  {O}}|e_{j}\rangle \langle e_{j}|\psi \rangle \\&=\sum _{{i,j}}\langle \psi |e_{i}\rangle \langle e_{i}|e_{j}\rangle \langle e_{j}|\psi \rangle O_{i}\\&=\sum _{i}|\langle e_{i}|\psi \rangle |^{2}O_{i}\\\end{aligned}}
$$
这表达式很像是一个算术平均值，它表明了期望值的物理意义：本征值 ${\displaystyle O_{i}}$ 是实验的可能结果，对应的系数$ {\displaystyle |\langle e_{i}|\psi \rangle |^{2}=|c_{i}|^{2}}$是这结果可能会发生的概率。汇总所有本征值与其对应的概率系数的乘积，就可以得到期望值。

QuTrunk中提供了线路期望值获取方法，通过调用circuit模块的如下函数来获取

- expval_pauli(paulis)计算泡利算子乘积的期望值。参数paulis(`Union`[`list`, `object`]) – oper_type (int)：Pauli运算符。target（int）：目标量子位的索引。返回值：泡利算子乘积的期望值
- expval_pauli_sum(pauli_coeffs)计算泡利算子乘积和的期望值。pauli_coeffs([`PauliCoeffs`] –维护一个PauliCoeff列表，每个PauliCoeff由一个系数和一系列pauli算子组成，PauliCoreff用于计算pauli乘积的和。返回值：返回泡利积的和。

如下是应用以上函数计算线路期望值的示例：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Ry, PauliZ, PauliI

circuit = QCircuit()
q = circuit.allocate(2)

Ry(1.23) * q[0]
pauli_str = [PauliZ(q[0]), PauliI(q[1])]
expect = circuit.expval_pauli(pauli_str)
print(expect)
```

输出的期望值结果为：

```python
0.33423772712450267
```

类似的计算线路的期望值之和，示例如下：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, Ry, PauliZ, PauliI, PauliX, PauliCoeff, PauliType, PauliCoeffs

circuit = QCircuit()
q = circuit.allocate(2)

H * q[0]
Ry(1.23) * q[1]

pauli_coeffs = PauliCoeffs() << PauliCoeff(0.12, [PauliType.PAULI_Z]) \
    << PauliCoeff(0.34, [PauliType.PAULI_X, PauliType.PAULI_I])

expect_sum = circuit.expval_pauli_sum(pauli_coeffs)
print(expect_sum)
```

获取的结果如下：

```
0.33999999999999997
```
