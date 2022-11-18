# QuTrunk使用指导-量子算符（qutrunk-0.1.13版本）

上一节我们介绍了QuTrunk的各种量子逻辑门和其操作方法，本章开始介绍量子算符。目前提供的算符有如下：

## 1、Classical算符

Classical算符用于制备量子经典态，默认情况下，QuTrunk分配的量子比特初始化的量子态为$\ket{0}$,如果用户需要根据需要制定特定状态，可以通过Classical算符来制备。Classical算符有一个参数state：电路需要初始化成的目标态，调用方法：`Classical(state) * qureg`

示例代码如下：

```python
#导入需要使用到的qutrunk的包
from qutrunk.circuit.ops import Classical
from qutrunk.circuit import QCircuit

#分配量子寄存器。构建2比特的量子线路
circuit = QCircuit()
qureg = circuit.allocate(2)

#打印构建线路的向量状态
print(circuit.get_statevector())

#执行Classical算符操作，将目标状态设置为1，二进制为0b01
Classical(1) * qureg

#打印线路图
circuit.print()
circuit.draw()
#输出执行Classical算符操作后的向量
print(circuit.get_statevector())
```

程序运行结果如下：

![image-20221114170356983](C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20221114170356983.png)

执行操作前，向量值是$\begin{pmatrix}
1&0&0&0
\end{pmatrix}$,对其执行Classical操作之后，最终输出的状态向量为：$\begin{pmatrix}
0&1&0&0
\end{pmatrix}$

Classical算符通常与其他算符一起使用，完成一定的功能。例如ADD算符中，将量子态初始化为初始化值，然后在此基础上执行ADD，后面的算符中将详细介绍。

## 2、PLUS算符

PLUS算符就是给所有的量子比特做了一个H门，只能作用于线路开始，下面通过示例通过PLUS算法制备plus态

```python
from qutrunk.circuit.ops import PLUS
from qutrunk.circuit import QCircuit

circuit = QCircuit()
qureg = circuit.allocate(4)
PLUS * qureg
circuit.print(unroll=False)
print(circuit.get_statevector())
```

输出的结果如下：

<img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20221115101047479.png" alt="image-20221115101047479" style="zoom:67%;" />

可以看到分配的4个量子比特均被执行了H门操作。PLUS算符通常用于除初始化线路时的批量H门操作

## 3、ADD算符

通过ADD算符可以执行加法计算，ADD算符有一个参数number：增加到线路的值即加数，调用的方法：`ADD(number) * qreg`

如下是调用ADD算符进行加法计算的示例：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Measure, All
from qutrunk.circuit.ops import Classical
from qutrunk.circuit.ops import ADD

#定义加法函数
def run_addition(num_qubits, init_value, number=0):
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    Classical(init_value) * qr

    ADD(number) * qr

    All(Measure) * qr
    res = circuit.run()
    print(res.get_outcome())

    return circuit

=
if __name__ == "__main__":
    circuit = run_addition(4, 3, 4) #4个量子比特，执行3+4
```

本示例中，定义了一个加法函数，这个函数的参数有3个：num_qubits即量子比特数，执行后，输出的结果如下，打印的结果是二进制$0b0111=2^2+2^1+2^0=7$

```python
0b0111
```

4比特最大支持加数值为0~15，如果需要更大加法数的计算也可以调整比特数，例如如果需要计算100+100 ，我可以将比特数增加到8

```python
circuit = run_addition(8,100, 100)
```

重新执行后，计算结果打印如下：

```python
0b11001000
```

$0b11001000=2^7+2^6+2^3=128+64+8=200$ 计算准确

## 4、INC算符（Increment）

INC算符即自增算符，相当于C语言中`++`算符，调用方法：`INC * qreg`使用方法如下：

```Python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Measure, All
from qutrunk.circuit.ops import Classical
from qutrunk.circuit.ops import INC


def increment(num_qubits, init_value):
    # Create quantum circuit
    circuit = QCircuit()

    # Allocate quantum qubits
    qr = circuit.allocate(num_qubits)

    # Set initial amplitudes to classical state with init_value
    Classical(init_value) * qr

    # Apply quantum gates
    INC * qr

    # Measure all quantum qubits
    All(Measure) * qr

    # Run quantum circuit
    res = circuit.run()

    # Print measure result like:
    # 0b0001
    print(res.get_outcome())

    return circuit


if __name__ == "__main__":
    # Run locally
    circuit = increment(4, 10)

    # Dram quantum circuit
    circuit.draw()
```

本示例中，定义了一个自减函数，这个函数的参数有2个：num_qubits即量子比特数和执行自减的初始化值，执行后，输出的结果如下，打印的结果是二进制

<img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20221114180250750.png" alt="image-20221114180250750" style="zoom:67%;" />

如果需要执行较大的值，可以增加量子比特数，例如如果初始化值为100，可以指定量子比特为8

```python
circuit = increment(8, 100)
```

<img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20221114180438733.png" alt="image-20221114180438733" style="zoom:67%;" />



## 5、DEC算符（Decrement）

DEC算符是自减算符，相当于`--`,应用算符方法`DEC * qreg`, 

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Measure, All
from qutrunk.circuit.ops import Classical
from qutrunk.circuit.ops import DEC


def decrement(num_qubits, init_value):
    circuit = QCircuit()
    qr = circuit.allocate(num_qubits)

    Classical(init_value) * qr

    DEC * qr

    All(Measure) * qr
    res = circuit.run()
    print(res.get_outcome())

    return circuit


if __name__ == "__main__":
    # Run locally
    circuit = decrement(4, 13)

    # Draw quantum circuit
    circuit.draw()
```

和ADD算符一样，支持的初始值数据大小与量子比特数量相关，示例中量子比特指定为4（最大支持值为15）,示例中将初始值设置为13

计算结果如下：

<img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20221115102316455.png" alt="image-20221115102316455" style="zoom:67%;" />

如果需要计算超过100的，比特数跟着调整，例如设置比特数为8，初始值为100，计算结果如下：

<img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20221115102543906.png" alt="image-20221115102543906" style="zoom: 50%;" />

## 6、AMP算符

AMP 算符用于将量子态初始化成特定的振幅态，有3个参数classicvector：振幅态列表，startind：振幅开始的index，numamps：振幅数量，应用AMP算符方法：`AMP(classicvetor,startind,numamps)`, 示例代码如下：

```python
from qutrunk.circuit.ops import AMP
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, All, Measure

circuit = QCircuit()
qureg = circuit.allocate(2)

AMP([1-2j, 2+3j, 3-4j, 0.5+0.7j], 1, 2) * qureg
print(circuit.get_statevector())
```

经过AMP算符后，输出的结果如下

<img src="C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20221115110903230.png" alt="image-20221115110903230" style="zoom:67%;" />

## 6、QAA算符

QAA算符为量子振幅放大算符，有2个参数，iterations:QAA迭代次数，marked_index:标记qubit的index，使用QAA算符的方法为：`QAA(iterations,marked_index)`,示例代码如下：

```python
from qutrunk.circuit.ops import QAA
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, All

circuit = QCircuit()
qureg = circuit.allocate(4)

All(H) * qureg
QAA(3, 7) * qureg

for i in range(2 ** len(qureg)):
    print(circuit.get_prob(i))
```

首先四个量子位均匀叠加，然后选择状态值7作为标记值。执行三次QAA迭代计算，操作后获得的结果是7的对应状态超过96%。

打印结果如下：

```python
prob of state |7> = 0.4726562499999991
prob of state |7> = 0.9084472656249968
prob of state |7> = 0.9613189697265575
0.002578735351562488
0.0025787353515624874
0.002578735351562488
0.0025787353515624874
0.002578735351562488
0.0025787353515624874
0.002578735351562488
0.9613189697265575
0.002578735351562488
0.0025787353515624874
0.002578735351562488
0.0025787353515624874
0.002578735351562488
0.0025787353515624874
0.002578735351562488
0.0025787353515624853
```

## 7、QFT算符

QFT 算符为量子傅里叶变换算符，其应用方法为：`QFT * qreg`,QFT算符目前支持最大的量子比特数为20。示例代码：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import All, Measure
from qutrunk.circuit.ops import QFT


def full_qft():
    circuit = QCircuit()
    qreg = circuit.allocate(8)

    QFT * qreg

    circuit.draw(line_length=1000)
    state = circuit.get_statevector()
    print(state)

    All(Measure) * qreg

    res = circuit.run(shots=1000)
    print(res.get_counts())


if __name__ == "__main__":
    full_qft()
```

输出结果如下：

![image-20221115114056030](C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20221115114056030.png)

## 8、QPE算符

QPE算符即量子相位估计算符，只有一个参数unitary：标准门，使用方法为`QPE(unitary) * qreg `, 以下是一个用到QPE算符计算$T$门相位估计的示例:

```python
from math import pi
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import NOT, Barrier, P, All, Measure
from qutrunk.circuit.ops import QPE


def _bin_int(itrable):
    return int("".join(map(str, reversed(itrable))), base=2)

def run_qpe(backend=None):
    """Estimate T-gate phase."""
    # allocate
    qc = QCircuit(backend=backend)
    q1, q2 = qc.allocate([4, 1])

    # Prepare our eigenstate |psi>
    NOT * q2[0]
    Barrier * q1
    Barrier * q2
    # apply QPE
    QPE(P(pi/4)) * (q1, q2)

    # measure q1
    All(Measure) * q1

    # print circuit
    # qc.print()

    # run circuit
    qc.run(shots=100)

    # print result
    print(q1.to_cl())
    # calculate the value of theta
    f = _bin_int(q1.to_cl())
    theta = f / 2 ** len(q1)
    print("θ=", theta)
    
    return qc

if __name__ == "__main__":
    circuit = run_qpe()
    circuit.draw()
```

输出的结果如下：

![image-20221115142733292](C:\Users\huang\AppData\Roaming\Typora\typora-user-images\image-20221115142733292.png)



