# QuTrunk 量子编程介绍

​         使用QuTrunk编写量子计算程序，一般需要经过三个主要步骤，首先构建量子线路，再次运行量子线路，最后输出结果。下面是之前一个bell_pair算法例子，下面会针对各个步骤做详细说明：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure

qc= QCircuit()
qr = qc.allocate(2) # allocate

H * qr[0]   # apply gate
CNOT * (qr[0], qr[1])
Measure * qr[0]
Measure * qr[1]

qc.print()   # print circuit
res = qc.run(shots=1024) # run circuit

print(res.get_counts()) # 打印输出统计结果

qc.draw() #打印量子线路图
```

上面示例程序运行后输出结果如下：

```python
qreg q[2]
creg c[2]
H | q[0]
CX(1) | (q[0], q[1])
Measure | q[0]
Measure | q[1]
[{"00": 522}, {"11": 502}]
      ┌───┐      ┌─┐   
q[0]: ┤ H ├──■───┤M├───
      └───┘┌─┴──┐└╥┘┌─┐
q[1]: ─────┤ CX ├─╫─┤M├
           └────┘ ║ └╥┘
c[0]: ════════════╩══╬═
                  0  ║
c[1]: ═══════════════╩═
                     1
```

整个程序执行有以下几个步骤：

1. 导入程序包

   程序需要使用的基本元素通过如下方式导入：

   ```python
   from qutrunk.circuit import QCircuit
   from qutrunk.circuit.gates import H, CNOT, Measure
   ```
   
   - QCircuit：是量子线路 ，维护对所有量子比特的各种门操作及操作时序，代表了整个量子算法的实现,其语法是：
   
     ```python
     class circuit.circuit.QCircuit(backend=None, density=False, name:Optional[str]=None, resource:Optional[bool]=False)
     ```
   
     backend用于运行量子电路。density则是创建密度矩阵Qureg对象，表示一组可以进入噪声和混合状态的量子位。本例子中backend采用默认none即使用本地local资源。
   
   - gates是量子门，例如常用的Pauli-X，Pauli-Y，Pauli-Z，H，CNOT，Measure，Rx，Ry，Rz等。
   
2. 初始化变量：allocate分配2个量子比特，并将其初始化状态置为0状态

   ```python
   qc=QCircuit()
   qr = qc.allocate(2) # allocate
   ```

3. 添加门操作

   - H | qr[0] 是对第一个量子比特q[0]执行H门操作，将其置于叠加态
   - CNOT | (qr[0], qr[1])对控制量子位q[0]和目标量子位q[1]执行受控非操作，使其置于量子纠缠状态

4. 对两个执行门操作后的2个量子比特添加measure测量

5. 将程序操作指令按QASM指令打印输出，并执行量子线路运算，执行次数为1024次（shots=1024）

6. 统计运行结果，并可视化输出，运行结果如下，00出现的次数是522次，11出现的次数书502次。

   ```python
   [{"00": 522}, {"11": 502}]
   ```

7. qc.draw()最后执行打印输出量子线路图

```python
      ┌───┐      ┌─┐   
q[0]: ┤ H ├──■───┤M├───
      └───┘┌─┴──┐└╥┘┌─┐
q[1]: ─────┤ CX ├─╫─┤M├
           └────┘ ║ └╥┘
c[0]: ════════════╩══╬═
                  0  ║
c[1]: ═══════════════╩═
                     1
```

如需详细了解QuTrunk的操作指导，[请点击]()