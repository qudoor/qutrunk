# QuTrunk使用指导-量子计算后端（qutrunk-0.1.13版本）

QuTrunk的backends模块即量子计算后端模块，用于执行量子线路，支持Python本地后端，qusprout和qusaas（目前还在集成中还未推出）两种远程后端以及第三方后端(目前支持IBM，后期还在支持AWS braket)：

- 本地local后端：Backendlocal
- 远程后端：Qusprout后端：BackendQuSprout；QuSaaS后端（开发中，后续推出）
- 第三方后端：IBM后端：BackendIBM；AWS braket后端（对接开发中，后续推出）

下面分别介绍后端的调用方法。

## 1、本地local后端

backends模块中本地local后端为Backendlocal，如果需要使用本地local后端，在程序中先调用这个模块，然后程序中指定后端为local，如果程序中不指定后端，则默认使用本地local作为后端，下面的示例代码中注释行有说明，构建线路的时候括号内参数为空即可，本示例使用bell_pair算法示例：

```python
from qutrunk.backends import BackendLocal #导入qutrunk的backends模块中的Backendlocal后端
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure# new QCircuit object

qc = QCircuit(backend=BackendLocal()) #指定计算后端为local
# or use as default
# qc = QCircuit()
qr = qc.allocate(2)
H * qr[0]
CNOT * (qr[0], qr[1])
Measure * qr[0]
Measure * qr[1]
res = qc.run(shots=1024)
print("==========circuit running result=========")
print(res.get_counts())
print("===========circuit running info==========")
print(res.excute_info())
qc.draw()
```

程序运行结果如下：

![image-20221115161759872](image\image-20221115161759872.png)

## 2、远程后端

### 2.1 QuSprout后端

QuSprout 是启科量子自主研发的一款免费、开源的量子计算模拟后端。它支持多个量子线路的任务管理、MPI多进程并行计算，在量子线路执行效率上较高。用户在 使用QuTrunk 量子编程框架生成量子线路后默认是使用本地local后端完成计算的，在面对大型复杂线路计算时本地local后端在执行效率会存在瓶颈，这种情况下用户可以指定后端连接到 QuSprout进行模拟计算，以达到更高的执行效率。

调用QuSprout后端的方法比较简单，和local一样，首先需要导入backends模块中的BackendQuSprout，然后在构建电路的时候指定后端为QuSprout即可。下面我们以GHZ态制备的示例来说明如何调用QuSprout进行模拟计算：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import CX, Measure, H, Barrier, All
from qutrunk.backends import BackendQuSprout #导入BackendQuSprout

def run_ghz(backend=None):
    # Create quantum circuit
    qc = QCircuit(name="ghz", backend=backend)

    # Allocate quantum qubits
    qr = qc.allocate(3)

    # Create a GHZ state
    H * qr[0]
    CX * (qr[0], qr[1])
    CX * (qr[0], qr[2])

    Barrier * qr

    # Measure all the qubits
    All(Measure) * qr

    # Run quantum circuit with 1024 times
    res = qc.run(shots=1024)

    # Print measure results like:
    # [{"000": 527}, {"111": 497}]

    print("==========circuit running result=========")
    print(res.get_counts())
    print("===========circuit running info==========")
    print(res.excute_info()) #打印线路执行信息，包括使用的后端类型

    return qc

if __name__ == "__main__":
    # Run locally
    circuit = run_ghz(backend=BackendQuSprout())#指定计算后端后QuSprout
     
    # Dram quantum circuit
    circuit.draw()
```

执行结果如下：

![image-20221115153145224](image\image-20221115153145224.png)

### 2.1 QuSaaS后端

QuSaaS是启科量子开发的一个开发者社区平台，提供启科各软件产品的API访问，QuTrunk的计算后端也可以通过QuSaaS远端访问实现，目前该功能还在开发完善中，后续教程会补充上。

## 3、第三方后端

### 3.1 IBM后端

#### 3.1.1 IBMid账号注册及API token生成

使用IBM的后端，需要用户先到IBM的网站上注册IBM的账号IBMid，访问地址：[IBM Quantum](https://quantum-computing.ibm.com/)

<img src="image\image-20221115153750950.png" alt="image-20221115153750950" style="zoom:67%;" />

注册完IBM账号后，登录到IBM quamtum主页，然后在账号详情下生成token，https://quantum-computing.ibm.com/account

打开页面后在API token下面点击生成token即可生成一个token，

<img src="image\image-20221115154531007.png" alt="image-20221115154531007" style="zoom:67%;" />

点击token旁边的复制按钮复制下token备用

#### 3.1.2 IBM后端调用及程序运行

本示例中仍然使用bell_pair示例来演示：

```python
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import H, CNOT, Measure
from qutrunk.backends import BackendQuSprout,BackendIBM

def run_bell_pair(backend=None):
    # allocate
    qc = QCircuit(backend=backend)
    qr = qc.allocate(2)

    # apply gate
    H | qr[0]
    CNOT | (qr[0], qr[1])
    Measure | qr[0]
    Measure | qr[1]

    # print circuit
    qc.print()

    # run circuit
    res = qc.run(shots=1024)
    print(res)
    return qc

#-----------------------------main-------------- 
# IBMQ作为计算后端
token = "替换成复制下来的token" 
circuit = run_bell_pair(backend=BackendIBM(token=token))

# 打印电路
circuit.draw()

```

输出结果如下：

![image-20221115174150003](image\image-20221115174150003.png)

返回的结果为json格式，统计结果为00为517次，11为507次`{'counts': {'0x0': 517, '0x3': 507}`

#### 3.1.3 IBMQ平台查询job执行信息

通过上面示例程序终端打印的信息可以看到，我们调用IBMQ后端计算资源成功，并生成了Job ID: 63735eaa754e4624994cf410.通过这个ID我们可以到IBMQ平台页面查询job执行情况。我们登录IBMQ平台，点击左上角的导航按钮展开导航，然后选Jobs

<img src="image\image-20221115160458822.png" alt="image-20221115160458822" style="zoom:67%;" />

打开Jobs页面，可以看到jobs列表，找到我们刚执行的jobid并点击打开，列表中第一个即为我们刚才执行的示例程序的job

![image-20221115174403571](image\image-20221115174403571.png)

打开job页面后展示信息如下：

![image-20221115174427153](image\image-20221115174427153.png)

Details可以查询到job创建时间，执行时间，Provider和runmode信息，以及状态时间线等信息。

Histogram是运行1024次后两个状态统计的次数，鼠标放到柱状上可以显示$\ket{00}$状态显示的次数为517,$\ket{11}$状态显示的次数为507

ciruit部分为打印的线路图，提供3种线路格式：图形化，QASM格式以及Qiskit格式

图形格式：

<img src="image\image-20221115161515110.png" alt="image-20221115161515110" style="zoom:67%;" />

QASM格式：

<img src="image\image-20221115161534072.png" alt="image-20221115161534072" style="zoom: 67%;" />

Qiskit格式：

<img src="image\image-20221115161553181.png" alt="image-20221115161553181" style="zoom:50%;" />

从IBM的页面的运行结果看，与我们在第一节本地local计算的结果基本一致。

### 3.1 AWS Braket后端

AWS Braket后端目前还在开发中，后续开发完后本章节会继续补充调用方法