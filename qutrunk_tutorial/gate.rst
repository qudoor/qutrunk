量子门
==========

单量子比特逻辑门
------------------
.. tabularcolumns:: |m{0.06\textwidth}<{\centering}|c|c|

.. list-table:: 
   :align: center
   :class: longtable 
   
   * - 门                                                     
     - 功能                    
     - 数学定义
     - 使用案例
   * - ``H``                                                   
     - 常用于使量子比特处于叠加态                     
     - :math:`\begin{bmatrix} 1/\sqrt {2} & 1/\sqrt {2} \\ 1/\sqrt {2} & -1/\sqrt {2} \end{bmatrix}\quad` 
     - H * qr[0]
   * - ``X``                                                   
     - 进行取反操作, 量子比特绕布洛赫球的x轴旋转pi角度                     
     - :math:`\begin{bmatrix}0 &1 \\  1&0\end{bmatrix}` 
     -  X * qr[0]
   * - ``Y``                                                   
     - 量子比特绕布洛赫球的y轴旋转pi角度                    
     - :math:`\begin{bmatrix}0 &-j \\  j&0\end{bmatrix}` 
     - Y * qr[0]
   * - ``Z``                                                   
     - 量子比特绕布洛赫球的z轴旋转pi角度                    
     - :math:`\begin{bmatrix}1 &0 \\  0&-1\end{bmatrix}` 
     - Z * qr[0] 
   * - ``P``                                                   
     - 将量子比特0>态和1>态的相位根据给定的角度进行移动                    
     - :math:`\begin{bmatrix} 1  & 0 \\ 0 & e^{i\theta }  \end{bmatrix}` 
     - P(alpha) * qr[0]
   * - ``S``                                                   
     - 量子比特绕布洛赫球的z轴旋转pi/2角度              
     - :math:`\begin{bmatrix} 1 & 0\\ 0 & 1j\end{bmatrix}` 
     - S * qr[0]
   * - ``T``                                                   
     - 量子比特绕布洛赫球的z轴旋转pi/4角度                   
     - :math:`\begin{bmatrix} 1 & 0\\ 0 & e^\frac{i\pi}{4}\end{bmatrix}` 
     - T * qr[0]
   * - ``U1``                                                   
     - 对单个量子比特绕z轴旋转                
     - :math:`\begin{bmatrix} 1 &0\\ 0 &e^{j\lambda }\end{bmatrix}`            
     - U1(pi/2) * qr[0]
   * - ``U2``                                                   
     - 对单个量子比特绕x+z轴旋转                  
     - :math:`\begin{bmatrix} \sqrt{2}  &-e^{j\lambda }\times \sqrt{2}  \\e^{j\phi  }\times \sqrt{2} &e^{j\lambda+j\phi}\times \sqrt{2} \end{bmatrix}`                 
     - U2(pi/2, pi/2) * qr[0]
   * - ``U3``                                                   
     - 通用单量子比特旋转门                   
     - :math:`\begin{bmatrix} \cos(\theta/2) &e^{j\lambda} \times \sin(\theta/2) \\ e^{j\phi }\times \sin(\theta/2) & e^{j\lambda+j\phi}\times \cos(\theta/2) \end{bmatrix}\quad`          
     - U3(pi/2,pi/2,pi/2) * qr[0]
   * - ``I``                                                   
     - 对单量子比特应用单位矩阵                 
     - :math:`\begin{bmatrix} 1 & 0\\ 0 & 1\end{bmatrix}` 
     - I * qr[0]
   * - ``R``                                                   
     - 绕cos(theta) + sin(theta)轴旋转角度phi               
     - :math:`\begin{bmatrix}\cos (\theta /2 ) &-je^{-j\times \phi} \times\sin\theta/2   \\-je^{j\times \phi} \times\sin\theta/2 &\cos\theta  /2 \end{bmatrix}` 
     - R(pi / 2, pi / 2) * qr[0]
   * - ``X1``                                                   
     - 应用单量子比特X1门               
     - :math:`\begin{bmatrix} \frac{1}{\sqrt{2} }  & -1j*\frac{1}{\sqrt{2} }\\ -1j*\frac{1}{\sqrt{2} } & \frac{1}{\sqrt{2} }\end{bmatrix}` 
     - X1 * qr[0]
   * - ``Y1``                                                   
     - 应用单量子比特Y1门                
     - :math:`\begin{bmatrix} \frac{1}{\sqrt{2} }  & -1j*\frac{1}{\sqrt{2} }\\ \frac{1}{\sqrt{2} } & \frac{1}{\sqrt{2} }\end{bmatrix}` 
     -  Y1 * qr[0]
   * - ``Z1``                                                   
     - 应用单量子比特Z1门              
     - :math:`\begin{bmatrix} 1j*\frac{π}{4} & 0 \\ 0 & e^{1j*\frac{π}{4}} \end{bmatrix}` 
     - Z1 * qr[0]


多量子比特逻辑门
------------------
.. tabularcolumns:: |m{0.06\textwidth}<{\centering}|c|c|

.. list-table:: 
   :align: center
   :class: longtable 
   
   * - 门                                                     
     - 功能                    
     - 数学定义
     - 使用案例
   * - ``CX``                                                   
     - qr[0]作为控制位，qr[1]为目标位，如果qr[0]为1则对qr[1]进行取反，如果qr[0]为0则不做任何操作                  
     - :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 0 & 1\\ 0 & 0 & 1 & 0\end{bmatrix}` 
     - CX * (qr[0], qr[1])
   * - ``Toffoli``                                                   
     - qr[0], qr[1]作为控制位，qr[2]为目标位, 如果qr[0], qr[1]均为1则对qr[1]进行取反，否则不做任何操作                
     - :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 0 & 1\\ 0 & 0 & 1 & 0\end{bmatrix}` 
     - Toffoli * (qr[0], qr[1], qr[2])   
   * - ``Measure``                                                   
     - 测量单个量子比特，将其随机压缩为0或1                 
     - None            
     - Measure * qr[0]    
   * - ``Ry``                                                   
     - 量子比特绕布洛赫球的y轴旋转theta角度                   
     - :math:`\begin{bmatrix} \cos (\alpha /2) & -\sin\alpha /2 \\\sin\alpha /2 & \cos (\alpha /2) \end{bmatrix}` 
     - Ry(alpha) * qr[0]
   * - ``Rz``                                                   
     - 量子比特绕布洛赫球的z轴旋转theta角度                  
     - :math:`\begin{bmatrix}e^{-j\alpha /2}  &0 \\ 0 &e^{j\alpha /2}\end{bmatrix}` 
     - Rz(alpha) * qr[0]
   * - ``Tdg``                                                   
     - 对T门的反向操作, 绕布洛赫球的z轴反方向旋转pi/4角度                   
     - :math:`\begin{bmatrix} 1 & 0\\0  &e^{j\times \frac{\pi }{4} } \end{bmatrix}`           
     - Tdg * qr[0]
   * - ``Swap``                                                   
     - 交换两个量子比特的状态                  
     - :math:`\begin{bmatrix}1 & 0 & 0 & 0\\ 0 & 0 & 1 & 0\\ 0 & 1 & 0 & 0\\ 0 & 0 & 0 & 1\end{bmatrix}` 
     - Swap * (qr[0], qr[1])
   * - ``SqrtSwap``                                                   
     - 对两个量子比特做sqrt交换                   
     - :math:`\begin{bmatrix}1 & 0 & 0 & 0\\ 0 & 0.5+0.5j & 0.5-0.5j & 0\\ 0 & 0.5-0.5j & 0.5+0.5j & 0\\ 0 & 0 & 0 & 1\end{bmatrix}` 
     - SqrtSwap * (qr[0], qr[1])
   * - ``SqrtX``                                                   
     - 平方根X门                   
     - :math:`\begin{bmatrix}1 + 1j & 1 - 1j\\ 1 - 1j & 1 + 1j\end{bmatrix}` 
     - SqrtX * qr[0]
   * - ``Rxx``                                                   
     - 两个量子比特绕x^x旋转，旋转角度为theta             
     - :math:`\begin{bmatrix} e^{- j\alpha /2}  & 0 & 0 & 0\\ 0 & e^{ j\alpha /2}  & 0 &0 \\0 & 0 & e^{ j\alpha /2} & 0\\ 0 & 0 & 0 &e^{- j\alpha /2}\end{bmatrix}`            
     - Rxx(alpha) * (qr[0], qr[1])
   * - ``Ryy``                                                   
     - 两个量子比特绕y^y旋转，旋转角度为theta                  
     - :math:`\begin{bmatrix}\cos \alpha /2  & 0 & 0 &j\times \sin \alpha /2 \\ 0 & \cos \alpha /2 & -j\times \sin \alpha /2 & 0\\ 0 & -j\times \sin \alpha /2 &\cos \alpha /2   &0 \\ j\times \sin \alpha /2  & 0 & 0 &\cos \alpha /2 \end{bmatrix}`           
     - Ryy(alpha) * (qr[0], qr[1])
   * - ``Rzz``                                                   
     - 两个量子比特绕z^z旋转，旋转角度为theta                 
     - :math:`\begin{bmatrix}1 + 1j & 1 - 1j\\ 1 - 1j & 1 + 1j\end{bmatrix}`            
     - Rzz(alpha) * (qr[0], qr[1])
   * - ``Barrier``                                                   
     - 分隔量子比特，阻止量子线路对相应量子比特做优化等处理              
     - None              
     - Barrier * (qr[0], qr[1])
   * - ``CH``                                                   
     - 阿达玛门控制                  
     - :math:`\begin{bmatrix} 1/\sqrt {2} & 0 & 1/\sqrt {2} & 0 \\ 0 & 1 & 0 & 0 \\ 1/\sqrt {2} & 0 & 1/\sqrt {2} & 0 \\ 0 & 0 & 0 & 1\end{bmatrix}\quad`              
     - CH * (qr[0], qr[1])
   * - ``CP``                                                   
     - 控制相位门                   
     - :math:`\begin{bmatrix} 1 & 0 & 0 & 0\\ 0 & 1 & 0 & 0 \\ 0 &  0& e^{j\alpha }  & 0\\ 0 &  0& 0 &1\end{bmatrix}`                            
     - CP(pi / 2) * (qr[0], qr[1])
   * - ``CR``                                                   
     - 控制旋转门                   
     - :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & \exp(j\alpha ) \end{bmatrix}\quad`
     - CR(pi / 2) * (qr[0], qr[1])
   * - ``CRx``                                                   
     - 控制Rx门               
     - :math:`\begin{bmatrix} \cos (\alpha /2) & 0 & -j\sin \alpha  & 0\\ 0 & 1 & 0 & 0 \\ -j\sin \alpha  &  0& \cos (\alpha /2)  & 0\\ 0 &  0& 0 &1\end{bmatrix}`              
     - CRx(pi / 2) * (qr[0], qr[1])
   * - ``CRy``                                                   
     - 控制Ry门                   
     - :math:`\begin{bmatrix} \cos(\theta/2) & 0 & -\sin(\theta/2) & 0 \\ 0 & 1 & 0 & 0 \\ \sin(\theta/2)  & 0 & \cos(\theta/2) & 0 \\ 0 & 0 & 0 & 1\end{bmatrix}\quad`             
     - CRy(pi / 2) * (qr[0], qr[1])
   * - ``CRz``                                                   
     - 控制Rz门                  
     - :math:`\begin{bmatrix} e^\frac{-i\theta}{2} & 0 & 0 & 0\\0 & 1 & 0 & 0\\ 0 & 0 & e^\frac{i\theta}{2} & 0 \\ 0 & 0 & 0 & 1\end{bmatrix}`            
     - CRz(pi / 2) * (qr[0], qr[1])
   * - ``CSx``                                                   
     - 控制√X门  
     - None        
     - None                       
   * - ``CU``                                                   
     - 控制U门                 
     - :math:`\begin{bmatrix} 1 & 0 & 0 &0 \\ 0& e^{j\gamma }\times \cos \theta /2  & 0 &-e^{j(\gamma+\lambda ) }\times \sin  \theta /2 \\ 0 & 0 &1  &0 \\ 0 & e^{j(\gamma+\phi ) }\times \sin  \theta /2 &  0& 0 & e^{j(\gamma+\phi+\lambda  ) }\times \cos \theta /2\end{bmatrix}`         
     - CU(pi / 2, pi / 2, pi / 2, pi / 2) * (qr[0], qr[1])
   * - ``CU1``                                                   
     - 控制U1门                   
     - :math:`\begin{bmatrix} 1 & 0 & 0 &0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 1 & 0\\ 0& 0  &  0&e^{j\alpha }\end{bmatrix}`        
     - CU1(pi / 2) * (qr[1], qr[2])
   * - ``CU3``                                                   
     - 控制U3门            
     - :math:`\begin{bmatrix}1  & 0 & 0 & 0 \\0 &\sin(\alpha/2) &0&-e^{1j*\theta }*\sin (\alpha/2)\\ 0& 0 & 1 &0 \\0  & e^{1j*\phi  }*\sin (\alpha/2) & 0 &e^{1j*(\phi +\lambda )}*\cos (\alpha/2)  \end{bmatrix}`                      
     - CU3(pi / 2, pi / 2, pi / 2) * (qr[0], qr[1])
   * - ``CY``                                                   
     - 控制Y门                   
     - :math:`\begin{bmatrix} 0 & 0 & -1j & 0 \\ 0 & 1 & 0 & 0\\ 1j & 0 & 0 & 0\\ 0 & 0 & 0 & 1\end{bmatrix}`            
     - CY * (qr[0], qr[1])
   * - ``CZ``                                                   
     - 多控制Z型门            
     - :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & -1 & 0\\ 0 & 0 & 0 & 1\end{bmatrix}`        
     - CZ * (qr[0], qr[1])
   * - ``MCX``                                                   
     - 多控制X(非)门，前两个量子比特为控制位             
     - :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 0 & 1\\ 0 & 0 & 1 & 0\end{bmatrix}`           
     - MCX(2) * (qr[0], qr[1], qr[2])
   * - ``MCZ``                                                   
     - 多控制Z门，前两个量子比特为控制位          
     - :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & -1 & 0\\ 0 & 0 & 0 & 1\end{bmatrix}`                  
     - MCZ(2) * (qr[0], qr[1], qr[2])
   * - ``CSwap``                                                   
     - 受控交换门，第一个量子比特为控制位            
     - :math:`\begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ \end{bmatrix}\quad`                                
     - CSwap * (qr[0], qr[1], qr[2])
   * - ``CSqrtX``                                                   
     - 控制√X门             
     - :math:`\begin{bmatrix} (1+1j)/2 & 0 & (1-1j)/2 &0 \\ 0 & 1 &  0& 0\\ (1-1j)/2  & 0 & (1+1j)/2 & 0\\ 0 & 0 &0  &1\end{bmatrix}`                        
     - CSqrtX * (qr[0], qr[1])
   * - ``SqrtXdg``                                                   
     - Sqrt(X)门逆操作             
     - :math:`\begin{bmatrix}1-1j  &1+1j \\(1-1j)/2  &(1+1j)/2 \end{bmatrix}`                           
     - SqrtXdg * qr[0]
   * - ``ISwap``                                                   
     - 在量子比特a和b之间执行iSWAP门                 
     - :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\alpha  & -j\times \sin\alpha  & 0 \\ 0 & -j\times \sin\alpha  & \cos\alpha  & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}\quad` 
     - iSwap(pi / 2) * (qr[0], qr[1])
   * - ``Rx``                                                   
     - 量子比特绕布洛赫球的x轴旋转theta角度                   
     - :math:`\begin{bmatrix} \cos(\theta/2) & -1i×\sin(\theta/2) \\ -1i×\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}\quad` 
     - Rx(alpha) * qr[0]
   * - ``Sdg``                                                   
     - 对S门的反向操作, 绕布洛赫球的z轴反方向旋转pi/2角度            
     - :math:`\begin{bmatrix} 1 & 0\\ 0 & 1j\end{bmatrix}` 
     - Sdg * qr[0]