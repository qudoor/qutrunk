单比特量子逻辑门
`````````````````````````````````````````````````
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