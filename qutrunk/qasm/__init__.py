"""
QASM转化为qutrunk量子程序  convert_qasm_to_qutrunk
主要步骤：
1 读取包含QASM内容的文件
2 调用convert_qasm_to_qutrunk()方法
3 调用convert_qutrunk_to_qasm()方法以把量子程序转为QASM，通过比较输入和生成的QASM是否相同，判断QASM是否正确转化成量子程序。
"""

from .qasm import Qasm