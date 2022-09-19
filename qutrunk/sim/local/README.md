# 构建说明：
1. mkdir build   # local目录下
2. cd build
3. cmake ..
4. cmake --build . 
# step 4 默认编译是debug模式
# step 4 cmake --build . --config Release 可以更改为release模式
# Local debug need specify RelWithDebInfo mode like "cmake --build . --config RelWithDebInfo" to generate pdb file, but still need specify Release when check in code so that   dll do not contains debug information

如果缺少构建环境，比如cmake, c/c++编译器，则需要根据出错提示安装相应的工具