# -a 拷贝所有文件到目标路径
echo building quest... 

dest_dir=./build
if [ -d ${dest_dir} ];then 
    rm -rf ${dest_dir}
fi

mkdir build
cd build
cmake ..
make

echo build quest done!