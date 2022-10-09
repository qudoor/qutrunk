echo [thrift] updating python file...

dest_dir=../
if [ -d ${dest_dir} ];then 
    rm -rf ${dest_dir}/rpc
fi

src_dir=./gen-py
if [ -d ${src_dir} ];then 
    rm -rf ${src_dir}
fi

thrift --gen py ecode.thrift
rm -rf ${dest_dir}/code
cp -r ./gen-py/code ${dest_dir}/code
# rm -rf ./gen-py

thrift --gen py:package_prefix='qutrunk.sim.qusprout.' qusproutdata.thrift
rm -rf ${dest_dir}/qusproutdata 
cp -r ./gen-py/qusproutdata ${dest_dir}/qusproutdata
# rm -rf ./gen-py

thrift --gen py:package_prefix='qutrunk.sim.qusprout.' qusprout.thrift
rm -rf ${dest_dir}/qusprout 
cp -r ./gen-py/qusprout ${dest_dir}/qusprout
# rm -rf ./gen-py

echo [thrift] update python file done!
