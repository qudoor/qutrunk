namespace py qusproutdata

include "ecode.thrift"

//执行指令的方式
enum ExecCmdType {
    ExecTypeDefault = 0,
    ExecTypeCpuSingle = 1,
    ExecTypeCpuMpi = 2,
    ExecTypeGpuSingle = 3
}

//振幅数据
struct Amplitude {
    //振幅实部
    1: required list<double> reals

    //振幅虚部
    2: required list<double> imags

    //振幅开始索引
    3: required i32 startind

    //振幅结束索引
    4: required i32 numamps
}

//自定义矩阵
struct Matrix {
    //矩阵实部
    1: required list<list<double>> reals

    //矩阵虚部
    2: required list<list<double>> imags
    
    //是否为酉矩阵
    3: required bool unitary
}

//扩展指令
struct Cmdex {
    //振幅数据
    1: optional Amplitude amp
    //矩阵数据
    2: optional Matrix mat
}

//指令数据
struct Cmd {
    //指令门类型
    1: required string gate

    //目标集
    2: required list<i32> targets

    //控制集
    3: required list<i32> controls

    //旋转集
    4: required list<double> rotation

    //指令详情描述
    5: required string desc

    //是否执行逆操作
    6: required bool inverse

    //扩展命令
    7: optional Cmdex cmdex
}

//指令集
struct Circuit {
    //指令集
    1: required list<Cmd> cmds
}

//单个比特测量结果
struct MeasureQubit {
    //量子比特
    1: required i32 idx

    //测量结果
    2: required i32 value
}

//所有比特测量结果
struct MeasureQubits {
    //测量结果
    1: required list<MeasureQubit> measure
}

//指令结果集
struct MeasureResult {
    //多次采样测量结果
    1: required list<MeasureQubits> measures
}

//任务初始化信息
struct InitQubitsReq {
    //任务id
    1: required string id

    //qubit数量
    2: required i32 qubits

    //密度矩阵
    3: optional bool density

    //执行指令的方式
    4: optional ExecCmdType exec_type

    //mpi执行的host列表
    5: optional list<string> hosts
}

struct InitQubitsResp {
    //返回码
    1: required ecode.BaseCode base
}

//发送任务
struct SendCircuitCmdReq {
    //任务id
    1: required string id

    //指令集
    2: optional Circuit circuit

    //是否释放该请求资源
    3: optional bool final
}

struct SendCircuitCmdResp {
    //返回码
    1: required ecode.BaseCode base
}

//取消任务
struct CancelCmdReq {
    //任务id
    1: required string id
}

struct CancelCmdResp {
    //返回码
    1: required ecode.BaseCode base
}

//获取振幅
struct GetProbAmpReq {
    //任务id
    1: required string id

    //qubit索引
    2: required i64 index
}

struct GetProbAmpResp {
    //返回码
    1: required ecode.BaseCode base

    //振幅
    2: optional double amp
}

//获取所有qubit的概率
struct GetProbOfAllOutcomReq {
    //任务id
    1: required string id

    //概率目标列表
    2: required list<i32> targets
}

struct GetProbOfAllOutcomResp {
    //返回码
    1: required ecode.BaseCode base

    //概率
    2: optional list<double> pro_outcomes
}

//获取所有的计算状态
struct GetAllStateReq {
    //任务id
    1: required string id
}

struct GetAllStateResp {
    //返回码
    1: required ecode.BaseCode base

    //状态
    2: optional list<string> all_state
}

//执行任务
struct RunCircuitReq {
    //任务id
    1: required string id

    //运行次数
    2: required i32 shots
}

struct RunCircuitResp {
    //返回码
    1: required ecode.BaseCode base 

    //返回结果
    2: optional MeasureResult result
}

//泡利算子操作类型
enum PauliOperType {
    POT_PAULI_I = 0, 
    POT_PAULI_X = 1, 
    POT_PAULI_Y = 2, 
    POT_PAULI_Z = 3
}

//获取泡利算子乘积的期望值
struct GetExpecPauliProdReq {
    //任务id
    1: required string id

    //期望值信息
    2: required list<PauliProdInfo> pauli_prod
}

struct GetExpecPauliProdResp {
    //返回码
    1: required ecode.BaseCode base 

    //期望值
    2: optional double expect
}

struct PauliProdInfo {
    //泡利算子操作类型
    1: required PauliOperType oper_type

    //目标比特位
    2: required i32 target
}

//获取泡利算子乘积之和的期望值
struct GetExpecPauliSumReq {
    //任务id
    1: required string id

    //泡利算子操作类型，注意：oper_type_list的数量必须是qubitnum*term_coeff_list的数量
    2: required list<PauliOperType> oper_type_list

    //回归系数
    3: optional list<double> term_coeff_list
}

struct GetExpecPauliSumResp {
    //返回码
    1: required ecode.BaseCode base 

    //期望值
    2: optional double expect
}
