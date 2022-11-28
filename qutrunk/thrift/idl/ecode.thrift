namespace py code

enum ErrCode {
    //------公共错误码------
    //成功
    COM_SUCCESS = 0,  

    //未知错误  
    COM_OTHRE = 1,
    
    //不合法参数
    COM_INVALID_PARAM = 2,

    //超时
    COM_TIMEOUT = 3,

    //异常
    COM_EXCEPTION = 4,

    //不存在
    COM_NOT_EXIST = 5,

    //存在
    COM_IS_EXIST = 6,

    //缓存中
    COM_IS_QUEUE = 7,  

    //内存不足
    COM_MEM_NOT_ENOUGH = 8,

    //未初始化
    COM_NOT_INIT = 9,

    //机器未注册
    COM_NOT_REGISTER = 10,

    //无可用的资源
    COM_NOT_RESOURCE = 11,

    //数据解析异常
    COM_PRASE_ERROR = 12,  
}

const map<ErrCode, string> ErrMsg = {
    ErrCode.COM_SUCCESS: "success",
    ErrCode.COM_OTHRE: "unknown error",
    ErrCode.COM_INVALID_PARAM: "invalid parameter",
    ErrCode.COM_TIMEOUT: "timeout", 
    ErrCode.COM_EXCEPTION: "exception", 
    ErrCode.COM_NOT_EXIST: "non-existence",
    ErrCode.COM_IS_EXIST: "existence",
    ErrCode.COM_IS_QUEUE: "be queuing",
    ErrCode.COM_MEM_NOT_ENOUGH: "out of memory", 
    ErrCode.COM_NOT_INIT: "uninitialized",
    ErrCode.COM_NOT_REGISTER: "unregistered",
    ErrCode.COM_NOT_RESOURCE: "out of resource",
    ErrCode.COM_PRASE_ERROR: "parse error",
}

//返回信息
struct BaseCode {
    //返回码
    1: required ErrCode code

    //返回描述
    2: required string msg
}