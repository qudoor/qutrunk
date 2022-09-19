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

    //------quroot错误码------
    //内存不足
    QUROOT_MEM_NOT_ENOUGH = 100,

    //未初始化
    QUROOT_NOT_INIT = 101,

    //机器未注册
    QUROOT_NOT_REGISTER = 102,

    //无可用的资源
    QUROOT_NOT_RESOURCE = 103,
}

const map<ErrCode, string> ErrMsg = {
    ErrCode.COM_SUCCESS: "成功", 
    ErrCode.COM_OTHRE: "未知错误", 
    ErrCode.COM_INVALID_PARAM: "不合法参数", 
    ErrCode.COM_TIMEOUT: "超时", 
    ErrCode.COM_EXCEPTION: "异常", 
    ErrCode.QUROOT_MEM_NOT_ENOUGH: "内存不足", 
    ErrCode.QUROOT_NOT_INIT: "未初始化",
    ErrCode.QUROOT_NOT_REGISTER: "机器未注册",
    ErrCode.QUROOT_NOT_RESOURCE: "无可用的资源",
}

//返回信息
struct BaseCode {
    //返回码
    1: required ErrCode code

    //返回描述
    2: required string msg
}