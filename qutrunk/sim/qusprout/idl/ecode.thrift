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
    ErrCode.COM_SUCCESS: "成功", 
    ErrCode.COM_OTHRE: "未知错误", 
    ErrCode.COM_INVALID_PARAM: "不合法参数", 
    ErrCode.COM_TIMEOUT: "超时", 
    ErrCode.COM_EXCEPTION: "异常", 
    ErrCode.COM_NOT_EXIST: "不存在",
    ErrCode.COM_IS_EXIST: "存在",
    ErrCode.COM_IS_QUEUE: "缓存中",
    ErrCode.COM_MEM_NOT_ENOUGH: "内存不足", 
    ErrCode.COM_NOT_INIT: "未初始化",
    ErrCode.COM_NOT_REGISTER: "机器未注册",
    ErrCode.COM_NOT_RESOURCE: "无可用的资源",
    ErrCode.COM_PRASE_ERROR: "数据解析异常",
}

//返回信息
struct BaseCode {
    //返回码
    1: required ErrCode code

    //返回描述
    2: required string msg
}