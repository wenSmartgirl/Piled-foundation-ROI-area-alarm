/**
 * @file rb_cas_utils.h
 * @brief 该文件由程序自动生成, 主要是接口实现
 *
 * @author jwzhou (zhou24388@163.com)
 * @version v0.5.0
 * @date 2023-06-07 17:27:55.454717
 *
 * @copyright Copyright (c) 2023 Hefei Source Intelligence Technology Co,.Ltd.
 */
#include "rb_cas_api.h"
#include "rb_cas_gconfig.h"
#include "rb_cas_para.h"
#include "rb_cas_utils.h"

// 算法句柄结构体
typedef struct _rb_cas_handle_s
{
    CRbCASUtils *__casobj;
}RB_CAS_Handle_S;


/**
 * @brief 
 * 
 * @param  phCAS            算法句柄
 * @param  s32W             图像宽度
 * @param  s32H             图像高度
 * @return RB_EXPORTS 
 */
RB_STATUS_CODE_E RB_CAS_Create(RB_HANDLE* phCAS, RB_S32 s32W, RB_S32 s32H)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    // 创建cas实例
    RB_CAS_Handle_S* hCAS = new RB_CAS_Handle_S;
    if (NULL == hCAS)
    {
        return RB_FAILURE;
    }
    
    hCAS->__casobj = new CRbCASUtils(s32W, s32H);
    if (NULL == hCAS->__casobj)
    {
        return RB_FAILURE;
    }
    
    // 返回句柄
    *phCAS = (RB_HANDLE)hCAS;

    return s32Ret;
}

/**
 * @brief 
 * 
 * @param  hCAS             算法句柄
 * @return 算法状态
 */
RB_STATUS_CODE_E RB_CAS_Destroy(RB_HANDLE hCAS)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    RB_CAS_Handle_S* pHandle = (RB_CAS_Handle_S*)hCAS;
    if (NULL == pHandle)
    {
        return RB_FAILURE;
    }
    
    if (NULL == pHandle->__casobj)
    {
        return RB_FAILURE;
    }
    else
    {
        delete pHandle->__casobj;
        pHandle->__casobj = NULL;
    }

    delete pHandle;
    pHandle = NULL;

    return s32Ret;
}


/**
 * @brief 
 * 
 * @param  pcLibVerStr      算法库版本号
 * @return 算法状态
 */
RB_STATUS_CODE_E RB_CAS_GetLibVer(RB_S8* pcLibVerStr)
{
        sprintf((char*)pcLibVerStr, "\n=============== LIB INFO =====================\n"
            "Project: %s\n"
            "Version: %s\n"
            "Commit md5: %s\n"
            "Build DateTime: %s\n"
            "==============================================\n", 
            PROJECT_NAME,
            LIB_VERSION, 
            LIB_COMMIT_MD5,
            BUILD_DATETIME);

    return RB_SUCCESS;
}

/**
 * @brief 
 * 
 * @param  hCAS             算法句柄
 * @param  pstCASPara       CAS参数
 * @return 算法状态
 */
RB_STATUS_CODE_E RB_CAS_Config(RB_HANDLE hCAS, const RB_CAS_Para_S* pstCASPara)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    RB_CAS_Handle_S* pHandle = (RB_CAS_Handle_S*)hCAS;
    if (NULL == pHandle || NULL == pstCASPara)
    {
        return RB_FAILURE;
    }

    // 配置参数
    s32Ret = (RB_STATUS_CODE_E)(pHandle->__casobj->config(pstCASPara));

    return s32Ret;
}

/**
 * @brief 
 * 
 * @param  hCAS             算法句柄
 * @param  pstImage         图像数据
 * @return 算法状态
 */
RB_STATUS_CODE_E RB_CAS_Process(RB_HANDLE hCAS, const RB_IMAGE_S* pstImage)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    RB_CAS_Handle_S* pHandle = (RB_CAS_Handle_S*)hCAS;
    if (NULL == pHandle || NULL == pstImage || NULL == pstImage->pData)
    {
        return RB_FAILURE;
    }

#ifdef SHOW_DEBUG_TIME
    double dtStart = __get_current_time_proc();
#endif // SHOW_DEBUG_TIME

    // 处理图像
    s32Ret = (RB_STATUS_CODE_E)(pHandle->__casobj->process(pstImage));
    if (s32Ret != RB_SUCCESS)
    {
        return s32Ret;
    }

#ifdef SHOW_DEBUG_TIME
    double dtEnd = __get_current_time_proc();
    std::cout<<"++++++  In CAS API: Process Time Span ++++++ "<< (dtEnd - dtStart) <<" ms"<<std::endl;
    dtStart = __get_current_time_proc();
#endif // SHOW_DEBUG_TIME

    return s32Ret;
}

/**
 * @brief 
 * 
 * @param  hCAS             算法句柄
 * @param  pstCASResult     CAS结果
 * @return 算法状态 
 */
RB_STATUS_CODE_E RB_CAS_GetResults(RB_HANDLE hCAS, RB_CAS_Result_S* pstCASResult)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    RB_CAS_Handle_S* pHandle = (RB_CAS_Handle_S*)hCAS;
    if (NULL == pHandle || NULL == pstCASResult)
    {
        return RB_FAILURE;
    }

    // 获取结果，通过参数返回出来
    s32Ret = (RB_STATUS_CODE_E)(pHandle->__casobj->get_results(pstCASResult));

    return s32Ret;
}
