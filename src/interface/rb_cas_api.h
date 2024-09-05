/**
 * @file rb_cas_api.h
 * @brief 该文件由程序自动生成，接口头文件
 *
 * @author jwzhou (zhou24388@163.com)
 * @version v0.5.0
 * @date 2023-06-07 17:27:55.454597
 *
 * @copyright Copyright (c) 2023 Hefei Source Intelligence Technology Co,.Ltd.
 */

#ifndef __RB_CAS_API_H__
#define __RB_CAS_API_H__

#include "rb_common.h"
#include "rb_cas_para.h"


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief 
 * 
 * @param  phCAS            算法句柄
 * @param  s32W             图像宽度
 * @param  s32H             图像高度
 * @return RB_EXPORTS 
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAS_Create(RB_HANDLE* phCAS, RB_S32 s32W, RB_S32 s32H);

/**
 * @brief 
 * 
 * @param  hCAS             算法句柄
 * @return 算法状态
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAS_Destroy(RB_HANDLE hCAS);


/**
 * @brief 
 * 
 * @param  pcLibVerStr      算法库版本号
 * @return 算法状态
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAS_GetLibVer(RB_S8* pcLibVerStr);

/**
 * @brief 
 * 
 * @param  hCAS             算法句柄
 * @param  pstCASPara       CAS参数
 * @return 算法状态
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAS_Config(RB_HANDLE hCAS, const RB_CAS_Para_S* pstCASPara);

/**
 * @brief 
 * 
 * @param  hCAS             算法句柄
 * @param  pstImage         图像数据
 * @return 算法状态
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAS_Process(RB_HANDLE hCAS, const RB_IMAGE_S* pstImage);

/**
 * @brief 
 * 
 * @param  hCAS             算法句柄
 * @param  pstCASResult     CAS结果
 * @return 算法状态 
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAS_GetResults(RB_HANDLE hCAS, RB_CAS_Result_S* pstCASResult);


#ifdef __cplusplus
}
#endif

#endif // !__RB_CAS_API_H__
