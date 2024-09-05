/**
 * @file rb_cas_para.h
 * @brief 该文件由程序自动生成，参数头文件
 *
 * @author jwzhou (zhou24388@163.com)
 * @version v0.5.0
 * @date 2023-06-07 17:27:55.454573
 *
 * @copyright Copyright (c) 2023 Hefei Source Intelligence Technology Co,.Ltd.
 */
#ifndef __RB_CAS_PARA_H__
#define __RB_CAS_PARA_H__

#include <stdio.h>
#include <iostream>
#include <vector>
#include "rb_common.h"

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CAS_ROI 4  /* 最多ROI个数 */
#define MAX_PED_NUM 32 /* 最多检测32个人 */
#define MAX_VER_NUM 8  /* ROI最多定点数 */

    // 该参数为设置危险灵敏度
    // 危险灵敏度为设置人员与危险区域的距离进行调整
    // 该处主要以人的BoundingBox与ROI的相对重叠面积为阈值
    typedef enum __rb_cas_danger_sense_e
    {
        RB_CAS_DANGER_SENS_LOW = -1,    /* 低灵敏度，非常靠近才算危险，  */
        RB_CAS_DANGER_SENS_DEFAULT = 0, /* 默认灵敏度，默认距离 */
        RB_CAS_DANGER_SENS_HIGH = 1,    /* 高灵敏度，较远距离就算危险 */
    } RB_CAS_DangerSens_E;

    // CAS参数
    typedef struct __rb_cas_para_s
    {
        RB_S32 s32BufferLen;                            /* 默认为5，可能用于平滑结果使用 */
        RB_S32 s32ROINum;                               /* 默认为1, */
        std::vector<RB_POINT_S> vecROIPts[MAX_CAS_ROI]; /* 设置的ROI点集, new added by jw@0721，每个点集至少3个点*/
        RB_CAS_DangerSens_E enSens;                     /* 默认灵敏度, 相交面积大于目标的10% */
    } RB_CAS_Para_S;

    // 人的属性
    typedef enum __rb_cas_ped_attr_e
    {
        RB_CAS_PED_ATTR_FREE = 0,    /* 自由人, 不靠近任何敏感区域 */
        RB_CAS_PED_ATTR_CLOSETO = 1, /* 靠近敏感区域 */
        RB_CAS_PED_ATTR_STANDIN = 2, /* 在敏感区域 */
    } RB_CAS_Ped_Attr_E;

    // output
    typedef struct __rb_cas_ped_s
    {
        RB_FLOAT f32Prob;           /* 人的分数 */
        RB_RECT_S stPedBoundingBox; /* 人的矩形框 */
        RB_S32 s32labelID;

    } RB_CAS_Ped_S;


    // 结果
    typedef struct __rb_cas_results_s
    {
        std::vector<RB_CAS_Ped_S> vecPedSet;             /* 自由人, 不靠近任何ROI */
    } RB_CAS_Result_S;

#ifdef __cplusplus
}
#endif
#endif // !__RB_CAS_PARA_H__
