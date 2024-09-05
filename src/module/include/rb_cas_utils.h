/**
 * @file rb_cas_utils.h
 * @brief 该文件由程序自动生成，内部核心业务逻辑类声明
 *
 * @author jwzhou (zhou24388@163.com)
 * @version v0.5.0
 * @date 2023-06-07 17:27:55.454623
 *
 * @copyright Copyright (c) 2023 Hefei Source Intelligence Technology Co,.Ltd.
 */
#ifndef __RB_CAS_UTILS_H__
#define __RB_CAS_UTILS_H__

#include "rb_cas_gconfig.h"
#include "rb_cas_para.h"
#include "rb_cas_detector.h"

typedef struct __rb_roi_tgt_intersect_s
{
    RB_S32 s32ROIID;
    RB_S32 s32TgtID;
    RB_FLOAT f32IntersectArea;
    cv::Point ptContoursCenter;
} RB_ROI_Tgt_Inter_S;

class CRbCASUtils
{
public:
    /* 构造函数 */
    CRbCASUtils(RB_S32 s32W, RB_S32 s32H);

    // 参数配置
    int config(const RB_CAS_Para_S *pstPara);

    // 处理
    int process(const RB_IMAGE_S *pstImage);

    // 获得结果
    int get_results(RB_CAS_Result_S *pstResult);


    // 删除目标检测器
    ~CRbCASUtils();

private:
    /* 清理目标信息 */
    void __clean_objs();

    /* 初始化参数 */
    void __init_para();

    /* 初始化模型，加载模型参数 */
    void __init_model();

    /* 设置尺寸以及利用尺寸初始化 */
    void __set_size(RB_S32 s32W, RB_S32 s32H);

    /* 规则后处理 */
    int __post_process();

private:
    /* 图像宽 */
    RB_S32     m_s32ImW;          
    
    /* 图像高 */
    RB_S32     m_s32ImH;          
    
    /* 缓冲区大小 */
    RB_S32     m_s32BufLen;

    /* ROI的个数 */
    RB_S32     m_s32ROINum;       

    /* 不同灵敏度下的阈值 */
    RB_DOUBLE  m_dOverlapRatio; 

    /* 危险区域的灵敏度*/
    RB_CAS_DangerSens_E  m_enDangerSens;   

    /* 所有ROI的点集 */
    std::vector<RB_POINT_S>   m_avecROIPolyGon[MAX_CAS_ROI];    

    /* 检测返回的结果 */
    RB_CAS_Result_S      m_stResults;   

#ifdef WITH_NCNN
    // 检测器
    CYOLOV5_NCNN_Detector* m_PCYOLOV5DetNet;
#else
#ifdef WITH_RKNN
    /* 目标检测器 */
    CYOLOV5_NPU_Detector*  m_PCYOLOV5DetNet;
#endif
#endif

    /* 检测结果 */
    std::vector<RB_DetTarget_S> m_vecObjs;

    // 输入的图像
    cv::Mat m_bgrIn;

    /* 掩码图像的缩放比例 */
    RB_FLOAT   m_f32MaskScale;  
    RB_S32     m_s32ScaleW;
    RB_S32     m_s32ScaleH;

    /* ROI的掩码 */
    cv::Mat    m_matROIMask;    
    std::vector<cv::Mat> m_vecMatTgtMask;

    std::vector<RB_ROI_Tgt_Inter_S> m_vecIntersect;
};

#endif // !__RB_CAS_UTILS_H__
