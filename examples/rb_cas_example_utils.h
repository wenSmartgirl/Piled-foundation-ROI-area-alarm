/**
 * @file rb_cas_example_utils.h
 * @brief 该文件由程序自动生成，图像和视频 demo 应用逻辑实现的接口类
 *
 * @author jwzhou (zhou24388@163.com)
 * @version v0.5.0
 * @date 2023-06-07 17:27:55.454753
 *
 * @copyright Copyright (c) 2023 Hefei Source Intelligence Technology Co,.Ltd.
 */

#ifndef __CAS_EXAMPLE_UTILS_H__
#define __CAS_EXAMPLE_UTILS_H__

#include "rb_common.h"
#include "rb_cas_api.h"
#include "rb_cas_gconfig.h"
#include "rb_cas_para.h"

/* 算法用例处理的类 */
class CRbcasExample
{
public:
    /* 通过图像宽高创建对象 */
    CRbcasExample(const int w, const int h);

    /* 通过图像创建对象 */
    CRbcasExample(cv::Mat matImgInit);

    /* 释放资源 */
    ~CRbcasExample();

    /* 配置ROI信息，其他信息默认 */
    int config(cv::Mat matImgConfig, bool bManual = false );

    /* 用例处理 */
    int process();

    /* 处理每一帧数据 */
    int process(cv::Mat matImgProcess);

    /* 显示结果 */
    int show_results(int wait_time=0, float show_ratio=1.f);

public: 
    /* 图像操作对象 */
    cv::Mat m_matResult;

private:
    /* 初始化数据 */
    int __init(int w, int h);

    /* 鼠标选择ROI区域 */
    cv::Rect mouse_selectROI(cv::Mat matImgConfig);

    /* 鼠标选择点集 */
    std::vector<cv::Point> mouse_selecPtsSet(cv::Mat matImgConfig);

private:
    /* 图像宽高 */
    int m_iImgWidth;
    int m_iImgHeight;

    /* ROI */
    cv::Rect m_rcROI;

    /* 算法处理的图像 */
    cv::Mat m_matImg;

    /* 区域点集 */
    std::vector<cv::Point> m_vPoints;

    /* 算法句柄 */
    RB_HANDLE m_AlgoHandle;

    /* 内部参数 */
    RB_CAS_Para_S m_stPara;

    /* 内部结果 */
    RB_CAS_Result_S m_stResults;

};

#endif 
