/**
 * @file rb_detector.cpp
 * @brief 
 * 
 * @author jw.zhou (zhou24388@163.com)
 * @version 1.0.0
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023 Hefei Source Intelligence Technology Co,.Ltd.
 * 
 * @par 修改日志:
 * Date     :  2023-05-31
 * version  :  {version}
 * Author   :  jw.zhou
 * Log      :  修改内容
 * copyright:  {copyright}
 */

#include "rb_cas_detector.h"

#define MAX_STRIDE 64

#ifdef WITH_NCNN
/* 初始化模型和各种参数 */
#ifdef LOAD_MEM_MODEL
CYOLOV5_NCNN_Detector::CYOLOV5_NCNN_Detector(const unsigned char *paramem, const unsigned char *modelmem)
{
    // 初始化参数
    __init_para();
    // 通过内存方式初始化模型
    // 加载模型
    __load_para_model_mem(paramem, modelmem);
}

// 通过内存加载模型
int CYOLOV5_NCNN_Detector::__load_para_model_mem(const unsigned char *paramen, const unsigned char *modelmem)
{
    int ret = 0;
    ret |= m_netYOLOv5.load_param(paramen);
    ret |= m_netYOLOv5.load_model(modelmem);
    m_netYOLOv5.opt.use_vulkan_compute = true;

    return ret;
}

#else // LOAD_FILE_MODEL

CYOLOV5_NCNN_Detector::CYOLOV5_NCNN_Detector(const char *parampath, const char *modelpath)
{
    // 初始化参数
    __init_para();
    // 通过参数名称和模型名称初始化模型
    // 加载模型
    __load_para_model_file(parampath, modelpath);
}

int CYOLOV5_NCNN_Detector::__load_para_model_file(const char *parampath, const char *modelpath)
{
    int ret = 0;
    ret |= m_netYOLOv5.load_param(parampath);
    ret |= m_netYOLOv5.load_model(modelpath);
    m_netYOLOv5.opt.use_vulkan_compute = true;

    return ret;
}
#endif // end LOAD_MEM_MODEL

// 初始化网络模型的参数
void CYOLOV5_NCNN_Detector::__init_para()
{
    m_s32Stride8 = 8;
    m_s32Stride16 = 16;
    m_s32Stride32 = 32;

    m_s32TgtSize = TGT_IM_SZ;
    m_f32ProbThr = PROB_THR;
    m_f32NMSThr = NMS_THR;

    m_af32Norms[0] = (1 / 255.f);
    m_af32Norms[1] = (1 / 255.f);
    m_af32Norms[2] = (1 / 255.f);

    m_matAnchors8 = ncnn::Mat(6);
    m_matAnchors8[0] = 10.f;
    m_matAnchors8[1] = 13.f;
    m_matAnchors8[2] = 16.f;
    m_matAnchors8[3] = 30.f;
    m_matAnchors8[4] = 33.f;
    m_matAnchors8[5] = 23.f;

    m_matAnchors16 = ncnn::Mat(6);
    m_matAnchors16[0] = 30.f;
    m_matAnchors16[1] = 61.f;
    m_matAnchors16[2] = 62.f;
    m_matAnchors16[3] = 45.f;
    m_matAnchors16[4] = 59.f;
    m_matAnchors16[5] = 119.f;

    m_matAnchors32 = ncnn::Mat(6);
    m_matAnchors32[0] = 116.f;
    m_matAnchors32[1] = 90.f;
    m_matAnchors32[2] = 156.f;
    m_matAnchors32[3] = 198.f;
    m_matAnchors32[4] = 373.f;
    m_matAnchors32[5] = 326.f;
}

CYOLOV5_NCNN_Detector::~CYOLOV5_NCNN_Detector()
{
    // 释放资源
}

int CYOLOV5_NCNN_Detector::__pre_process(const cv::Mat matIm)
{
    m_matBGRIn = matIm.clone();

    m_s32ImgW = matIm.cols;
    m_s32ImgH = matIm.rows;
    m_f32Scale = 1.f;

    // letterbox pad to multiple of MAX_STRIDE
    int w = m_s32ImgW;
    int h = m_s32ImgH;

    if (w > h)
    {
        m_f32Scale = (float)m_s32TgtSize / w;
        w = m_s32TgtSize;
        h = h * m_f32Scale;
    }
    else
    {
        m_f32Scale = (float)m_s32TgtSize / h;
        h = m_s32TgtSize;
        w = w * m_f32Scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(matIm.data, ncnn::Mat::PIXEL_BGR2RGB, m_s32ImgW, m_s32ImgH, w, h);

    // pad to m_s32TgtSize rectangle
    // yolov5/utils/datasets.py letterbox
    m_s32PadW = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    m_s32PadH = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

    ncnn::copy_make_border(in, in_pad, m_s32PadH / 2, m_s32PadH - m_s32PadH / 2, 
        m_s32PadW / 2, m_s32PadW - m_s32PadW / 2, ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(0, m_af32Norms);

    return 0;
}


int CYOLOV5_NCNN_Detector::process(const cv::Mat matIm, std::vector<RB_DetTarget_S> &objects)
{
    if (matIm.empty())
    {
        std::cout<<"Please confirm the image is OK. "<<std::endl;
        return -1;
    }

    // 前处理
    __pre_process(matIm);

    // 创建模型
    ncnn::Extractor ex = m_netYOLOv5.create_extractor();

    std::vector<RB_DetTarget_S> vecObjs8;
    std::vector<RB_DetTarget_S> vecObjs16;
    std::vector<RB_DetTarget_S> vecObjs32;

    ncnn::Mat out_s8, out_s16, out_s32;

#ifdef LOAD_MEM_MODEL
    ex.input(NCNN_PARAM_ID_SPACE::BLOB_in0, in_pad);
    ex.extract(NCNN_PARAM_ID_SPACE::BLOB_out0, out_s8);
    ex.extract(NCNN_PARAM_ID_SPACE::BLOB_out1, out_s16);
    ex.extract(NCNN_PARAM_ID_SPACE::BLOB_out2, out_s32);
#else
    ex.input("in0", in_pad);
    ex.extract("out0", out_s8);
    ex.extract("out1", out_s16);
    ex.extract("out2", out_s32);
#endif // end of LOAD_MEM_MODEL

    // 清除vector
    m_vecProposals.clear();    
    std::vector<RB_DetTarget_S>().swap(m_vecProposals);    

    // stride 8
    __generate_proposals(m_matAnchors8, m_s32Stride8, in_pad, out_s8, m_f32ProbThr, vecObjs8);
    m_vecProposals.insert(m_vecProposals.end(), vecObjs8.begin(), vecObjs8.end());

    // stride 16
    __generate_proposals(m_matAnchors16, m_s32Stride16, in_pad, out_s16, m_f32ProbThr, vecObjs16);
    m_vecProposals.insert(m_vecProposals.end(), vecObjs16.begin(), vecObjs16.end());

    // stride 32
    __generate_proposals(m_matAnchors32, m_s32Stride32, in_pad, out_s32, m_f32ProbThr, vecObjs32);
    m_vecProposals.insert(m_vecProposals.end(), vecObjs32.begin(), vecObjs32.end());

    // 后处理
    __post_process();

    // 输出
    get_results(objects);

    return 0;
}


void CYOLOV5_NCNN_Detector::__post_process()
{
    m_vecObjs.clear();
    std::vector<RB_DetTarget_S>().swap(m_vecObjs);

    // sort all proposals by score from highest to lowest
    __qsort_descent(m_vecProposals);

    // apply nms with m_f32NMSThr
    std::vector<int> picked;
    __nms_sorted_bboxes(m_vecProposals, picked);

    int count = picked.size();
    // std::cout<<"count:"<<count<<std::endl;
    for (int i = 0; i < count; i++)
    {
        RB_DetTarget_S objTmp = m_vecProposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objTmp.rect.x - (m_s32PadW / 2)) / m_f32Scale;
        float y0 = (objTmp.rect.y - (m_s32PadH / 2)) / m_f32Scale;
        float x1 = (objTmp.rect.x + objTmp.rect.width - (m_s32PadW / 2)) / m_f32Scale;
        float y1 = (objTmp.rect.y + objTmp.rect.height - (m_s32PadH / 2)) / m_f32Scale;

        // clip
        x0 = std::max(std::min(x0, (float)(m_s32ImgW - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(m_s32ImgH - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(m_s32ImgW - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(m_s32ImgH - 1)), 0.f);

        objTmp.rect.x = x0;
        objTmp.rect.y = y0;
        objTmp.rect.width = x1 - x0;
        objTmp.rect.height = y1 - y0;

        m_vecObjs.push_back(objTmp);
    }
}



// NMS 合并目标框
void CYOLOV5_NCNN_Detector::__nms_sorted_bboxes(const std::vector<RB_DetTarget_S> &vecObjs,
    std::vector<int> &picked)
{
    picked.clear();

    const int n = vecObjs.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = vecObjs[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const RB_DetTarget_S &a = vecObjs[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const RB_DetTarget_S &b = vecObjs[picked[j]];
            float inter_area = __intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > m_f32NMSThr)
            {
                keep = 0;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }
}

// 快速排序，将目标按照分数排序
void CYOLOV5_NCNN_Detector::__qsort_descent_inplace(std::vector<RB_DetTarget_S> &vecObjs, int left, int right)
{
    int i = left;
    int j = right;
    float p = vecObjs[(left + right) / 2].f32Prob;

    while (i <= j)
    {
        while (vecObjs[i].f32Prob > p)
            i++;

        while (vecObjs[j].f32Prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(vecObjs[i], vecObjs[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                __qsort_descent_inplace(vecObjs, left, j);
        }
#pragma omp section
        {
            if (i < right)
                __qsort_descent_inplace(vecObjs, i, right);
        }
    }
}

// 生成 proposal
void CYOLOV5_NCNN_Detector::__generate_proposals(const ncnn::Mat &anchors, int stride,
        const ncnn::Mat &in_pad, const ncnn::Mat &feat_blob, float prob_threshold,
        std::vector<RB_DetTarget_S> &objects)
{

    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    RB_DetTarget_S objTmp;
                    objTmp.rect.x = x0;
                    objTmp.rect.y = y0;
                    objTmp.rect.width = x1 - x0;
                    objTmp.rect.height = y1 - y0;
                    objTmp.s32labelID = class_index;
                    objTmp.f32Prob = confidence;

                    objects.push_back(objTmp);
                }
            }
        }
    }
}


void CYOLOV5_NCNN_Detector::get_results(std::vector<RB_DetTarget_S> & objects)
{
    objects.clear();
    std::vector<RB_DetTarget_S>().swap(objects);
    objects.insert(objects.end(), m_vecObjs.begin(), m_vecObjs.end());

}

#ifdef SHOW_DEBUG_IMG
void CYOLOV5_NCNN_Detector::draw_objects()
{
    cv::namedWindow("Detector", cv::WINDOW_NORMAL);
    cv::Mat image = m_matBGRIn.clone();
    char text[256];
    for (int i = 0; i < m_vecObjs.size(); i++)
    {
        RB_DetTarget_S obj = m_vecObjs[i];
        sprintf(text, "%d: %.2f%%", obj.s32labelID, obj.f32Prob * 100);
#ifdef SHOW_DEBUG_INFO
        // printf("%s\n", text);
#endif
        int x1 = obj.rect.x;
        int y1 = obj.rect.y;
        int x2 = x1 + obj.rect.width;
        int y2 = y1 + obj.rect.height;
        rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
        putText(image, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("Detector", image);
    // cv::waitKey(0);
}
#endif // end of SHOW_DEBUG_IMG

#else
#ifdef WITH_RKNN
/**
 * @brief Construct a new cyolov5 npu detector::cyolov5 npu detector object
 *
 */
CYOLOV5_NPU_Detector::CYOLOV5_NPU_Detector()
{
    resize_buf = nullptr;

    // 初始化 stride 和 anchor
    __init_stride_anchor();

    // 初始化模型
    __init_model();
}

/**
 * @brief Destroy the cyolov5 npu detector::cyolov5 npu detector object
 *
 */
CYOLOV5_NPU_Detector::~CYOLOV5_NPU_Detector()
{
    // 释放 rknn
    rknn_destroy(ctx);

    // 释放空间
    if (resize_buf)
    {
        free(resize_buf);
    }

    // 释放模型数据
    if (model_data)
    {
        free(model_data);
    }
}

/**
 * @brief 加载模型和创建网络
 *
 * @return int
 */
int CYOLOV5_NPU_Detector::__init_model()
{
    int ret = 0;
    model_data_size = 0;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

#ifdef SHOW_DEBUG_INFO
    printf("Loading mode...\n");
#endif

    // 根据文件名加载模型文件
    model_data = __load_model(MODEL_NAME, &model_data_size);

    // 初始化 rknn
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        return -1;
    }

#ifdef SHOW_DEBUG_INFO
    // 查询 SDK 信息
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        return -1;
    }
#endif

    // 查询输入输出信息
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {

        return -1;
    }

#ifdef SHOW_DEBUG_INFO
 
#endif

    // 输入信息
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
#ifdef SHOW_DEBUG_INFO
        __dump_tensor_attr(&(input_attrs[i]));
#endif
    }

    // 输出信息
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
#ifdef SHOW_DEBUG_INFO
        __dump_tensor_attr(&(output_attrs[i]));
#endif
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
#ifdef SHOW_DEBUG_INFO

#endif
        m_NetC = input_attrs[0].dims[1];
        m_NetH = input_attrs[0].dims[2];
        m_NetW = input_attrs[0].dims[3];
    }
    else
    {
#ifdef SHOW_DEBUG_INFO

#endif
        m_NetH = input_attrs[0].dims[1];
        m_NetW = input_attrs[0].dims[2];
        m_NetC = input_attrs[0].dims[3];
    }

    int imsz = m_ImgH * m_ImgW * m_ImgC;
    int netinsz = m_NetH * m_NetW * m_NetC;

#ifdef SHOW_DEBUG_INFO
    printf("model input height=%d, width=%d, channel=%d\n", m_NetH, m_NetW, m_NetC);
#endif

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = netinsz;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    resize_buf = malloc(netinsz);
    memset(resize_buf, 0x00, netinsz);

    // 清空数据
    out_scales.clear();
    out_zps.clear();
    std::vector<float>().swap(out_scales);
    std::vector<int32_t>().swap(out_zps);
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    return 0;
}

// TODO:
// TODO:
// int CYOLOV5_NPU_Detector::__pre_process_rga(const cv::Mat matIm)
// {
//     int ret = 0;

//     // 图像预处理, 也可以根据需求将图像转换成网络需要的样子，目前只是做图像颜色空间转换
//     cv::cvtColor(m_matBGRIn, m_matRGBIn, cv::COLOR_BGR2RGB);

//     m_ImgH = m_matBGRIn.rows;
//     m_ImgW = m_matBGRIn.cols;
//     m_ImgC = 3;

//     int netInSz = m_NetH * m_NetW * m_NetC;
//     int imInSz = m_ImgH * m_ImgW * m_ImgC;

//     if (imInSz != netInSz)
//     {
//         rga_buffer_t dst_tmp;
//         im_rect dst_rect_tmp;

//         // resize 后再扩边
//         int newImW, newImH;
//         float ratio = std::min(float(m_NetW) / float(m_ImgW), float(m_NetH) / float(m_ImgH));
//         m_scaleW = m_scaleH = ratio;
//         newImW = int(m_ImgW * ratio);
//         newImH = int(m_ImgH * ratio);

//         void* resize_buf_tmp = malloc(newImW * newImH * m_ImgC * sizeof(char));
//         memset(resize_buf_tmp, 0x00, newImW * newImH * m_ImgC * sizeof(char));
//         src = wrapbuffer_virtualaddr((void *)m_matRGBIn.data, m_ImgW, m_ImgH, RK_FORMAT_RGB_888);
//         dst_tmp = wrapbuffer_virtualaddr((void *)resize_buf_tmp, newImW, newImH, RK_FORMAT_RGB_888);
//         dst = wrapbuffer_virtualaddr((void *)resize_buf, m_NetW, m_NetH, RK_FORMAT_RGB_888);

//         ret = imcheck(src, dst_tmp, src_rect, dst_rect_tmp);
//         if (IM_STATUS_NOERROR != ret)
//         {
//             printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
//             return -1;
//         }

//         // 缩放图像
//         IM_STATUS STATUS = imresize(src, dst_tmp);

//         // 扩边
//         m_bottomPadd = m_NetH - newImH;
//         m_rightPadd = m_NetW - newImW;
//         m_topPadd = m_bottomPadd>>1;
//         m_leftPadd = m_rightPadd>>1;
//         m_bottomPadd -= m_topPadd;
//         m_rightPadd -= m_leftPadd;

//         ret = imcheck(dst_tmp, dst, dst_rect_tmp, dst_rect);
//         if (IM_STATUS_NOERROR != ret)
//         {
//             printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
//             return -1;
//         }

//         ret = immakeBorder(dst_tmp, dst, m_topPadd, m_bottomPadd, m_leftPadd, m_rightPadd, IM_BORDER_CONSTANT);
//         if (ret != IM_STATUS_SUCCESS)
//         {
//             printf("%d, border error! %s", __LINE__, imStrError((IM_STATUS)ret));
//             return -1;
//         }

//         // 释放临时空间
//         if (resize_buf_tmp)
//         {
//             free(resize_buf_tmp);
//         }

//         inputs[0].buf = resize_buf;
//     }
//     else
//     {
//         inputs[0].buf = (void *)m_matRGBIn.data;
//     }

//     return ret;
// }

/**
 * @brief 对输入图像进行预处理
 * TODO:
 * 1. 图像 padding
 * 2. 图像缩放
 * 3. 图像颜色空间转换
 *
 * @param  matIm            My Param doc
 */
int CYOLOV5_NPU_Detector::__pre_process(const cv::Mat matIm)
{
    m_matBGRIn = matIm.clone();
    m_ImgH = m_matBGRIn.rows;
    m_ImgW = m_matBGRIn.cols;
    m_ImgC = 3;

    int netInSz = m_NetH * m_NetW * m_NetC;
    int imInSz = m_ImgH * m_ImgW * m_ImgC;

    int ret = 0;

    /* 显示时间 */
#ifdef SHOW_DEBUG_TIME
    double dtStart = __get_current_time_proc();
#endif // SHOW_DEBUG_TIME

        // 图像预处理, 也可以根据需求将图像转换成网络需要的样子，目前只是做图像颜色空间转换
        cv::cvtColor(m_matBGRIn, m_matRGBIn, cv::COLOR_BGR2RGB);

//     /* 显示时间 */
#ifdef SHOW_PROC_TIME
        double dtEnd = __get_current_time_proc();
        std::cout << "++++++ in detector  __pre_process Image Time Span ++++++ " << (dtEnd - dtStart) << " ms" << std::endl;
        dtStart = __get_current_time_proc();
#endif // SHOW_PROC_TIME

    std::cout << "hello" << std::endl;

    if (imInSz != netInSz)
    {

        src = wrapbuffer_virtualaddr((void *)m_matRGBIn.data, m_ImgW, m_ImgH, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void *)resize_buf, m_NetW, m_NetH, RK_FORMAT_RGB_888);

   
        ret = imcheck(src, dst, src_rect, dst_rect);

  
        if (IM_STATUS_NOERROR != ret)
        {
    
            return -1;
        }


            
        IM_STATUS STATUS = imresize(src, dst);
  
        m_scaleW = (float)m_NetW / (float)m_ImgW;
        m_scaleH = (float)m_NetH / (float)m_ImgH;
        m_rightPadd = 0;
        m_leftPadd = 0;
        m_topPadd = 0;
        m_bottomPadd = 0;

    

        // cv::Mat resize_img(cv::Size(m_NetW, m_NetH), CV_8UC3, resize_buf);
        inputs[0].buf = resize_buf;

        //         cv::Mat matPadResizedIm;   // 缩放后扩边的图像
        // // #define USE_PADDING
        // #ifdef USE_PADDING
        //         cv::Mat matUnPadResizedIm; // 没有扩边的缩放图像
        //         int newImW, newImH;
        //         // 先缩放再扩边, newshape=(netw, neth), oldshape=(imgw, imgh)
        //         float ratio = std::min(float(m_NetW)/float(m_ImgW), float(m_NetH)/float(m_ImgH));
        //         m_scaleW = m_scaleH = ratio;
        //         newImW = int(m_ImgW * ratio);
        //         newImH = int(m_ImgH * ratio);
        //         cv::resize(m_matBGRIn, matUnPadResizedIm, cv::Size(newImW, newImH));

        //         m_bottomPadd = m_NetH - newImH;
        //         m_rightPadd = m_NetW - newImW;
        //         m_topPadd = m_bottomPadd>>1;
        //         m_leftPadd = m_rightPadd>>1;
        //         m_bottomPadd -= m_topPadd;
        //         m_rightPadd -= m_leftPadd;

        // 图像扩边
        //         cv::copyMakeBorder(matUnPadResizedIm, matPadResizedIm, m_topPadd, m_bottomPadd, m_leftPadd,
        //                         m_rightPadd, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        // #else
        //         cv::resize(m_matBGRIn, matPadResizedIm, cv::Size(m_NetW, m_NetH));
        //         m_scaleW = (float)m_NetW / (float)m_ImgW;
        //         m_scaleH = (float)m_NetH / (float)m_ImgH;
        //         m_rightPadd = 0;
        //         m_leftPadd = 0;
        //         m_topPadd = 0;
        //         m_bottomPadd = 0;
        // #endif

        // cv::imwrite("unpad.jpg", matPadResizedIm);

        // 颜色空间转换
        // cv::cvtColor(matPadResizedIm, m_matRGBIn, cv::COLOR_BGR2RGB);
    }
    else
    {
        // m_scaleW = m_scaleH = 1.f;
        // m_topPadd = 0;
        // m_rightPadd = 0;
        // m_bottomPadd = 0;
        // m_leftPadd = 0;

        // 颜色空间转换
        // cv::cvtColor(m_matBGRIn, m_matRGBIn, cv::COLOR_BGR2RGB);
    }

    // 赋值
    // inputs[0].buf = (void *)m_matRGBIn.data;

    return 0;
}

/**
 * @brief
 *
 * @param  matIm            My Param doc
 * @param  objects          My Param doc
 * @return int
 */
int CYOLOV5_NPU_Detector::process2(const cv::Mat matIm, std::vector<RB_DetTarget_S> &objects)
{
    int ret = 0;
    if (matIm.empty())
    {
        printf("read image failure!\n");
        return -1;
    }

/* 显示时间 */
#ifdef SHOW_PROC_TIME
    double dtStart = __get_current_time_proc();
#endif /* SHOW_PROC_TIME */

    // 图像预处理, 也可以根据需求将图像转换成网络需要的样子，目前只是做图像颜色空间转换
    __pre_process(matIm);
    // __pre_process_rga(matIm);

    /* 显示时间 */
#ifdef SHOW_PROC_TIME
    double dtEnd = __get_current_time_proc();
    std::cout << "++++++ in detector  __pre_process Image Time Span ++++++ " << (dtEnd - dtStart) << " ms" << std::endl;
    dtStart = __get_current_time_proc();
#endif // SHOW_PROC_TIME

    rknn_inputs_set(ctx, io_num.n_input, inputs);
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);


//=========================================================


#ifdef SHOW_PROC_TIME
    dtEnd = __get_current_time_proc();
    std::cout << "++++++ in detector  process net Time Span ++++++ " << (dtEnd - dtStart) << " ms" << std::endl;
    dtStart = __get_current_time_proc();
#endif // SHOW_PROC_TIME

    // 后处理
    __post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf);
    get_results(objects);
std::cout<<"===================== get_results==========="<<objects.size()<<std::endl;
#ifdef SHOW_PROC_TIME
    dtEnd = __get_current_time_proc();
    std::cout << "++++++ in detector  __post_process net Time Span ++++++ " << (dtEnd - dtStart) << " ms" << std::endl;
#endif // SHOW_PROC_TIME

    // 释放资源
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    return 0;
}

/**
 * @brief 获得目标集合
 *
 * @param  objects          My Param doc
 * @return int
 */
int CYOLOV5_NPU_Detector::get_results(std::vector<RB_DetTarget_S> &objects)
{
    objects.clear();
    std::vector<RB_DetTarget_S>().swap(objects);
    objects.insert(objects.end(), m_vecObjs.begin(), m_vecObjs.end());
    return 0;
}

#ifdef SHOW_DEBUG_IMG
/**
 * @brief 显示结果
 *
 * @param  bgr              My Param doc
 */
void CYOLOV5_NPU_Detector::draw_objects()
{
    // cv::namedWindow("SHOW_Result", cv::WINDOW_NORMAL);
    cv::Mat image = m_matBGRIn.clone();
    char text[256];
    for (int i = 0; i < m_vecObjs.size(); i++)
    {
        RB_DetTarget_S obj = m_vecObjs[i];
        sprintf(text, "%d: %.2f%%", obj.s32labelID, obj.f32Prob * 100);
#ifdef SHOW_DEBUG_INFO
        printf("%s\n", text);
#endif
        int x1 = obj.rect.x;
        int y1 = obj.rect.y;
        int x2 = x1 + obj.rect.width;
        int y2 = y1 + obj.rect.height;
        rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
        putText(image, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    // cv::imshow("SHOW_Result", image);
    // cv::waitKey(0);
}
#endif

#ifdef SHOW_DEBUG_INFO
void CYOLOV5_NPU_Detector::__dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims,
           attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
#endif

unsigned char *CYOLOV5_NPU_Detector::__load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

unsigned char *CYOLOV5_NPU_Detector::__load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {

        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = __load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

void CYOLOV5_NPU_Detector::__nms_sorted_bboxes(const std::vector<RB_DetTarget_S> &vecObjs, std::vector<int> &picked)
{
    picked.clear();

    const int n = vecObjs.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = vecObjs[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const RB_DetTarget_S &a = vecObjs[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const RB_DetTarget_S &b = vecObjs[picked[j]];
            float inter_area = __intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > NMS_THR)
            {
                keep = 0;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }
}

void CYOLOV5_NPU_Detector::__qsort_descent_inplace(std::vector<RB_DetTarget_S> &vecObjs, int left, int right)
{
    int i = left;
    int j = right;
    float p = vecObjs[(left + right) / 2].f32Prob;

    while (i <= j)
    {
        while (vecObjs[i].f32Prob > p)
            i++;

        while (vecObjs[j].f32Prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(vecObjs[i], vecObjs[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                __qsort_descent_inplace(vecObjs, left, j);
        }
#pragma omp section
        {
            if (i < right)
                __qsort_descent_inplace(vecObjs, i, right);
        }
    }
}

void CYOLOV5_NPU_Detector::__generate_proposals(int8_t *input, int *anchor, int stride, int32_t zp,
                                                float scale, std::vector<RB_DetTarget_S> &vecObjs)
{
    int grid_h = m_NetH / stride;
    int grid_w = m_NetW / stride;



    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = __qnt_f32_to_affine(__unsigmoid(PROB_THR), zp, scale);

    RB_DetTarget_S stObjtmp;
    cv::Rect_<float> recttmp;

    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                int8_t *in_ptr = input + offset;
                int8_t box_score = in_ptr[4 * grid_len];
                float box_conft = __sigmoid(__deqnt_affine_to_f32(box_score, zp, scale));

                if (box_score < thres_i8)
                {
                    continue;
                }

                int8_t maxClassProbs = in_ptr[5 * grid_len];
                int maxClassId = 0;
                for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                {
                    int8_t prob = in_ptr[(5 + k) * grid_len];
                    if (prob > maxClassProbs)
                    {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }

                float class_conf = __sigmoid(__deqnt_affine_to_f32(maxClassProbs, zp, scale));
                float box_conf = __sigmoid(__deqnt_affine_to_f32(box_score, zp, scale));
                float prob__ = class_conf * box_conf;


                if (prob__ > PROB_THR)
                {
                    float dx = __sigmoid(__deqnt_affine_to_f32(*in_ptr, zp, scale));
                    float dy = __sigmoid(__deqnt_affine_to_f32(in_ptr[grid_len], zp, scale));
                    float dw = __sigmoid(__deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale));
                    float dh = __sigmoid(__deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale));

                    float box_x = (dx * 2.f - 0.5f + j) * (float)stride;
                    float box_y = (dy * 2.f - 0.5f + i) * (float)stride;

                    float box_w = (dw * 2.f) * (dw * 2.f) * (float)anchor[a * 2];
                    float box_h = (dh * 2.f) * (dh * 2.f) * (float)anchor[a * 2 + 1];

                    box_x = box_x - box_w * 0.5f;
                    box_y = box_y - box_h * 0.5f;

                    recttmp.x = box_x;
                    recttmp.y = box_y;
                    recttmp.width = box_w;
                    recttmp.height = box_h;

                    stObjtmp.s32labelID = maxClassId;
                    stObjtmp.rect = recttmp;
                    stObjtmp.f32Prob = prob__;
                    vecObjs.push_back(stObjtmp);

                }
            }
        }
    }
}

void CYOLOV5_NPU_Detector::__post_process(int8_t *input0, int8_t *input1, int8_t *input2)
{
    // 不同 stride 下的目标集合
    std::vector<RB_DetTarget_S> vecPropS8;
    std::vector<RB_DetTarget_S> vecPropS16;
    std::vector<RB_DetTarget_S> vecPropS32;
    std::vector<RB_DetTarget_S> vecProps;

    // std::cout<<"input0 "<<sizeof(input0)/sizeof(input0[0])<<std::endl;

    // 根据不同的 stride 产生不同的 proposal
    __generate_proposals(input0, (int *)m_anchor8, m_stride8, out_zps[0], out_scales[0], vecPropS8);
    vecProps.insert(vecProps.end(), vecPropS8.begin(), vecPropS8.end());

    __generate_proposals(input1, (int *)m_anchor16, m_stride16, out_zps[1], out_scales[1], vecPropS16);
    vecProps.insert(vecProps.end(), vecPropS16.begin(), vecPropS16.end());

    __generate_proposals(input2, (int *)m_anchor32, m_stride32, out_zps[2], out_scales[2], vecPropS32);
    vecProps.insert(vecProps.end(), vecPropS32.begin(), vecPropS32.end());

    // 按照 prob 的大小进行降序排序
    __qsort_descent(vecProps);

    // 执行 nms
    std::vector<int> vecPicked;
    __nms_sorted_bboxes(vecProps, vecPicked);

    // 清空结果，赋值出去
    m_vecObjs.clear();
    std::vector<RB_DetTarget_S>().swap(m_vecObjs);
    int count = vecPicked.size();
    for (int i = 0; i < count; i++)
    {
        RB_DetTarget_S objTmp = vecProps[vecPicked[i]];
        float x1 = objTmp.rect.x;
        float y1 = objTmp.rect.y;
        float x2 = objTmp.rect.x + objTmp.rect.width - 1;
        float y2 = objTmp.rect.y + objTmp.rect.height - 1;
        int id = objTmp.s32labelID;
        float prob = objTmp.f32Prob;

        objTmp.rect.x = (int)((__clamp(x1, 0, m_NetW) - m_leftPadd) / m_scaleW);
        objTmp.rect.y = (int)((__clamp(y1, 0, m_NetH) - m_topPadd) / m_scaleH);
        objTmp.rect.width = (int)((__clamp(x2, 0, m_NetW) - m_leftPadd) / m_scaleW) - objTmp.rect.x + 1;
        objTmp.rect.height = (int)((__clamp(y2, 0, m_NetH) - m_topPadd) / m_scaleH) - objTmp.rect.y + 1;
        objTmp.s32labelID = id;
        objTmp.f32Prob = prob;

        m_vecObjs.push_back(objTmp);
    }
}

#endif // end of #ifdef WITH_RKNN

#endif  // end of else
