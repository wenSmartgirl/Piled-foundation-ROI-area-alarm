
#include <iostream>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/centroid.h>
#include "depthai/depthai.hpp"
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "ntkgf.h"
#include "rb_cah_example_utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include "rb_cah_detector.h"

using namespace std;
constexpr auto FPS = 20.0;
bool SPARSE=false;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
const std::vector<std::string> kClassNames = {
    "person",
    "lifting"//
};

struct GroundCircle {
    float cx;
    float cz;
    float r;
    int label;
    float prob;
    cv::Rect rect; // 目标在图像中的二维边界框
};

// 点云数据->映射
static GroundCircle make_circle(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, 
                                int label, float prob, const cv::Rect &rect,
                                float default_radius, bool use_cloud_radius = false, // 是否使用点云半径
                                float max_radius = 1.5f) // max_radius 最大半径
{
    GroundCircle circle;
    circle.label = label;
    circle.prob = prob;
    circle.rect = rect;
    if (!cloud || cloud->points.empty()) {
        circle.cx = circle.cz =0.0f;
        circle.r = default_radius;
        return circle;
    } 

    // 质心
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    circle.cx = centroid[0]/1000.0f; // 转换为米
    circle.cz = centroid[2]/1000.0f;

    if (!use_cloud_radius) {
        circle.r = default_radius;
        return circle;
    }

    // 计算点云真实的半径
    float max_r = 0.0f;
    for(const auto &p : cloud->points) {
        float dx = p.x/1000.0f - circle.cx;
        float dz = p.z/1000.0f - circle.cz;
        float dist = std::sqrt(dx*dx + dz*dz);
        if (dist > max_r) {
            max_r = dist;
        }
    }

    // 限制半径范围
    circle.r = std::min(max_r, max_radius);
    if (circle.r < 0.1f) {
        circle.r = default_radius; // 如果半径太小，使用默认半径
    }
    return circle;
}

// 指针指向点云数据  / / 体素大小  / / 体素网格滤波器阈值  / / 体素网格滤波器最大点数
void Process_PointCloud(pcl::PointCloud<PointT>::Ptr& cloud, PointT &min_point, PointT &max_point,
                        float voxel_size,int threshold, int max_points)
{
    // 计算点云的边界
    pcl::getMinMax3D(*cloud, min_point, max_point);

    //应用VoxelGrip进行下采样 减少数据量
    pcl::VoxelGrid<PointT> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_grid_filter.setInputCloud(cloud);
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
    voxel_grid_filter.filter(*cloud_filtered);

    // 法线估计
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud_filtered);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    normal_estimation.compute(*cloud_normals);
}



int main(int argc, char* argv[])
{
    CYOLOV5_NPU_Detector_CAH detector;
    //load model
    // if (detector.__init_model() < 0) {
    //     std::cerr << "model init fail" << std::endl;
    //     return -1;
    // }

     //create
    auto pipeline = dai::Pipeline();
    auto camRgb     = pipeline.create<dai::node::ColorCamera>();
    auto monoLeft   = pipeline.create<dai::node::ColorCamera>();
    auto monoRight  = pipeline.create<dai::node::ColorCamera>();
    auto depth      = pipeline.create<dai::node::StereoDepth>();
    auto pointCloud = pipeline.create<dai::node::PointCloud>();

    // 创建同步节点和输出节点
    auto sync = pipeline.create<dai::node::Sync>();
    auto xOut = pipeline.create<dai::node::XLinkOut>();
    xOut->input.setBlocking(false);

    //Properties
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1200_P);
    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    std::tuple<int,int> isp_scale_center(1,3);
    camRgb->setIspScale(isp_scale_center);
    camRgb->setFps(FPS);

    monoLeft->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1200_P);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_B);
    // monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_C);
    std::tuple<int,int> isp_scale_Left(1,3);
    monoLeft->setIspScale(isp_scale_Left);
    monoLeft->setFps(FPS);


    monoRight->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1200_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_C);
    // monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_B);
    std::tuple<int,int> isp_scale_Right(1,3);
    monoRight->setIspScale(isp_scale_Right);
    monoRight->setFps(FPS);


    depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DETAIL);
    depth->setDepthAlign(dai::CameraBoardSocket::CAM_A);
    int w = 640;
    int h = 400;
    depth->setOutputSize(w,h);
    // depth->setPreviewSize(1280, 800);  //w=1280，h=800
    // 

    //set
    pointCloud->initialConfig.setSparse(SPARSE);

    //link
    monoLeft->isp.link(depth->left);
    monoRight->isp.link(depth->right);
    depth->depth.link(pointCloud->inputDepth);
    camRgb->isp.link(sync->inputs["rgb"]);
    pointCloud->outputPointCloud.link(sync->inputs["pdata"]);
    sync->out.link(xOut->input);
    xOut->setStreamName("out");

    // device
    dai::Device device(pipeline);
    auto q = device.getOutputQueue("out", 4, false);

    // bool first = true;
    /* 初始化算法句柄 */

    //cv::Mat m_matImg = cv::imread("1.jpg");
    cv::Mat src=cv::Mat::zeros(cv::Size(w,h),CV_8UC3);


    CRbcahExample *example = new CRbcahExample(src);

    /* 配置算法 */
    example->config(src, false);
    RB_CAH_Result_S m_stResults;


    cv::namedWindow("color",cv::WINDOW_NORMAL);

    while(true) {
        auto inMessage = q->get<dai::MessageGroup>();//取出信息
        auto inColor = inMessage->get<dai::ImgFrame>("rgb");
        auto colorFrame = inColor->getCvFrame(); // 把depthai的图像帧转换为cv::mat

        float HEIGHT_THRE = 0.3f;

        // 图像预处理
        cv::Mat resized_image;
        cv::resize(colorFrame, resized_image, cv::Size(640, 640));
        cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);//BGR->RGB
        
        //执行推理
        // std::vector<RB_DetTarget_S> dets;
        // if (detector.process2(resized_image) < 0) {
        //     std::cerr << "推理失败" << std::endl;
        //     continue;
        //     // return -1;
        // }


        //获取点云
        auto inPointCloud=inMessage->get<dai::PointCloudData>("pdata");
        auto points= inPointCloud->getPoints();
        auto width=inPointCloud->getWidth();    
        auto height=inPointCloud->getHeight();  


        cv::Mat redst;
        cv::resize(colorFrame, redst, cv::Size(width, height));// 彩色图 colorFrame 按指定尺寸缩放到 redst

        /* 算法处理 */
        example->process(redst);

        // 生成地面的圆
        std::vector<GroundCircle> persons,liftings;
        float PERSON_RADIUS = 0.3f; // 人的默认半径
        float LIFTING_RADIUS = 0.5f; // 物体的默认
        bool USE_CLOUD_RADIUS = false; // 人是否使用点云半径

        std::vector<PointCloud::Ptr> vcloud;
        std::vector<int> vlabel;
        std::vector<float> vprob;
        for (int target=0;target<example->m_stResults.vecPedSet.size();target++)
        {

            RB_CAH_Ped_S stTmp = example->m_stResults.vecPedSet[target];
            PointCloud::Ptr cloud (new PointCloud);
            vlabel.push_back(stTmp.s32labelID);
            vprob.push_back(stTmp.f32Prob);
            for (int y_Row = 0; y_Row < height; y_Row++)
            {
                for (int x_Col = 0; x_Col < width; x_Col++)
                {
                    // 二维坐标转换为一维数组
                    int pose_ = y_Row * width + x_Col;
                    
                    int u0,v0,u1,v1;
                    u0=stTmp.stPedBoundingBox.stTopLeft.s32X;
                    v0=stTmp.stPedBoundingBox.stTopLeft.s32Y;
                    u1=stTmp.stPedBoundingBox.stBottomRight.s32X;
                    v1=stTmp.stPedBoundingBox.stBottomRight.s32Y;
                    if(x_Col>u0 && x_Col<u1 && y_Row>v0 && y_Row<v1)
                    {
                        PointT p;
                        p.x = -points[pose_].x;
                        p.y = points[pose_].y;
                        p.z = points[pose_].z;
                        // 生成点云
                        cv::Vec3b pixel = redst.at<cv::Vec3b>(y_Row, x_Col);
                        p.b = p.g =p.r = 255;
                        cloud->points.push_back(p);
                    }
                }
            }
            cloud->height = 1;
            cloud->width = cloud->points.size();
            cloud->is_dense = true; // 没有无效点
            vcloud.push_back(cloud);
        }

        for(int target=0;target<example->m_stResults.vecPedSet.size();target++)
        {
            RB_CAH_Ped_S st = example->m_stResults.vecPedSet[target];
            if(vcloud[target]->points.empty()) continue;
            
            // 2D
            cv::Rect r;
            r.x = st.stPedBoundingBox.stTopLeft.s32X;
            r.y = st.stPedBoundingBox.stTopLeft.s32Y;
            r.width = st.stPedBoundingBox.stBottomRight.s32X - r.x;
            r.height = st.stPedBoundingBox.stBottomRight.s32Y - r.y;

            float def_radius = (st.s32labelID==0)? PERSON_RADIUS : LIFTING_RADIUS;
            bool usefp = (st.s32labelID==0)? USE_CLOUD_RADIUS : false; // 人的半径
            
            GroundCircle g = make_circle(vcloud[target],st.s32labelID,st.f32Prob,r,def_radius,usefp);
            
            if(st.s32labelID == 1 ) liftings.push_back(g);
            else persons.push_back(g);

            cv::rectangle(redst,r,cv::Scalar(0,255,0),2);
            std::string label =kClassNames[st.s32labelID]+" "+cv::format("%.2f",st.f32Prob);
            cv::putText(redst,label,cv::Point(r.x,r.y-5),
                                    cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),1);

            
        }
        // 相交 报警
        bool danger = false;
        for(const auto&p : persons)
        {
            for(const auto&h : liftings)
            {
                float dx = p.cx - h.cx;
                float dz = p.cz - h.cz;
                float dist = sqrt(dx*dx + dz*dz);
                if(dist < (p.r + h.r))
                {
                    danger = true;
                    std::cout<<"警告！目标危险！"<<std::endl;
                    std::cout<<"目标距离危险物"<< dist <<"m"<<std::endl;
                    // 绘制相交
                    cv::rectangle(redst,p.rect,cv::Scalar(0,0,255),3);
                    cv::rectangle(redst,h.rect,cv::Scalar(0,0,255),3);
                }
            }
        }

        // 显示报警
        if(danger)
        {
            cv::putText(redst,"Danger",cv::Point(10,100),cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,0,255),2);
        }



    cv::imshow("color",redst);
    int key = cv::waitKey(1);
    if(key == 'q' || key == 'Q') {
        return 0;
    }
       

    }

  

    /* 删除算法句柄 */
    delete example;


    return 0;
}

