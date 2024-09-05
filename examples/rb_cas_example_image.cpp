/**
 * @file rb_cas_example_image.cpp
 * @brief 该文件由程序自动生成，图像文件的 demo
 *
 * @author jwzhou (zhou24388@163.com)
 * @version v0.5.0
 * @date 2023-06-07 17:27:55.454835
 *
 * @copyright Copyright (c) 2023 Hefei Source Intelligence Technology Co,.Ltd.
 */

#include <iostream>
#include "rb_cas_example_utils.h"

using namespace std;

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " image_path" << endl;
        return -1;
    }


    cv::Mat src = cv::imread(argv[1]);
    if (src.empty())
    {
        cout << "read image failed" << endl;
        return -1;
    }

    /* 初始化算法句柄 */
    CRbcasExample *example = new CRbcasExample(src);

    /* 配置算法 */
    example->config(src, false);

    /* 算法处理 */
    example->process(src);

    /* 显示结果 */
    example->show_results(0);
  

    /* 删除算法句柄 */
    delete example;


    return 0;
}

