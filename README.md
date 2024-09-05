## 项目结构 （Architecture）

```bash
.
├── assets           # 存放测试图片或者视频以及模型文件
│   ├── images       # 测试图片
│   │   └── test.png 
│   └── models       # 模型文件
│       ├── cas.rknn
├── CMakeLists.txt   # 根目录的 CMakeLists.txt
├── configs          # 配置的文件，全局参数配置文件，会将编译参数设置进去
│   ├── rb_cas_gconfig.h.in
│   └── rb_cas_signature.cpp.in
├── examples         # 示例文件
│   ├── CMakeLists.txt
│   ├── cvui.h
│   ├── rb_cas_example_image.cpp
│   ├── rb_cas_example_utils.cpp
│   ├── rb_cas_example_utils.h
│   └── rb_cas_example_video.cpp
├── LICENSE          # 
├── README.md        # readme
├── src              # 源代码
│   ├── CMakeLists.txt
│   ├── interface    # 接口文件
│   │   ├── rb_common.h
│   │   ├── rb_cas_api.h
│   │   └── rb_cas_para.h
│   └── module       # 模块代码
│       ├── include  # 头文件
│       │   ├── rb_cas_detector.h
│       │   └── rb_cas_utils.h
│       └── source   # 源文件
│           ├── rb_cas_detector.cpp
│           ├── rb_cas_api.cpp
│           └── rb_cas_utils.cpp
├── toolchains      # 编译工具链选项
│   └── host.toolchain.cmake
├── ubuntu_build.sh # ubuntu上开发的
```


## 概述 ( Summary)


### **主要功能 (Main Function)**


### **参数配置**


### **适用范围 (Scope of Application)**


### **代码规范 (Code Specification)**


### **核心技术 (Core Technology )**


## 安装、编译与测试 (Install/Compile/Test)


### 关于依赖


## 参考 (References)


