## **一. 人脸关键点检测中输入图片的size设置为什么要强行设置为64的n倍？**

//图像处理核函数入口 tivxTargetKernelExecute 在文件 /home/tda4/ti-processor-sdk-rtos-j721e-evm-08_02_00_05/tiovx/source/framework/vx_target_kernel.c中

knl->process_func（）该函数会跳转到tivxKernelMscScaleProcess函数

//疑似主要图像处理数据入口yuv or rgb 在文件/home/tda4/ti-processor-sdk-rtos-j721e-evm-08_02_00_05/tiovx/kernels_j7/hwa/vpac_msc/vx_vpac_msc_multi_scale_output_target_sim.c 中

tivxKernelMscScaleProcess（xxx）

//图像数据结构体

tivx_obj_desc_image_t

//图像数据类型转换buf函数 其是将原始数据由8位左移4位，变成12位。其调用缘由是ti c-model只支持12位数据操作

lse_reformat_in（） /home/tda4/ti-processor-sdk-rtos-j721e-evm-08_02_00_05/tiovx/kernels_j7/hwa/arm/vx_kernels_hwa_target_utils.c 文件中

//在tivx_obj_desc_t这个结构体中，表明了输入数据要强制性做64字节对齐

typedef struct _tivx_obj_desc_t {

    /*! \brief ID of object in shared memory */
    volatile uint16_t obj_desc_id;
    
    /*! \brief Type of object descritor, see \ref tivx_obj_desc_type_e */
    volatile uint16_t type;
    
    /*! \brief object descriptor ID of the scope in which this object is created
     *         For element of pyramid and object array, this is obj_desc_id
     *         of excompassing pyramid, object array
     */
    volatile uint16_t scope_obj_desc_id;
    
    /*! \brief number of input nodes that have consumed this obj_desc, used in pipelining mode only for data references */
    volatile uint16_t in_node_done_cnt;
    
    /*! \brief Host reference, accessible only on HOST side */
    volatile uint64_t host_ref;
    
    /*! \brief reference flags */
    volatile uint32_t flags;
    
    /*! \brief holds the index ID in the case that this is an element within a pyramid or object array */
    volatile uint32_t element_idx;
    
    /*! \brief reference timestamp */
    volatile uint64_t timestamp;
    
    /*! \brief holds the CPU ID of the OpenVX host for this object descriptor */
    volatile uint32_t host_cpu_id;
    
    /*! \brief reserved to make 64b aligned */
    volatile uint32_t rsv0;
    
    /*! \brief holds the index of the IPC port on OpenVX host for this object descriptor for each remote CPU
     *
     *         A connection to each remote CPU could be on a different host_port_id, hence this needs
     *         to hold host_port_id for each remote CPU, this array is indexed by TIOVX CPU ID.
     *         TIVX_OBJ_DESC_MAX_HOST_PORT_ID_CPU MUST be <= TIVX_CPU_ID_MAX
     * */
    volatile uint16_t host_port_id[TIVX_OBJ_DESC_MAX_HOST_PORT_ID_CPU];

} tivx_obj_desc_t;

//而在vxGetObjectArrayItem这个函数中，会进行vx_image的图片数据指针生成。在这个函数内部，调用了ownReferenceGetHandleFromObjDescId函数，在这个函数中，则采用了tivx_obj_desc_t的结构来处理输入数据。因此最终导致输入数据的对齐方式被强制改为64b对齐的方式。

vx_reference VX_API_CALL vxGetObjectArrayItem(vx_object_array objarr, vx_uint32 index)

copyScalerInput-》vxMapImagePatch能看到图片stride的具体信息，在tivx_obj_desc_image_t这个结构体中。

//推理tidl的接口函数为TIDL_process，函数存在于/home/tda4/ti-processor-sdk-rtos-j721e-evm-08_02_00_05/tidl_j721e_08_02_00_11/ti_dl/algo/src/tidl_alg.c



## 二.量化记录

量化核心函数入口/home/tda4/ti-processor-sdk-rtos-j721e-evm-08_02_00_05/tidl_j721e_08_02_00_11/ti_dl/utils/tidlModelImport/tidl_import_main.cpp

TIDL_import_backend（）

其是做完了网络梳理，去除冗余网络之后的一个网络输入

内部转入TIDL_import_quantize（）接口

然后统计各个网络层中的参数最大最小值。TIDL_findMaxQuantizationScale（）

真实量化操作在TIDL_importQuantLayerParams函数入口

=========================================================

TIDL_QuantizeSignedMax（）bias量化接口

TIDL_GetMaxQuantScale（）获取量化因子

      if (pData > 0)
      {
        param = (pData *  quantPrec + QUAN_STYLE2_ROUND);
      }
      else
      {
        param = (pData *  quantPrec - QUAN_STYLE2_ROUND);
      }

参数进行量化，最后输出保存

============================================================

TIDL_QuantPerChannelWeight（） || TIDL_QuantizeSignedMax（） weight 量化接口

TIDL_GetMaxQuantScale（）获取量化因子，进行量化，同上

=========================================================

核心求量化因子的函数TIDL_GetMaxQuantScale 内容如下：

  float absRange = (fabs(max) > fabs(min)) ? fabs(max) : fabs(min);

  /* If absolute range is below minimum and treat it as zero */
  if ( absRange < TIDL_MINIMUM_QUANTIZATION_RANGE)
  {
    absRange = 0;
  }
  else if (gParams.quantizationStyle == TIDL_QuantStyleP2Dynamic)
  {
    absRange = (float)ceil(log((double)absRange) / log((double)2));
    absRange = pow(2.0, (double)absRange);
  }
  else
  {

  }
  float maxWeightsScalePossible = -1;
  if (absRange != 0)
  {
    maxWeightsScalePossible = ((1.0*(1 << (weightBits - 1))) / absRange);
  }
  return (maxWeightsScalePossible);

这里可以看到，如果不做pow2，那么如果量化因子本身是小于1时，会导致计算出来的maxweights比原始移位的值更大。

最后计算出来的值，限制在8bit数据范围内。

超过的，直接按-127~127来取值



## 三.tidl yolo结果叠加解析

/home/tda4/ti-processor-sdk-rtos-j721e-evm-08_02_00_05/tidl_j721e_08_02_00_11/ti_dl/test/src/tidl_image_postproc.c 处理函数文件在这个c文件

int32_t tidl_tb_postProc(int32_t width, int32_t height, int32_t n, int32_t frameCount, void * fPtr, int32_t elementType)这个函数



## 四.编译选项文件说明

**tiovx/psdkra_tools_path.mak** - 给出其余编译文件对应的路径缩写定义

比如：

```
PSDK_PATH ?= $(abspath ..)

CUSTOM_KERNEL_PATH ?= $(TIOVX_PATH)/kernels_j7

VXLIB_PATH ?= $(PSDK_PATH)/vxlib_c66x_1_1_5_0
J7_C_MODELS_PATH ?= $(PSDK_PATH)/j7_c_models
TIDL_PATH ?= $(PSDK_PATH)/tidl_j7_01_03_00_11/ti_dl
……
```

**tiovx/build_flags.mak** - 给出编译是pc emulation 还是device 方式，debug还是release 方式。核心就改这几处。



## 五.修改kernel中的代码输出在tda4板子上起效的步骤

以/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/kernels/img_proc/c66/vx_image_preprocessing_target.c为例，我们需要在异常情况下增加打印输出。

    if(status!=TIADALG_PROCESS_SUCCESS)
    {
        VX_PRINT(VX_ZONE_ERROR, "blk_width = %d\n", blk_width);
        VX_PRINT(VX_ZONE_ERROR, "blk_height = %d\n", blk_height);
        VX_PRINT(VX_ZONE_ERROR, "blk_stride = %d\n", blk_stride);
        VX_PRINT(VX_ZONE_ERROR, "data_type = %d\n", data_type);
        VX_PRINT(VX_ZONE_ERROR, "prms->nodeParams.color_conv_flag = %d\n", prms->nodeParams.color_conv_flag);
        VX_PRINT(VX_ZONE_ERROR, "prms->nodeParams.scale_val0 = %f\n", prms->nodeParams.scale_val[0]);
        VX_PRINT(VX_ZONE_ERROR, "prms->nodeParams.mean_pixel0 = %f\n", prms->nodeParams.mean_pixel[0]);
    
        VX_PRINT(VX_ZONE_ERROR, "[IMG_PROC] tiadalg failed !!!\n");
        status = VX_FAILURE;
    }

那么首先需要编译lib库本身

执行：make vx_target_kernels_img_proc_c66

生成了vx_target_kernels_img_proc_c66.lib后 由于lib也是提供给其他程序用于编译，此处这个库是作用于c6x_1芯片。因此再执行

make vision_apps

让关联该库文件的可执行文件重新链接该lib文件

然后执行：make linux_fs_install_sd

这个步骤中其会将

# #copy remote firmware files for c6x_1
cp /opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/out/J7/C66/SYSBIOS/release/vx_app_tirtos_linux_c6x_1.out /opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/targetfs//lib/firmware/j7-c66_0-fw
/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/ti-cgt-c6000_8.3.7/bin/strip6x -p /opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/targetfs//lib/firmware/j7-c66_0-fw

而j7-c66_0-fw就是我们板端 关联c66芯片的文件，其位置在 /lib/firmware目录下。

**注意：**

**改动tiovx里的代码要先在vision_apps目录make tiovx**，然后再执行上面的步骤，实现更新j7-c66_0-fw



## 六.记录一次增加openvx node 节点的步骤

项目上需要，需要在openvx上面增加一个处理原始画面，截取roi区域用于后续处理的需求。

因此在取到视频流之后，需要送入自己的节点进行crop处理。

如下记录基础处理步骤。

1.在/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/tiovx/kernels/openvx-core/c66x 路径下仿照vx_color_convert_host.c 增加一个自己的节点实体处理函数文件 “vx_img_crop_target.c”，对应在相应include文件夹下面建立对应的tivx_kernel_img_crop.h 头

头文件中，核心将自己的参数构成宏定义确定。比如我们这里传了4个参数，输入图片，输出图片，crop的起始x和y。

在c文件中。核心关注函数tivxKernelImgCrop（）他就是节点每帧都会进入的函数。函数内注意图片的地址，要按照map的方式进行获取。

        for (i = 0; i < src_desc->planes; i++)
        {
            VX_PRINT(VX_ZONE_ERROR, "src plane = %d, memsize = %d\n", i, src_desc->mem_size[i]);
            src_desc_target_ptr[i] = tivxMemShared2TargetPtr(&src_desc->mem_ptr[i]);
            tivxCheckStatus(&status, tivxMemBufferMap(src_desc_target_ptr[i], src_desc->mem_size[i],
                                                      (vx_enum)VX_MEMORY_TYPE_HOST, (vx_enum)VX_READ_ONLY));
        }
       // tivxSetPointerLocation(src_desc, src_desc_target_ptr, (uint8_t**)&src_addr);
    
        for (i = 0; i < dst_desc->planes; i++)
        {
            VX_PRINT(VX_ZONE_ERROR, "dst plane = %d, memsize = %d\n", i, dst_desc->mem_size[i]);
            dst_desc_target_ptr[i] = tivxMemShared2TargetPtr(&dst_desc->mem_ptr[i]);
            tivxCheckStatus(&status, tivxMemBufferMap(dst_desc_target_ptr[i], dst_desc->mem_size[i],
                (vx_enum)VX_MEMORY_TYPE_HOST, (vx_enum)VX_WRITE_ONLY));
        }

然后可以定义自己的处理函数。

由于我们这里在dsp上系统没有实现crop，因此我们实际crop是在cpu上直接运行的memcpy。

2.处理完c66x后，我们还需要 /opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/apps/dl_demos/app_tidl_cam_od/app_crop_module.c 即应用路径下面，增加我们的处理函数

该c文件主要做得就是初始化app_init_crop 响应节点建立app_create_graph_crop 

初始化过程中，需要注意: 需要准确知晓自己在 create即后续实际每帧处理的时候的数据类型。然后好确定自己的out_buf的类型。因为out_buf是在初始化申请的空间。因此

        cap_yuv_outImg= vxCreateImage(
                                context, 
                                sensorOutParams.sensorInfo.raw_params.width, 
                                sensorOutParams.sensorInfo.raw_params.height, 
                                VX_DF_IMAGE_NV12/*VX_DF_IMAGE_UYVY*/
                         );
    需要根据不同的输入确定自己图片类型。比如我们这里如果原始的capture输入，那么数据类型应该是VX_DF_IMAGE_UYVY，如果是ldc处理之后的输入，那么数据类型就是VX_DF_IMAGE_NV12

在app_create_graph_crop ，就引入了，我们节点建立的关键函数：vxImgCropNode，其就关联到我们c66和host（后续讲）的实体节点建立

该函数位置位于：/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/tiovx/source/framework/vx_node_api.c

对应头文件也需要增加声明。对应我们的需求，我们增加了函数：

```
VX_API_ENTRY vx_node VX_API_CALL vxImgCropNode(vx_graph graph, vx_image input, vx_image output, vx_int32 startx, vx_int32 starty)
{
    vx_scalar start_x = vxCreateScalar(vxGetContext((vx_reference)graph), (vx_enum)VX_TYPE_INT32, &startx);
    vx_scalar start_y = vxCreateScalar(vxGetContext((vx_reference)graph), (vx_enum)VX_TYPE_INT32, &starty);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)start_x,
        (vx_reference)start_y,     
    };
    vx_node node = tivxCreateNodeByKernelEnum(graph, (vx_enum)VX_KERNEL_IMG_CROP, params, dimof(params));
       
    vxReleaseScalar(&start_x);
    vxReleaseScalar(&start_y);
    
    return node;

}
可以看到，我们这里节点的输入就是input，output，startx，starty
特别注意：针对其他参数的传入，需要转一遍vx_scalar的方式传入。
```

该函数内tivxCreateNodeByKernelEnum函数中的VX_KERNEL_IMG_CROP 就关联到我方的vx_img_crop_target.c

其定义需要在该文件下增加：/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/tiovx/include/VX/vx_kernels.h

3.在板子运行前，类似的节点文件，板子会进行一次初始化。其会优先调用/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/tiovx/kernels/openvx-core/c66x/vx_kernels_openvx_core_target.c这个函数，将所有在列表内的gTivx_target_kernel_list的函数进行注册tivxRegisterTargetKernels

这样板子才能认识这些函数，也就是这样，相当于设定了我们自己增加的函数入口。因此，需要在gTivx_target_kernel_list中，增加我们的 {tivxAddTargetKernelImgCrop, tivxRemoveTargetKernelImgCrop}描述。

4.板子在运行注册时，不是我们想象中的只需要修改c66目录下的target文件即可，同样，还需要修改host文件下的内容。因此我们还需要在/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/tiovx/kernels/openvx-core/host/vx_img_crop_host.c 路径下增加我们的host.c文件。

该文件要注意，如果参数数量有变化，一定要注意      vxAddParameterToKernel这个函数的调用。

```
  if (status == (vx_status)VX_SUCCESS)
        {
            status = vxAddParameterToKernel(kernel,
                        index,
                        (vx_enum)VX_INPUT,
                        (vx_enum)VX_TYPE_SCALAR,
                        (vx_enum)VX_PARAMETER_STATE_REQUIRED
            );
            index++;
        }
```

调试发现host文件中核心的起作用的函数是

        if (status == (vx_status)VX_SUCCESS)
        {
            //VX_PRINT(VX_ZONE_WARNING, "vxFinalizeKernel, From img crop host file!!\n");
            status = vxFinalizeKernel(kernel);
        }


该函数的调用在target.c中没有。

5.如上修改完成后需要执行三次编译操作：

1）/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps# make tiovx 即将我们在vx下面改动进行编译。

2）/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps# make vision_apps 关联vx库的文件重新连接一下vx编出来的库。更新一些out文件。

3）/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps# make linux_fs_install_sd 编译上板子的关键文件，其最终输出的文件目录在/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/targetfs/lib/firmware 关联我们vx改动的文件最后会被重命名为j7-c66_0-fw 之类。在我们开发板目录中对应opt/lib/firmware中

6.补充说明。在我们/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/apps/dl_demos/app_tidl_cam_od/main.c 文件下，核心关联节点设置的函数是static vx_status app_create_graph(AppObj *obj)

我们在该函数中增加了

    //进行输入画面裁剪，获取roi区域
    if(status == VX_SUCCESS)
    {
        //status = app_create_graph_crop(obj->graph, &obj->cropObj, obj->captureObj.raw_image_arr[0]);
        status = app_create_graph_crop(obj->graph, &obj->cropObj, obj->ldcObj.output_arr);
        APP_PRINTF("Crop graph done!\n");
    }
    可以看到我们这里的输入是采用ldc的输出。
        if(status == VX_SUCCESS)
        {
            //status = app_create_graph_scaler(obj->context, obj->graph, &obj->scalerObj, obj->ldcObj.output_arr);
            status = app_create_graph_scaler(obj->context, obj->graph, &obj->scalerObj, obj->cropObj.crop_image_arr[0]);
            APP_PRINTF("Scaler graph done!\n");
        }
     scaler的输入则被我们改变成我们crop之后的输出。
     
     这里同样要注意。scaler的输入要求是nv12，因此我们crop的输出要作为scaler的输入的话就必须改为nv12



## 七.记录交叉编译板子端的FFmpeg的步骤

1.首先编译x264库

在这个网址下载：http://download.videolan.org/pub/videolan/x264/snapshots/

使用的是x264-snapshot-20191217-2245-stable.tar.bz2

放到home下

tar -vxf  x264-snapshot-20191217-2245-stable.tar.bz2

```
./configure --prefix=/home/ffmpeg/x264/ --cross-prefix=/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu- --host=arm-linux-gnueabihf --disable-asm --enable-shared --enable-static --enable-pic
```

以上几个参数核心就是cross-prefix设置。这里注意，其设置的是一个交叉编译bin文件对应的路径以及“arm-none-linux-gnueabihf-”这个是所有的gcc，cc之类的bin文件的前缀所以一定要有“-”这个符号。去到这个目录就能发现这些文件了。

然后常规的

make -j4

make install

就会在指定目录生成include bin lib文件了。我们这里动态库和静态库都生成了

但需要指出。静态库后续连接编译的时候，系统的dl库会报一些dlcolse之类的错。没有花心思去解决了。

编译好x264之后，开始编译FFmpeg。

2.编译FFmpeg

在这个网址下载：http://ffmpeg.org/releases/

使用的是ffmpeg-4.4.2.tar.bz2

放到home下解压

然后注意，由于我们x264不是默认系统路径即没有装到/usr/local下面，所以需要指定x264编译和连接的路径。在执行config之前，先在终端中执行：

```
export PKG_CONFIG_PATH=/home/ffmpeg/x264/lib/pkgconfig/
```

执行完毕后再执行：

```
./configure --enable-cross-compile --arch=aarch64 --prefix=/home/ffmpeg/ffmpeg/ --cross-prefix=/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu- --target-os=linux --enable-shared --enable-libx264 --enable-gpl --extra-cflags=-I/home/ffmpeg/x264/include --extra-ldflags=-L/home/ffmpeg/x264/lib
```

注意由于我们板端程序编译的是64位的，所以这里arch要设置为aarch64。另外我们交叉编译的目标系统设置为linux。

还有因为要用x264，关联x264需要让gpl起效。另外就只指定x264的路径了。

执行后，make -j4 然后install

这样FFmpeg的动态静态库就生成了。

3.在我们的程序内将FFmpeg使用起来。

以tidl_cam_od为例。打开/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/apps/dl_demos/app_tidl_cam_od/concerto.mak

```
ifeq ($(TARGET_CPU),A72)
FFMPEG_PATH = /home/ffmpeg
FFMPEG_INCLUDE_PATH=$(FFMPEG_PATH)/ffmpeg/include
FFMPEG_LIB_PATH=$(FFMPEG_PATH)/ffmpeg/lib

IDIRS += $(FFMPEG_INCLUDE_PATH)

#这些库目前拷贝到了/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/lib/J7/A72/LINUX/release下面
# STATIC_LIBS += x264
# STATIC_LIBS += avcodec
# STATIC_LIBS += avformat
# STATIC_LIBS += swscale
# STATIC_LIBS += avfilter
# STATIC_LIBS += avutil
# STATIC_LIBS += swresample
SYS_SHARED_LIBS += x264
SYS_SHARED_LIBS += avformat
SYS_SHARED_LIBS += avcodec
SYS_SHARED_LIBS += avutil
SYS_SHARED_LIBS += swresample
SYS_SHARED_LIBS += avfilter
SYS_SHARED_LIBS += swscale
endif
```

增加如上编译选项。即指定了我们的FFmpeg是在编译arm程序的时候引入。

然后之前在使用静态库的时候报错dl相关的库。所以我们后来选择使用动态库加载方式。

动态库注意，其读取动态库的路径是/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/aarch64-none-linux-gnu/libc/usr/lib64

因此我们需要将我们生成好的FFmpeg和x264的库都扔到这个目录下来。

注意。不能单独扔libx264.so，还要把母文件一起放进去。不然他只是一个软连接，不要被文件信息给骗了。

放好之后，我们就可以愉快的编译生成out了

/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps# make vx_app_tidl_cam_od

然后在板端，由于我们的程序放在了/opt/visionapp下面，所以对应所有的x264 FFmpeg so都要放一份进去。



## 八.记录外部节点配置参数意义

```
        graph_parameter_index = 0;
#ifndef USE_FFMPEG
        add_graph_parameter_by_node_index(obj->graph, obj->captureObj.node, 1);
        obj->captureObj.graph_parameter_index = graph_parameter_index;
#else
		//注意：这个地方设置的index取决于节点内部设置时的输入对应的位置，比如输入是在0号位置，那么index就要设置为0
        add_graph_parameter_by_node_index(obj->graph, obj->cropObj.node, 0);
        obj->cropObj.graph_parameter_index = graph_parameter_index;
#endif
        graph_parameters_queue_params_list[graph_parameter_index].graph_parameter_index = graph_parameter_index;
        graph_parameters_queue_params_list[graph_parameter_index].refs_list_size = APP_BUFFER_Q_DEPTH;

 #ifndef  USE_FFMPEG
        graph_parameters_queue_params_list[graph_parameter_index].refs_list = (vx_reference*)&obj->captureObj.raw_image_arr[0];
#else
        graph_parameters_queue_params_list[graph_parameter_index].refs_list = (vx_reference*)&obj->cropObj.input_images[0];
#endif
        graph_parameter_index++;
```

注意，这里对应

```
VX_API_ENTRY vx_node VX_API_CALL vxImgCropNode(vx_graph graph, vx_image input, vx_image output, vx_int32 startx, vx_int32 starty)
{
    vx_scalar start_x = vxCreateScalar(vxGetContext((vx_reference)graph), (vx_enum)VX_TYPE_INT32, &startx);
    vx_scalar start_y = vxCreateScalar(vxGetContext((vx_reference)graph), (vx_enum)VX_TYPE_INT32, &starty);

    vx_reference params[] = {
        (vx_reference)input,//因为vxImgCropNode添加的时候input放在了0号位置，所以上面的add_graph_parameter_by_node_index设置为0
        (vx_reference)output,
        (vx_reference)start_x,
        (vx_reference)start_y,     
    };
    vx_node node = tivxCreateNodeByKernelEnum(graph, (vx_enum)VX_KERNEL_IMG_CROP, params, dimof(params));
       
    vxReleaseScalar(&start_x);
    vxReleaseScalar(&start_y);
    
    return node;

}
```

同理

        if(status == VX_SUCCESS)
        {
            status = tivxSetNodeParameterNumBufByIndex(obj->cropObj.node, 1, 6);//这个地方的1代表output的位置index
        }
    对应上面的param可以看到，输出缓存index就是1.



## 九.记录一次修改已有节点输出的流程和注意点

由于我们od检测需要将检测到的框返回到主函数内，原有的app_post_proc_module模块只是将od目标检测框在传入的图像上绘制出来。对应框的位置却没有返回，外部需要获取的话还需要再重新解析一遍结果buf，因此需要将tivxDrawBoxDetectionsNode（）函数进行修改。

```
VX_API_ENTRY vx_node VX_API_CALL tivxDrawBoxDetectionsNode(vx_graph             graph,
                                                           vx_user_data_object  configuration,
                                                           vx_tensor            input_tensor,
                                                           vx_image             input_image,
                                                           vx_image             output_image,
                                                           vx_user_data_object  results)
{
  vx_reference prms[] = {
          (vx_reference)configuration,
          (vx_reference)input_tensor,
          (vx_reference)input_image,
          (vx_reference)output_image,
          (vx_reference)results
  };
  在这里增加一个vx_user_data_object  results 参数传入
```

对应我们在app_init_post_proc（）中对object的结构申请空间

由于默认情况下没有od检测的结构。因此在/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/kernels/img_proc/include/TI/tivx_img_proc_kernels.h头文件中，我们增加对应结果的结构体

```
typedef struct
{
  //位置
  vx_int32 box_left;
  vx_int32 box_top;
  vx_int32 box_width;
  vx_int32 box_height;

  //类别
  vx_int32 box_index;

  //分数
  vx_float32 box_score;
} tivxODtargetInfo;

typedef struct
{

  /** List of class ID's in descending order */
  tivxODtargetInfo box_info[100];
  /** Number of top results to consider */
  vx_int32 num_box;
} tivxODPostProcOutput;
```

一顿操作之后，我们将/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/kernels/img_proc/host/tivx_draw_box_detections_host.c文件中增加了一个参数输出代码段

```
        if (status == VX_SUCCESS)
        {
            status = vxAddParameterToKernel(kernel,
                        index,
                        VX_OUTPUT,
                        VX_TYPE_USER_DATA_OBJECT,
                        VX_PARAMETER_STATE_REQUIRED
            );
            index++;
        }
        注意数据类型
```

然后去到/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/kernels/img_proc/c66/vx_draw_box_detections_target.c文件中，处理我们要输出的结构体数据：

```
        tivx_obj_desc_user_data_object_t *results_desc;
        void *results_target_ptr;
        
        results_desc = (tivx_obj_desc_user_data_object_t *)obj_desc[TIVX_KERNEL_DRAW_BOX_DETECTIONS_OUTPUT_RESULT_IDX];
        results_target_ptr = tivxMemShared2TargetPtr(&results_desc->mem_ptr);
        tivxMemBufferMap(results_target_ptr, results_desc->mem_size, VX_MEMORY_TYPE_HOST, VX_WRITE_ONLY);
        
        tivxODPostProcOutput *pResults = (tivxODPostProcOutput *)results_target_ptr;
        
        int bi = 0;
        pResults->num_box = numBox;
        for ( bi = 0; bi < numBox; bi++)
        {
            //VX_PRINT(VX_ZONE_ERROR, "get new obj  label = %d score = %f\n", boxptr[bi].classIndex, boxptr[bi].score);
           // VX_PRINT(VX_ZONE_ERROR, "box = [l = %d, t = %d, r = %d, b = %d]\n", boxptr[bi].x, boxptr[bi].y, boxptr[bi].x + boxptr[bi].w, boxptr[bi].y + boxptr[bi].h);
            pResults->box_info[bi].box_left = boxptr[bi].x;
            pResults->box_info[bi].box_top = boxptr[bi].y;
            pResults->box_info[bi].box_width = boxptr[bi].w;
            pResults->box_info[bi].box_height = boxptr[bi].h;
            pResults->box_info[bi].box_index = boxptr[bi].classIndex;
            pResults->box_info[bi].box_score = boxptr[bi].score;
         }
 		tivxMemBufferUnmap(results_target_ptr, results_desc->mem_size, VX_MEMORY_TYPE_HOST, VX_WRITE_ONLY);
```

这样就把数据回到了我们传入的buf中。

在/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/apps/dl_demos/app_tidl_cam_od/main.c中，我们需要做如下一些处理

        //输出结果导出来--增加导出结果的节点
        add_graph_parameter_by_node_index(obj->graph, obj->postProcObj.node, 4);
        obj->postProcObj.graph_parameter_index = graph_parameter_index;
        graph_parameters_queue_params_list[graph_parameter_index].graph_parameter_index = graph_parameter_index;
        graph_parameters_queue_params_list[graph_parameter_index].refs_list_size = 1;//APP_BUFFER_Q_DEPTH;
        graph_parameters_queue_params_list[graph_parameter_index].refs_list = (vx_reference*)&obj->postProcObj.results;
        graph_parameter_index++;

然后再pipeline函数内进行结果处理

        //编入队列
        if(status == VX_SUCCESS)
        {
            status = vxGraphParameterEnqueueReadyRef(obj->graph, postProcObj->graph_parameter_index, (vx_reference*)&postProcObj->results, 1);
        }
        
        然后解出来
                /* Dequeue output */
            if(status == VX_SUCCESS)
            {
                status = vxGraphParameterDequeueDoneRef(obj->graph, postProcObj->graph_parameter_index, (vx_reference*)&results, 1, &num_refs);
            }
            vx_map_id map_id_results;
            tivxODPostProcOutput *pResults;
    
            if(status == VX_SUCCESS)
            {
                status = vxMapUserDataObject(results, 0, sizeof(tivxODPostProcOutput), &map_id_results,
                                        (void **)&pResults, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
            }
    
            if(pResults != NULL)
            {
    
                if (pResults->num_box > 0)
                {
                    printf("-------detect  = %d box !!!------------\n", pResults->num_box);
                    vx_int32 i;
                    for (i = 0; i < pResults->num_box; i++)
                    {
                        printf("box %d-------class index  = %d score = %f !!!------------\n", i, pResults->box_info[i].box_index, pResults->box_info[i].box_score);
                    }
                }
            }

基本流程就OK了



有个坑：在app_create_graph_post_proc函数内，当tivxDrawBoxDetectionsNode参数发生变化之后，

    printf("app_create_graph_post_proc ------ 3--------------\n");
    vx_bool replicate[] = {vx_false_e, vx_true_e, vx_true_e, vx_true_e, vx_true_e};
    vxReplicateNode(graph, postProcObj->node, replicate, 5);
    
    printf("app_create_graph_post_proc ------ 4--------------\n");

这个部分也需要对应变化成5个输入，原来是4个。如果不这样处理。那么会导致app_create_graph_post_proc的输出会出现闪屏现象。



## 十.记录tda4 opencv 编译调用流程和注意点

下载opencv版本：https://opencv.org/releases/ 我们用的OpenCV – 4.5.5

打开cmake-gui

https://javay.blog.csdn.net/article/details/51272989?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-51272989-blog-113643822.pc_relevant_multi_platform_whitelistv4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-51272989-blog-113643822.pc_relevant_multi_platform_whitelistv4&utm_relevant_index=7

参考这个文章，配置好我们的交叉编译gcc和g++ 还有lib 搜索位置

目前我们设置的交叉编译文件的位置如下：

gcc：/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc

g++：/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++

搜索目录：/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/

configure 然后 generate，注意：一般会把world编译勾选上，build_png jpeg 都勾上，这样就不用麻烦的设置调用so顺序

完事用shell转到我们设置的build目录

执行make -j4

make install

报错：找不到png相关

/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/../lib/gcc/aarch64-none-linux-gnu/9.2.1/../../../../aarch64-none-linux-gnu/bin/ld: ../../lib/libopencv_world.so.4.5.5: undefined reference to `png_do_expand_palette_rgb8_neon'

参考：https://blog.csdn.net/u010571709/article/details/122103591

打开[opencv](https://so.csdn.net/so/search?q=opencv&spm=1001.2101.3001.7020)源码目录，编辑文件 `vim 3rdparty/libpng/pngpriv.h`

```
 130 /* #  if (defined(__ARM_NEON__) || defined(__ARM_NEON)) && \ */ 注释
 131 # if defined(PNG_ARM_NEON) && (defined(__ARM_NEON__) || defined(__ARM_NEON)) && \ 新加
 132    defined(PNG_ALIGNED_MEMORY_SUPPORTED)
 133 #     define PNG_ARM_NEON_OPT 2
 134 #  else
 135 #     define PNG_ARM_NEON_OPT 0
 136 #  endif
 137 #endif

```

问题解决。

***************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

在项目内进行c/c++混合编译

c++代码 接口 h

```
#ifndef _APP_OPENCV_MODULE
#define _APP_OPENCV_MODULE

#pragma once

#ifdef __cplusplus
extern "C" {
#endif
int opencv_add(int a, int b);

#ifdef __cplusplus
}
#endif

#endif
```

接口 cpp

```
#ifdef __cplusplus
extern "C" {
#endif

using namespace cv;
int opencv_add(int a, int b)
{
    open_adas* testClass = new open_adas();

    int ret = testClass->process(a,b);

    cout << ret << endl;
    
    delete testClass;
    return ret;
}
#ifdef __cplusplus
}
#endif
```

在makefile中 concerto.mak

```
……
OPENCV_PATH=/home/opencv/opencv-4.5.5-arm/arm-install
OPENCV_INCLUDE_PATH=$(OPENCV_PATH)/include/opencv4/
OPENCV_LIB_PATH=$(OPENCV_PATH)/lib
IDIRS += $(OPENCV_INCLUDE_PATH)

SYS_SHARED_LIBS += opencv_world

CPPSOURCES  :=
CPPFLAGS    := --std=c++11

CPPSOURCES    += open_adas.cpp
CPPSOURCES    += opencv_fun.cpp
……
```

然后编译好的opencv的库，放到/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/aarch64-none-linux-gnu/libc/usr/lib64路径下，即sys找寻路径的地址。

在我们c文件中，调用我们cpp文件

```
//是否用opencv的接口
#define OPENCV_DEBUG

#ifdef OPENCV_DEBUG
#include "opencv_fun.h"
int opencv_add(int a, int b);//声明
#endif

……
//就可以找地方来调用了
int ret = opencv_add(1,2);
……
```

特别需要注意的一点：

在编译板端的程序时，需要将环境变量中的target端的include目录映射进去，而这个目录会导致正常的x86编译失败（如编译opencv x86版本的库的时候，会出现include目录异常的情况）

对应修改的文件为：vim ~/.bashrc

```
……
#在编译板子程序的时候需要设置如下三个环境变量
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/pdk_jacinto_07_01_00_45/packages/ti/drv/udma/lib/j721e_hostemu/c7x-hostemu/release
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/targetfs/usr/include/
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/opt/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/pdk_jacinto_07_01_00_45/packages/
……
在编译x86环境时，这三个量需要注释掉。
然后执行 source ~/.bashrc
```



## 十一.记录tda4 使用yolop模型流程和注意点

yolop模型同时整合了目标检测/freespace分割/车道线检测，其网络主要构成可参见对应的yoloP 模型

其中原始的yoloP目标检测部分采用的是yolov5的网络结构，然后对应的计算回归和损失的地方，由于需要在tda4 7.1版本sdk上运行，因此需要进行修改

                # Regression
                #yolo5 的处理方式
                # pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                #yolo3的处理方式，这里是考虑移植到ti上，板端7.01的yolo模型支持
                pxy = ps[:, :2].sigmoid()
                pwh = torch.exp(ps[:, 2:4]) * anchors[i]
                
         参见：loss.py 92行 同样在yoloP detect函数中的forward，也需要进行修改
         		xy,wh,conf = x[i].split((2,2,self.nc+1), 4)
         		xy = (xy.sigmoid() + self.grid[i])*self.stride[i]
         		wh = torch.exp(wh)*self.anchor_grid[i]
         		conf = conf.sigmoid()
         		y = torch.cat((xy,wh,conf),4)
         参见：yolop.py 260行
    ！如上改动在tda4 8.2版本sdk是不需要的。目前因为是要适配7.1版本sdk所以只能按之前yolov3的方式改动。

同时，YOLOP原始的网络结构在SSP操作上与ti要求的不一致，focus操作也是yolov5才支持的写法。因此都根据之前ti支持的写法进行了网络结构的改写。

改写完之后进行训练。

训练，目标检测部分是转成了coco类型，freespace和line检测都是通过mask标定的方式生成的label。同时读取，其loss的计算如下

loss = lbox（目标框位置） + lobj（目标置信度） + lcls（类别loss） + lseg_da（freespace） + lseg_ll（line点置信度） + liou_ll（line点构成的区域与gt之间的iou） 构成

在做freespace和line检测时，其采用的损失函数为 nn.BCEWithLogitsLoss（）该函数已经做了一次sigmoid，而yoloP处理detect的时候，其针对seg的结果又做了一次sigmoid（参见yolop 611），所以这里可能存在一个bug（但目前保留原有的训练方式反而能获取到正确的结果）还需要进一步验证不同的训练方式。



训练完成后，要注意，在export_onnx.py中 除了修改对应的网络结构外，在seg导出的时候，一定要将原始的2维度输出，通过argmax函数转为1维输出。具体参见128行。这样导出来的模型才可以直接在ti板子上直接获取到分割结果。



好，导出onnx之后，转入到tidl里面去，转bin

首先prototxt 主要针对yolo 目标检测部分，跟yolov3的一致 这些anchor都来至于yolop网络结构，节点对应的输出就是8 16 32倍下采样对应的输出

```
name: "yolo_v3"
tidl_yolo {
  yolo_param {
    input: "onnx::Reshape_751"
    anchor_width: 3.0
    anchor_width: 5.0
    anchor_width: 4.0
    anchor_height: 9.0
    anchor_height: 11.0
    anchor_height: 20.0
  }
  yolo_param {
    input: "onnx::Reshape_826"
    anchor_width: 7.0
    anchor_width: 6.0
    anchor_width: 12.0
    anchor_height: 18.0
    anchor_height: 39.0
    anchor_height: 31.0
  }
  yolo_param {
    input: "onnx::Reshape_901"
    anchor_width: 19.0
    anchor_width: 38.0
    anchor_width: 68.0
    anchor_height: 50.0
    anchor_height: 81.0
    anchor_height: 157.0
  }
  detection_output_param {
    num_classes: 1
    share_location: true
    background_label_id: -1
    nms_param {
      nms_threshold: 0.45
      top_k: 200
    }
    code_type: CENTER_SIZE
    keep_top_k: 100
    confidence_threshold: 0.25
  }
  name: "yolo_v3"
  in_width: 640
  in_height: 384
  #output: "detections"
}
```

然后修改 import的配置。

```
modelType          = 2
numParamBits       = 8
numFeatureBits     = 8
quantizationStyle  = 3#保留跟yolov3一致的量化方案
#quantizationStyle  = 2
inputNetFile       = "../../test/testvecs/models/public/onnx/yolop4-640-384.onnx"
outputNetFile      = "../../test/testvecs/config/tidl_models/onnx/tidl_net_yolop_6438.bin"
outputParamsFile   = "../../test/testvecs/config/tidl_models/onnx/tidl_io_yolop_6438_"
inDataNorm  = 1
inMean = 124 116 104 #这是yolop指定的预处理方式
inScale = 0.017125 0.0175 0.017429
inDataFormat = 1
inWidth  = 640
inHeight = 384 
inNumChannels = 3
numFrames = 1
inData  =   "../../test/testvecs/config/detection_bdd.txt"
perfSimConfig = ../../test/testvecs/config/import/device_config.cfg
inElementType = 0
#outDataNamesList = "det_out,drive_area_seg,lane_line_seg"
outDataNamesList = "drive_area_seg,lane_line_seg"#这里要注意了，不能把det_out放这里，因为det_out的输出是从metaArch出来的。
#outDataNamesList = "drive_area_seg"
metaArchType = 4
metaLayersNamesList =  "../../test/testvecs/models/public/onnx/yolop4-640-384.prototxt"
postProcType = 3
```

这样就可以获取到对应的bin文件了。

bin文件可以单独测试od 或者单独测试 seg 注意修改postProcType就好了，postProcType = 3的时候全部结果导出的时候od是不会做叠加的。

进行bin文件调用

因为这个时候从tidl输出已经是三个输出了，od/seg/line，因此核心需要修改post_pro后处理，我们是将所有的绘制操作都在后处理进行，因此修改了vx_draw_box_detections_target.c，当然同时需要修改其他对应的接口。

    postProcObj->node = tivxDrawBoxDetectionsNode(graph,
                                                        postProcObj->config,
                                                        input_tensor,
                                                        input_tensor_seg,
                                                        input_tensor_line,
                                                        input_image,
                                                        output_image, 
                                                        result);
                                                 #接口被我们改成了这个样子

要小心

        //输出结果导出来
        //add_graph_parameter_by_node_index(obj->graph, obj->postProcObj.node, 4);//注意因为我们加了两个输入
        add_graph_parameter_by_node_index(obj->graph, obj->postProcObj.node, 6);
        obj->postProcObj.graph_parameter_index = graph_parameter_index;
        graph_parameters_queue_params_list[graph_parameter_index].graph_parameter_index = graph_parameter_index;
        graph_parameters_queue_params_list[graph_parameter_index].refs_list_size = 1;//APP_BUFFER_Q_DEPTH;
        graph_parameters_queue_params_list[graph_parameter_index].refs_list = (vx_reference*)&obj->postProcObj.results;
        graph_parameter_index++;
        和
                if (status == VX_SUCCESS)
            {
                //status = tivxSetNodeParameterNumBufByIndex(obj->postProcObj.node, 3, 2);
                status = tivxSetNodeParameterNumBufByIndex(obj->postProcObj.node, 5, 2);
            }
            因为我们输入多加了两个，所以对应的index要做修改

在vx_draw_box_detections_target内部，目标检测我们沿用原有的处理方式，针对seg我们直接解析，由于其输出是按点判断是否是seg点，因此循环放到最外以加速处理，这里画点叠加的开销还是比较大的，如果不需要叠加，可以将此处的运算省下来。

```
#ifdef DRAW_SEG
        //直接解析seg
        {
            output_sizes[0] = ioBufDesc->outWidth[1] + ioBufDesc->outPadL[1] + ioBufDesc->outPadR[1];
            output_sizes[1] = ioBufDesc->outHeight[1] + ioBufDesc->outPadT[1] + ioBufDesc->outPadB[1];
            output_sizes[2] = ioBufDesc->outNumChannels[1];
            {
                vx_uint8 *seg_buffer;
                vx_uint8 *line_buffer;
                seg_buffer = (vx_uint8 *)input_tensor_seg_target_ptr;
                line_buffer = (vx_uint8 *)input_tensor_line_target_ptr;
                //因为seg 和line的输出是一样的，所以，统一按 seg的参数进行处理
                //int32_t nChannelNum =  ioBufDesc->outNumChannels[1];
                int32_t nWidth =  ioBufDesc->outWidth[1];
                int32_t nHeight =  ioBufDesc->outHeight[1];

                vx_uint8  *pOut_seg = NULL;
                vx_uint8  *pOut_line = NULL;
                vx_int32 m,i, j;

                for (m = 0; m < 1; m++)
                {
                    //pOut_seg = (vx_uint8 *)seg_buffer + (nWidth * nHeight * m) + (ioBufDesc->outPadT[k] * nWidth) + ioBufDesc->outPadL[k];
                    pOut_seg = (vx_uint8 *)seg_buffer + (ioBufDesc->outPadT[1] * output_sizes[0]) + ioBufDesc->outPadL[1];
                    pOut_line = (vx_uint8 *)line_buffer + (ioBufDesc->outPadT[1] * output_sizes[0]) + ioBufDesc->outPadL[1];

                    int start_h = nHeight / 2;
                    pOut_seg = pOut_seg + start_h * nWidth;
                    pOut_line = pOut_line + start_h * nWidth;

                    //因为DVR视角上半区为天空，根据测试视频观测，所以加速判断
                    for (i = start_h; i < nHeight; i++)
                    {
                        for (j = 0; j < nWidth; j++)
                        {
                            vx_uint8 valueSeg = *(pOut_seg + j);
                            vx_uint8 valueLine = *(pOut_line + j);

                            if (valueSeg > 0)
                            {
                                vx_uint8 color_map[3];
                                color_map[0] = 180;
                                color_map[1] = 80;
                                color_map[2] = 40;
                                vx_int32 x = (vx_int32)(width * 1.0 / nWidth * j);
                                vx_int32 y = (vx_int32)(height * 1.0 / nHeight * i);
                                drawPoint(data_ptr_1, data_ptr_2, width, height, x, y, color_map);
                                // VX_PRINT(VX_ZONE_ERROR, "freespace x = %d y = %d seg_value = %d\n", x, y, valueSeg);
                            }
                            if (valueLine > 0)
                            {
                                //  Y = 16  + 0.183 * R + 0.614 * g + 0.062 * b                //
                                // Cb = 128 - 0.101 * R - 0.339 * g + 0.439 * b                //
                                // Cr = 128 + 0.439 * R - 0.399 * g - 0.040 * b  
                                vx_uint8 color_map[3];
                                color_map[0] = 62;
                                color_map[1] = 102;
                                color_map[2] = 239;
                                vx_int32 x = (vx_int32)(width * 1.0 / nWidth * j);
                                vx_int32 y = (vx_int32)(height * 1.0 / nHeight * i);
                                drawPoint(data_ptr_1, data_ptr_2, width, height, x, y, color_map);
                                // VX_PRINT(VX_ZONE_ERROR, "lane x = %d y = %d line value = %d\n", x, y, valueSeg);
                            }
                        }
                        pOut_seg += nWidth;
                        pOut_line += nWidth;
                    }
                }
            }
        }
#endif
```

如果不需要内部解析，那么我们也可以在main里面处理。我们也写了相关处理函数：vx_status dataGetKeyValue( AppObj *obj,  CropObj    *cropObj)

这里目前处理效率不是很高，还需进一步改进。

目前目标检测在板端检测效果非常差，而onnx效果在pc上测试效果正常，因此还需要判断下具体产生问题的原因。

ps：

经过初步判定yoloP的原始训练的anchor尺寸非常小，并且都是竖条状，目前修改为coco的anchor，效果明显有改善。
