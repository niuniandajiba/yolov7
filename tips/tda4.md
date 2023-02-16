# TDA4

## 一、vision_app 相关

### make sdk



## 二、模型转换相关

### YOLOP



## 三、pc emulation build

### 8.04

按照user_guide来操作，一切正常

### 7.01

#### 1. make tidl_rt

`Linking /tda4/sdk7.01/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/tidl_j7_01_03_00_11/ti_dl/rt/out/PC/x86_64/LINUX/debug/libvx_tidl_rt.so`
`/usr/bin/ld: cannot find -l:dmautils.lib`

按照user_guide来操作，make sdk时出现问题，最后发现是/tda4/sdk7.01/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/tidl_j7_01_03_00_11/makerules/config.mk:129的PDK_INSTALL_PATH路径出现了问题，导致没有读到库，将里面的“/pdk”改成“/pdk_jacinto_07_01_00_45”解决。

#### 2. make tiovx

`/usr/bin/ld: cannot find -lapp_utils_iss`

`/usr/bin/ld: cannot find -lvx_app_ptk_demo_common`

前者添加库的路径，后者把8.04的拿来了，居然能用。

更改文件：/tda4/sdk7.01/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/tiovx/kernels_j7/concerto_inc.mak:70

`LDIRS += $(VISION_APPS_PATH)/tmplib`, 然后把库放这下面

#### 3. run

`"VX_ZONE_ERROR:[ownReleaseReferenceInt:307]`

原因：/tda4/sdk7.01/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/vision_apps/apps/dl_demos/app_tidl_od/app_display_module.c:68中，判断条件`&&(displayObj->display_option == 1)`让返回的status=VX_FAILURE,程序终止。

## 四、停车位检测模型



### 8.04

转换8bits模型，网络在sigmoid层后截断，读取sTIDL_IOBufDesc_t中的outTensorScale，output/=outTensorScale，后处理手动实现，结果无误。

### 7.01

#### 找outTensorScale

sTIDL_IOBufDesc_t结构中不含outTensorScale，需要自己去找。

/home/chenqiang/tda4/sdk8.04/ti-processor-sdk-rtos-j721e-evm-08_04_00_06/tidl_j721e_08_04_00_16/ti_dl/utils/tidlModelImport/tidl_import_common.cpp:1505

```c
gIOParams.outTensorScale[numDataBuf] = tIDLNetStructure->TIDLLayers[i].outData.tensorScale;
```

可以知道实际上就是最后一层的outData.tensorScale.

之后改写/home/chenqiang/tda4/sdk8.04/ti-processor-sdk-rtos-j721e-evm-08_04_00_06/tidl_j721e_08_04_00_16/ti_dl/utils/tidlTensorRangeUpdate/tidl_tensor_range_update.c:108,加入

```c
printf("tensorscale: %f\n", TIDLLayer->outData.tensorScale);
```

**记得注释**掉下面的updateTensorRange的两行。

之后make，在./out中找到可执行文件并传路径进去，得到：![](C:\Users\6000022857\Pictures\tensorScale.png)

#### 转换经历

转换8bits和16bits模型，在sigmoid后截断，解析输出发现全是00和ff，原因为该层输出范围为(0,0.23),tensorScale=770,导致大于0.23的数字都被错误缩放。（可能是7里面的sigmoid实现有问题，8就是正常的）

转换8bits模型，在sigmoid前截断，tensorScale=3.01，最小单位仅为0.33的样子，对回归模型影响感觉会很大。

转换16bits模型，在sigmoid前截断，tensorScale=1541.79，精确度略有下降，在可接受的范围内。

#### 手动实现后处理

多的部分在于手动实现sigmoid，这里参考tidl中/tda4/sdk7.01/ti-processor-sdk-rtos-j721e-evm-07_01_00_11/tidl_j7_01_03_00_11/ti_dl/algo/src/tidl_softmax.c中的实现：

```c
略
```

（最后没有用这种麻烦的方法,快速写了个下面这种的)

```c
vx_float32 sigmoid_out(vx_float32 x)
{
    return (vx_float32)(1.0 / (1.0 + exp(-x)));
}
```

注：16位模型中后处理的stride需要多加注意。

#### 实际跑起来

16位模型下输出结果只有半边是对的，所有右边的车位都看不见，经过逐节点排查，发现scaler这个节点输出无问题，它之后的preproc节点输出就有问题，（8位下无问题）原因目前猜测是16位模型会影响preproc节点的处理，提供错误的tensorstride信息从而使得到的三通道rgb图只剩一半。

## 五、跑板子

`[I][2021-06-01 10:02:25][sendCommInitMsgToMcu:572] Send Comm Init Cmd
cat: can't open '/sys//class/cat: can't open '/sys//class/mmc_host/mmc0/mmc0:0001/lh_sd_card_status': No such file or directory`