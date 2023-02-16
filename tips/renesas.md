# Renesas

## CNN_TOOLCHAIN

命令行没看懂怎么用，就用CNN_Fronted_GUI了

### GUI

#### 出现了各种各样的读取库的错误：

- GLIBC_2.29，使用命令strings /path/to/libc.so.6 | grep GLIBC确认版本，最简单的解决方法是用ubuntu20.04.
- ERROR: file too short，把/path/to/lib下面的各种库文件重新软链接一遍后解决。（可能的原因：在windows下解压会导致里面的软链接失效，最好在ubuntu下解压）
- 运行过程中的找不到libgenericAPI_RCAR_V3U1.so，find ./ -name {libname}后在GUI终端export LD_LIBRARY_PATH={path/to/libname}解决。

#### onnx::opset_version问题：

用sample中opset=14的onnx模型无问题，~~但是自己一旦用opset>=9的模型就不行，一直在运行中但是不输出结果~~（已解决），原因：默认mode='nearest'，在导入时会被当成Identity Node，从而导致这个层实际上没有用，输出FM大小对不上。![img](C:\Users\6000022857\Documents\typora_image\renesas_maual_cnn_framework_resize.png)

![img](C:\Users\6000022857\Documents\typora_image\onnx_resize_node.png)



另外文档要求ONNXv3模型（[对应opset=8/9](https://onnxruntime.ai/docs/reference/compatibility.html)），经测试支持的ONNX版本不限于v3。

#### Upsample问题：

因为导出torch.nn.Upsample->onnx::Resize需要opset>=11,所以先用之前在tda2上尝试过的反卷积层实验。

用opset=8的yolo模型，导入时之前训练的tda2转换前模型的反卷积层出现问题（注：tda2上反卷积层要求CIn=COut=NumGroups）![img](C:\Users\6000022857\Documents\typora_image\renesas_error_convtranspose_onnx.png)

用tda2转换后的caffe模型导入，反卷积层得到类似但不同的问题：![img](C:\Users\6000022857\Documents\typora_image\renesas_error_convtranspose_caffe.png)

onnx::resize不支持mode='nearset', 'linear', 'cubic'，但是瑞萨有自己特定的叫做upscale的caffe层，强行在prototxt中将interpolate层改为upscale后转换通过，但是结果不对，在模型前加入bn结果也还是不对。（输出的int8.4置信度20*20全为128）

注：将反卷积层的groups改为1后通过，但是推理结果和pc端对不上的问题在官方提供的sample和我们自己的模型中都有出现，已经问了瑞萨相关人员，正在等待官方人员回复。

## V3H StarterKit

以上碰见的问题先不管，总算是跑出了一个转换后的可执行文件，看起来应该是在板子上运行的，就想办法传到板子上去。

### 传文件

先用usb串口连接上电脑，然后按文档中设置：

*buad_rate 115200, 8bit data, parity none, stop 1 bit, flow control none*

连接上后ifconfig看连接端口，然后就可以用ssh连接或者sftp传东西了。

传上去后在板端运行也有GLIBC版本的问题，看附带的sdk里面有一个更新系统的办法，就尝试一下。

### yocto

1. 找个尖锐物拨板子上的开关，切换成SCIF download模式后，把一系列文件写进512-Mbit SPI里面。（先传flash_writer_tool.mot，再传一系列文件）（用MobaXterm没法传文件，用SecureCRT传）
2. 在linux host端（可以是虚拟机）准备好tftp和nfs服务，把Image和xxx.dtb文件放在设置的tftp设置的路径下，新系统整个放在nfs设置的路径下。
3. 把板子的开关拨回来，然后在uboot界面更改bootargs以及bootcmd，更改为挂载启动
4. 重新上电，板子成功挂载启动，然后用`mount mmcblk0p1 /fs`把emmc挂载在/fs这个路径下，把里面的东西删了之后把整个系统拷进去。
5. 继续更改bootargs和bootcmd，改成从本机读取Image以及本机启动
6. 重新上电，系统更新完成。

### 运行

helloworld_sample运行成功，但是后面的SAMPLE_CVE和SAMPLE_CNN都运行失败在同一个地方<R_OSAL_MmngrOpen: code 5>，原因还在找，但听说不用瑞萨了，所以不管了。

## 总结

因为不用瑞萨的板子了，所以先不管这些遗留问题了

遗留问题：

1. cnn输出结果和pc端对不上。
2. 无法在板端运行cve和cnn的sample程序。
3. 莫名其妙的很多层转不通。
