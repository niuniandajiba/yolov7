# 1、输出是caffe2而tidl用的是caffe1有些结构不一样

## （1）输入结构参考如下修改

```
input_shape {
  dim: 1
  dim: 3
  dim: 320
  dim: 768
}
```

# （2）“Pooling”层删除“pooling\_param”中的“round\_mode”参数

```
layer {
  name: "max_pool1"
  type: "Pooling"
  bottom: "relu_blob32"
  top: "max_pool_blob1"
  pooling_param {
    pool: MAX
    kernel_size: 5
    stride: 1
    pad: 2
    round_mode: FLOOR
  }
}
```

# 2、输入增加一个batchnorm层

由于输入为rgb需要做归一化到0-1，这里用一个batchnorm层替代

## （1）参考如下，在第一个卷积前增加batchnorm层

```
layer {
  name: "batch_norm0"
  type: "BatchNorm"
  bottom: "blob1"
  top: "batch_norm_blob0"
  batch_norm_param {
    use_global_stats: true
    eps: 0.001
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "batch_norm_blob0"
  top: "conv_blob1"
  convolution_param {
    num_output: 16
    bias_term: false
    pad: 2
    kernel_size: 6
    group: 1
    stride: 2
    weight_filler {
      type: "xavier"
    }
    dilation: 1
  }
}
```

## （2）执行脚本修改权重文件

```python
import numpy as np
import caffe
caffe_root = r'E:\project\caffe-windows'
sys.path.insert(0, caffe_root + 'python')
caffe.set_mode_cpu()
network_path = ""
model_weights = ""
net = caffe.Net(network_path, model_weights, caffe.TEST)
add_mean = np.array([0.0]*3)
add_var = np.array([255.0*255.0]*3)
add_scale = np.array([1.0])
net.params['batch_norm0'][0].data.flat = add_mean.flat
net.params['batch_norm0'][1].data.flat = add_var.flat
net.params['batch_norm0'][2].data.flat = add_scale.flat
res = net.forward()
net.save(model_weights)
```

# 3、再删除不兼容参数

## （1）将“Flatten”层删除“flatten\_param ”中的“end\_axis”参数

```
layer {
  name: "flatten3"
  type: "Flatten"
  bottom: "relu_blob57"
  top: "flatten_blob3"
  flatten_param {
    axis: 1
    end_axis: -1
  }
}
```