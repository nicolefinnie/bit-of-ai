## Quant it Until You Make it!


### How to run it?

You need to [install poetry](https://python-poetry.org/docs/) for installing python packages.

```sh
cd $PROJECT_ROOT_DIR
poetry run python pipeline.py resnet18

```

### What models does it support?

- Simple CNN that can deal with CIFAR-10
- ResNet 18
- GPT2 (but I don't have time to do calibration on the language data yet - TODO)

### What does it do?

- It asks you if you want to optimize the graph first (though I was told by a quantization expert that you should
try to quantize it first before optimizingthe graph )

```sh

$ python pipeline.py simplecnn
🧐 Should I trace the model 'resnet18' for you? (y/n) [y]: 
🔍 Should I analyze the model 'resnet18' for optimization? (y/n) [y]: 
🔍 Found optimization opportunity: Conv2d layer1.0.conv2 and BatchNorm2d layer1.0.bn2!
🔍 Found optimization opportunity: Conv2d layer1.1.conv2 and BatchNorm2d layer1.1.bn2!
🔍 Found optimization opportunity: Conv2d layer2.0.conv2 and BatchNorm2d layer2.0.bn2!
🔍 Found optimization opportunity: Conv2d layer2.1.conv2 and BatchNorm2d layer2.1.bn2!
🔍 Found optimization opportunity: Conv2d layer3.0.conv2 and BatchNorm2d layer3.0.bn2!
🔍 Found optimization opportunity: Conv2d layer3.1.conv2 and BatchNorm2d layer3.1.bn2!
🔍 Found optimization opportunity: Conv2d layer4.0.conv2 and BatchNorm2d layer4.0.bn2!
🔍 Found optimization opportunity: Conv2d layer4.1.conv2 and BatchNorm2d layer4.1.bn2!
⚙️  Should I optimize the model 'resnet18' for you? (y/n) [y]: 
Start to fuse Conv2d and BatchNorm2d...
⚙️  Start to fuse Conv2d and BatchNorm2d...
🛠️  Before optimization:
opcode         name                   target                                                      args                                   kwargs
-------------  ---------------------  ----------------------------------------------------------  -------------------------------------  --------
placeholder    x                      x                                                           ()                                     {}
call_module    conv1                  conv1                                                       (x,)                                   {}
call_module    bn1                    bn1                                                         (conv1,)                               {}
call_module    relu                   relu                                                        (bn1,)                                 {}
call_module    maxpool                maxpool                                                     (relu,)                                {}
call_module    layer1_0_conv1         layer1.0.conv1                                              (maxpool,)                             {}
call_module    layer1_0_bn1           layer1.0.bn1                                                (layer1_0_conv1,)                      {}
call_module    layer1_0_relu          layer1.0.relu                                               (layer1_0_bn1,)                        {}
call_module    layer1_0_conv2         layer1.0.conv2                                              (layer1_0_relu,)                       {}
call_module    layer1_0_bn2           layer1.0.bn2                                                (layer1_0_conv2,)                      {}
call_function  add                    <built-in function add>                                     (layer1_0_bn2, maxpool)                {}
call_module    layer1_0_relu_1        layer1.0.relu                                               (add,)                                 {}
call_module    layer1_1_conv1         layer1.1.conv1                                              (layer1_0_relu_1,)                     {}
call_module    layer1_1_bn1           layer1.1.bn1                                                (layer1_1_conv1,)                      {}
call_module    layer1_1_relu          layer1.1.relu                                               (layer1_1_bn1,)                        {}
call_module    layer1_1_conv2         layer1.1.conv2                                              (layer1_1_relu,)                       {}
call_module    layer1_1_bn2           layer1.1.bn2                                                (layer1_1_conv2,)                      {}
call_function  add_1                  <built-in function add>                                     (layer1_1_bn2, layer1_0_relu_1)        {}
call_module    layer1_1_relu_1        layer1.1.relu                                               (add_1,)                               {}
call_module    layer2_0_conv1         layer2.0.conv1                                              (layer1_1_relu_1,)                     {}
call_module    layer2_0_bn1           layer2.0.bn1                                                (layer2_0_conv1,)                      {}
call_module    layer2_0_relu          layer2.0.relu                                               (layer2_0_bn1,)                        {}
call_module    layer2_0_conv2         layer2.0.conv2                                              (layer2_0_relu,)                       {}
call_module    layer2_0_bn2           layer2.0.bn2                                                (layer2_0_conv2,)                      {}
call_module    layer2_0_downsample_0  layer2.0.downsample.0                                       (layer1_1_relu_1,)                     {}
call_module    layer2_0_downsample_1  layer2.0.downsample.1                                       (layer2_0_downsample_0,)               {}
call_function  add_2                  <built-in function add>                                     (layer2_0_bn2, layer2_0_downsample_1)  {}
call_module    layer2_0_relu_1        layer2.0.relu                                               (add_2,)                               {}
call_module    layer2_1_conv1         layer2.1.conv1                                              (layer2_0_relu_1,)                     {}
call_module    layer2_1_bn1           layer2.1.bn1                                                (layer2_1_conv1,)                      {}
call_module    layer2_1_relu          layer2.1.relu                                               (layer2_1_bn1,)                        {}
call_module    layer2_1_conv2         layer2.1.conv2                                              (layer2_1_relu,)                       {}
call_module    layer2_1_bn2           layer2.1.bn2                                                (layer2_1_conv2,)                      {}
call_function  add_3                  <built-in function add>                                     (layer2_1_bn2, layer2_0_relu_1)        {}
call_module    layer2_1_relu_1        layer2.1.relu                                               (add_3,)                               {}
call_module    layer3_0_conv1         layer3.0.conv1                                              (layer2_1_relu_1,)                     {}
call_module    layer3_0_bn1           layer3.0.bn1                                                (layer3_0_conv1,)                      {}
call_module    layer3_0_relu          layer3.0.relu                                               (layer3_0_bn1,)                        {}
call_module    layer3_0_conv2         layer3.0.conv2                                              (layer3_0_relu,)                       {}
call_module    layer3_0_bn2           layer3.0.bn2                                                (layer3_0_conv2,)                      {}
call_module    layer3_0_downsample_0  layer3.0.downsample.0                                       (layer2_1_relu_1,)                     {}
call_module    layer3_0_downsample_1  layer3.0.downsample.1                                       (layer3_0_downsample_0,)               {}
call_function  add_4                  <built-in function add>                                     (layer3_0_bn2, layer3_0_downsample_1)  {}
call_module    layer3_0_relu_1        layer3.0.relu                                               (add_4,)                               {}
call_module    layer3_1_conv1         layer3.1.conv1                                              (layer3_0_relu_1,)                     {}
call_module    layer3_1_bn1           layer3.1.bn1                                                (layer3_1_conv1,)                      {}
call_module    layer3_1_relu          layer3.1.relu                                               (layer3_1_bn1,)                        {}
call_module    layer3_1_conv2         layer3.1.conv2                                              (layer3_1_relu,)                       {}
call_module    layer3_1_bn2           layer3.1.bn2                                                (layer3_1_conv2,)                      {}
call_function  add_5                  <built-in function add>                                     (layer3_1_bn2, layer3_0_relu_1)        {}
call_module    layer3_1_relu_1        layer3.1.relu                                               (add_5,)                               {}
call_module    layer4_0_conv1         layer4.0.conv1                                              (layer3_1_relu_1,)                     {}
call_module    layer4_0_bn1           layer4.0.bn1                                                (layer4_0_conv1,)                      {}
call_module    layer4_0_relu          layer4.0.relu                                               (layer4_0_bn1,)                        {}
call_module    layer4_0_conv2         layer4.0.conv2                                              (layer4_0_relu,)                       {}
call_module    layer4_0_bn2           layer4.0.bn2                                                (layer4_0_conv2,)                      {}
call_module    layer4_0_downsample_0  layer4.0.downsample.0                                       (layer3_1_relu_1,)                     {}
call_module    layer4_0_downsample_1  layer4.0.downsample.1                                       (layer4_0_downsample_0,)               {}
call_function  add_6                  <built-in function add>                                     (layer4_0_bn2, layer4_0_downsample_1)  {}
call_module    layer4_0_relu_1        layer4.0.relu                                               (add_6,)                               {}
call_module    layer4_1_conv1         layer4.1.conv1                                              (layer4_0_relu_1,)                     {}
call_module    layer4_1_bn1           layer4.1.bn1                                                (layer4_1_conv1,)                      {}
call_module    layer4_1_relu          layer4.1.relu                                               (layer4_1_bn1,)                        {}
call_module    layer4_1_conv2         layer4.1.conv2                                              (layer4_1_relu,)                       {}
call_module    layer4_1_bn2           layer4.1.bn2                                                (layer4_1_conv2,)                      {}
call_function  add_7                  <built-in function add>                                     (layer4_1_bn2, layer4_0_relu_1)        {}
call_module    layer4_1_relu_1        layer4.1.relu                                               (add_7,)                               {}
call_module    avgpool                avgpool                                                     (layer4_1_relu_1,)                     {}
call_function  flatten                <built-in method flatten of type object at 0x7f0f7d83a500>  (avgpool, 1)                           {}
call_module    fc                     fc                                                          (flatten,)                             {}
output         output                 output                                                      (fc,)                                  {}
None
✅ After optimization:
opcode         name                   target                                                      args                                     kwargs
-------------  ---------------------  ----------------------------------------------------------  ---------------------------------------  --------
placeholder    x                      x                                                           ()                                       {}
call_module    conv1                  conv1                                                       (x,)                                     {}
call_module    relu                   relu                                                        (conv1,)                                 {}
call_module    maxpool                maxpool                                                     (relu,)                                  {}
call_module    layer1_0_conv1         layer1.0.conv1                                              (maxpool,)                               {}
call_module    layer1_0_relu          layer1.0.relu                                               (layer1_0_conv1,)                        {}
call_module    layer1_0_conv2         layer1.0.conv2                                              (layer1_0_relu,)                         {}
call_function  add                    <built-in function add>                                     (layer1_0_conv2, maxpool)                {}
call_module    layer1_0_relu_1        layer1.0.relu                                               (add,)                                   {}
call_module    layer1_1_conv1         layer1.1.conv1                                              (layer1_0_relu_1,)                       {}
call_module    layer1_1_relu          layer1.1.relu                                               (layer1_1_conv1,)                        {}
call_module    layer1_1_conv2         layer1.1.conv2                                              (layer1_1_relu,)                         {}
call_function  add_1                  <built-in function add>                                     (layer1_1_conv2, layer1_0_relu_1)        {}
call_module    layer1_1_relu_1        layer1.1.relu                                               (add_1,)                                 {}
call_module    layer2_0_conv1         layer2.0.conv1                                              (layer1_1_relu_1,)                       {}
call_module    layer2_0_relu          layer2.0.relu                                               (layer2_0_conv1,)                        {}
call_module    layer2_0_conv2         layer2.0.conv2                                              (layer2_0_relu,)                         {}
call_module    layer2_0_downsample_0  layer2.0.downsample.0                                       (layer1_1_relu_1,)                       {}
call_function  add_2                  <built-in function add>                                     (layer2_0_conv2, layer2_0_downsample_0)  {}
call_module    layer2_0_relu_1        layer2.0.relu                                               (add_2,)                                 {}
call_module    layer2_1_conv1         layer2.1.conv1                                              (layer2_0_relu_1,)                       {}
call_module    layer2_1_relu          layer2.1.relu                                               (layer2_1_conv1,)                        {}
call_module    layer2_1_conv2         layer2.1.conv2                                              (layer2_1_relu,)                         {}
call_function  add_3                  <built-in function add>                                     (layer2_1_conv2, layer2_0_relu_1)        {}
call_module    layer2_1_relu_1        layer2.1.relu                                               (add_3,)                                 {}
call_module    layer3_0_conv1         layer3.0.conv1                                              (layer2_1_relu_1,)                       {}
call_module    layer3_0_relu          layer3.0.relu                                               (layer3_0_conv1,)                        {}
call_module    layer3_0_conv2         layer3.0.conv2                                              (layer3_0_relu,)                         {}
call_module    layer3_0_downsample_0  layer3.0.downsample.0                                       (layer2_1_relu_1,)                       {}
call_function  add_4                  <built-in function add>                                     (layer3_0_conv2, layer3_0_downsample_0)  {}
call_module    layer3_0_relu_1        layer3.0.relu                                               (add_4,)                                 {}
call_module    layer3_1_conv1         layer3.1.conv1                                              (layer3_0_relu_1,)                       {}
call_module    layer3_1_relu          layer3.1.relu                                               (layer3_1_conv1,)                        {}
call_module    layer3_1_conv2         layer3.1.conv2                                              (layer3_1_relu,)                         {}
call_function  add_5                  <built-in function add>                                     (layer3_1_conv2, layer3_0_relu_1)        {}
call_module    layer3_1_relu_1        layer3.1.relu                                               (add_5,)                                 {}
call_module    layer4_0_conv1         layer4.0.conv1                                              (layer3_1_relu_1,)                       {}
call_module    layer4_0_relu          layer4.0.relu                                               (layer4_0_conv1,)                        {}
call_module    layer4_0_conv2         layer4.0.conv2                                              (layer4_0_relu,)                         {}
call_module    layer4_0_downsample_0  layer4.0.downsample.0                                       (layer3_1_relu_1,)                       {}
call_function  add_6                  <built-in function add>                                     (layer4_0_conv2, layer4_0_downsample_0)  {}
call_module    layer4_0_relu_1        layer4.0.relu                                               (add_6,)                                 {}
call_module    layer4_1_conv1         layer4.1.conv1                                              (layer4_0_relu_1,)                       {}
call_module    layer4_1_relu          layer4.1.relu                                               (layer4_1_conv1,)                        {}
call_module    layer4_1_conv2         layer4.1.conv2                                              (layer4_1_relu,)                         {}
call_function  add_7                  <built-in function add>                                     (layer4_1_conv2, layer4_0_relu_1)        {}
call_module    layer4_1_relu_1        layer4.1.relu                                               (add_7,)                                 {}
call_module    avgpool                avgpool                                                     (layer4_1_relu_1,)                       {}
call_function  flatten                <built-in method flatten of type object at 0x7f0f7d83a500>  (avgpool, 1)                             {}
call_module    fc                     fc                                                          (flatten,)                               {}
output         output                 output                                                      (fc,)                                    {}
None

```

- It asks you if you want to perform post quantization on the optimized graph

```sh
⚙️  Should I quantize the model for you? (y/n) [y]: 
⚙️ Analyzing quantization strategies layer by layer.
🟢 Layer: 
   ➡️  Strategy:
   - Default: Facebook General Matrix Multiplication (FBGEMM)
   - Rationale: Default low-precision strategy for x86.

🟢 Layer: conv1
   ➡️  Strategy:
   - Weight: int8 (per-channel quantization)
   - Activation: int8 (per-tensor quantization)
   - Rationale: Optimized for target hardware (Conv2D sensitive).

🟢 Layer: pool
   ➡️  Strategy:
   - Activation: int8 (per-tensor quantization)
   - Rationale: Pooling layers don't require weight quantization.

🟢 Layer: conv2
   ➡️  Strategy:
   - Weight: int8 (per-channel quantization)
   - Activation: int8 (per-tensor quantization)
   - Rationale: Optimized for target hardware (Conv2D sensitive).

🟢 Layer: fc1
   ➡️  Strategy:
   - Weight: customized int4
   - Activation: int8 (per-tensor quantization)
   - Rationale: Optimized for Linear layers with customized int4 weight.

🟢 Layer: fc2
   ➡️  Strategy:
   - Weight: customized int4
   - Activation: int8 (per-tensor quantization)
   - Rationale: Optimized for Linear layers with customized int4 weight.

/home/fin2rng/.cache/pypoetry/virtualenvs/bit-of-ai-ih5nOhlw-py3.10/lib/python3.10/site-packages/torch/ao/quantization/quantize.py:320: UserWarning: None of the submodule got qconfig applied. Make sure you passed correct configuration through `qconfig_dict` or by assigning the `.qconfig` attribute directly on submodules
  warnings.warn("None of the submodule got qconfig applied. Make sure you "
Files already downloaded and verified
🔄 Calibrating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [00:05<00:00, 209.31it/s]
Files already downloaded and verified
🔄 Evaluating Quantization Error: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 313/313 [00:01<00:00, 186.64it/s]
📊 Average quantization error over dataset: 0.000000133
```

- If the post quantization error is too high, it asks you want to try QAT

```sh
⚙️  Post quantization error is higher than the threshold 1e-07,Should I perform QAT for you? (y/n) [y]: 
Files already downloaded and verified
🟢 Layer: 
   ➡️  Strategy:
   - Default: Facebook General Matrix Multiplication (FBGEMM)
   - Rationale: Default low-precision strategy for x86.

🟢 Layer: conv1
   ➡️  Strategy:
   - Weight: int8 (per-channel quantization)
   - Activation: int8 (per-tensor quantization)
   - Rationale: Optimized for target hardware (Conv2D sensitive).

🟢 Layer: pool
   ➡️  Strategy:
   - Activation: int8 (per-tensor quantization)
   - Rationale: Pooling layers don't require weight quantization.

🟢 Layer: conv2
   ➡️  Strategy:
   - Weight: int8 (per-channel quantization)
   - Activation: int8 (per-tensor quantization)
   - Rationale: Optimized for target hardware (Conv2D sensitive).

🟢 Layer: fc1
   ➡️  Strategy:
   - Weight: customized int4
   - Activation: int8 (per-tensor quantization)
   - Rationale: Optimized for Linear layers with customized int4 weight.

🟢 Layer: fc2
   ➡️  Strategy:
   - Weight: customized int4
   - Activation: int8 (per-tensor quantization)
   - Rationale: Optimized for Linear layers with customized int4 weight.

🔄 Finetuning Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [00:07<00:00, 170.39it/s]
Epoch 1/5, Loss: 4.15846953125
🔄 Finetuning Epoch 2: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [00:07<00:00, 172.53it/s]
Epoch 2/5, Loss: 4.158469140625
🔄 Finetuning Epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [00:07<00:00, 172.24it/s]
Epoch 3/5, Loss: 4.158470703125
🔄 Finetuning Epoch 4: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [00:07<00:00, 169.51it/s]
Epoch 4/5, Loss: 4.15846640625
🔄 Finetuning Epoch 5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [00:07<00:00, 168.29it/s]
Epoch 5/5, Loss: 4.158467578125
🏁 Finetuning completed after 5 epochs!
Files already downloaded and verified
🔄 Evaluating Quantization Error: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 313/313 [00:01<00:00, 186.53it/s]
📊 Average quantization error over dataset: 0.000000133
🎉 QAT is more effective than post quantization with this dataset.
```

