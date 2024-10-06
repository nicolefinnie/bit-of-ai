## Design Questions

### Design an end-to-end pipeline to take a high-level ML model (e.g., in PyTorch#) and deploy it on custom AI hardware. What would the stages of compilation, optimization, and deployment look like?


### Answer:

In our real project, we initially trained models in PyTorch and mirrored the architecture in a custom C++ inference framework. For the core operations like matrix multiplication and batch normalization, we leveraged the Eigen library. To load the weights into this C++ framework, we built a converter that transforms the PyTorch weights into a binary format that our inference engine can read, adding layer signatures to ensure correctness. This worked well for smaller models, and we used CMake to compile the inference framework for different target architectures, such as x86 (testing) and ARM Cortex M6 (target hardware).

However, this approach doesn't scale well for larger models like LLMs. For these, we moved to a modern pipeline using ONNX. After converting the PyTorch model to ONNX, we compile the ONNX model to a TensorRT format for inference on NVIDIA Orin chips, allowing for more efficient model deployment on embedded devices.

For model optimization, we employed an in-house Neural Architecture Search (NAS) tool to search for the best architecture that performs well on downstream tasks. To optimize the architecture for hardware, we could leverage tools like torch.fx to generate an Intermediate Representation (IR) of the model. This allows us to inspect specific operations for potential optimizations, such as avoiding unnecessary format conversions (e.g., switching between NHWC and NCHW), which can be detected and addressed in the IR.

Additionally, post-training quantization is also a key step in our optimization pipeline, helping to reduce the model size and improve efficiency without sacrificing too much accuracy.
