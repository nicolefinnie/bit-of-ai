"""This module is a tutorial on how to implement a simple quantization aware training (QAT) in PyTorch.
without using `torch.quantization` module.

Everyone can QAT la la la!
"""
import torch
import torch.nn as nn


class QuantizedLayer:
    def __init__(self, quantization_bit: int):
        self.quantization_bit = quantization_bit
        self.scale = None
        self.zero_point = 0.0
        #(2^bit-1 -  (2^bit)) according to  2's compliment
        # e.g. 8 bit would be (max-min) / (127 - (-128))) 
        self.min = -2**(quantization_bit)
        self.max = 2**(quantization_bit) - 1
    
    def calibrate(self, x: torch.Tensor):
        """Calibrate the scale and zero point of the layer using the first test batches.
        Symmetrically Quantize the pass-in tensor.
        
        For QAT, it doesn't make sense to do asymmetric quantization since quantization errors will be learned
        by the model.
        We still keep the tensor in the original precision, but we quantize the value to a lower precision.
        In this case the tensor is still in float32.
        """
        x_min = x.min().item()
        x_max = x.max().item()
        self.scale = (x_max - x_min) / (self.max - self.min)


    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x_q = torch.clamp((x/self.scale).round() + self.zero_point,self.min, self.max)
        # We want to operate x_q in the same precision during training
        assert(x_q.dtype == x.dtype)
        return x_q

    def dequantize_tensor(self, x_q: torch.Tensor) -> torch.Tensor:
        """Dequantize the pass-in tensor back to float32."""
        x = (x_q.to(torch.float32) - self.zero_point)*self.scale
        assert(x.dtype == x_q.dtype)
        return x
    
class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, quantization_layer, **kwargs):
        super(QuantizedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.quantization = quantization_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """For QAT, we always quantize the input tensor, weight, and dequantize the output
        so that the model can learn the quantization errors for each layer in a high precision
        """
        x_q = self.quantization.quantize_tensor(x)
        weight_q = self.quantization.quantize_tensor(self.conv.weight)
        if self.conv.bias is not None:
            bias_q = self.quantization.quantize_tensor(self.conv.bias)
        else:
            bias_q = None
        output_q = nn.functional.conv2d(x_q, weight_q, bias_q, self.conv.stride, self.conv.padding, 
                                        self.conv.dilation, self.conv.groups)
        de_output = self.quantization.dequantize_tensor(output_q)

        return de_output

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, quantization_layer, **kwargs):
        super(QuantizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.quantization = quantization_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       # quantize intput tensor
       x_q = self.quantization.quantize_tensor(x)
       weight_q = self.quantization.quantize_tensor(self.linear.weight)
       if self.linear.bias is not None:
           bias_q = self.quantization.quantize_tensor(self.linear.bias)
       else:
           bias_q = None
       output_q = nn.functional.linear(x_q, weight_q, bias_q)
       de_output = self.quantization.dequantize_tensor(output_q)

       return de_output

class QunatizedReLU(nn.Module):
    def __init__(self, quantization_layer):
        super(QunatizedReLU, self).__init__()
        self.quantization = quantization_layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The input has been quantized, so we don't need to quantize it again.
        We just need to map to the quantized range"""
        x_q = self.quantization.quantize_tensor(x)
        relu_q = torch.clamp(x_q.to(torch.float32), min=self.quantization.zero_point).to(x.dtype)
        de_relu = self.quantization.dequantize_tensor(relu_q)
        return de_relu


class QuantizedModel(nn.Module):
    """
        input shape = (B, 3, 32, 32)
        output feature = (input feature + 2*padding - kernel_size) / stride + 1
        so after the first layer [B, 16, 30, 30]
        after the second layer [B, 32, 28, 28]
    """
    def __init__(self):
        super(QuantizedModel, self).__init__()
        self.quantization_layer = QuantizedLayer(quantization_bit=8)
        self.conv1 = QuantizedConv2d(in_channels=3, out_channels=16, kernel_size=3, quantization_layer=self.quantization_layer)
        self.conv2 = QuantizedConv2d(in_channels=16, out_channels=4, kernel_size=3, quantization_layer=self.quantization_layer)
        self.relu = QunatizedReLU(quantization_layer=self.quantization_layer)
        self.linear1 = QuantizedLinear(in_features=4*28*28, out_features=16, quantization_layer=self.quantization_layer)
        self.linear2 = QuantizedLinear(in_features=16, out_features=1, quantization_layer=self.quantization_layer)

    def forward(self, x: torch.Tensor):
        x_q = self.conv1(x)
        x_q = self.relu(x_q)
        x_q = self.conv2(x_q)
        x_q = self.relu(x_q)
        x_q = x_q.flatten(1)
        x_q = self.linear1(x_q)
        x_q = self.relu(x_q)
        x_q = self.linear2(x_q)
        return x_q
    
    def calibrate_model(self, x: torch.Tensor):
        """Calibrate the model using the first test batches.
        
        It makes sense that every layer has its scale and zero point.
        """
        print('Calibrating the model...')
        self.quantization_layer.calibrate(x)
        print(f'Quantization scale: {self.quantization_layer.scale}')
        print(f'Quantization zero point: {self.quantization_layer.zero_point}')
        

def post_quantization(output: torch.Tensor, scale: float = 1.0, zero_point: float = 0.0) -> torch.Tensor:
    """Post quantization of the output tensor to calculate the error."""
  
    output_q = torch.clamp((output / scale).round() + zero_point, -128, 127).to(torch.int8)
    output_dequantized = (output_q.to(torch.float32) - zero_point) * scale
    quantization_error = torch.abs(output - output_dequantized)
    print(f'Quantization Error: {quantization_error.mean().item()}')

    # Verify how close the post-quantized output is to the original output
    if quantization_error.mean().item() < 1e-3:
        print("QAT is cool stuff")
    else:
        # I didn't train the model, hahah
        print("The error is way too high, can you tune/retrain your model, dude? .")


def qat_eval():
    """ This is a simple test to show how to do quantization aware training (QAT) in PyTorch.
    I know the training loop is still missing, that's why the quantization error is high.
    I'll add it when I have time again, but you can see the model output would be quantization-friendly
    because it already lies between -128 and 127 for int8 quantization. 

    If you use the coolest seed 42, this is what you get:

        Quantization scale: 0.017354835502788744
        Quantization zero point: 0.0
        De-quantized output min: -123.86145782470703, max: 92.9872055053711
        Quantization Error: 0.24925534427165985
    """
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QuantizedModel().to(device)
    model.eval()
    input_tensor = torch.randn(64, 3, 32, 32).to(device)
    model.calibrate_model(input_tensor)
    
    # Test only to show the de-quantized output is quantization-friendly, e.g. it can be easily quantized to 8-bit
    with torch.no_grad():
        output = model(input_tensor)
        print(f'De-quantized output min: {output.min()}, max: {output.max()}')
        post_quantization(output)

if __name__=='__main__':
    qat_eval()