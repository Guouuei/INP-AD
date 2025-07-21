import tensorrt as trt
import os

def build_tensorrt_engine(onnx_model_path, engine_file_path, precision='fp16'):
    """
    从ONNX模型构建TensorRT引擎。

    参数：
        onnx_model_path (str): 输入ONNX模型的路径。
        engine_file_path (str): 保存输出TensorRT引擎的路径。
        precision (str): 'fp32', 'fp16' 或 'int8'。默认为 'fp16'。
                         注意：'int8' 需要校准。
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # 您可以将其更改为 INFO 以获取更详细的输出

    # 创建构建器和网络
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        print(f"正在从 {onnx_model_path} 加载ONNX模型...")
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("错误：解析ONNX文件失败。")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("ONNX模型加载成功。")

        # 配置构建器
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB，根据您的模型需求调整

        # 设置精度
        if precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.flags |= 1 << int(trt.BuilderFlag.FP16)
                print("正在使用FP16精度构建引擎。")
            else:
                print("警告：此平台不支持FP16，回退到FP32。")
        elif precision == 'int8':
            if builder.platform_has_fast_int8:
                config.flags |= 1 << int(trt.BuilderFlag.INT8)
                # 如果使用INT8，您需要在此处实现Int8校准器。
                # 为简单起见，本示例不包含校准器。
                print("正在使用INT8精度构建引擎（需要校准）。")
                print("警告：本示例中未实现INT8校准。")
            else:
                print("警告：此平台不支持INT8。")

        print("正在构建TensorRT引擎...")
        engine = builder.build_engine(network, config)

        if engine:
            print("TensorRT引擎构建成功。")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print(f"TensorRT引擎已保存到 {engine_file_path}")
        else:
            print("错误：构建TensorRT引擎失败。")

        return engine

if __name__ == '__main__':
    # 假设ONNX模型在同一目录中，或者您提供完整路径
    # 根据您的ONNX导出代码的输出。
    base_path = r"F:\INP-Former-main\saved_results\INP-Former-Single-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=392_Crop=392_INP_num=6\front"
    onnx_model_name = "model_final.onnx" # 根据您的 export_onnx 函数
    engine_name = "model_final_int8.engine"

    onnx_model_path = os.path.join(base_path, onnx_model_name)
    engine_file_path = os.path.join(base_path, engine_name)

    # 在尝试转换之前，请确保ONNX文件存在
    if not os.path.exists(onnx_model_path):
        print(f"错误：ONNX模型未找到于 {onnx_model_path}。请先运行ONNX导出。")
    else:
        build_tensorrt_engine(onnx_model_path, engine_file_path, precision='int8') # 或 'fp32'