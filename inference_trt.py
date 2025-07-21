import os
from time import time
import cv2
import numpy as np
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动管理CUDA上下文的创建和销毁


# 假设所有辅助函数 (get_gaussian_kernel, cosine_similarity 等) 保持不变。
# 为保证代码完整性，此处一并提供。

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = np.arange(kernel_size)
    x_grid = np.repeat(x_coord, kernel_size).reshape(kernel_size, kernel_size)
    y_grid = x_grid.T
    xy_grid = np.stack([x_grid, y_grid], axis=-1).astype(np.float32)
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * np.pi * variance)) * np.exp(-np.sum((xy_grid - mean) ** 2., axis=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.reshape(kernel_size, kernel_size)
    return gaussian_kernel


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x1_norm = np.linalg.norm(x1, axis=dim, keepdims=True).clip(min=eps)
    x2_norm = np.linalg.norm(x2, axis=dim, keepdims=True).clip(min=eps)
    dot_product = np.sum(x1 * x2, axis=dim, keepdims=True)
    similarity = dot_product / (x1_norm * x2_norm)
    similarity = (np.round(1 - similarity, decimals=4))
    return np.squeeze(similarity, axis=dim)


def resize_with_align_corners(image, out_size):
    # 此函数期望一个二维 numpy 数组
    in_height, in_width = image.shape
    out_height, out_width = out_size
    x_indices = np.linspace(0, in_width - 1, out_width).astype(np.float32)
    y_indices = np.linspace(0, in_height - 1, out_height).astype(np.float32)
    map_x, map_y = np.meshgrid(x_indices, y_indices)
    resized_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return resized_image


def resize_without_align_corners(image, out_size):
    # 此函数假设输入是一个 4D numpy 数组 (B, C, H, W)
    batch_size, channels, _, _ = image.shape
    out_height, out_width = out_size
    resized_images = np.zeros((batch_size, channels, out_height, out_width), dtype=image.dtype)
    for b in range(batch_size):
        for c in range(channels):
            resized_images[b, c] = cv2.resize(image[b, c], (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    return resized_images


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = cosine_similarity(fs, ft)  # 假设 a_map 此时是 (B, H, W) 或 (H, W)

        # 核心修改：在添加额外维度之前进行 resize
        # 如果 a_map 是 (B, H, W)，你需要遍历 Batch 维度或处理单个样本
        # 假设你的 Batch Size 是 1，并且 cosine_similarity 返回的 a_map 是 (H, W)
        # 如果它返回的是 (1, H, W)，你需要先 Squeeze
        if a_map.ndim == 3 and a_map.shape[0] == 1:
            a_map_2d = np.squeeze(a_map, axis=0)  # 变为 (H, W)
        elif a_map.ndim == 2:
            a_map_2d = a_map  # 已经是 (H, W)
        else:
            raise ValueError(f"a_map 形状不符合预期，期望 (H, W) 或 (1, H, W)，实际为 {a_map.shape}")

        a_map_resized = resize_with_align_corners(a_map_2d, out_size)

        # 现在，将 resize 后的 a_map 添加 Batch 和 Channel 维度以保持一致性
        a_map_resized = np.expand_dims(a_map_resized, axis=0)  # 添加 Channel 维度
        a_map_resized = np.expand_dims(a_map_resized, axis=0)  # 添加 Batch 维度
        a_map_list.append(a_map_resized)

    anomaly_map = np.round(np.mean(np.concatenate(a_map_list, axis=1), axis=1, keepdims=True), decimals=4)
    return anomaly_map, a_map_list


class TRT_inference:
    """
    使用 PyCUDA 进行内存管理的 TensorRT 推理类 (已更新 API)。
    """

    def __init__(self, trt_engine_path: str):
        self.trt_engine_path = trt_engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.output_names = []  # 新增：用于存储输出张量的名称
        self.load_engine()

    def load_engine(self):
        """从文件加载 TensorRT 引擎并分配内存 (使用新版 API)。"""
        runtime = trt.Runtime(self.logger)
        with open(self.trt_engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if not self.engine:
            raise RuntimeError(f"从 {self.trt_engine_path} 加载 TensorRT 引擎失败")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        print("--- 引擎绑定信息 ---")
        # 使用新版 API (num_io_tensors 和 get_tensor_*)
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            tensor_mode = self.engine.get_tensor_mode(tensor_name)

            # 打印每个绑定的信息，帮助你调试和确认第5个输出是什么
            print(f"  索引 {i}: 名称='{tensor_name}', 模式={tensor_mode}, 形状={tensor_shape}, 类型={tensor_dtype}")

            volume = abs(trt.volume(tensor_shape))
            host_mem = cuda.pagelocked_empty(volume, dtype=tensor_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            mem_info = {
                'host_mem': host_mem,
                'device_mem': device_mem,
                'shape': tensor_shape,
                'name': tensor_name,
                'dtype': tensor_dtype
            }
            if tensor_mode == trt.TensorIOMode.INPUT:
                self.inputs.append(mem_info)
            else:  # 是输出
                self.outputs.append(mem_info)
                self.output_names.append(tensor_name)  # 存储输出名称
        print("--------------------")

    def infer(self, input_data: np.ndarray):
        """
        对输入数据执行推理。
        """
        if len(self.inputs) != 1:
            raise ValueError(f"期望 1 个输入, 但检测到 {len(self.inputs)} 个")

        input_binding = self.inputs[0]

        if input_data.dtype != input_binding['dtype']:
            input_data = input_data.astype(input_binding['dtype'])

        np.copyto(input_binding['host_mem'], input_data.ravel())

        cuda.memcpy_htod_async(input_binding['device_mem'], input_binding['host_mem'], self.stream)

        # 注意: execute_async_v2 的 bindings 列表长度和顺序必须与引擎的 I/O 张量总数一致
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out_binding in self.outputs:
            cuda.memcpy_dtoh_async(out_binding['host_mem'], out_binding['device_mem'], self.stream)

        self.stream.synchronize()

        results = []
        for out_binding in self.outputs:
            # get_tensor_shape 获取的是编译时的形状，对于动态形状，需要从 context 获取运行时形状
            binding_index = self.engine.get_binding_index(out_binding['name'])
            output_shape = self.context.get_binding_shape(binding_index)
            output_data = out_binding['host_mem'].reshape(output_shape)
            results.append(output_data.copy())

        return results

    def __del__(self):
        """释放CUDA内存和其他资源。"""
        if self.stream:
            self.stream.synchronize()

        for mem_info in self.inputs + self.outputs:
            if mem_info.get('device_mem'):
                mem_info['device_mem'].free()

def pre_process(image_path, input_size):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    modified_image = cv2.resize(image, (input_size, input_size))
    modified_image = modified_image.astype(np.float32) / 255.0
    modified_image = (modified_image - mean) / std
    modified_image = np.transpose(modified_image, (2, 0, 1))
    modified_image = np.expand_dims(modified_image, axis=0).astype(np.float32)
    return modified_image


def visualize(output_folder_path, image_path, anomaly_map_image):
    origin_image_bgr = cv2.imread(image_path)  # 直接使用cv2读取的BGR图像
    origin_height, origin_width = origin_image_bgr.shape[:2]

    heat_map = min_max_norm(anomaly_map_image)
    heat_map_resized = cv2.resize(heat_map, (origin_width, origin_height))
    heat_map_image = cvt2heatmap(heat_map_resized * 255)

    overlay = cv2.addWeighted(origin_image_bgr, 0.6, heat_map_image, 0.4, 0)

    base_filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_folder_path, f"overlay_{base_filename}"), overlay)
    cv2.imwrite(os.path.join(output_folder_path, f"heatmap_{base_filename}"), heat_map_image)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    if a_max - a_min == 0:
        return np.zeros_like(image, dtype=np.float32)
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heat_map = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heat_map


def main_process(image_folder_path, output_folder_path, trt_engine_path, input_size, max_ratio, visualize_output):
    os.makedirs(output_folder_path, exist_ok=True)
    all_files = sorted(
        [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    trt_model = TRT_inference(trt_engine_path)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4)

    print(f"\n开始对 {len(all_files)} 张图片进行推理...")
    for idx, file in enumerate(all_files):
        start_time = time()
        image_path = os.path.join(image_folder_path, file)
        base_name = os.path.splitext(file)[0]
        input_image = pre_process(image_path, input_size)

        # all_outputs 的顺序与 trt_model.outputs (元数据列表) 的顺序一致
        all_outputs = trt_model.infer(input_image)

        # ==================== 新的分配逻辑 ====================
        # 1. 根据形状过滤掉不需要的输出（比如标量）
        feature_maps = []
        for i, output_data in enumerate(all_outputs):
            # 我们只保留4维的特征图 (B, C, H, W)
            if len(output_data.shape) == 4:
                feature_maps.append(output_data)

        # 2. 检查过滤后是否剩下4个特征图
        if len(feature_maps) != 4:
            raise ValueError(f"期望找到 4 个特征图输出, 但在过滤后实际找到 {len(feature_maps)} 个。请检查模型结构。")

        # 3. 假设前两个是en，后两个是de
        en = feature_maps[0:2]
        de = feature_maps[2:4]
        # =======================================================

        anomaly_map, _ = cal_anomaly_maps(en, de, input_size)
        anomaly_map = resize_without_align_corners(anomaly_map, (256, 256))
        anomaly_map = anomaly_map[0, 0, :, :]
        anomaly_map = cv2.filter2D(anomaly_map, -1, gaussian_kernel)
        anomaly_map = np.round(anomaly_map, decimals=4)

        anomaly_map_image_for_viz = anomaly_map.copy()

        if max_ratio == 0:
            sp_score = np.max(anomaly_map)
        else:
            anomaly_map_flat = anomaly_map.ravel()
            num_elements_to_take = max(1, int(len(anomaly_map_flat) * max_ratio))
            sp_score = np.sort(anomaly_map_flat)[-num_elements_to_take:].mean()

        if visualize_output:
            visualize(output_folder_path, image_path, anomaly_map_image_for_viz)

        end_time = time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"{idx + 1:05d} | {elapsed_time: >7.2f} ms | 图片: {base_name}, 异常分数: {sp_score:.4f}")

    print("推理完成。")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='用于异常检测的 TensorRT 推理脚本')
    parser.add_argument('--image_folder_path', type=str, default=r'F:\INP-Former-main\Mydatasets\onnx_inf\front', help='输入图片文件夹的路径')
    parser.add_argument('--output_folder_path', type=str, default=r'F:\INP-Former-main\Mydatasets\onnx_inf\front_results_trt', help='保存可视化结果的路径')
    parser.add_argument('--trt_engine_path', type=str,default=r'F:\INP-Former-main\saved_results\INP-Former-Single-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=392_Crop=392_INP_num=6\front\model_final_32.engine' , help='TensorRT 引擎文件 (.engine) 的路径')
    parser.add_argument('--input_size', type=int, default=392, help='模型推理的输入尺寸')
    parser.add_argument('--max_ratio', type=float, default=0.01, help='用于计算异常分数的最大比率')
    parser.add_argument('--visualize_output', action='store_true', help='设置此项以可视化输出结果')
    args = parser.parse_args()

    main_process(
        image_folder_path=args.image_folder_path,
        output_folder_path=args.output_folder_path,
        trt_engine_path=args.trt_engine_path,
        input_size=args.input_size,
        max_ratio=args.max_ratio,
        visualize_output=args.visualize_output
    )