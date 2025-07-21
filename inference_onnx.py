import os
from time import time
import cv2
import numpy as np
import onnxruntime
import argparse

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
    in_height, in_width = image.shape[-2:]
    out_height, out_width = out_size

    x_indices = np.linspace(0, in_width - 1, out_width).astype(np.float32)
    y_indices = np.linspace(0, in_height - 1, out_height).astype(np.float32)
    map_x, map_y = np.meshgrid(x_indices, y_indices)

    resized_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return resized_image

def resize_without_align_corners(image, out_size):
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

    for idx, i in enumerate(range(len(ft_list))):
        fs = fs_list[i]
        ft = ft_list[i]

        a_map = cosine_similarity(fs, ft)
        a_map = np.squeeze(a_map)
        a_map = resize_with_align_corners(a_map, out_size)
        a_map = np.expand_dims(a_map, axis=0)
        a_map = np.expand_dims(a_map, axis=0)
        a_map_list.append(a_map)

    anomaly_map = np.round(np.mean(np.concatenate(a_map_list, axis=1), axis=1, keepdims=True), decimals=4)

    return anomaly_map, a_map_list

class ONNX_inference:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.options, self.providers = self.set_options()
        self.session, self.inputs, self.outputs = self.load_model()

    def set_options(self):
        options = onnxruntime.SessionOptions()
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested', 'cudnn_conv_algo_search': 'HEURISTIC'}), 'CPUExecutionProvider']

        return options, providers

    def load_model(self):
        session = onnxruntime.InferenceSession(self.model_file_path, self.options, self.providers)
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]

        return session, inputs, outputs


def pre_process(image_path, input_size):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    modified_image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    modified_image = modified_image.astype(np.float32) / 255.0
    modified_image = (modified_image - mean) / std
    modified_image = np.transpose(modified_image, (2, 0, 1))
    modified_image = np.expand_dims(modified_image, axis=0).astype(np.float32)

    return modified_image

def visualize(output_folder_path, image_path, anomaly_map_image):
    origin_image = cv2.imread(image_path)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    origin_height, origin_width = origin_image.shape[:2]
    
    heat_map = min_max_norm(anomaly_map_image)
    heat_map_resized = cv2.resize(heat_map, (origin_width, origin_height))
    heat_map_image = cvt2heatmap(heat_map_resized * 255)

    overlay = cv2.addWeighted(origin_image, 0.6, heat_map_image, 0.4, 0)

    overlay_save_path = os.path.join(output_folder_path, f"overlay_{os.path.basename(image_path)}")
    cv2.imwrite(overlay_save_path, overlay)

    heat_map_save_path = os.path.join(output_folder_path, f"heatmap_{os.path.basename(image_path)}")
    cv2.imwrite(heat_map_save_path, heat_map_image)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def cvt2heatmap(gray):
    heat_map = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heat_map

def main_process(image_folder_path, output_folder_path, onnx_model_path, input_size, max_ratio, visualize_output):
    
    os.makedirs(output_folder_path, exist_ok=True)
    
    all_files = os.listdir(image_folder_path)
    
    onnx_model = ONNX_inference(onnx_model_path)
    
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4)

    for idx, file in enumerate(all_files):
        start_time = time()
        image_path = os.path.join(image_folder_path, file)
        base_name = file.split(".")[0]
        input_image = pre_process(image_path, input_size)
        
        outputs = onnx_model.session.run(onnx_model.outputs, {onnx_model.inputs[0]: input_image})

        en = outputs[0:2]
        de = outputs[2:4]

        anomaly_map, _ = cal_anomaly_maps(en, de, input_size)
        anomaly_map = resize_without_align_corners(anomaly_map, (256, 256))

        anomaly_map = anomaly_map[0, 0, :, :]
        anomaly_map = cv2.filter2D(anomaly_map, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT)
        anomaly_map = np.round(anomaly_map, decimals=4)
        anomaly_map_image = anomaly_map
        
        if max_ratio == 0:
            sp_score = np.max(anomaly_map.ravel())
        else:
            anomaly_map = anomaly_map.ravel()
            sp_score = np.sort(anomaly_map)[-int(anomaly_map.shape[0] * max_ratio):]
            sp_score = sp_score.mean()

        if visualize_output:
            visualize(output_folder_path, image_path, anomaly_map_image)
        
        end_time = time()
        elapsed_time = (end_time - start_time) * 1000
        
        print(f"{idx:05d} | {elapsed_time} ms | Image: {base_name}, Anomaly Score: {sp_score:.4f}")

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='ONNX Inference for Anomaly Detection')
    
    parser.add_argument('--image_folder_path', type=str, default=r'F:\INP-Former-main\Mydatasets\onnx_inf\front', help='Path to input image folder')
    parser.add_argument('--output_folder_path', type=str,default=r'F:\INP-Former-main\Mydatasets\onnx_inf\front_results',  help='Path to save visualized outputs')
    parser.add_argument('--onnx_model_path', type=str,default=r'F:\INP-Former-main\saved_results\INP-Former-Single-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=392_Crop=392_INP_num=6\front\model_final.onnx',  help='Path to the ONNX model file')
    parser.add_argument('--input_size', type=int, default=392, help='Input size for model inference')
    parser.add_argument('--max_ratio', type=float, default=0.01, help='Max ratio used for score thresholding')
    parser.add_argument('--visualize_output', action='store_true', help='Flag to visualize the results')

    args = parser.parse_args()

    main_process(
        image_folder_path=args.image_folder_path,
        output_folder_path=args.output_folder_path,
        onnx_model_path=args.onnx_model_path,
        input_size=args.input_size,
        max_ratio=args.max_ratio,
        visualize_output=args.visualize_output
    )