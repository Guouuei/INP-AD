import os
import shutil


def find_and_move_matched_masks(folder1, folder2, destination_folder):
    """
    在 folder1 中查找 xxx_mask.bmp 文件，
    如果在 folder2 中找到对应的 xxx.bmp 文件，则将 xxx_mask.bmp 剪切到 destination_folder。

    参数:
    folder1 (str): 第一个文件夹的路径 (包含 _mask.bmp 文件)。
    folder2 (str): 第二个文件夹的路径 (包含 .bmp 文件)。
    destination_folder (str): 匹配的 _mask.bmp 文件将被移动到的目标文件夹。
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        try:
            os.makedirs(destination_folder)
            print(f"创建目标文件夹: {destination_folder}")
        except OSError as e:
            print(f"创建目标文件夹 {destination_folder} 失败: {e}")
            return

    # 检查 folder1 和 folder2 是否存在
    if not os.path.isdir(folder1):
        print(f"错误: 文件夹1 '{folder1}' 不存在。")
        return
    if not os.path.isdir(folder2):
        print(f"错误: 文件夹2 '{folder2}' 不存在。")
        return

    print(f"开始在 '{folder1}' 中查找 *_mask.bmp 文件...")
    moved_files_count = 0

    # 遍历 folder1 中的所有文件
    for filename_mask in os.listdir(folder1):
        if filename_mask.endswith("_mask.bmp"):
            # 提取 xxx 部分 (例如从 xxx_mask.bmp 提取 xxx)
            prefix = filename_mask[:-9]  # 移除 "_mask.bmp" (9个字符)

            # 构建在 folder2 中对应的文件名
            corresponding_filename_bmp = prefix + ".bmp"
            path_corresponding_bmp = os.path.join(folder2, corresponding_filename_bmp)

            # 检查对应的 .bmp 文件是否存在于 folder2
            if os.path.isfile(path_corresponding_bmp):
                print(f"找到匹配: '{filename_mask}' (在 {folder1}) -> '{corresponding_filename_bmp}' (在 {folder2})")

                source_mask_path = os.path.join(folder1, filename_mask)
                destination_mask_path = os.path.join(destination_folder, filename_mask)

                try:
                    # 剪切 (移动) _mask.bmp 文件
                    shutil.move(source_mask_path, destination_mask_path)
                    print(f"已将 '{filename_mask}' 从 '{folder1}' 移动到 '{destination_folder}'")
                    moved_files_count += 1
                except Exception as e:
                    print(f"移动文件 '{filename_mask}' 失败: {e}")
            # else:
            #     print(f"未找到 '{filename_mask}' 在 '{folder1}' 的对应文件 '{corresponding_filename_bmp}' 在 '{folder2}'")

    if moved_files_count == 0:
        print("没有找到匹配的文件，或者没有文件被移动。")
    else:
        print(f"操作完成。总共移动了 {moved_files_count} 个文件。")


if __name__ == "__main__":
    # --- 请在这里配置您的文件夹路径 ---
    # 例如: folder1_path = "C:/Users/YourUser/Desktop/Folder_With_Masks"
    # 例如: folder2_path = "/mnt/data/Folder_With_Images"

    folder1_path = r"F:\INP-Former-main\Mydatasets\dianrong\front\ground_truth\abnormal"  # 包含 xxx_mask.bmp 文件的文件夹
    folder2_path = r"F:\INP-Former-main\Mydatasets\dianrong\front\test\abnormal"  # 包含 xxx.bmp 文件的文件夹

    # 匹配的 _mask.bmp 文件将被移动到这个文件夹
    # 这个文件夹会自动创建在脚本运行的目录下
    destination_masks_path = r"F:\INP-Former-main\Mydatasets\dianrong\front\tets_mask"

    find_and_move_matched_masks(folder1_path, folder2_path, destination_masks_path)


