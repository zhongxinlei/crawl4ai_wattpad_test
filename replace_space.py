import os


def replace_filename_spaces(folder_path):
    """
    递归替换文件夹中所有文件名中的空格为下划线

    参数:
    folder_path (str): 目标文件夹路径

    特性说明:
    - 自动处理子文件夹中的文件
    - 保留文件扩展名不变
    - 显示详细的重命名过程
    - 包含错误处理机制
    """
    for root, dirs, files in os.walk(folder_path):
        # 处理文件名
        for filename in files:
            # 仅当文件名包含空格时处理
            if ' ' in filename:
                # 构造新文件名（保留扩展名）
                new_name = filename.replace(' ', '_')

                # 构建完整路径
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, new_name)

                try:
                    os.rename(old_path, new_path)
                    print(f"成功重命名：{old_path} -> {new_path}")
                except Exception as e:
                    print(f"重命名失败：{old_path} | 错误：{str(e)}")

        # 可选：处理目录名（需要时取消注释）
        """
        for i in range(len(dirs)):
            if ' ' in dirs[i]:
                new_dir = dirs[i].replace(' ', '_')
                old_dir_path = os.path.join(root, dirs[i])
                new_dir_path = os.path.join(root, new_dir)

                try:
                    os.rename(old_dir_path, new_dir_path)
                    dirs[i] = new_dir  # 更新遍历列表
                    print(f"成功重命名目录：{old_dir_path} -> {new_dir_path}")
                except Exception as e:
                    print(f"目录重命名失败：{old_dir_path} | 错误：{str(e)}")
        """


if __name__ == "__main__":
    # 使用示例
    target_folder = "raw_novel/"  # 替换为实际路径
    replace_filename_spaces(target_folder)
    print("文件名空格替换操作完成！")