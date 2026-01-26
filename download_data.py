#!/usr/bin/env python3
"""
数据集下载脚本 - 智能垃圾分类系统
Dataset Download Script - Intelligent Waste Classification System

功能:
1. 检查并配置 Kaggle API
2. 自动下载垃圾分类数据集
3. 解压并整理数据目录结构
4. 验证数据集完整性
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

# 数据集配置
DATASET_NAME = "mostafaabla/garbage-classification"
DATA_DIR = "./data/garbage_classification"
EXPECTED_CLASSES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash',
    'white-glass', 'brown-glass'
]


def print_header(title: str) -> None:
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(step: int, description: str) -> None:
    """打印步骤信息"""
    print(f"\n[步骤 {step}] {description}")
    print("-" * 40)


def check_kaggle_installed() -> bool:
    """检查 Kaggle 是否已安装"""
    try:
        import kaggle
        return True
    except ImportError:
        return False


def install_kaggle() -> bool:
    """安装 Kaggle 包"""
    print("正在安装 kaggle 包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        print("kaggle 包安装成功!")
        return True
    except subprocess.CalledProcessError:
        print("错误: 安装 kaggle 包失败")
        return False


def get_kaggle_config_path() -> Path:
    """获取 Kaggle 配置文件路径"""
    if os.name == 'nt':  # Windows
        return Path(os.environ.get('USERPROFILE', '')) / '.kaggle' / 'kaggle.json'
    else:  # Linux/Mac
        return Path.home() / '.kaggle' / 'kaggle.json'


def check_kaggle_config() -> Tuple[bool, str]:
    """
    检查 Kaggle API 配置（支持环境变量和 JSON 文件两种方式）

    Returns:
        Tuple[bool, str]: (是否已配置, 配置方式说明)
    """
    # 方式1: 检查新版环境变量 KAGGLE_API_TOKEN
    if os.environ.get('KAGGLE_API_TOKEN'):
        return True, "环境变量 KAGGLE_API_TOKEN"

    # 方式2: 检查旧版环境变量 KAGGLE_USERNAME + KAGGLE_KEY
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        return True, "环境变量 KAGGLE_USERNAME/KAGGLE_KEY"

    # 方式3: 检查 kaggle.json 文件
    config_path = get_kaggle_config_path()

    if not config_path.exists():
        return False, str(config_path)

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'username' in config and 'key' in config:
                return True, f"配置文件 {config_path}"
            else:
                return False, "配置文件格式错误，缺少 username 或 key"
    except json.JSONDecodeError:
        return False, "配置文件 JSON 格式错误"
    except Exception as e:
        return False, str(e)


def setup_kaggle_config() -> bool:
    """交互式设置 Kaggle API 配置（支持新版 Token 和旧版 JSON 文件）"""
    print_header("Kaggle API 配置向导")

    print("""
要下载 Kaggle 数据集，需要配置 API 密钥。

请按以下步骤获取 API Token：
1. 登录 https://www.kaggle.com
2. 点击右上角头像 → "Settings"
3. 找到 "API" 部分，点击 "Create New Token"
4. 复制显示的 Token（格式如: KGAT_xxxxxxxxxxxx）
""")

    print("\n请选择配置方式:")
    print("  1. 输入 API Token（新版，推荐）")
    print("  2. 使用 kaggle.json 文件（旧版）")
    print("  3. 手动下载数据集（跳过 API 配置）")

    choice = input("\n请输入选项 (1/2/3): ").strip()

    if choice == '1':
        return setup_with_token()
    elif choice == '2':
        return setup_with_json_file()
    elif choice == '3':
        print("\n请手动下载数据集:")
        print(f"  https://www.kaggle.com/datasets/{DATASET_NAME}")
        print("  下载后解压到 ./data/garbage_classification/ 目录")
        return False
    else:
        print("无效选项")
        return False


def setup_with_token() -> bool:
    """使用 API Token 配置"""
    print("\n请输入你的 Kaggle API Token")
    print("（格式如: KGAT_a7e3efaace0004e8883eb90af7b15b37）")

    token = input("\nAPI Token: ").strip()

    if not token:
        print("错误: Token 不能为空")
        return False

    if not token.startswith('KGAT_'):
        print("警告: Token 格式可能不正确，但将继续尝试...")

    # 设置环境变量
    os.environ['KAGGLE_API_TOKEN'] = token
    print("Token 已设置到环境变量")

    # 同时询问是否要保存到配置文件（需要用户名）
    print("\n是否同时保存到配置文件？（下次运行时无需再输入）")
    save_choice = input("保存配置? (y/n): ").strip().lower()

    if save_choice == 'y':
        username = input("请输入你的 Kaggle 用户名: ").strip()
        if username:
            config_path = get_kaggle_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # 从 token 提取 key（去掉 KGAT_ 前缀，或直接使用整个 token）
            key = token

            config_data = {"username": username, "key": key}
            try:
                with open(config_path, 'w') as f:
                    json.dump(config_data, f)
                if os.name != 'nt':
                    os.chmod(config_path, 0o600)
                print(f"配置已保存到: {config_path}")
            except Exception as e:
                print(f"保存配置文件失败: {e}")

    return True


def setup_with_json_file() -> bool:
    """使用 kaggle.json 文件配置（旧版方式）"""
    print("\n如果你有 kaggle.json 文件，请提供路径")
    print("（旧版 Kaggle 会下载此文件，新版需要手动创建）")

    # 询问 kaggle.json 文件位置
    default_download_path = Path.home() / "Downloads" / "kaggle.json"
    json_path = input(f"\n请输入 kaggle.json 文件路径\n(直接回车使用默认: {default_download_path}): ").strip()

    if not json_path:
        json_path = str(default_download_path)

    json_path = Path(json_path).expanduser()

    if not json_path.exists():
        print(f"\n错误: 找不到文件 {json_path}")
        print("\n如果你只有 API Token，请选择方式 1 重新运行")
        return False

    # 复制到正确位置
    config_path = get_kaggle_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(json_path, config_path)
        # 设置权限 (仅限 Unix)
        if os.name != 'nt':
            os.chmod(config_path, 0o600)
        print(f"\n配置文件已复制到: {config_path}")
        return True
    except Exception as e:
        print(f"\n错误: 复制配置文件失败 - {e}")
        return False


def download_dataset() -> bool:
    """使用 Kaggle API 下载数据集"""
    print("正在下载数据集...")
    print(f"数据集: {DATASET_NAME}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        # 下载到当前目录
        api.dataset_download_files(DATASET_NAME, path=".", unzip=False)
        print("数据集下载完成!")
        return True
    except Exception as e:
        print(f"错误: 下载失败 - {e}")
        return False


def extract_dataset() -> bool:
    """解压数据集到指定目录"""
    zip_file = "garbage-classification.zip"

    if not os.path.exists(zip_file):
        print(f"错误: 找不到 {zip_file}")
        return False

    print(f"正在解压 {zip_file}...")

    # 创建数据目录
    data_parent = Path(DATA_DIR).parent
    data_parent.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # 获取 zip 内容结构
            namelist = zip_ref.namelist()

            # 临时解压目录
            temp_dir = data_parent / "_temp_extract"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            zip_ref.extractall(temp_dir)

        # 查找实际的数据目录
        target_dir = Path(DATA_DIR)
        if target_dir.exists():
            shutil.rmtree(target_dir)

        # 寻找包含类别文件夹的目录
        found_data_dir = None
        for root, dirs, files in os.walk(temp_dir):
            # 检查是否包含预期的类别目录
            matching_classes = [d for d in dirs if d in EXPECTED_CLASSES]
            if len(matching_classes) >= 6:  # 至少找到一半的类别
                found_data_dir = Path(root)
                break

        if found_data_dir:
            shutil.move(str(found_data_dir), str(target_dir))
            print(f"数据已解压到: {target_dir}")
        else:
            # 如果找不到，使用整个临时目录
            # 查找第一层目录
            subdirs = list(temp_dir.iterdir())
            if len(subdirs) == 1 and subdirs[0].is_dir():
                shutil.move(str(subdirs[0]), str(target_dir))
            else:
                shutil.move(str(temp_dir), str(target_dir))
            print(f"数据已解压到: {target_dir}")

        # 清理
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"错误: 解压失败 - {e}")
        return False


def verify_dataset() -> Tuple[bool, List[str], List[str]]:
    """
    验证数据集完整性

    Returns:
        Tuple[bool, List[str], List[str]]: (是否完整, 找到的类别, 缺失的类别)
    """
    data_path = Path(DATA_DIR)

    if not data_path.exists():
        return False, [], EXPECTED_CLASSES

    found_classes = []
    missing_classes = []

    for class_name in EXPECTED_CLASSES:
        class_path = data_path / class_name
        if class_path.exists() and class_path.is_dir():
            # 检查是否有图片文件
            images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + \
                     list(class_path.glob("*.jpeg")) + list(class_path.glob("*.JPG"))
            if images:
                found_classes.append(f"{class_name} ({len(images)} images)")
            else:
                missing_classes.append(f"{class_name} (empty)")
        else:
            missing_classes.append(class_name)

    return len(missing_classes) == 0, found_classes, missing_classes


def print_dataset_info() -> None:
    """打印数据集信息"""
    data_path = Path(DATA_DIR)

    print("\n数据集信息:")
    print("-" * 40)

    total_images = 0
    for class_name in sorted(os.listdir(data_path)):
        class_path = data_path / class_name
        if class_path.is_dir():
            images = list(class_path.glob("*.[jJ][pP][gG]")) + \
                     list(class_path.glob("*.[pP][nN][gG]")) + \
                     list(class_path.glob("*.[jJ][pP][eE][gG]"))
            count = len(images)
            total_images += count
            print(f"  {class_name:15} : {count:5} 张图片")

    print("-" * 40)
    print(f"  {'总计':15} : {total_images:5} 张图片")
    print(f"  类别数: {len(EXPECTED_CLASSES)}")


def cleanup_zip() -> None:
    """询问是否删除 zip 文件"""
    zip_file = "garbage-classification.zip"
    if os.path.exists(zip_file):
        response = input(f"\n是否删除下载的 zip 文件以节省空间? (y/n): ").strip().lower()
        if response == 'y':
            os.remove(zip_file)
            print(f"已删除: {zip_file}")


def main():
    """主函数"""
    print_header("垃圾分类数据集下载工具")

    # 检查数据集是否已存在
    is_valid, found, missing = verify_dataset()
    if is_valid:
        print("\n数据集已存在且完整!")
        print_dataset_info()
        return
    elif found:
        print(f"\n发现部分数据，但有 {len(missing)} 个类别缺失")
        response = input("是否重新下载完整数据集? (y/n): ").strip().lower()
        if response != 'y':
            return

    # 步骤 1: 检查 Kaggle 安装
    print_step(1, "检查 Kaggle 环境")

    if not check_kaggle_installed():
        print("Kaggle 未安装")
        if not install_kaggle():
            print("\n请手动安装: pip install kaggle")
            return
    else:
        print("Kaggle 已安装")

    # 步骤 2: 检查 API 配置
    print_step(2, "检查 Kaggle API 配置")

    is_configured, config_info = check_kaggle_config()
    if is_configured:
        print(f"API 已配置: {config_info}")
    else:
        print(f"API 未配置: {config_info}")
        if not setup_kaggle_config():
            print("\n无法完成配置，请参考 README.md 手动配置 Kaggle API")
            return

    # 步骤 3: 下载数据集
    print_step(3, "下载数据集")

    if not download_dataset():
        print("\n下载失败，请检查网络连接或手动下载:")
        print(f"  https://www.kaggle.com/datasets/{DATASET_NAME}")
        return

    # 步骤 4: 解压数据集
    print_step(4, "解压数据集")

    if not extract_dataset():
        print("\n解压失败，请手动解压 garbage-classification.zip")
        return

    # 步骤 5: 验证数据集
    print_step(5, "验证数据集")

    is_valid, found, missing = verify_dataset()
    if is_valid:
        print("数据集验证成功!")
        print_dataset_info()
    else:
        print(f"\n警告: 以下类别缺失或为空:")
        for m in missing:
            print(f"  - {m}")

    # 清理
    cleanup_zip()

    print_header("完成")
    print(f"\n数据集路径: {os.path.abspath(DATA_DIR)}")
    print("\n现在可以运行以下命令开始训练:")
    print("  python main.py")


if __name__ == "__main__":
    main()
