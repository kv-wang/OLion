#!/usr/bin/env python3
"""
上传 imagenet-vae 数据集到 ModelScope
数据集路径: /root/autodl-tmp/imagenet-vae
目标数据集: wangzixiao/imagenet_image_latent
"""

import os
from pathlib import Path
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.file_download import model_file_download
import json

def upload_dataset_to_modelscope(
    local_path: str,
    dataset_name: str,
    namespace: str = "wangzixiao",
    commit_message: str = "Upload imagenet-vae dataset"
):
    """
    上传数据集到 ModelScope
    
    Args:
        local_path: 本地数据集路径
        dataset_name: 数据集名称
        namespace: 命名空间（用户名）
        commit_message: 提交信息
    """
    from modelscope.hub.api import HubApi
    
    # 初始化 API
    api = HubApi()
    
    # 检查登录状态
    try:
        user_info = api.get_user_info()
        print(f"已登录用户: {user_info.get('Name', 'Unknown')}")
    except Exception as e:
        print(f"未登录或登录已过期，请先运行: modelscope login")
        print(f"错误信息: {e}")
        return False
    
    dataset_id = f"{namespace}/{dataset_name}"
    print(f"准备上传数据集到: {dataset_id}")
    print(f"本地路径: {local_path}")
    
    # 检查本地路径
    if not os.path.exists(local_path):
        print(f"错误: 本地路径不存在: {local_path}")
        return False
    
    # 创建数据集（如果不存在）
    try:
        api.create_dataset(
            dataset_name=dataset_name,
            namespace=namespace,
            dataset_type='generic',
            chineseName='ImageNet Image Latent Dataset',
            license='MIT'
        )
        print(f"数据集 {dataset_id} 创建成功或已存在")
    except Exception as e:
        print(f"创建数据集时出现错误（可能已存在）: {e}")
    
    # 上传文件
    try:
        # 使用 git 方式上传（推荐用于大文件）
        print("开始上传文件...")
        print("注意: 对于大文件，建议使用 git-lfs")
        
        # 方法1: 使用 API 上传（适合小文件）
        # 对于大文件，建议使用 git 方式
        
        # 方法2: 使用 git 命令上传（推荐）
        print("\n推荐使用以下 git 命令上传:")
        print(f"cd {local_path}")
        print(f"git init")
        print(f"git lfs install")
        print(f"git remote add origin https://www.modelscope.cn/datasets/{dataset_id}.git")
        print(f"git add .")
        print(f"git commit -m '{commit_message}'")
        print(f"git push origin main")
        
        return True
        
    except Exception as e:
        print(f"上传过程中出现错误: {e}")
        return False


def create_upload_script(local_path: str, dataset_name: str, namespace: str = "wangzixiao"):
    """创建上传脚本"""
    script_content = f"""#!/bin/bash
# 上传数据集到 ModelScope
# 数据集: {namespace}/{dataset_name}

cd {local_path}

# 初始化 git 仓库
if [ ! -d .git ]; then
    git init
    git lfs install
    git remote add origin https://www.modelscope.cn/datasets/{namespace}/{dataset_name}.git
fi

# 添加所有文件
git add .

# 提交
git commit -m "Upload imagenet-vae dataset"

# 推送到 ModelScope
git push origin main
"""
    
    script_path = f"/root/autodl-tmp/upload_{dataset_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"上传脚本已创建: {script_path}")
    return script_path


if __name__ == "__main__":
    local_path = "/root/autodl-tmp/imagenet-vae"
    dataset_name = "imagenet_image_latent"
    namespace = "wangzixiao"
    
    print("=" * 60)
    print("ModelScope 数据集上传工具")
    print("=" * 60)
    print(f"本地路径: {local_path}")
    print(f"目标数据集: {namespace}/{dataset_name}")
    print("=" * 60)
    
    # 创建上传脚本
    script_path = create_upload_script(local_path, dataset_name, namespace)
    
    print("\n步骤:")
    print("1. 安装 ModelScope SDK: pip install modelscope")
    print("2. 登录 ModelScope: modelscope login")
    print("3. 安装 git-lfs: apt-get install git-lfs 或 yum install git-lfs")
    print(f"4. 运行上传脚本: bash {script_path}")
    print("\n或者手动执行以下命令:")
    print(f"   cd {local_path}")
    print(f"   git init")
    print(f"   git lfs install")
    print(f"   git remote add origin https://www.modelscope.cn/datasets/{namespace}/{dataset_name}.git")
    print(f"   git add .")
    print(f"   git commit -m 'Upload imagenet-vae dataset'")
    print(f"   git push origin main")

