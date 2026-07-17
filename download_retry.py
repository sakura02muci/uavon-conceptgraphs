#!/usr/bin/env python
"""使用Python API下载数据集，带重试机制"""

from huggingface_hub import snapshot_download
import os

def download_with_retry(repo_id, local_dir, max_retries=5):
    """带重试的下载函数"""
    for attempt in range(max_retries):
        try:
            print(f"\n[尝试 {attempt + 1}/{max_retries}] 下载 {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                resume_download=True,
                max_workers=4
            )
            print(f"✅ {repo_id} 下载完成!")
            return True
        except Exception as e:
            print(f"❌ 失败: {e}")
            if attempt < max_retries - 1:
                print(f"等待10秒后重试...")
                import time
                time.sleep(10)
    return False

if __name__ == "__main__":
    datasets = [
        ("Kyaren/UAV-ON-envs-train", "TRAIN_ENVS"),
        ("Kyaren/UAV-ON-envs-test", "TEST_ENVS"),
        ("Kyaren/UAV-ON-dataset", "DATASET"),
    ]
    
    for repo_id, local_dir in datasets:
        os.makedirs(local_dir, exist_ok=True)
        download_with_retry(repo_id, local_dir)
