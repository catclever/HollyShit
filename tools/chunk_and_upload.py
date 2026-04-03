"""
【脚本功能】：全能型特征矩阵切分与魔搭上传器 (Chunk & Upload Multi-Tool)
【核心设计】：
    - `split`: 将单体巨大的 `.npy` 切分成带压缩的 `.npz` 切片序列，不吃内存。
    - `push`: 将单个文件或整个文件夹上传至 ModelScope 模型库/数据集库。
    - `pipeline`: 切一块 -> 传一块 -> 删一块，实现零增量存储开销。
【运行示例】：
    python tools/chunk_and_upload.py split --input ./bge_huge.npy --chunk_size 500000 --delete_source
    python tools/chunk_and_upload.py push --target ./chunk_0.npz --repo_id kael/phase0 --token XXX --delete_source
    python tools/chunk_and_upload.py pipeline --input ./bge.npy --chunk_size 500000 --repo_id kael/phase0 --token XXX --delete_chunks
"""

import os
import shutil
import argparse
import numpy as np

class ChunkUploader:
    def __init__(self, ms_repo_id=None, ms_token=None):
        self.repo_id = ms_repo_id
        self.ms_token = ms_token
        self.api = None
        
        if self.repo_id and self.ms_token:
            try:
                from modelscope.hub.api import HubApi
                self.api = HubApi()
                self.api.login(self.ms_token)
                print(f"[ModelScope] 鉴权成功！已挂载目标库：{self.repo_id}")
            except ImportError:
                print("[FATAL] 缺少 modeolscope 库，请运行 `pip install modelscope`")
                exit(1)

    def split_npy(self, input_path, chunk_size, out_dir, delete_source=False):
        """将庞大的单体 .npy 零拷贝读取并按 chunk_size 分割成压缩的 .npz"""
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        print(f"[SPLIT] 准备加载母文件: {input_path}")
        # mmap_mode="r" 极其关键，它允许我们在不吃爆内存的情况下读取 T 级别的数据
        mmap_arr = np.load(input_path, mmap_mode="r")
        total_elements = mmap_arr.shape[0]
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        print(f"[SPLIT] 母矩阵维度: {mmap_arr.shape}，步长: {chunk_size}")

        chunk_files = []
        for start_idx in range(0, total_elements, chunk_size):
            end_idx = min(start_idx + chunk_size, total_elements)
            # 文件名强制对齐索引，形如：bge_chunk_000000_0500000.npz
            chunk_name = f"{base_name}_chunk_{start_idx:07d}_{end_idx:07d}.npz"
            chunk_path = os.path.join(out_dir, chunk_name)
            
            print(f"[SPLIT] 正在切割落盘 [{start_idx} : {end_idx}] -> {chunk_path}")
            # 从 memmap 里切出一片，然后保存为 npz，附带轻微压缩
            chunk_data = mmap_arr[start_idx:end_idx].astype(np.float16)
            np.savez_compressed(chunk_path, features=chunk_data)
            chunk_files.append(chunk_path)
            
            # 手动执行 Python 垃圾回收提示，尽早释放 Numpy 内存切片
            del chunk_data
            
        print("[SPLIT] 所有切块分割完毕！")
        
        if delete_source:
            # 必须彻底释放 mmap 引用才能删除文件
            del mmap_arr
            print(f"[⚠ DELETE_SOURCE] 正在销毁母文件: {input_path}")
            os.remove(input_path)
            
        return chunk_files

    def push(self, target_path, delete_source=False):
        """将本地的文件或者文件夹推送到 ModelScope"""
        if not self.api:
            raise RuntimeError("未配置 ModelScope repo_id 或 token！")
            
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"找不到需要上传的目标：{target_path}")

        print(f"\n[PUSH] 开始向 ModelScope 推送: {target_path}")
        
        # ModelScope 的 push_model 通常需要针对一个目录操作
        # 所以如果目标是一个独立的文件，我们建一个临时目录中转
        is_file = os.path.isfile(target_path)
        upload_dir = target_path
        
        if is_file:
            temp_dir = os.path.join(os.path.dirname(target_path), f"ms_tmp_{os.path.basename(target_path)}")
            os.makedirs(temp_dir, exist_ok=True)
            # 采用 copy 而不是 move，以防上传失败导致文件迷失
            shutil.copy2(target_path, temp_dir)
            upload_dir = temp_dir
            
        try:
            # 执行底层推送指令，如果是初次推送遇到空仓库会自动初始化
            self.api.push_model(model_id=self.repo_id, model_dir=upload_dir)
            print("[PUSH] 上传成功 100% 🚀")
        except Exception as e:
            print(f"[PUSH ERROR] 上传失败: {e}")
            if is_file: shutil.rmtree(upload_dir)
            return False
            
        # 清理临时目录
        if is_file:
            shutil.rmtree(upload_dir)
            
        if delete_source:
             print(f"[⚠ DELETE_SOURCE] 正在销毁已上传的源文件: {target_path}")
             if is_file:
                 os.remove(target_path)
             else:
                 shutil.rmtree(target_path)
                 
        return True

    def pipeline(self, input_path, chunk_size, out_dir, delete_chunks=False, delete_source=False):
        """终极流水线：切出一块 -> 立刻上传 -> 按照指示删除块 -> 继续下一块"""
        print(f"=== 启动极限流水线，目标仓库：{self.repo_id} ===")
        mmap_arr = np.load(input_path, mmap_mode="r")
        total_elements = mmap_arr.shape[0]
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for start_idx in range(0, total_elements, chunk_size):
            end_idx = min(start_idx + chunk_size, total_elements)
            chunk_name = f"{base_name}_chunk_{start_idx:07d}_{end_idx:07d}.npz"
            chunk_path = os.path.join(out_dir, chunk_name)
            
            # STEP 1: 切割
            print(f"\n[PIPELINE - STEP 1] 切割落盘 [{start_idx} : {end_idx}] -> {chunk_name}")
            chunk_data = mmap_arr[start_idx:end_idx].astype(np.float16)
            np.savez_compressed(chunk_path, features=chunk_data)
            del chunk_data
            
            # STEP 2: 上传
            print(f"[PIPELINE - STEP 2] 上传至 ModelScope")
            success = self.push(chunk_path, delete_source=delete_chunks)
            if not success:
               print(f"[FATAL] 块 {chunk_name} 上传失败，中断流水线以防数据丢失。")
               exit(1)
               
        print("\n[PIPELINE] 全流水线执行完毕，所有块均已云端同步！")
        
        if delete_source:
            del mmap_arr
            print(f"[⚠ DELETE_SOURCE] 正在销毁起点的母文件: {input_path}")
            os.remove(input_path)


def main():
    parser = argparse.ArgumentParser(description="Chunk & Upload Multi-Tool对于高维矩阵矩阵切片")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # =============== SubParser: SPLIT ===============
    parser_split = subparsers.add_parser("split", help="仅切割本地矩阵，不发生网络传输")
    parser_split.add_argument("--input", type=str, required=True, help="巨大的源 npy 路径")
    parser_split.add_argument("--chunk_size", type=int, default=500000, help="每个切块容纳的条目数")
    parser_split.add_argument("--out_dir", type=str, default="./chunks", help="切片输出目录")
    parser_split.add_argument("--delete_source", action="store_true", help="所有切片完成后，物理删除原巨大输入文件")

    # =============== SubParser: PUSH ===============
    parser_push = subparsers.add_parser("push", help="仅将本地文件或目录暴力推送到 ModelScope")
    parser_push.add_argument("--target", type=str, required=True, help="需要上传的单个文件，或包含文件的目录")
    parser_push.add_argument("--repo_id", type=str, required=True, help="ModelScope 目标仓库 (如 kael/phase0-embeddings)")
    parser_push.add_argument("--token", type=str, required=True, help="ModelScope Access Token")
    parser_push.add_argument("--delete_source", action="store_true", help="上传 100% 成功后，立刻物理删除该 target!")

    # =============== SubParser: PIPELINE ===============
    parser_pipe = subparsers.add_parser("pipeline", help="【硬盘告急专用】切出一块->瞬间上传->瞬间删除本地切块->继续下一切块")
    parser_pipe.add_argument("--input", type=str, required=True, help="巨大的源 npy 路径")
    parser_pipe.add_argument("--chunk_size", type=int, default=500000, help="每个切块容纳的条目数")
    parser_pipe.add_argument("--out_dir", type=str, default="./chunks", help="切块缓冲目录")
    parser_pipe.add_argument("--repo_id", type=str, required=True, help="ModelScope 目标仓库")
    parser_pipe.add_argument("--token", type=str, required=True, help="ModelScope Access Token")
    parser_pipe.add_argument("--delete_chunks", action="store_true", help="【极其重要】边切边传边删，只占极小硬盘")
    parser_pipe.add_argument("--delete_source", action="store_true", help="全部大功告成后，干掉输入的母文件")

    args = parser.parse_args()

    # 初始化 Uploader（按需注入 token）
    repo_id = getattr(args, "repo_id", None)
    token = getattr(args, "token", None)
    uploader = ChunkUploader(repo_id, token)

    if args.command == "split":
        uploader.split_npy(args.input, args.chunk_size, args.out_dir, args.delete_source)
    elif args.command == "push":
        uploader.push(args.target, args.delete_source)
    elif args.command == "pipeline":
        uploader.pipeline(args.input, args.chunk_size, args.out_dir, args.delete_chunks, args.delete_source)

if __name__ == "__main__":
    main()
