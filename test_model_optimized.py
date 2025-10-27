#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版本的模型测试代码 - 使用Flash Attention 2和分桶批量推理
"""

import os
import sys
import json
import argparse
import time
import torch
import gc
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from collections import defaultdict

def str2bool(s: str) -> bool:
    return s.lower() in ("1", "true", "t", "y", "yes")

def parse_gen_args():
    """解析生成参数"""
    parser = argparse.ArgumentParser(description="优化版本的模型测试，支持Flash Attention 2和批量推理")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="最大生成的新token数")
    parser.add_argument("--min-new-tokens", type=int, default=1, help="最小生成的新token数")
    parser.add_argument("--do-sample", type=str2bool, default=True, help="是否启用采样")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.8, help="核采样阈值")
    parser.add_argument("--top-k", type=int, default=40, help="top-k 采样阈值")
    parser.add_argument("--num-beams", type=int, default=1, help="束搜索的beam数")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="重复惩罚系数")
    parser.add_argument("--batch-size", type=int, default=4, help="批量推理的批次大小")
    parser.add_argument("--bucket-size", type=int, default=8, help="分桶的桶大小")
    parser.add_argument("--use-flash-attention", type=str2bool, default=True, help="是否使用Flash Attention 2")
    return vars(parser.parse_args())

def load_test_data(file_path: str = None):
    """加载测试集数据"""
    if file_path is None:
        test_data_path = "/root/autodl-tmp/test_labels.json"
    else:
        test_data_path = file_path
    
    if not os.path.exists(test_data_path):
        print(f"✗ 测试数据文件不存在: {test_data_path}")
        return None
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"✓ 成功加载测试数据，共 {len(test_data)} 个样本")
        return test_data
    except Exception as e:
        print(f"✗ 加载测试数据失败: {e}")
        return None

def load_model_and_tokenizer(base_model_path: str, adapter_path: str, use_flash_attention: bool = True):
    """加载模型和tokenizer"""
    print(f"正在加载基础模型: {base_model_path}")
    print(f"LoRA适配器路径: {adapter_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置padding side为left以获得正确的生成结果
    tokenizer.padding_side = "left"
    
    # 加载模型配置
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("✓ 启用 Flash Attention 2")
    
    # 加载基础模型
    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    
    # 加载LoRA适配器
    print("加载LoRA适配器...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # 合并适配器权重
    print("合并适配器权重...")
    model = model.merge_and_unload()
    
    # 设置为评估模式
    model.eval()
    
    print(f"✓ 模型加载完成，参数量: {model.num_parameters():,}")
    return model, tokenizer

def prepare_inputs(samples: List[Dict], tokenizer) -> List[str]:
    """准备输入文本"""
    texts = []
    for sample in samples:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '').strip()
        
        if input_text:
            question = f"{instruction}\n\n{input_text}"
        else:
            question = instruction
            
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)
    
    return texts

def get_optimal_batch_size(token_length: int) -> int:
    """根据序列长度获取最优batch_size（基于性能测试结果）"""
    if token_length < 1024:        # short
        return 20  # 使用与medium相同的配置
    elif token_length < 2048:      # medium
        return 20  # 最优配置：吞吐量0.37 samples/s
    elif token_length < 4096:      # long  
        return 12  # 最优配置：吞吐量0.24 samples/s
    elif token_length < 8192:      # very_long
        return 4   # 最优配置：吞吐量0.10 samples/s
    else:                          # extra_long
        return 2   # 最优配置：吞吐量0.07 samples/s

def bucket_samples_by_length(samples: List[Dict], tokenizer) -> List[Tuple[List[Dict], int]]:
    """根据输入长度对样本进行分桶，并返回每个桶的最优batch_size"""
    print("🔄 正在对样本进行智能分桶...")
    
    # 定义长度分组区间
    length_buckets = {
        'short': [],      # < 1024 tokens (batch_size=20)
        'medium': [],     # 1024-2048 tokens (batch_size=20)
        'long': [],       # 2048-4096 tokens (batch_size=12)
        'very_long': [],  # 4096-8192 tokens (batch_size=4)
        'extra_long': []  # > 8192 tokens (batch_size=2)
    }
    
    # 计算每个样本的输入长度并分组
    for sample in samples:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '').strip()
        
        if input_text:
            question = f"{instruction}\n\n{input_text}"
        else:
            question = instruction
            
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 估算token长度
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_length = len(tokens)
        
        # 根据长度分组
        if token_length < 1024:
            length_buckets['short'].append((token_length, sample))
        elif token_length < 2048:
            length_buckets['medium'].append((token_length, sample))
        elif token_length < 4096:
            length_buckets['long'].append((token_length, sample))
        elif token_length < 8192:
            length_buckets['very_long'].append((token_length, sample))
        else:
            length_buckets['extra_long'].append((token_length, sample))
    
    # 打印分组统计（优化后的格式）
    print("📊 序列长度分布统计:")
    bucket_configs = {
        'short': (20, "短序列"),
        'medium': (20, "中等序列"), 
        'long': (12, "长序列"),
        'very_long': (4, "很长序列"),
        'extra_long': (2, "超长序列")
    }
    
    for bucket_name, (batch_size, display_name) in bucket_configs.items():
        count = len(length_buckets[bucket_name])
        if count > 0:
            print(f"  📦 {display_name}: {count} 个样本 (最优batch_size={batch_size})")
    
    # 在每个长度组内按长度排序并创建批次
    final_buckets = []
    
    for group_name, group_samples in length_buckets.items():
        if not group_samples:
            continue
            
        # 按长度排序
        group_samples.sort(key=lambda x: x[0])
        
        # 获取该组的最优batch_size
        optimal_batch_size = bucket_configs[group_name][0]
        
        # 创建批次
        current_bucket = []
        for length, sample in group_samples:
            current_bucket.append(sample)
            
            if len(current_bucket) >= optimal_batch_size:
                final_buckets.append((current_bucket, optimal_batch_size))
                current_bucket = []
        
        # 处理剩余样本
        if current_bucket:
            final_buckets.append((current_bucket, optimal_batch_size))
    
    total_batches = len(final_buckets)
    avg_batch_size = sum(len(bucket[0]) for bucket in final_buckets) / total_batches if total_batches > 0 else 0
    
    print(f"✅ 智能分桶完成: {total_batches} 个批次，平均批次大小 {avg_batch_size:.1f}")
    
    return final_buckets

def batch_generate(model, tokenizer, texts: List[str], gen_args: Dict) -> List[str]:
    """批量生成"""
    # 对文本进行tokenize
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成参数
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # 构建生成参数，只包含存在的参数
    generate_kwargs = {
        "max_new_tokens": gen_args.get("max_new_tokens", 4096),
        "min_new_tokens": gen_args.get("min_new_tokens", 1),
        "do_sample": gen_args.get("do_sample", True),
        "temperature": gen_args.get("temperature", 0.2),
        "top_p": gen_args.get("top_p", 0.8),
        "top_k": gen_args.get("top_k", 40),
        "num_beams": gen_args.get("num_beams", 1),
        "repetition_penalty": gen_args.get("repetition_penalty", 1.1),
        "pad_token_id": pad_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    
    # 只有当参数存在且不为0时才添加
    if gen_args.get("no_repeat_ngram_size", 0) > 0:
        generate_kwargs["no_repeat_ngram_size"] = gen_args["no_repeat_ngram_size"]
    
    if gen_args.get("early_stopping", False):
        generate_kwargs["early_stopping"] = gen_args["early_stopping"]
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **generate_kwargs
        )
    
    # 解码输出
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return outputs

def load_model_with_flash_attention(base_model_path: str, adapter_path: str, use_flash_attention: bool = True):
    """加载模型并启用Flash Attention 2"""
    print("加载基础模型...")
    
    # 模型配置
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    # 如果使用Flash Attention 2
    if use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("✓ 启用Flash Attention 2")
        except Exception as e:
            print(f"⚠ Flash Attention 2启用失败，使用默认attention: {e}")
    
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    
    # 加载LoRA适配器
    print("加载LoRA适配器...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # 合并适配器权重
    print("合并适配器权重...")
    model = model.merge_and_unload()
    
    # 设置为评估模式
    model.eval()
    print("✓ 模型加载成功！")
    
    return model



def run_optimized_inference(model, tokenizer, test_samples: List[Dict], gen_args: Dict = None):
    """运行优化版本的推理"""
    print(f"🚀 开始优化推理，共 {len(test_samples)} 个样本...")
    
    # 分桶处理（使用最优batch_size）
    buckets = bucket_samples_by_length(test_samples, tokenizer)
    non_empty_buckets = [bucket for bucket in buckets if bucket[0]]  # 过滤空桶
    print(f"📊 分为 {len(non_empty_buckets)} 个非空批次")
    
    # 使用传入的生成参数，如果没有则使用默认值
    if gen_args is None:
        gen_args = {
            "max_new_tokens": 4096,  # 修正为与test_model.py一致的默认值
            "min_new_tokens": 1,
            "do_sample": True,
            "temperature": 0.2,      # 修正为与test_model.py一致
            "top_p": 0.8,            # 修正为与test_model.py一致
            "top_k": 40,             # 修正为与test_model.py一致
            "num_beams": 1,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 0,
            "early_stopping": False,
        }
    
    # 添加tokenizer相关参数
    gen_args.update({
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True
    })
    
    print(f"📋 生成参数: max_new_tokens={gen_args['max_new_tokens']}, temperature={gen_args['temperature']}, top_p={gen_args['top_p']}, top_k={gen_args['top_k']}")
    
    all_results = []
    start_time = time.time()
    
    for i, (bucket_samples, optimal_batch_size) in enumerate(non_empty_buckets):
        print(f"🔄 处理批次 {i+1}/{len(non_empty_buckets)}: {len(bucket_samples)} 个样本, batch_size={optimal_batch_size}")
        
        # 准备批量输入
        texts = prepare_inputs(bucket_samples, tokenizer)
        
        # 批量推理
        outputs = batch_generate(model, tokenizer, texts, gen_args)
        
        # 组合结果
        for j, (sample, output) in enumerate(zip(bucket_samples, outputs)):
            result = {
                'sample_id': len(all_results) + 1,
                'instruction': sample.get('instruction', ''),
                'input': sample.get('input', ''),
                'expected_output': sample.get('output', ''),
                'actual_output': output,
                'batch_id': i + 1,
                'batch_size': optimal_batch_size
            }
            all_results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n✅ 优化推理完成！")
    print(f"⏱️  总耗时: {total_time:.2f}s")
    print(f"⚡ 平均每样本: {total_time/len(test_samples):.2f}s")
    print(f"🚀 吞吐量: {len(test_samples)/total_time:.2f} 样本/秒")
    
    return {
        'results': all_results,
        'performance': {
            'total_time': total_time,
            'avg_time_per_sample': total_time / len(test_samples),
            'throughput': len(test_samples) / total_time,
            'total_samples': len(test_samples)
        }
    }

def test_single_optimized_model(base_model_path: str, adapter_path: str, test_data: List[Dict], gen_args: Dict = None):
    """测试单个优化模型"""
    model_name = os.path.basename(adapter_path)
    print(f"\n" + "="*80)
    print(f"🚀 开始测试模型: {model_name}")
    print("="*80)
    
    model = None
    try:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU内存使用情况: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
        # 加载模型和tokenizer
        print("🔧 加载模型和tokenizer...")
        model, tokenizer = load_model_and_tokenizer(base_model_path, adapter_path, use_flash_attention=True)
        
        # 运行优化版本推理测试
        print("\n" + "="*80)
        print("🔥 开始优化版本推理测试")
        print("="*80)
        results = run_optimized_inference(model, tokenizer, test_data, gen_args)
        
        # 保存结果
        output_path = f"/root/autodl-tmp/optimized_results_{model_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {output_path}")
        
        # 显示性能统计
        perf = results['performance']
        print(f"\n" + "="*80)
        print(f"📈 {model_name} 性能统计总结")
        print("="*80)
        print(f"📊 总样本数: {perf['total_samples']}")
        print(f"⏱️  总耗时: {perf['total_time']:.2f}s")
        print(f"⚡ 平均每样本: {perf['avg_time_per_sample']:.2f}s")
        print(f"🚀 吞吐量: {perf['throughput']:.2f} 样本/秒")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"✗ 测试模型 {model_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 确保释放模型内存
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"模型 {model_name} 内存已释放")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🚀 优化版本推理测试启动")
    print("="*80)
    
    # 解析命令行参数
    gen_args = parse_gen_args()
    print(f"📋 使用生成参数: {gen_args}")
    
    # 配置
    base_model_path = "/root/autodl-tmp/Qwen3-14B"
    saves_path = "/root/autodl-tmp/saves/qwen3-14B-lora-model-parallel-1016"
    test_data_path = "/root/autodl-tmp/test_labels_CE.json"
    
    # 获取所有需要测试的模型路径并排序
    model_paths = [os.path.join(saves_path, d) for d in os.listdir(saves_path) if d.startswith("checkpoint-")]
    model_paths.sort(key=lambda x: int(x.split('-')[-1]))  # 按checkpoint数字排序
    model_paths.append(saves_path)  # 添加最终模型

    print(f"基础模型路径: {base_model_path}")
    print(f"找到 {len(model_paths)} 个模型需要测试")
    
    # 加载测试数据
    print("📊 加载测试数据...")
    test_data = load_test_data(test_data_path)
    if test_data is None:
        print("❌ 测试数据加载失败，退出测试")
        return
    print(f"✅ 成功加载 {len(test_data)} 个测试样本")
    
    # 测试所有模型
    success_count = 0
    for i, adapter_path in enumerate(model_paths):
        model_name = os.path.basename(adapter_path)
        print(f"\n=== 开始测试模型 {i+1}/{len(model_paths)}: {model_name} ===")
        
        # 测试单个模型，传递生成参数
        if test_single_optimized_model(base_model_path, adapter_path, test_data, gen_args):
            success_count += 1
        
        # 在模型之间强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n" + "="*80)
    print("🎉 所有模型测试完成")
    print("="*80)
    print(f"成功测试了 {success_count}/{len(model_paths)} 个模型")
    print("="*80)

if __name__ == "__main__":
    main()