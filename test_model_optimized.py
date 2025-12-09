#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized version of model testing code - Using Flash Attention 2 and bucketed batch inference
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
    """Parse generation arguments"""
    parser = argparse.ArgumentParser(description="Optimized model testing, supporting Flash Attention 2 and batch inference")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum new tokens to generate")
    parser.add_argument("--min-new-tokens", type=int, default=1, help="Minimum new tokens to generate")
    parser.add_argument("--do-sample", type=str2bool, default=True, help="Whether to enable sampling")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling threshold")
    parser.add_argument("--num-beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty coefficient")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for batch inference")
    parser.add_argument("--bucket-size", type=int, default=8, help="Bucket size for bucketing")
    parser.add_argument("--use-flash-attention", type=str2bool, default=True, help="Whether to use Flash Attention 2")
    return vars(parser.parse_args())

def load_test_data(file_path: str = None):
    """Load test dataset"""
    if file_path is None:
        test_data_path = "/root/autodl-tmp/test_labels.json"
    else:
        test_data_path = file_path
    
    if not os.path.exists(test_data_path):
        print(f"✗ Test data file does not exist: {test_data_path}")
        return None
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"✓ Successfully loaded test data, total {len(test_data)} samples")
        return test_data
    except Exception as e:
        print(f"✗ Failed to load test data: {e}")
        return None

def load_model_and_tokenizer(base_model_path: str, adapter_path: str, use_flash_attention: bool = True):
    """Load model and tokenizer"""
    print(f"Loading base model: {base_model_path}")
    print(f"LoRA adapter path: {adapter_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for correct generation results
    tokenizer.padding_side = "left"
    
    # Load model configuration
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("✓ Enabled Flash Attention 2")
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge adapter weights
    print("Merging adapter weights...")
    model = model.merge_and_unload()
    
    # Set to evaluation mode
    model.eval()
    
    print(f"✓ Model loaded successfully, parameters: {model.num_parameters():,}")
    return model, tokenizer

def prepare_inputs(samples: List[Dict], tokenizer) -> List[str]:
    """Prepare input texts"""
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
    """Get optimal batch_size based on sequence length (based on performance test results)"""
    if token_length < 1024:        # short
        return 20  # Use same configuration as medium
    elif token_length < 2048:      # medium
        return 20  # Optimal configuration: throughput 0.37 samples/s
    elif token_length < 4096:      # long  
        return 12  # Optimal configuration: throughput 0.24 samples/s
    elif token_length < 8192:      # very_long
        return 4   # Optimal configuration: throughput 0.10 samples/s
    else:                          # extra_long
        return 2   # Optimal configuration: throughput 0.07 samples/s

def bucket_samples_by_length(samples: List[Dict], tokenizer) -> List[Tuple[List[Dict], int]]:
    """Bucket samples by input length and return optimal batch_size for each bucket"""
    print("🔄 Smart bucketing samples...")
    
    # Define length grouping intervals
    length_buckets = {
        'short': [],      # < 1024 tokens (batch_size=20)
        'medium': [],     # 1024-2048 tokens (batch_size=20)
        'long': [],       # 2048-4096 tokens (batch_size=12)
        'very_long': [],  # 4096-8192 tokens (batch_size=4)
        'extra_long': []  # > 8192 tokens (batch_size=2)
    }
    
    # Calculate input length for each sample and group them
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
        
        # Estimate token length
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_length = len(tokens)
        
        # Group by length
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
    
    # Print grouping statistics (optimized format)
    print("📊 Sequence length distribution statistics:")
    bucket_configs = {
        'short': (20, "Short sequence"),
        'medium': (20, "Medium sequence"), 
        'long': (12, "Long sequence"),
        'very_long': (4, "Very long sequence"),
        'extra_long': (2, "Extra long sequence")
    }
    
    for bucket_name, (batch_size, display_name) in bucket_configs.items():
        count = len(length_buckets[bucket_name])
        if count > 0:
            print(f"  📦 {display_name}: {count} samples (optimal batch_size={batch_size})")
    
    # Sort by length within each group and create batches
    final_buckets = []
    
    for group_name, group_samples in length_buckets.items():
        if not group_samples:
            continue
            
        # Sort by length
        group_samples.sort(key=lambda x: x[0])
        
        # Get optimal batch_size for this group
        optimal_batch_size = bucket_configs[group_name][0]
        
        # Create batches
        current_bucket = []
        for length, sample in group_samples:
            current_bucket.append(sample)
            
            if len(current_bucket) >= optimal_batch_size:
                final_buckets.append((current_bucket, optimal_batch_size))
                current_bucket = []
        
        # Handle remaining samples
        if current_bucket:
            final_buckets.append((current_bucket, optimal_batch_size))
    
    total_batches = len(final_buckets)
    avg_batch_size = sum(len(bucket[0]) for bucket in final_buckets) / total_batches if total_batches > 0 else 0
    
    print(f"✅ Smart bucketing completed: {total_batches} batches, average batch size {avg_batch_size:.1f}")
    
    return final_buckets

def batch_generate(model, tokenizer, texts: List[str], gen_args: Dict) -> List[str]:
    """Batch generation"""
    # Tokenize texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generation parameters
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Build generation arguments, only include existing parameters
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
    
    # Only add if parameter exists and is not 0
    if gen_args.get("no_repeat_ngram_size", 0) > 0:
        generate_kwargs["no_repeat_ngram_size"] = gen_args["no_repeat_ngram_size"]
    
    if gen_args.get("early_stopping", False):
        generate_kwargs["early_stopping"] = gen_args["early_stopping"]
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **generate_kwargs
        )
    
    # Decode outputs
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return outputs

def load_model_with_flash_attention(base_model_path: str, adapter_path: str, use_flash_attention: bool = True):
    """Load model and enable Flash Attention 2"""
    print("Loading base model...")
    
    # Model configuration
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    # If using Flash Attention 2
    if use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("✓ Enabled Flash Attention 2")
        except Exception as e:
            print(f"⚠ Failed to enable Flash Attention 2, using default attention: {e}")
    
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge adapter weights
    print("Merging adapter weights...")
    model = model.merge_and_unload()
    
    # Set to evaluation mode
    model.eval()
    print("✓ Model loaded successfully!")
    
    return model

def run_optimized_inference(model, tokenizer, test_samples: List[Dict], gen_args: Dict = None):
    """Run optimized inference"""
    print(f"🚀 Starting optimized inference, total {len(test_samples)} samples...")
    
    # Bucket processing (using optimal batch_size)
    buckets = bucket_samples_by_length(test_samples, tokenizer)
    non_empty_buckets = [bucket for bucket in buckets if bucket[0]]  # Filter empty buckets
    print(f"📊 Divided into {len(non_empty_buckets)} non-empty batches")
    
    # Use passed generation arguments, if not provided use default values
    if gen_args is None:
        gen_args = {
            "max_new_tokens": 4096,  # Fixed to be consistent with test_model.py
            "min_new_tokens": 1,
            "do_sample": True,
            "temperature": 0.2,      # Fixed to be consistent with test_model.py
            "top_p": 0.8,            # Fixed to be consistent with test_model.py
            "top_k": 40,             # Fixed to be consistent with test_model.py
            "num_beams": 1,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 0,
            "early_stopping": False,
        }
    
    # Add tokenizer related parameters
    gen_args.update({
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True
    })
    
    print(f"📋 Generation arguments: max_new_tokens={gen_args['max_new_tokens']}, temperature={gen_args['temperature']}, top_p={gen_args['top_p']}, top_k={gen_args['top_k']}")
    
    all_results = []
    start_time = time.time()
    
    for i, (bucket_samples, optimal_batch_size) in enumerate(non_empty_buckets):
        print(f"🔄 Processing batch {i+1}/{len(non_empty_buckets)}: {len(bucket_samples)} samples, batch_size={optimal_batch_size}")
        
        # Prepare batch inputs
        texts = prepare_inputs(bucket_samples, tokenizer)
        
        # Batch inference
        outputs = batch_generate(model, tokenizer, texts, gen_args)
        
        # Combine results
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
    
    print(f"\n✅ Optimized inference completed!")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⚡ Average per sample: {total_time/len(test_samples):.2f}s")
    print(f"🚀 Throughput: {len(test_samples)/total_time:.2f} samples/second")
    
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
    """Test single optimized model"""
    model_name = os.path.basename(adapter_path)
    print(f"\n" + "="*80)
    print(f"🚀 Start testing model: {model_name}")
    print("="*80)
    
    model = None
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
        # Load model and tokenizer
        print("🔧 Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(base_model_path, adapter_path, use_flash_attention=True)
        
        # Run optimized inference test
        print("\n" + "="*80)
        print("🔥 Start optimized inference test")
        print("="*80)
        results = run_optimized_inference(model, tokenizer, test_data, gen_args)
        
        # Save results
        output_path = f"/root/autodl-tmp/optimized_results_{model_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        
        # Display performance statistics
        perf = results['performance']
        print(f"\n" + "="*80)
        print(f"📈 {model_name} Performance Statistics Summary")
        print("="*80)
        print(f"📊 Total samples: {perf['total_samples']}")
        print(f"⏱️  Total time: {perf['total_time']:.2f}s")
        print(f"⚡ Average per sample: {perf['avg_time_per_sample']:.2f}s")
        print(f"🚀 Throughput: {perf['throughput']:.2f} samples/second")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure model memory is released
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"Model {model_name} memory released")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("🚀 Optimized inference test started")
    print("="*80)
    
    # Parse command line arguments
    gen_args = parse_gen_args()
    print(f"📋 Using generation arguments: {gen_args}")
    
    # Configuration
    base_model_path = "/root/autodl-tmp/Qwen3-14B"
    saves_path = "/root/autodl-tmp/saves/checkpoint-1476"
    test_data_path = "/root/autodl-tmp/inference_dataset_filtered_8192.json"
    
    # Get all model paths to test and sort them
    model_paths = [os.path.join(saves_path, d) for d in os.listdir(saves_path) if d.startswith("checkpoint-")]
    model_paths.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    model_paths.append(saves_path)  # Add final model

    print(f"Base model path: {base_model_path}")
    print(f"Found {len(model_paths)} models to test")
    
    # Load test data
    print("📊 Loading test data...")
    test_data = load_test_data(test_data_path)
    if test_data is None:
        print("❌ Failed to load test data, exiting test")
        return
    print(f"✅ Successfully loaded {len(test_data)} test samples")
    
    # Test all models
    success_count = 0
    for i, adapter_path in enumerate(model_paths):
        model_name = os.path.basename(adapter_path)
        print(f"\n=== Start testing model {i+1}/{len(model_paths)}: {model_name} ===")
        
        # Test single model, pass generation arguments
        if test_single_optimized_model(base_model_path, adapter_path, test_data, gen_args):
            success_count += 1
        
        # Force garbage collection between models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n" + "="*80)
    print("🎉 All models tested completed")
    print("="*80)
    print(f"Successfully tested {success_count}/{len(model_paths)} models")
    print("="*80)

if __name__ == "__main__":
    main()
