#!/usr/bin/env python3
"""
Response Generation for Adversarial Attack Evaluation
Given an adversarial image and benchmark prompts, generate model responses.
"""

import argparse
import json
import os
import torch
from tqdm import tqdm


def generate_llava(image_path: str, input_file: str, output_file: str,
                  model_path: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
    """Generate responses using LLaVA-1.5."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from PIL import Image
    
    print(f"[*] Loading LLaVA from {model_path}")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    
    image = Image.open(image_path).convert("RGB")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    prompts = data if isinstance(data, list) else data.get('prompts', [])
    
    results = []
    for entry in tqdm(prompts, desc="LLaVA generating"):
        question = entry['question']
        entry_id = entry.get('id', 'unknown')
        
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=True,
                                       temperature=0.7, top_p=0.9)
        
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response
        
        results.append({"id": entry_id, "question": question, "answer": answer})
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved {len(results)} responses to {output_file}")


def generate_qwen(image_path: str, input_file: str, output_file: str,
                  model_path: str = "Qwen/Qwen-VL-Chat", device: str = "cuda"):
    """Generate responses using Qwen-VL."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    
    print(f"[*] Loading Qwen-VL from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16).eval()
    
    image_path_abs = os.path.abspath(image_path)
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    prompts = data if isinstance(data, list) else data.get('prompts', [])
    
    results = []
    for entry in tqdm(prompts, desc="Qwen-VL generating"):
        question = entry['question']
        entry_id = entry.get('id', 'unknown')
        
        query = tokenizer.from_list_format([{'image': image_path_abs}, {'text': question}])
        
        with torch.no_grad():
            response, _ = model.chat(tokenizer, query=query, history=None)
        
        results.append({"id": entry_id, "question": question, "answer": response})
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved {len(results)} responses to {output_file}")


def generate_internvl(image_path: str, input_file: str, output_file: str,
                      model_path: str = "OpenGVLab/InternVL2-8B", device: str = "cuda",
                      max_new_tokens: int = 100):
    """Generate responses using InternVL2."""
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from PIL import Image
    from torchvision import transforms
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    
    print(f"[*] Loading InternVL2 from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.llm_config.attn_implementation = 'eager'
    
    model = AutoModel.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    pixel_values = transform(image).unsqueeze(0).to(device, torch.bfloat16)
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    prompts = data if isinstance(data, list) else data.get('prompts', [])
    
    gen_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    results = []
    
    for entry in tqdm(prompts, desc="InternVL2 generating"):
        question = entry['question']
        entry_id = entry.get('id', 'unknown')
        
        with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, question,
                               gen_config, num_patches_list=[1])
        
        results.append({"id": entry_id, "question": question, "answer": response})
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved {len(results)} responses to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for evaluation")
    parser.add_argument("--model", type=str, required=True,
                       choices=["llava", "qwen", "internvl"],
                       help="Model type")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input JSON file with {question, id} entries")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Adversarial image path")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSON file for responses")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()
    
    model_path = args.model_path or {
        "llava":   "llava-hf/llava-1.5-7b-hf",
        "qwen":    "Qwen/Qwen-VL-Chat",
        "internvl": "OpenGVLab/InternVL2-8B",
    }[args.model]
    
    if args.model == "llava":
        generate_llava(args.image_path, args.input_file, args.output_file,
                      model_path, args.device)
    elif args.model == "qwen":
        generate_qwen(args.image_path, args.input_file, args.output_file,
                     model_path, args.device)
    elif args.model == "internvl":
        generate_internvl(args.image_path, args.input_file, args.output_file,
                        model_path, args.device, args.max_new_tokens)


if __name__ == "__main__":
    main()
