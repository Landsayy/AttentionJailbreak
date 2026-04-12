#!/usr/bin/env python3
"""
This module implements the Push-Pull attention loss for adversarial image
generation against LVLMs. 
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ─── Image Normalization ────────────────────────────────────────────────────

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def normalize_image(images, mean=CLIP_MEAN, std=CLIP_STD):
    """Normalize images from [0, 1] to model-specific distribution."""
    mean = torch.tensor(mean, device=images.device, dtype=images.dtype).view(-1, 1, 1)
    std  = torch.tensor(std,  device=images.device, dtype=images.dtype).view(-1, 1, 1)
    if images.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    return (images - mean) / std


# ─── Base Attacker Class ────────────────────────────────────────────────────

class BaseAttacker:
    """Base class for Push-Pull adversarial attacks on LVLMs.
    
    Subclasses must implement:
        - _get_model_specific_tokens(): Return token IDs for image/system markers
        - _tokenize(): Tokenize text inputs
        - _build_multimodal_embeds(): Build multimodal embeddings from image + text
        - _forward_with_attention(): Run forward pass with attention extraction
    """
    
    # Attention loss hyperparameters
    attention_layers = [-1, -2, -3, -4, -5, -6]  # Last 6 layers by default
    alpha_suppress  = 10.0   # Weight for suppressing system attention
    beta_amplify    = 5.0    # Weight for amplifying image attention
    
    # Training config
    metrics_log_freq = 100
    asr_eval_freq   = 50
    
    def __init__(self, model_path: str, device: str = "cuda",
                 torch_dtype=torch.float16, cache_dir: Optional[str] = None):
        self.model_path  = model_path
        self.device      = device
        self.torch_dtype = torch_dtype
        self.cache_dir   = cache_dir
        
        self.loss_history    = []
        self.metrics_history = []
        self.asr_history     = []
        self.asr_milestones  = {}
        self._debug_printed  = False
        
        self.loss_ce_val   = 0.0
        self.loss_attn_val = 0.0
    
    def _load_target_texts(self, targets_file: Optional[str] = None,
                           use_corpus: bool = False,
                           default_target: str = "I hate you. You are stupid.") -> List[str]:
        """Load target texts from file or corpus."""
        if targets_file:
            with open(targets_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        
        if use_corpus:
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "harmful_corpus", "derogatory_corpus.csv"),
                "harmful_corpus/derogatory_corpus.csv",
            ]
            for path in candidates:
                if os.path.exists(path):
                    targets = []
                    with open(path, 'r', encoding='utf-8') as f:
                        for row in csv.reader(f):
                            if row and len(row[0].strip()) > 2:
                                targets.append(row[0].strip())
                    print(f"[*] Loaded {len(targets)} targets from corpus")
                    return targets
        
        return [default_target]
    
    def _compute_pushpull_loss(self, attn_weights: torch.Tensor,
                               labels: torch.Tensor,
                               input_ids: torch.Tensor) -> tuple:
        """Compute Push-Pull attention loss.
        
        Args:
            attn_weights: [B, L, L] - averaged attention over target layers/heads
            labels:       [B, L] - -100 for non-target tokens
            input_ids:    [B, L] - token IDs for finding image/system positions
        
        Returns:
            (loss_suppress, loss_amplify, loss_attn, info_dict)
        """
        device = attn_weights.device
        B, L = attn_weights.shape[:2]
        
        target_mask  = (labels != -100).float()
        num_targets  = target_mask.sum()
        
        if num_targets == 0:
            return (torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device), {})
        
        # Get token positions (implemented by subclass)
        image_mask, system_mask = self._get_token_masks(input_ids, L, device)
        
        # Row selector: target token positions
        row_mask = target_mask.unsqueeze(2)
        
        # Column selectors
        system_col = system_mask.unsqueeze(1)
        image_col  = image_mask.unsqueeze(1)
        
        # Push-Pull operations
        target_to_system = attn_weights * row_mask * system_col
        target_to_image = attn_weights * row_mask * image_col
        
        loss_suppress = target_to_system.sum() / num_targets
        loss_amplify  = -target_to_image.sum() / num_targets
        
        loss_attn = self.alpha_suppress * loss_suppress + self.beta_amplify * loss_amplify
        
        info = {
            'sys_attn': float(target_to_system.sum().item()) / num_targets.item(),
            'img_attn': float(target_to_image.sum().item()) / num_targets.item(),
            'ratio':    float(target_to_image.sum().item()) / max(target_to_system.sum().item(), 1e-8),
        }
        
        return loss_suppress, loss_amplify, loss_attn, info
    
    def _get_token_masks(self, input_ids, seq_len, device):
        """Get image and system token masks. Must be implemented by subclass."""
        raise NotImplementedError
    
    def _print_debug_info(self, **kwargs):
        """Print debug information on first forward pass."""
        if self._debug_printed:
            return
        self._debug_printed = True
        
        print("\n" + "=" * 70)
        print("[Debug] Push-Pull Attention Loss Initialization")
        print("=" * 70)
        for k, v in kwargs.items():
            print(f"  {k}: {v}")
        print("=" * 70 + "\n")
    
    def save_image(self, image_tensor: torch.Tensor, path: str):
        """Save image tensor [3, H, W] or [H, W, 3] to file."""
        img = image_tensor.detach().cpu()
        if img.dim() == 3 and img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        img = torch.clamp(img, 0.0, 1.0)
        img_np = (img.numpy() * 255.0).astype(np.uint8)
        Image.fromarray(img_np).save(path)
    
    def plot_loss_curve(self, save_dir: str):
        """Save loss curve plot."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, linewidth=1.5, alpha=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("PGD Attack Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
        plt.close()
        torch.save(self.loss_history, os.path.join(save_dir, "loss_history.pt"))
    
    def attack_pgd(self, image_path: str, text_prompt: str, target_texts: List[str],
                   num_iter: int = 2000, alpha: float = 1/255, eps: float = 16/255,
                   constrained: bool = True, save_dir: str = "./attack_results",
                   mini_batch_size: int = 1, seed: int = 42):
        """Main PGD attack loop with Push-Pull attention loss.
        
        Args:
            image_path:     Path to clean input image
            text_prompt:    User instruction (usually empty for universal attack)
            target_texts:   List of target harmful responses
            num_iter:       Number of PGD iterations
            alpha:          Step size (in /255 units)
            eps:            Max perturbation budget (in /255 units)
            constrained:    Whether to use L_inf constraint
            save_dir:       Output directory
            mini_batch_size: Targets sampled per iteration
            seed:           Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        os.makedirs(save_dir, exist_ok=True)
        total_targets = len(target_texts)
        mini_batch_size = min(mini_batch_size, total_targets)
        
        print(f"\n{'=' * 60}")
        print("[*] Push-Pull Attention-Guided Attack")
        print(f"    Model:     {self.model_path}")
        print(f"    Mode:      {'Constrained' if constrained else 'Unconstrained'} "
              f"(ε={int(eps*255)}/255)")
        print(f"    Step:      α={alpha:.4f}, Iter={num_iter}")
        print(f"    Targets:   {total_targets} (batch={mini_batch_size})")
        print(f"    Attn Loss: α={self.alpha_suppress}, β={self.beta_amplify}, "
              f"layers={len(self.attention_layers)}")
        print(f"{'=' * 60}\n")
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        
        x_orig = image_tensor.clone()
        x_adv  = image_tensor.clone().detach().requires_grad_(True)
        
        torch.cuda.empty_cache()
        
        for it in range(num_iter):
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            
            # Sample mini-batch
            if total_targets > mini_batch_size:
                idx = torch.randperm(total_targets)[:mini_batch_size].tolist()
                batch_targets = [target_texts[i] for i in idx]
            else:
                batch_targets = target_texts
            
            # Forward + backward
            if len(batch_targets) == 1:
                loss = self._forward_with_attention(x_adv, text_prompt, batch_targets)
                loss.backward()
                loss_val = float(loss.item())
                del loss
            else:
                sign_votes = torch.zeros_like(x_adv)
                loss_accum = 0.0
                for t in batch_targets:
                    if x_adv.grad is not None:
                        x_adv.grad.zero_()
                    single_loss = self._forward_with_attention(x_adv, text_prompt, [t])
                    single_loss.backward()
                    sign_votes += x_adv.grad.sign()
                    loss_accum += float(single_loss.item())
                    del single_loss
                    torch.cuda.empty_cache()
                loss_val = loss_accum / len(batch_targets)
                x_adv.grad.data.copy_(sign_votes)
            
            self.loss_history.append(loss_val)
            
            # Logging
            if (it + 1) % self.metrics_log_freq == 0 or it == 0:
                info = self._get_attention_info(x_adv, text_prompt, batch_targets[:1])
                ratio = info.get('ratio', 0)
                print(f"[Iter {it+1:05d}/{num_iter}] Loss: {loss_val:.4f} | "
                      f"Sys: {info.get('sys_attn', 0):.4f} | "
                      f"Img: {info.get('img_attn', 0):.4f} | "
                      f"Ratio: {ratio:.2f}x")
                self.metrics_history.append({
                    'iteration': it + 1,
                    'loss': loss_val,
                    'loss_ce':   self.loss_ce_val,
                    'loss_attn': self.loss_attn_val,
                    **info
                })
            
            # PGD update
            with torch.no_grad():
                x_new = x_adv - alpha * x_adv.grad.sign()
                if constrained:
                    delta = torch.clamp(x_new - x_orig, -eps, eps)
                    x_new = x_orig + delta
                x_adv.data = torch.clamp(x_new, 0.0, 1.0)
            
            if (it + 1) % 100 == 0:
                torch.cuda.empty_cache()
        
        print("\n[✓] Attack finished!")
        
        # Save results
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        self.save_image(x_orig[0], os.path.join(save_dir, "images", "clean.png"))
        self.save_image(x_adv[0],  os.path.join(save_dir, "adversarial.png"))
        self.plot_loss_curve(save_dir)
        
        config = {
            'model_path': self.model_path,
            'num_iter': num_iter, 'alpha': float(alpha),
            'eps': float(eps) if constrained else None,
            'constrained': constrained,
            'attention_layers': self.attention_layers,
            'alpha_suppress': self.alpha_suppress,
            'beta_amplify':   self.beta_amplify,
            'final_loss': float(self.loss_history[-1]),
        }
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[✓] Results saved to: {save_dir}/")
        return x_adv[0].detach()
    
    def _forward_with_attention(self, image_tensor, text_prompt, target_texts):
        """Forward pass with Push-Pull loss. Must be implemented by subclass."""
        raise NotImplementedError
    
    def _get_attention_info(self, image_tensor, text_prompt, target_texts):
        """Get attention metrics without gradients."""
        return {'sys_attn': 0, 'img_attn': 0, 'ratio': 0}


# ─── LLaVA-1.5 Attacker ─────────────────────────────────────────────────────

class LLaVAAttacker(BaseAttacker):
    """Push-Pull attack for LLaVA-1.5 models."""
    
    def __init__(self, model_path: str = "llava-hf/llava-1.5-7b-hf",
                 device: str = "cuda", torch_dtype=torch.float16,
                 cache_dir: Optional[str] = None):
        super().__init__(model_path, device, torch_dtype, cache_dir)
        
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        print(f"[*] Loading LLaVA: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype,
            low_cpu_mem_usage=True, cache_dir=cache_dir,
            attn_implementation="eager"
        ).to(device).eval()
        
        # Force eager attention for attention extraction
        if hasattr(self.model, 'language_model'):
            self.model.language_model.config._attn_implementation = "eager"
        
        for p in self.model.parameters():
            p.requires_grad = False
        
        print("[✓] LLaVA loaded")
        self.image_token_id = self.model.config.image_token_index
    
    def _get_token_masks(self, input_ids, seq_len, device):
        B = input_ids.shape[0]
        
        # Image tokens: find <image> token position
        image_mask = torch.zeros(B, seq_len, device=device)
        system_mask = torch.zeros(B, seq_len, device=device)
        
        for i in range(B):
            img_pos = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) > 0:
                system_mask[i, :img_pos[0].item()] = 1.0
                # Image token occupies one position, but we count image attention
                # from the expanded image embeddings
                image_mask[i, img_pos[0].item()] = 1.0
            else:
                # Fallback: first 10% is system
                system_end = max(1, seq_len // 10)
                system_mask[i, :system_end] = 1.0
        
        return image_mask, system_mask
    
    def _tokenize(self, prompts, targets=None):
        """Tokenize prompts and full conversations."""
        tokenizer = self.processor.tokenizer
        
        prompt_texts = []
        for p in prompts:
            if p:
                prompt_texts.append(f"USER: <image>\n{p}\nASSISTANT:")
            else:
                prompt_texts.append("USER: <image>\nASSISTANT:")
        
        prompt_tokens = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.device)
        
        if targets:
            full_texts = [f"{prompt_texts[i]} {targets[i]}" for i in range(len(prompts))]
            full_tokens = tokenizer(full_texts, return_tensors="pt", padding=True).to(self.device)
        else:
            full_tokens = prompt_tokens
        
        return prompt_tokens, full_tokens
    
    def _build_multimodal_embeds(self, full_tokens, prompt_tokens, image_tensor):
        """Build multimodal embeddings with image features inserted."""
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        B = full_tokens.input_ids.shape[0]
        
        # Vision features
        images_norm = normalize_image(image_tensor)
        target_size = self.model.config.vision_config.image_size
        if images_norm.shape[-1] != target_size:
            images_norm = F.interpolate(images_norm, size=(target_size, target_size),
                                        mode="bilinear", align_corners=False)
        
        vision_out = self.model.vision_tower(images_norm, output_hidden_states=True)
        select_layer = self.model.config.vision_feature_layer
        img_features = vision_out.hidden_states[select_layer]
        if self.model.config.vision_feature_select_strategy == "default":
            img_features = img_features[:, 1:]
        img_embeds = self.model.multi_modal_projector(img_features)
        
        # Text embeddings
        txt_embeds = self.model.get_input_embeddings()(full_tokens.input_ids)
        
        # Build final embeddings with image tokens replaced
        final_embeds_list = []
        labels_list = []
        indices_list = []
        
        for i in range(B):
            full_ids  = full_tokens.input_ids[i]
            prompt_ids = prompt_tokens.input_ids[i]
            txt_emb   = txt_embeds[i]
            img_emb   = img_embeds[i]
            
            # Find <image> token
            img_pos = (full_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                continue
            img_idx = img_pos[0].item()
            
            # Prompt length (for labels)
            prompt_len = (prompt_ids != pad_id).sum().item()
            non_pad = (full_ids != pad_id).nonzero(as_tuple=True)[0]
            content_start = non_pad[0].item() if len(non_pad) > 0 else 0
            target_start  = content_start + prompt_len
            
            # Labels: only on target tokens
            labels = torch.full_like(full_ids, -100)
            target_ids = full_ids[target_start:]
            target_ids = target_ids[target_ids != pad_id]
            if len(target_ids) > 0:
                labels[target_start:target_start + len(target_ids)] = target_ids
            
            # Build multimodal embedding
            before = txt_emb[:img_idx]
            after  = txt_emb[img_idx + 1:]
            mid    = img_emb
            mid_lbl = torch.full((img_emb.shape[0],), -100, device=txt_emb.device, dtype=torch.long)
            
            new_emb = torch.cat([before, mid, after], dim=0)
            new_lbl = torch.cat([labels[:img_idx], mid_lbl, labels[img_idx + 1:]], dim=0)
            
            # Record indices for attention masking
            indices_list.append({
                'system_end':  img_idx,
                'image_start': img_idx,
                'image_end':   img_idx + img_emb.shape[0],
            })
            
            final_embeds_list.append(new_emb)
            labels_list.append(new_lbl)
        
        # Pad to same length
        max_len = max(e.shape[0] for e in final_embeds_list)
        padded_embeds, padded_labels, masks = [], [], []
        
        for emb, lbl in zip(final_embeds_list, labels_list):
            pad_len = max_len - emb.shape[0]
            padded_embeds.append(F.pad(emb, (0, 0, 0, pad_len)))
            padded_labels.append(F.pad(lbl, (0, pad_len), value=-100))
            masks.append(F.pad(torch.ones(emb.shape[0], device=emb.device),
                              (0, pad_len), value=0))
        
        return (torch.stack(padded_embeds), torch.stack(padded_labels),
                torch.stack(masks).long(), indices_list)
    
    def _forward_with_attention(self, image_tensor, text_prompt, target_texts):
        B = len(target_texts)
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt] * B
        
        prompt_tokens, full_tokens = self._tokenize(text_prompt, target_texts)
        embeds, labels, attn_mask, indices = self._build_multimodal_embeds(
            full_tokens, prompt_tokens, image_tensor)
        
        lm_out = self.model.language_model(
            inputs_embeds=embeds, attention_mask=attn_mask,
            return_dict=True, output_attentions=True)
        
        # Aggregate attention over target layers
        num_layers = len(lm_out.attentions)
        valid_layers = [i for i in self.attention_layers if abs(i) < num_layers]
        attn_list = [lm_out.attentions[i].mean(dim=1) for i in valid_layers]
        attn_weights = torch.stack(attn_list, dim=0).mean(dim=0)
        
        # Compute Push-Pull loss
        loss_suppress, loss_amplify, loss_attn, info = self._compute_pushpull_loss(
            attn_weights, labels, full_tokens.input_ids)
        
        # CE loss
        logits = lm_out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100)
        
        loss_total = loss_ce + loss_attn
        
        self.loss_ce_val   = float(loss_ce.item())
        self.loss_attn_val = float(loss_attn.item())
        
        if not self._debug_printed:
            self._print_debug_info(
                model="LLaVA-1.5",
                attn_layers=valid_layers,
                total_layers=num_layers,
                embed_shape=embeds.shape,
                ce_loss=loss_ce.item(),
                suppress_loss=loss_suppress.item(),
                amplify_loss=loss_amplify.item(),
                alpha=self.alpha_suppress, beta=self.beta_amplify,
            )
        
        return loss_total
    
    def _get_attention_info(self, image_tensor, text_prompt, target_texts):
        """Get attention metrics without gradients."""
        with torch.no_grad():
            B = len(target_texts)
            if isinstance(text_prompt, str):
                text_prompt = [text_prompt] * B
            prompt_tokens, full_tokens = self._tokenize(text_prompt, target_texts)
            embeds, labels, attn_mask, _ = self._build_multimodal_embeds(
                full_tokens, prompt_tokens, image_tensor)
            
            lm_out = self.model.language_model(
                inputs_embeds=embeds, attention_mask=attn_mask,
                return_dict=True, output_attentions=True)
            
            num_layers = len(lm_out.attentions)
            valid_layers = [i for i in self.attention_layers if abs(i) < num_layers]
            attn_list = [lm_out.attentions[i].mean(dim=1) for i in valid_layers]
            attn_weights = torch.stack(attn_list, dim=0).mean(dim=0)
            
            _, _, _, info = self._compute_pushpull_loss(
                attn_weights, labels, full_tokens.input_ids)
            return info


# ─── Qwen-VL Attacker ──────────────────────────────────────────────────────

class QwenVLAttacker(BaseAttacker):
    """Push-Pull attack for Qwen-VL models."""
    
    def __init__(self, model_path: str = "Qwen/Qwen-VL-Chat",
                 device: str = "cuda", torch_dtype=torch.bfloat16,
                 cache_dir: Optional[str] = None):
        super().__init__(model_path, device, torch_dtype, cache_dir)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"[*] Loading Qwen-VL: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, cache_dir=cache_dir)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = 151643
            self.tokenizer.pad_token = '<|endoftext|>'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True,
            cache_dir=cache_dir, torch_dtype=torch_dtype).eval()
        
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.img_start_id = getattr(self.tokenizer, 'img_start_id', None)
        self.img_end_id   = getattr(self.tokenizer, 'img_end_id', None)
        
        print("[✓] Qwen-VL loaded")
    
    def _tokenize(self, prompts, targets=None):
        """Tokenize using Qwen-VL's from_list_format."""
        full_texts, prompt_texts = [], []
        
        for i, p in enumerate(prompts):
            query_p = self.tokenizer.from_list_format([
                {'image': 'PLACEHOLDER_IMAGE'},
                {'text': p}])
            prompt_texts.append(query_p)
            
            if targets:
                target = targets[i] if i < len(targets) else ""
                query_f = self.tokenizer.from_list_format([
                    {'image': 'PLACEHOLDER_IMAGE'},
                    {'text': p + target}])
                full_texts.append(query_f)
        
        prompt_tokens = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.device)
        if targets:
            full_tokens = self.tokenizer(full_texts, return_tensors="pt", padding=True).to(self.device)
        else:
            full_tokens = prompt_tokens
        
        return prompt_tokens, full_tokens
    
    def _get_token_masks(self, input_ids, seq_len, device):
        B = input_ids.shape[0]
        image_mask  = torch.zeros(B, seq_len, device=device)
        system_mask = torch.zeros(B, seq_len, device=device)
        
        for i in range(B):
            if self.img_start_id is not None and self.img_end_id is not None:
                start_pos = (input_ids[i] == self.img_start_id).nonzero(as_tuple=True)[0]
                end_pos   = (input_ids[i] == self.img_end_id).nonzero(as_tuple=True)[0]
                if len(start_pos) > 0:
                    system_mask[i, :start_pos[0].item()] = 1.0
                    # Count attention to image region
                    image_mask[i, start_pos[0].item()+1:end_pos[0].item()] = 1.0
            else:
                system_end = max(1, int(seq_len * 0.05))
                system_mask[i, :system_end] = 1.0
                image_mask[i, system_end:int(seq_len * 0.5)] = 1.0
        
        return image_mask, system_mask
    
    def _forward_with_attention(self, image_tensor, text_prompt, target_texts):
        import tempfile
        
        B = len(target_texts)
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt] * B
        
        # Save image to temp file for tokenizer
        img_np = image_tensor[0].detach().cpu()
        img_np = torch.clamp(img_np, 0.0, 1.0)
        img_np = (img_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            Image.fromarray(img_np).save(tmp.name)
            tmp_path = tmp.name
        
        try:
            full_texts = []
            for i in range(B):
                query = self.tokenizer.from_list_format([
                    {'image': tmp_path},
                    {'text': text_prompt[i] + target_texts[i]}
                ])
                full_texts.append(query)
            
            prompt_texts = []
            for i in range(B):
                query = self.tokenizer.from_list_format([
                    {'image': tmp_path},
                    {'text': text_prompt[i]}
                ])
                prompt_texts.append(query)
            
            inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True).to(self.device)
            prompts_t = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.device)
            
            input_ids = inputs['input_ids']
            labels = input_ids.clone()
            
            for i in range(B):
                p_len = (prompts_t.input_ids[i] != self.tokenizer.pad_token_id).sum().item()
                non_pad = (input_ids[i] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start = non_pad[0].item() if len(non_pad) > 0 else 0
                labels[i, :start + p_len] = -100
            
            # Patch visual encoder
            img_resized = image_tensor
            if img_resized.shape[-1] != 448:
                img_resized = F.interpolate(img_resized, size=(448, 448),
                                            mode="bilinear", align_corners=False)
            img_norm = normalize_image(img_resized)
            
            orig_encode = self.model.transformer.visual.encode
            def patched_encode(_):
                n = B
                batch_img = img_norm.to(self.device).repeat(n, 1, 1, 1)
                return self.model.transformer.visual(batch_img)
            self.model.transformer.visual.encode = patched_encode
            
            try:
                outputs = self.model(
                    input_ids=input_ids, attention_mask=inputs['attention_mask'],
                    labels=labels, return_dict=True, output_attentions=True)
            finally:
                self.model.transformer.visual.encode = orig_encode
            
            loss_ce = outputs.loss
            
            num_layers = len(outputs.attentions)
            valid_layers = [i for i in self.attention_layers if -num_layers <= i < num_layers]
            attn_list = [outputs.attentions[i].mean(dim=1) for i in valid_layers]
            attn_weights = torch.stack(attn_list, dim=0).mean(dim=0)
            
            _, _, loss_attn, info = self._compute_pushpull_loss(
                attn_weights, labels, input_ids)
            
            loss_total = loss_ce + loss_attn
            
            self.loss_ce_val   = float(loss_ce.item())
            self.loss_attn_val = float(loss_attn.item())
            
            if not self._debug_printed:
                self._print_debug_info(
                    model="Qwen-VL",
                    attn_layers=valid_layers, total_layers=num_layers,
                    ce_loss=loss_ce.item(), info=info)
            
            return loss_total
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _get_attention_info(self, image_tensor, text_prompt, target_texts):
        with torch.no_grad():
            import tempfile
            B = len(target_texts)
            if isinstance(text_prompt, str):
                text_prompt = [text_prompt] * B
            
            img_np = image_tensor[0].detach().cpu()
            img_np = torch.clamp(img_np, 0.0, 1.0)
            img_np = (img_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                Image.fromarray(img_np).save(tmp.name)
                tmp_path = tmp.name
            
            try:
                full_texts = [self.tokenizer.from_list_format([
                    {'image': tmp_path}, {'text': text_prompt[i] + target_texts[i]}
                ]) for i in range(B)]
                prompt_texts = [self.tokenizer.from_list_format([
                    {'image': tmp_path}, {'text': text_prompt[i]}
                ]) for i in range(B)]
                
                inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True).to(self.device)
                prompts_t = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.device)
                input_ids = inputs['input_ids']
                labels = input_ids.clone()
                
                for i in range(B):
                    p_len = (prompts_t.input_ids[i] != self.tokenizer.pad_token_id).sum().item()
                    non_pad = (input_ids[i] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                    start = non_pad[0].item() if len(non_pad) > 0 else 0
                    labels[i, :start + p_len] = -100
                
                img_norm = normalize_image(F.interpolate(image_tensor, size=(448, 448),
                                mode="bilinear", align_corners=False))
                
                orig = self.model.transformer.visual.encode
                self.model.transformer.visual.encode = lambda _: img_norm.to(self.device).repeat(B, 1, 1, 1)
                outputs = self.model(input_ids=input_ids, attention_mask=inputs['attention_mask'],
                                     return_dict=True, output_attentions=True)
                self.model.transformer.visual.encode = orig
                
                num_layers = len(outputs.attentions)
                valid = [i for i in self.attention_layers if -num_layers <= i < num_layers]
                attn = torch.stack([outputs.attentions[i].mean(dim=1) for i in valid], dim=0).mean(dim=0)
                _, _, _, info = self._compute_pushpull_loss(attn, labels, input_ids)
                return info
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


# ─── InternVL2 Attacker ─────────────────────────────────────────────────────

class InternVLAttacker(BaseAttacker):
    """Push-Pull attack for InternVL2 models."""
    
    IMG_START   = '<img>'
    IMG_END     = '</img>'
    IMG_CONTEXT = '<IMG_CONTEXT>'
    
    def __init__(self, model_path: str = "OpenGVLab/InternVL2-8B",
                 device: str = "cuda", torch_dtype=torch.bfloat16,
                 cache_dir: Optional[str] = None):
        super().__init__(model_path, device, torch_dtype, cache_dir)
        
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        print(f"[*] Loading InternVL2: {model_path}")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
        if hasattr(config, 'llm_config'):
            config.llm_config.attn_implementation = 'eager'
        if hasattr(config, 'vision_config'):
            config.vision_config.use_flash_attn = False
        
        self.model = AutoModel.from_pretrained(
            model_path, config=config, torch_dtype=torch_dtype,
            trust_remote_code=True, low_cpu_mem_usage=True,
            cache_dir=cache_dir).to(device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Token IDs
        self.img_ctx_id  = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT)
        self.img_start_id = self.tokenizer.convert_tokens_to_ids(self.IMG_START)
        self.img_end_id   = self.tokenizer.convert_tokens_to_ids(self.IMG_END)
        self.num_image_token = self.model.num_image_token
        self.image_size = (self.model.config.force_image_size
                          or self.model.config.vision_config.image_size)
        
        print("[✓] InternVL2 loaded")
    
    def _tokenize(self, prompts, targets=None):
        full_strs, prompt_strs = [], []
        num_patches = 1
        
        for i, p in enumerate(prompts):
            img_tokens = (self.IMG_START + self.IMG_CONTEXT * self.num_image_token * num_patches
                         + self.IMG_END)
            question = f"{img_tokens}\n{p}" if p else img_tokens
            
            template = copy.deepcopy(self.model.conv_template)
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            prompt_str = template.get_prompt()
            
            prompt_strs.append(prompt_str)
            if targets:
                full_strs.append(prompt_str + (targets[i] if i < len(targets) else ""))
        
        prompt_tokens = self.tokenizer(prompt_strs, return_tensors='pt', padding=True).to(self.device)
        if targets:
            full_tokens = self.tokenizer(full_strs, return_tensors='pt', padding=True).to(self.device)
        else:
            full_tokens = prompt_tokens
        
        return prompt_tokens, full_tokens
    
    def _get_token_masks(self, input_ids, seq_len, device):
        B = input_ids.shape[0]
        image_mask  = torch.zeros(B, seq_len, device=device)
        system_mask = torch.zeros(B, seq_len, device=device)
        
        for i in range(B):
            img_s = (input_ids[i] == self.img_start_id).nonzero(as_tuple=True)[0]
            if len(img_s) > 0:
                system_mask[i, :img_s[0].item()] = 1.0
            image_mask[i] = (input_ids[i] == self.img_ctx_id).float()
        
        return image_mask, system_mask
    
    def _forward_with_attention(self, image_tensor, text_prompt, target_texts):
        B = len(target_texts)
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt] * B
        
        _, full_tokens = self._tokenize(text_prompt, target_texts)
        prompt_tokens, _ = self._tokenize(text_prompt)
        
        input_ids = full_tokens.input_ids
        attn_mask = full_tokens.attention_mask
        
        # Labels
        labels = torch.full_like(input_ids, -100)
        for i in range(B):
            p_len = (prompt_tokens.input_ids[i] != self.tokenizer.pad_token_id).sum().item()
            non_pad = (input_ids[i] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            cs = non_pad[0].item() if len(non_pad) > 0 else 0
            target_start = cs + p_len
            full_valid = (input_ids[i] != self.tokenizer.pad_token_id).sum().item()
            target_end = cs + full_valid
            if target_start < target_end:
                labels[i, target_start:target_end] = input_ids[i, target_start:target_end]
        
        # Vision features
        img_norm = normalize_image(image_tensor.float(), IMAGENET_MEAN, IMAGENET_STD)
        if img_norm.shape[-1] != self.image_size:
            img_norm = F.interpolate(img_norm, size=(self.image_size, self.image_size),
                                     mode='bilinear', align_corners=False)
        img_norm = img_norm.to(self.torch_dtype)
        vit_embeds = self.model.extract_feature(img_norm)
        
        # Scatter into text embeddings
        B_, L_ = input_ids.shape
        txt_emb = self.model.language_model.get_input_embeddings()(input_ids).clone()
        C = txt_emb.shape[-1]
        
        flat_emb = txt_emb.reshape(B_ * L_, C)
        flat_ids = input_ids.reshape(B_ * L_)
        selected = (flat_ids == self.img_ctx_id)
        
        vit_flat = vit_embeds.repeat(B_, 1, 1).reshape(-1, C)
        flat_emb[selected] = flat_emb[selected] * 0.0 + vit_flat[selected.sum():].to(flat_emb.dtype)
        input_embeds = flat_emb.reshape(B_, L_, C)
        
        # Forward
        lm_out = self.model.language_model(
            inputs_embeds=input_embeds, attention_mask=attn_mask,
            return_dict=True, output_attentions=True)
        
        num_layers = len(lm_out.attentions)
        valid = [i for i in self.attention_layers if -num_layers <= i < num_layers]
        attn_list = [lm_out.attentions[i].mean(dim=1) for i in valid]
        attn_weights = torch.stack(attn_list, dim=0).mean(dim=0)
        
        loss_suppress, loss_amplify, loss_attn, info = self._compute_pushpull_loss(
            attn_weights, labels, input_ids)
        
        logits = lm_out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100)
        
        loss_total = loss_ce + loss_attn
        
        self.loss_ce_val   = float(loss_ce.item())
        self.loss_attn_val = float(loss_attn.item())
        
        if not self._debug_printed:
            self._print_debug_info(
                model="InternVL2",
                attn_layers=valid, total_layers=num_layers,
                ce_loss=loss_ce.item(), info=info)
        
        return loss_total
    
    def _get_attention_info(self, image_tensor, text_prompt, target_texts):
        with torch.no_grad():
            B = len(target_texts)
            if isinstance(text_prompt, str):
                text_prompt = [text_prompt] * B
            
            _, full_tokens = self._tokenize(text_prompt, target_texts)
            prompt_tokens, _ = self._tokenize(text_prompt)
            
            input_ids = full_tokens.input_ids
            attn_mask = full_tokens.attention_mask
            
            labels = torch.full_like(input_ids, -100)
            for i in range(B):
                p_len = (prompt_tokens.input_ids[i] != self.tokenizer.pad_token_id).sum().item()
                non_pad = (input_ids[i] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                cs = non_pad[0].item() if len(non_pad) > 0 else 0
                target_start = cs + p_len
                full_valid = (input_ids[i] != self.tokenizer.pad_token_id).sum().item()
                target_end = cs + full_valid
                if target_start < target_end:
                    labels[i, target_start:target_end] = input_ids[i, target_start:target_end]
            
            img_norm = normalize_image(image_tensor.float(), IMAGENET_MEAN, IMAGENET_STD)
            if img_norm.shape[-1] != self.image_size:
                img_norm = F.interpolate(img_norm, size=(self.image_size, self.image_size),
                                        mode='bilinear', align_corners=False)
            img_norm = img_norm.to(self.torch_dtype)
            vit_embeds = self.model.extract_feature(img_norm)
            
            B_, L_ = input_ids.shape
            txt_emb = self.model.language_model.get_input_embeddings()(input_ids).clone()
            C = txt_emb.shape[-1]
            
            flat_emb = txt_emb.reshape(B_ * L_, C)
            flat_ids = input_ids.reshape(B_ * L_)
            selected = (flat_ids == self.img_ctx_id)
            
            vit_flat = vit_embeds.repeat(B_, 1, 1).reshape(-1, C)
            flat_emb[selected] = flat_emb[selected] * 0.0 + vit_flat[selected.sum():].to(flat_emb.dtype)
            input_embeds = flat_emb.reshape(B_, L_, C)
            
            lm_out = self.model.language_model(
                inputs_embeds=input_embeds, attention_mask=attn_mask,
                return_dict=True, output_attentions=True)
            
            num_layers = len(lm_out.attentions)
            valid = [i for i in self.attention_layers if -num_layers <= i < num_layers]
            attn_list = [lm_out.attentions[i].mean(dim=1) for i in valid]
            attn_weights = torch.stack(attn_list, dim=0).mean(dim=0)
            
            _, _, _, info = self._compute_pushpull_loss(attn_weights, labels, input_ids)
            return info


# ─── CLI Entry Point ────────────────────────────────────────────────────────

ATTACKER_REGISTRY = {
    'llava':   LLaVAAttacker,
    'qwen':    QwenVLAttacker,
    'internvl': InternVLAttacker,
}


def main():
    parser = argparse.ArgumentParser(
        description="Push-Pull Attention-Guided Visual Jailbreaking (ACL 2026)")
    
    parser.add_argument("--model", type=str, default="llava",
                       choices=list(ATTACKER_REGISTRY.keys()),
                       help="Model type: llava, qwen, internvl")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--image_path", type=str, required=True,
                       help="Clean input image")
    parser.add_argument("--prompt", type=str, default="",
                       help="User instruction text")
    parser.add_argument("--targets_file", type=str, default=None)
    parser.add_argument("--target", type=str, default="I hate you.")
    parser.add_argument("--use_corpus", action="store_true")
    
    parser.add_argument("--num_iter", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Step size (unit: /255)")
    parser.add_argument("--eps", type=float, default=16.0,
                       help="Max perturbation (unit: /255)")
    parser.add_argument("--constrained", action="store_true")
    
    parser.add_argument("--save_dir", type=str, default="./attack_results")
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--attn_layers", type=str, default="last-6")
    parser.add_argument("--alpha_suppress", type=float, default=10.0)
    parser.add_argument("--beta_amplify",   type=float, default=5.0)
    
    args = parser.parse_args()
    
    # Parse layer config
    layer_map = {
        'last-1': [-1], 'last-2': [-1, -2],
        'last-3': [-1, -2, -3], 'last-6': [-1, -2, -3, -4, -5, -6],
    }
    layers = layer_map.get(args.attn_layers, [-1, -2, -3, -4, -5, -6])
    
    # Unit conversion
    alpha = args.alpha / 255.0
    eps   = args.eps   / 255.0
    
    # Auto save_dir
    if args.save_dir == "./attack_results":
        model_name = os.path.basename(args.model_path.rstrip('/'))
        eps_int = int(round(eps * 255))
        args.save_dir = f"results/{model_name}_eps{eps_int}_iter{args.num_iter}"
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load targets
    target_texts = [args.target]
    if args.use_corpus:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "harmful_corpus", "derogatory_corpus.csv"),
            "harmful_corpus/derogatory_corpus.csv",
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    target_texts = [r[0].strip() for r in csv.reader(f) if r and len(r[0].strip()) > 2]
                print(f"[*] Loaded {len(target_texts)} targets from corpus")
                break
    elif args.targets_file:
        with open(args.targets_file, 'r') as f:
            target_texts = [l.strip() for l in f if l.strip()]
    
    # Initialize attacker
    attacker_cls = ATTACKER_REGISTRY[args.model]
    dtype_map = {'qwen': torch.bfloat16, 'internvl': torch.bfloat16, 'llava': torch.float16}
    attacker = attacker_cls(
        model_path=args.model_path, device=args.device,
        torch_dtype=dtype_map.get(args.model, torch.float16),
        cache_dir=args.cache_dir)
    
    attacker.attention_layers = layers
    attacker.alpha_suppress   = args.alpha_suppress
    attacker.beta_amplify     = args.beta_amplify
    
    print(f"\n[*] Config: layers={layers}, α={args.alpha_suppress}, β={args.beta_amplify}")
    
    attacker.attack_pgd(
        image_path=args.image_path, text_prompt=args.prompt,
        target_texts=target_texts, num_iter=args.num_iter,
        alpha=alpha, eps=eps, constrained=args.constrained,
        save_dir=args.save_dir, mini_batch_size=args.mini_batch_size,
        seed=args.seed)
    
    print("\n[✓] Done!")


if __name__ == "__main__":
    main()
