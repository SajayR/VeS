"""
Audio encoder using HuBERT for audio embedding extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import torch._dynamo


class AudioEmbedder(nn.Module):
    """
    Audio encoder that extracts embeddings from raw audio using HuBERT.
    
    Features:
    - Uses HuBERT as the base encoder
    - Applies downsampling and projection layers
    - Supports LoRA fine-tuning (currently commented out)
    - Handles attention mask downsampling
    """

    def __init__(self, embedding_dim: int = 256, hubert_name: str = "ntu-spml/distilhubert", device: str = "auto"):
        """
        Initialize AudioEmbedder.
        
        Args:
            embedding_dim: Final embedding dimension
            hubert_name: HuBERT model name from HuggingFace
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        super().__init__()
        
        self.hubert = AutoModel.from_pretrained(
            hubert_name,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        
        # LoRA configuration (currently disabled)
        # lora_cfg = LoraConfig(
        #     r=128,
        #     lora_alpha=128,
        #     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        #     bias="none",
        # )
        # self.hubert = get_peft_model(self.hubert, lora_cfg)
        
        self.hubert.gradient_checkpointing_enable()

        # Projection layers
        self.projection1 = nn.Linear(self.hubert.config.hidden_size, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.projection2 = nn.Linear(256, embedding_dim)
        
        self.downsample_factor = self._compute_downsample_factor()
        print(f"AudioEmbedder: Downsample factor: {self.downsample_factor}")
            
        # Enable gradients for all parameters
        for param in self.hubert.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True
        for param in self.layer_norm.parameters():
            param.requires_grad = True

        print(f"AudioEmbedder initialized with HuBERT ({hubert_name})")
    
    def _compute_downsample_factor(self) -> int:
        """
        Compute the downsampling factor from HuBERT's feature extractor.
        
        Returns:
            Downsampling factor (product of all stride values)
        """
        downsample_factor = 1
        if hasattr(self.hubert, 'feature_extractor'):
            for layer in self.hubert.feature_extractor.conv_layers:
                if hasattr(layer, 'conv'):
                    downsample_factor *= layer.conv.stride[0]
        else:
            downsample_factor = 320  # Default for HuBERT
        return downsample_factor
    
    def _downsample_attention_mask(self, attention_mask: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Downsample the attention mask to match the output sequence length.
        
        Args:
            attention_mask: Original attention mask (B, input_length)
            target_length: Target sequence length
            
        Returns:
            Downsampled attention mask (B, target_length)
        """
        if attention_mask is None:
            return None
        
        # Use adaptive pooling for robust downsampling
        mask_float = attention_mask.float().unsqueeze(1)  # (B, 1, input_length)
        downsampled_mask = F.adaptive_avg_pool1d(mask_float, target_length)  # (B, 1, target_length)
        downsampled_mask = (downsampled_mask.squeeze(1) > 0.5).long()  # (B, target_length)
        
        return downsampled_mask
    
    @torch._dynamo.disable()
    def forward(self, audio_input: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the audio encoder.
        
        Args:
            audio_input: Raw audio waveform (B, T)
            attention_mask: Audio attention mask (B, T)
            
        Returns:
            Tuple of:
                - audio_feats: Normalized audio embeddings (B, Na//REDUCTION, D)
                - output_attention_mask: Downsampled attention mask (B, Na//REDUCTION)
        """
        assert attention_mask is not None, "Attention mask is required"
        
        # Extract HuBERT features
        hubert_out = self.hubert(
            audio_input,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state  # (B, Na, H)
        
        # Apply temporal downsampling
        REDUCTION = 2
        hubert_out = hubert_out.transpose(1, 2)  # (B, H, Na)
        hubert_out = F.avg_pool1d(
            hubert_out, kernel_size=REDUCTION, stride=REDUCTION
        )  # (B, H, Na//2)
        hubert_out = hubert_out.transpose(1, 2)  # (B, Na//2, H)

        # Downsample attention mask accordingly
        mask_ds = self._downsample_attention_mask(
            attention_mask, hubert_out.size(1) * REDUCTION
        )
        output_attention_mask = mask_ds[:, ::REDUCTION]
        
        # Apply projection layers
        feats = self.layer_norm(self.projection1(hubert_out))
        feats = self.projection2(feats)
        feats = F.normalize(feats, dim=-1)  # (B, Na//2, D)
        
        return feats, output_attention_mask
    
    def unfreeze_hubert(self):
        """Unfreeze HuBERT encoder for fine-tuning (currently no-op)."""
        pass