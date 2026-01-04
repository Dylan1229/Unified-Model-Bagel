# Patch-based diffusion utilities

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class PatchInfo:
    """Metadata for patch-based diffusion processing."""
    num_patches: int                           # Total number of patches
    patches_per_image: List[int]               # Number of patches per image
    patch_h: int                               # Patches along height
    patch_w: int                               # Patches along width
    latent_patch_size: int                     # Size of each patch in latent space (tokens per side)
    original_h: int                            # Original latent height (in tokens)
    original_w: int                            # Original latent width (in tokens)
    latent_offsets: List[int]                  # Cumulative offset for each image's patches
    patch_indices: List[str]                   # Unique identifier for each patch


def compute_patch_grid(
    latent_h: int,
    latent_w: int,
    patch_size: int,
    latent_downsample: int,
) -> Tuple[int, int, int]:
    """
    Compute patch grid dimensions.
    
    Args:
        latent_h: Height of latent in tokens (H // latent_downsample)
        latent_w: Width of latent in tokens (W // latent_downsample)
        patch_size: Desired patch size in pixels
        latent_downsample: Downsample factor from pixels to latent tokens
        
    Returns:
        (patches_h, patches_w, latent_patch_size): Grid dimensions and patch token size
    """
    # Convert pixel patch_size to latent token patch size
    latent_patch_size = patch_size // latent_downsample
    
    # Ensure we have at least 1 patch and patch size doesn't exceed latent size
    latent_patch_size = max(1, min(latent_patch_size, min(latent_h, latent_w)))
    
    # Compute number of patches (ceil division to cover entire image)
    patches_h = (latent_h + latent_patch_size - 1) // latent_patch_size
    patches_w = (latent_w + latent_patch_size - 1) // latent_patch_size
    
    # Recompute patch size to evenly divide (use floor division for even patches)
    # For simplicity, we require latent dimensions to be divisible by patch count
    # If not perfectly divisible, we'll handle padding in split function
    
    return patches_h, patches_w, latent_patch_size


def split_latent_to_patches(
    packed_latent: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    patch_size: int,
    latent_downsample: int,
    latent_channel: int,
    latent_patch_size_model: int,
    device: torch.device = None,
) -> Tuple[torch.Tensor, PatchInfo]:
    """
    Split packed latent tensors into patches for patch-based diffusion.
    
    Args:
        packed_latent: Packed latent tensor of shape (total_tokens, feature_dim)
                       feature_dim can be latent_dim or hidden_size (for embedded features)
        image_sizes: List of (H, W) tuples for each image in pixels
        patch_size: Desired patch size in pixels (e.g., 256, 512)
        latent_downsample: Downsample factor from pixels to latent (e.g., 16 for VAE)
        latent_channel: Number of latent channels (e.g., 16) - used for reference only
        latent_patch_size_model: Model's patchify size (e.g., 2 for Bagel) - used for reference only
        device: Target device for tensors
        
    Returns:
        (patched_latent, patch_info): 
            - patched_latent: Tensor of shape (num_patches, tokens_per_patch, feature_dim)
            - patch_info: PatchInfo dataclass with metadata
    """
    if device is None:
        device = packed_latent.device
    
    # Use actual feature dimension from input tensor (works for both raw latent and embedded features)
    feature_dim = packed_latent.shape[-1]
    
    patches_list = []
    patches_per_image = []
    latent_offsets = [0]
    patch_indices = []
    
    token_offset = 0
    
    for img_idx, (H, W) in enumerate(image_sizes):
        # Compute latent dimensions for this image
        latent_h = H // latent_downsample
        latent_w = W // latent_downsample
        num_tokens = latent_h * latent_w
        
        # Extract this image's latent tokens
        image_latent = packed_latent[token_offset:token_offset + num_tokens]  # (h*w, feature_dim)
        
        # Reshape to 2D grid: (h, w, feature_dim)
        image_latent_2d = image_latent.view(latent_h, latent_w, feature_dim)
        
        # Compute patch grid
        patches_h, patches_w, latent_ps = compute_patch_grid(
            latent_h, latent_w, patch_size, latent_downsample
        )
        
        # Split into patches
        for ph in range(patches_h):
            for pw in range(patches_w):
                # Compute patch boundaries
                h_start = ph * latent_ps
                h_end = min((ph + 1) * latent_ps, latent_h)
                w_start = pw * latent_ps
                w_end = min((pw + 1) * latent_ps, latent_w)
                
                # Extract patch: (patch_h, patch_w, feature_dim)
                patch = image_latent_2d[h_start:h_end, w_start:w_end, :]
                
                # Flatten patch to (patch_tokens, feature_dim)
                patch_flat = patch.reshape(-1, feature_dim)
                patches_list.append(patch_flat)
                
                # Create unique patch identifier
                patch_indices.append(f"img{img_idx}-p{ph}-{pw}")
        
        num_patches_this_image = patches_h * patches_w
        patches_per_image.append(num_patches_this_image)
        latent_offsets.append(latent_offsets[-1] + num_patches_this_image)
        token_offset += num_tokens
    
    # Stack all patches: need to pad to same size if patches have different token counts
    max_patch_tokens = max(p.shape[0] for p in patches_list)
    padded_patches = []
    for patch in patches_list:
        if patch.shape[0] < max_patch_tokens:
            # Pad with zeros
            padding = torch.zeros(
                max_patch_tokens - patch.shape[0], feature_dim, 
                device=device, dtype=patch.dtype
            )
            patch = torch.cat([patch, padding], dim=0)
        padded_patches.append(patch)
    
    patched_latent = torch.stack(padded_patches, dim=0)  # (num_patches, max_tokens, feature_dim)
    
    # Get representative image for grid info (assuming all same size for now)
    H, W = image_sizes[0]
    latent_h = H // latent_downsample
    latent_w = W // latent_downsample
    patches_h, patches_w, latent_ps = compute_patch_grid(
        latent_h, latent_w, patch_size, latent_downsample
    )
    
    patch_info = PatchInfo(
        num_patches=len(patches_list),
        patches_per_image=patches_per_image,
        patch_h=patches_h,
        patch_w=patches_w,
        latent_patch_size=latent_ps,
        original_h=latent_h,
        original_w=latent_w,
        latent_offsets=latent_offsets,
        patch_indices=patch_indices,
    )
    
    return patched_latent, patch_info


def concat_patches_to_latent(
    patched_latent: torch.Tensor,
    patch_info: PatchInfo,
    image_sizes: List[Tuple[int, int]],
    latent_downsample: int,
    latent_channel: int,
    latent_patch_size_model: int,
) -> torch.Tensor:
    """
    Concatenate patches back into full latent tensors.
    
    Args:
        patched_latent: Tensor of shape (num_patches, tokens_per_patch, feature_dim)
                        feature_dim can be latent_dim or hidden_size
        patch_info: PatchInfo from split_latent_to_patches
        image_sizes: List of (H, W) tuples for each image in pixels
        latent_downsample: Downsample factor from pixels to latent
        latent_channel: Number of latent channels (for reference)
        latent_patch_size_model: Model's patchify size (for reference)
        
    Returns:
        packed_latent: Reconstructed tensor of shape (total_tokens, feature_dim)
    """
    # Use actual feature dimension from input tensor
    feature_dim = patched_latent.shape[-1]
    device = patched_latent.device
    dtype = patched_latent.dtype
    
    all_image_latents = []
    patch_idx = 0
    
    for img_idx, (H, W) in enumerate(image_sizes):
        latent_h = H // latent_downsample
        latent_w = W // latent_downsample
        
        # Reconstruct 2D latent grid
        image_latent_2d = torch.zeros(
            latent_h, latent_w, feature_dim, 
            device=device, dtype=dtype
        )
        
        # Get patch grid dimensions
        patches_h, patches_w, latent_ps = compute_patch_grid(
            latent_h, latent_w, 
            patch_info.latent_patch_size * latent_downsample,  # Convert back to pixel size
            latent_downsample
        )
        
        for ph in range(patches_h):
            for pw in range(patches_w):
                # Get this patch
                patch = patched_latent[patch_idx]  # (tokens_per_patch, feature_dim)
                
                # Compute boundaries
                h_start = ph * latent_ps
                h_end = min((ph + 1) * latent_ps, latent_h)
                w_start = pw * latent_ps
                w_end = min((pw + 1) * latent_ps, latent_w)
                
                patch_h = h_end - h_start
                patch_w = w_end - w_start
                actual_tokens = patch_h * patch_w
                
                # Reshape and place patch (only use actual tokens, ignore padding)
                patch_2d = patch[:actual_tokens].view(patch_h, patch_w, feature_dim)
                image_latent_2d[h_start:h_end, w_start:w_end, :] = patch_2d
                
                patch_idx += 1
        
        # Flatten back to (h*w, feature_dim)
        image_latent_flat = image_latent_2d.view(-1, feature_dim)
        all_image_latents.append(image_latent_flat)
    
    # Concatenate all images
    packed_latent = torch.cat(all_image_latents, dim=0)
    
    return packed_latent


def expand_for_patches(
    tensor: torch.Tensor,
    patch_info: PatchInfo,
    expand_dim: int = 0,
) -> torch.Tensor:
    """
    Expand a per-image tensor to per-patch by repeating.
    
    Args:
        tensor: Tensor with one entry per image (e.g., timesteps of shape (num_images,))
        patch_info: PatchInfo with patches_per_image
        expand_dim: Dimension along which to expand
        
    Returns:
        Expanded tensor with one entry per patch
    """
    expanded_parts = []
    for img_idx, num_patches in enumerate(patch_info.patches_per_image):
        img_tensor = tensor.select(expand_dim, img_idx)
        # Repeat for each patch of this image
        repeated = img_tensor.unsqueeze(expand_dim).expand(
            *([num_patches] + [-1] * (tensor.dim() - 1))
        )
        if expand_dim == 0:
            expanded_parts.append(repeated)
        else:
            expanded_parts.append(repeated)
    
    return torch.cat(expanded_parts, dim=expand_dim)


def create_patch_position_ids(
    image_sizes: List[Tuple[int, int]],
    patch_info: PatchInfo,
    latent_downsample: int,
    max_latent_size: int,
    get_flattened_position_ids_fn,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create position IDs for each patch's tokens.
    
    Args:
        image_sizes: List of (H, W) for each image
        patch_info: PatchInfo from splitting
        latent_downsample: Downsample factor
        max_latent_size: Maximum latent size for position embedding
        get_flattened_position_ids_fn: Function to compute 2D position IDs
        device: Target device
        
    Returns:
        Tensor of position IDs for all patch tokens
    """
    all_position_ids = []
    patch_idx = 0
    
    for img_idx, (H, W) in enumerate(image_sizes):
        latent_h = H // latent_downsample
        latent_w = W // latent_downsample
        
        # Get full image position IDs
        full_pos_ids = get_flattened_position_ids_fn(
            H, W, latent_downsample, max_num_patches_per_side=max_latent_size
        )
        full_pos_ids = full_pos_ids.view(latent_h, latent_w)
        
        patches_h, patches_w, latent_ps = compute_patch_grid(
            latent_h, latent_w,
            patch_info.latent_patch_size * latent_downsample,
            latent_downsample
        )
        
        for ph in range(patches_h):
            for pw in range(patches_w):
                h_start = ph * latent_ps
                h_end = min((ph + 1) * latent_ps, latent_h)
                w_start = pw * latent_ps
                w_end = min((pw + 1) * latent_ps, latent_w)
                
                # Extract position IDs for this patch
                patch_pos_ids = full_pos_ids[h_start:h_end, w_start:w_end].flatten()
                all_position_ids.append(patch_pos_ids)
                patch_idx += 1
    
    # Pad to same length
    max_len = max(p.shape[0] for p in all_position_ids)
    padded_pos_ids = []
    for pos_ids in all_position_ids:
        if pos_ids.shape[0] < max_len:
            padding = torch.zeros(max_len - pos_ids.shape[0], dtype=pos_ids.dtype, device=device)
            pos_ids = torch.cat([pos_ids, padding], dim=0)
        padded_pos_ids.append(pos_ids)
    
    return torch.stack(padded_pos_ids, dim=0)  # (num_patches, max_tokens)

