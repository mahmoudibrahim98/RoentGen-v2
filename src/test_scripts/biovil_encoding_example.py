"""
Example: How to encode images using microsoft/BiomedVLP-CXR-BERT-specialized (BioViL)

This script demonstrates the CORRECT way to encode chest X-ray images using the BioViL model
via the official hi-ml-multimodal library from Microsoft Health AI.

Installation:
    pip install hi-ml-multimodal

Reference:
    https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal
"""

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from health_multimodal.image.utils import get_image_inference, ImageModelType

print("="*70)
print("BioViL Image Encoding Example")
print("="*70)

# 1. Load the BioViL model using hi-ml-multimodal
print("\n1. Loading BioViL model...")
image_inference_engine = get_image_inference(image_model_type=ImageModelType.BIOVIL)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
image_inference_engine.model.to(device)
image_inference_engine.model.eval()

print(f"   ✓ Model loaded on {device}")
print(f"   ✓ Model type: {type(image_inference_engine.model)}")

# Get the transform (includes resize, normalize, etc.)
transform = image_inference_engine.transform

# 2. Create dummy chest X-ray images (grayscale, 512x512)
print("\n2. Creating dummy chest X-ray images...")
batch_size = 4
dummy_images_tensor = torch.randn(batch_size, 1, 512, 512).to(device)

# Normalize to [0, 1] if needed
if dummy_images_tensor.min() < 0:
    dummy_images_tensor = (dummy_images_tensor + 1.0) / 2.0

print(f"   ✓ Created {batch_size} dummy images with shape: {dummy_images_tensor.shape}")

# 3. Convert tensor to PIL Images
print("\n3. Converting tensors to PIL Images...")
pil_images = []
for i in range(batch_size):
    img = dummy_images_tensor[i]
    
    # Convert to numpy
    img_np = img.cpu().numpy()
    
    # Handle grayscale (single channel)
    if img_np.shape[0] == 1:
        img_np = img_np[0]  # Remove channel dimension -> [H, W]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='L')  # Grayscale
    else:
        # RGB image
        img_np = img_np.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')
    
    pil_images.append(pil_img)

print(f"   ✓ Converted to {len(pil_images)} PIL Images")
print(f"   ✓ Image mode: {pil_images[0].mode}, size: {pil_images[0].size}")

# 4. Apply BioViL transform and create batch
print("\n4. Applying BioViL preprocessing transform...")
batch = torch.stack([transform(img) for img in pil_images])
batch = batch.to(device)

print(f"   ✓ Preprocessed batch shape: {batch.shape}")

# 5. Extract image embeddings
print("\n5. Extracting image embeddings...")
with torch.no_grad():
    output = image_inference_engine.model(batch)  # Returns ImageModelOutput
    embeddings = output.projected_global_embedding  # [B, joint_feature_size]

print(f"   ✓ Embeddings shape: {embeddings.shape}")
print(f"   ✓ Embeddings dtype: {embeddings.dtype}")

# 6. L2 normalize (recommended for similarity computation)
print("\n6. L2 normalizing embeddings...")
embeddings_normalized = F.normalize(embeddings, p=2, dim=-1)

print(f"   ✓ Normalized embeddings shape: {embeddings_normalized.shape}")
print(f"   ✓ Embedding stats before normalization:")
print(f"      - Mean: {embeddings.mean():.4f}")
print(f"      - Std:  {embeddings.std():.4f}")
print(f"      - L2 norms: {embeddings.norm(p=2, dim=-1)}")
print(f"   ✓ L2 norms after normalization: {embeddings_normalized.norm(p=2, dim=-1)}")

# 7. Compute pairwise cosine similarity
print("\n7. Computing pairwise cosine similarity...")
similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())

print(f"   ✓ Similarity matrix shape: {similarity_matrix.shape}")
print(f"\n   Similarity matrix:")
print(similarity_matrix.cpu().numpy())

# 8. Example: Computing similarity between two sets of images
print("\n8. Example: Real vs Synthetic image similarity...")
real_images = torch.randn(2, 1, 512, 512).to(device)
synthetic_images = torch.randn(2, 1, 512, 512).to(device)

# Normalize to [0, 1]
real_images = (real_images + 1.0) / 2.0
synthetic_images = (synthetic_images + 1.0) / 2.0

# Convert to PIL
def tensor_to_pil(images_tensor):
    pil_imgs = []
    for i in range(images_tensor.shape[0]):
        img_np = images_tensor[i, 0].cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        pil_imgs.append(Image.fromarray(img_np, mode='L'))
    return pil_imgs

real_pil = tensor_to_pil(real_images)
synthetic_pil = tensor_to_pil(synthetic_images)

# Get embeddings
with torch.no_grad():
    real_batch = torch.stack([transform(img) for img in real_pil]).to(device)
    synthetic_batch = torch.stack([transform(img) for img in synthetic_pil]).to(device)
    
    real_emb = F.normalize(
        image_inference_engine.model(real_batch).projected_global_embedding,
        p=2, dim=-1
    )
    synthetic_emb = F.normalize(
        image_inference_engine.model(synthetic_batch).projected_global_embedding,
        p=2, dim=-1
    )

# Compute mean pairwise cosine similarity
similarities = torch.sum(real_emb * synthetic_emb, dim=1)
mean_similarity = similarities.mean().item()

print(f"   ✓ Mean cosine similarity between real and synthetic: {mean_similarity:.4f}")
print(f"   ✓ Individual similarities: {similarities.cpu().numpy()}")

print("\n" + "="*70)
print("✓ BioViL Image Encoding Complete!")
print("="*70)

print("\n📝 Summary:")
print("  1. Load model: get_image_inference(ImageModelType.BIOVIL)")
print("  2. Convert images to PIL format")
print("  3. Apply transform: image_inference_engine.transform(pil_image)")
print("  4. Get embeddings: model(batch).projected_global_embedding")
print("  5. L2 normalize: F.normalize(embeddings, p=2, dim=-1)")
print("  6. Compute similarity using dot product or cosine similarity")
print("\n📚 Key Points:")
print("  - Embeddings are in joint image-text latent space")
print("  - Always L2 normalize before computing similarity")
print("  - Works with both grayscale and RGB images")
print("  - Transform handles resize, normalization automatically")
