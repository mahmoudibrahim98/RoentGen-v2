"""
Validation metrics for RoentGen-v2 model checkpoints.

This module implements comprehensive validation metrics as described in the paper:
1. Text Prompt Alignment: Disease AUROC, sex/race accuracy, age RMSE
2. Real-Synthetic Image Similarity: FID, BioViL cosine similarity, MS-SSIM
3. Intra-Prompt Image Diversity: MS-SSIM and BioViL embedding similarity

References:
- Torch X-ray Vision (XRV): https://github.com/mlmed/torchxrayvision
- Sex prediction model: https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00032-4/fulltext
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.linalg import sqrtm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import warnings
from pathlib import Path


class TextPromptAlignmentMetrics:
    """
    Evaluates how accurately RoentGen-v2 adheres to provided text prompts.
    Uses pretrained classifiers from Torch X-ray Vision (XRV) library.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize XRV models for disease, sex, race, and age prediction.

        Args:
            device: Device to run inference on
        """
        self.device = device

        # Disease classification model (DenseNet-121)
        self.disease_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.disease_model.to(device)
        self.disease_model.eval()

        # Disease labels to evaluate
        self.disease_labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Edema",
            "Pneumothorax",
            "Effusion"
        ]
        print(self.disease_model.pathologies)
        # Get indices for these diseases in XRV model output
        self.disease_indices = [
            list(self.disease_model.pathologies).index(disease)
            for disease in self.disease_labels
        ]

        # Race classification model
        self.race_model = xrv.baseline_models.emory_hiti.RaceModel()
        self.race_model.to(device)
        self.race_model.eval()

        # Age prediction model
        # self.age_model = xrv.models.DenseNet(weights="densenet121-res224-age")
        self.age_model = xrv.baseline_models.riken.AgeModel()
        self.age_model.to(device)
        self.age_model.eval()

        # Sex prediction model (will be loaded separately using custom implementation)
        self.sex_model = None

        # Transform for XRV models (expects 224x224)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BOX),
            transforms.Normalize(mean=[0.5], std=[0.5])  # XRV expects normalized images
        ])

    def load_sex_model(self, checkpoint_path: str):
        """
        Load custom sex prediction model.

        Args:
            checkpoint_path: Path to sex model checkpoint
        """
        from roentgenv2.inference_code.sex_model import SexModelResNet
        self.sex_model = SexModelResNet.load_from_checkpoint(checkpoint_path, num_classes=2)

        # checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # self.sex_model.load_from_checkpoint(checkpoint_path)
        self.sex_model.to(self.device)
        self.sex_model.eval()

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for XRV models (512x512 -> 224x224).

        Args:
            images: Batch of images [B, C, H, W] in range [0, 1] or [-1, 1]

        Returns:
            Preprocessed images [B, C, 224, 224]
        """
        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Convert RGB to grayscale if needed
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)

        # Downsample to 224x224 using area interpolation
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="area"
        )

        # Normalize for XRV models
        images = (images - 0.5) / 0.5

        return images

    @torch.no_grad()
    def compute_disease_auroc(
        self,
        synthetic_images: torch.Tensor,
        disease_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute AUROC for disease classification on synthetic images.

        Args:
            synthetic_images: Synthetic images [B, C, H, W]
            disease_labels: Ground truth disease labels [B, num_diseases]

        Returns:
            Dictionary of disease name -> AUROC score
        """
        # Preprocess images
        images = self.preprocess_images(synthetic_images)

        # Get predictions
        predictions = self.disease_model(images)

        # Extract predictions for target diseases
        predictions = predictions[:, self.disease_indices].cpu().numpy()
        labels = disease_labels.cpu().numpy()

        # Compute AUROC for each disease
        aurocs = {}
        for i, disease in enumerate(self.disease_labels):
            # Skip if all labels are same (undefined AUROC)
            if len(np.unique(labels[:, i])) < 2:
                aurocs[disease] = np.nan
                continue

            try:
                aurocs[disease] = roc_auc_score(labels[:, i], predictions[:, i])
            except ValueError:
                aurocs[disease] = np.nan

        # Compute mean AUROC
        valid_aurocs = [v for v in aurocs.values() if not np.isnan(v)]
        aurocs["mean_auroc"] = np.mean(valid_aurocs) if valid_aurocs else np.nan

        return aurocs

    @torch.no_grad()
    def compute_sex_accuracy(
        self,
        synthetic_images: torch.Tensor,
        sex_labels: torch.Tensor
    ) -> float:
        """
        Compute sex classification accuracy.

        Args:
            synthetic_images: Synthetic images [B, C, H, W]
            sex_labels: Ground truth sex labels [B] (0=M, 1=F)

        Returns:
            Sex classification accuracy
        """
        if self.sex_model is None:
            warnings.warn("Sex model not loaded. Call load_sex_model() first.")
            return np.nan

        # Preprocess images
        images = self.preprocess_images(synthetic_images)

        # Get predictions
        predictions = self.sex_model(images)
        predicted_labels = (predictions > 0.5).long().cpu().numpy()
        true_labels = sex_labels.cpu().numpy()

        return accuracy_score(true_labels, predicted_labels)

    @torch.no_grad()
    def compute_race_accuracy(
        self,
        synthetic_images: torch.Tensor,
        race_labels: torch.Tensor
    ) -> float:
        """
        Compute race classification accuracy using XRV model.

        Args:
            synthetic_images: Synthetic images [B, C, H, W]
            race_labels: Ground truth race labels [B] (0=White, 1=Black, 2=Asian, 3=Hispanic)

        Returns:
            Race classification accuracy
        """
        # Preprocess images
        images = self.preprocess_images(synthetic_images)

        # Get predictions
        predictions = self.race_model(images)
        predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        true_labels = race_labels.cpu().numpy()

        return accuracy_score(true_labels, predicted_labels)

    @torch.no_grad()
    def compute_age_rmse(
        self,
        synthetic_images: torch.Tensor,
        age_labels: torch.Tensor
    ) -> float:
        """
        Compute age prediction RMSE in years.

        Args:
            synthetic_images: Synthetic images [B, C, H, W]
            age_labels: Ground truth ages in years [B]

        Returns:
            RMSE in years
        """
        # Preprocess images
        images = self.preprocess_images(synthetic_images)

        # Get predictions
        predictions = self.age_model(images).squeeze()

        # Compute RMSE
        rmse = torch.sqrt(F.mse_loss(predictions, age_labels.to(self.device)))

        return rmse.item()


class RealSyntheticSimilarityMetrics:
    """
    Evaluates similarity between real and synthetic images using:
    1. FID (Fréchet Inception Distance)
    2. BioViL cosine similarity
    3. MS-SSIM (Multi-Scale Structural Similarity Index)
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize models for similarity computation.

        Args:
            device: Device to run inference on
        """
        self.device = device

        # Inception v3 for FID
        self.inception_model = None  # Will be loaded lazily

        # BioViL encoder for medical image embeddings
        self.biovil_model = None  # Will be loaded lazily

    def _load_inception(self):
        """Lazy load Inception v3 model for FID computation."""
        if self.inception_model is None:
            from torchvision.models import inception_v3
            self.inception_model = inception_v3(pretrained=True, transform_input=False)
            self.inception_model.fc = nn.Identity()  # Remove final FC layer
            self.inception_model.to(self.device)
            self.inception_model.eval()

    def _load_biovil(self):
        """Lazy load BioViL model for medical image embeddings."""
        if self.biovil_model is None:
            try:
                from transformers import AutoModel, AutoProcessor

                # Load the full model
                model = AutoModel.from_pretrained(
                    "microsoft/BiomedVLP-CXR-BERT-specialized",
                    trust_remote_code=True
                )

                # Extract only the vision encoder
                if hasattr(model, 'vision_model'):
                    self.biovil_model = model.vision_model
                elif hasattr(model, 'vision_encoder'):
                    self.biovil_model = model.vision_encoder
                else:
                    # If we can't find vision encoder, use full model with processor
                    self.biovil_model = model
                    self.biovil_processor = AutoProcessor.from_pretrained(
                        "microsoft/BiomedVLP-CXR-BERT-specialized",
                        trust_remote_code=True
                    )

                self.biovil_model.to(self.device)
                self.biovil_model.eval()
            except Exception as e:
                warnings.warn(f"Could not load BioViL model: {e}")
                self.biovil_model = None

    @torch.no_grad()
    def compute_fid(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Compute Fréchet Inception Distance between real and synthetic images.

        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            batch_size: Batch size for processing

        Returns:
            FID score (lower is better, 0 is perfect match)
        """
        self._load_inception()

        # Get embeddings for both image sets
        real_embeddings = self._get_inception_embeddings(real_images, batch_size)
        synthetic_embeddings = self._get_inception_embeddings(synthetic_images, batch_size)

        # Compute mean and covariance
        mu_real = np.mean(real_embeddings, axis=0)
        sigma_real = np.cov(real_embeddings, rowvar=False)

        mu_synthetic = np.mean(synthetic_embeddings, axis=0)
        sigma_synthetic = np.cov(synthetic_embeddings, rowvar=False)

        # Compute FID
        ssdiff = np.sum((mu_real - mu_synthetic) ** 2.0)
        covmean = sqrtm(sigma_real.dot(sigma_synthetic))

        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma_real + sigma_synthetic - 2.0 * covmean)

        return float(fid)

    @torch.no_grad()
    def _get_inception_embeddings(
        self,
        images: torch.Tensor,
        batch_size: int
    ) -> np.ndarray:
        """
        Extract Inception v3 embeddings from images.

        Args:
            images: Images [N, C, H, W] in range [0, 1] or [-1, 1]
            batch_size: Batch size for processing

        Returns:
            Embeddings [N, 2048]
        """
        # Normalize to [0, 1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Resize to 299x299 for Inception
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

        # Convert grayscale to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Normalize for Inception
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std

        # Extract embeddings in batches
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            embedding = self.inception_model(batch)
            embeddings.append(embedding.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    @torch.no_grad()
    def compute_biovil_similarity(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between BioViL embeddings of paired real/synthetic images.

        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W] (paired with real)

        Returns:
            Mean cosine similarity (1.0 = identical, 0.0 = orthogonal)
        """
        self._load_biovil()

        if self.biovil_model is None:
            warnings.warn("BioViL model not available. Returning NaN.")
            return np.nan

        # Get embeddings
        real_embeddings = self._get_biovil_embeddings(real_images)
        synthetic_embeddings = self._get_biovil_embeddings(synthetic_images)

        # Compute pairwise cosine similarity
        real_embeddings = F.normalize(real_embeddings, dim=1)
        synthetic_embeddings = F.normalize(synthetic_embeddings, dim=1)

        similarities = torch.sum(real_embeddings * synthetic_embeddings, dim=1)

        return similarities.mean().item()

    @torch.no_grad()
    def _get_biovil_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract BioViL embeddings from images.

        Args:
            images: Images [N, C, H, W]

        Returns:
            Embeddings [N, D]
        """
        # Normalize to [0, 1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # BioViL expects RGB images
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Resize to BioViL input size (typically 224x224 or 512x512)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

        # Extract embeddings
        try:
            # Try direct image encoding (if we have vision encoder)
            outputs = self.biovil_model(pixel_values=images.to(self.device))

            # Handle different output formats
            if hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # Use CLS token or mean pooling
                embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            elif isinstance(outputs, torch.Tensor):
                embeddings = outputs
            else:
                embeddings = outputs[0][:, 0]  # Fallback

        except TypeError:
            # If that fails, it might need processor
            if hasattr(self, 'biovil_processor'):
                # Process images through the processor
                inputs = self.biovil_processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.biovil_model(**inputs)
                embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0][:, 0]
            else:
                raise

        return embeddings

    def compute_ms_ssim(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor
    ) -> float:
        """
        Compute Multi-Scale Structural Similarity Index between paired images.

        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W] (paired with real)

        Returns:
            Mean MS-SSIM (1.0 = identical, 0.0 = no similarity)
        """
        from pytorch_msssim import ms_ssim

        # Normalize to [0, 1]
        if real_images.min() < 0:
            real_images = (real_images + 1.0) / 2.0
        if synthetic_images.min() < 0:
            synthetic_images = (synthetic_images + 1.0) / 2.0

        # Compute MS-SSIM
        ms_ssim_val = ms_ssim(
            real_images,
            synthetic_images,
            data_range=1.0,
            size_average=True
        )

        return ms_ssim_val.item()


class IntraPromptDiversityMetrics:
    """
    Evaluates diversity of multiple images generated from the same prompt.
    Lower scores indicate higher diversity (more desirable).
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize diversity metrics.

        Args:
            device: Device to run inference on
        """
        self.device = device
        self.biovil_model = None

    def _load_biovil(self):
        """Lazy load BioViL model."""
        if self.biovil_model is None:
            try:
                from transformers import AutoModel, AutoProcessor

                # Load the full model
                model = AutoModel.from_pretrained(
                    "microsoft/BiomedVLP-CXR-BERT-specialized",
                    trust_remote_code=True
                )

                # Extract only the vision encoder
                if hasattr(model, 'vision_model'):
                    self.biovil_model = model.vision_model
                elif hasattr(model, 'vision_encoder'):
                    self.biovil_model = model.vision_encoder
                else:
                    # If we can't find vision encoder, use full model with processor
                    self.biovil_model = model
                    self.biovil_processor = AutoProcessor.from_pretrained(
                        "microsoft/BiomedVLP-CXR-BERT-specialized",
                        trust_remote_code=True
                    )

                self.biovil_model.to(self.device)
                self.biovil_model.eval()
            except Exception as e:
                warnings.warn(f"Could not load BioViL model: {e}")
                self.biovil_model = None

    def compute_intra_prompt_ms_ssim(
        self,
        images_per_prompt: List[torch.Tensor]
    ) -> float:
        """
        Compute mean pairwise MS-SSIM among images generated from same prompt.

        Args:
            images_per_prompt: List of image sets, where each set contains
                              multiple images [K, C, H, W] from same prompt

        Returns:
            Mean MS-SSIM across all prompts (lower = more diverse)
        """
        from pytorch_msssim import ms_ssim

        per_prompt_ssims = []

        for images in images_per_prompt:
            # Normalize to [0, 1]
            if images.min() < 0:
                images = (images + 1.0) / 2.0

            # Compute pairwise MS-SSIM
            K = len(images)
            pairwise_ssims = []

            for i in range(K):
                for j in range(i + 1, K):
                    img1 = images[i:i+1]
                    img2 = images[j:j+1]

                    ssim_val = ms_ssim(
                        img1,
                        img2,
                        data_range=1.0,
                        size_average=True
                    )
                    pairwise_ssims.append(ssim_val.item())

            # Average for this prompt
            per_prompt_ssims.append(np.mean(pairwise_ssims))

        # Average across all prompts
        return float(np.mean(per_prompt_ssims))

    @torch.no_grad()
    def compute_intra_prompt_biovil_similarity(
        self,
        images_per_prompt: List[torch.Tensor]
    ) -> float:
        """
        Compute mean pairwise BioViL cosine similarity among images from same prompt.

        Args:
            images_per_prompt: List of image sets [K, C, H, W] per prompt

        Returns:
            Mean cosine similarity (lower = more diverse)
        """
        self._load_biovil()

        if self.biovil_model is None:
            warnings.warn("BioViL model not available. Returning NaN.")
            return np.nan

        per_prompt_sims = []

        for images in images_per_prompt:
            # Get embeddings
            embeddings = self._get_biovil_embeddings(images)
            embeddings = F.normalize(embeddings, dim=1)

            # Compute pairwise cosine similarity
            K = len(embeddings)
            pairwise_sims = []

            for i in range(K):
                for j in range(i + 1, K):
                    sim = torch.sum(embeddings[i] * embeddings[j])
                    pairwise_sims.append(sim.item())

            # Average for this prompt
            per_prompt_sims.append(np.mean(pairwise_sims))

        # Average across all prompts
        return float(np.mean(per_prompt_sims))

    @torch.no_grad()
    def _get_biovil_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract BioViL embeddings."""
        # Normalize to [0, 1]
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Convert to RGB
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Resize
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

        # Extract embeddings
        try:
            # Try direct image encoding (if we have vision encoder)
            outputs = self.biovil_model(pixel_values=images.to(self.device))

            # Handle different output formats
            if hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # Use CLS token or mean pooling
                embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            elif isinstance(outputs, torch.Tensor):
                embeddings = outputs
            else:
                embeddings = outputs[0][:, 0]  # Fallback

        except TypeError:
            # If that fails, it might need processor
            if hasattr(self, 'biovil_processor'):
                # Process images through the processor
                inputs = self.biovil_processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.biovil_model(**inputs)
                embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0][:, 0]
            else:
                raise

        return embeddings


class ValidationMetricsRunner:
    """
    Main runner for computing all validation metrics on a validation set.
    Coordinates text alignment, similarity, and diversity metrics.
    """

    def __init__(
        self,
        device: str = "cuda",
        sex_model_path: Optional[str] = None
    ):
        """
        Initialize all metric computers.

        Args:
            device: Device to run inference on
            sex_model_path: Path to sex prediction model checkpoint
        """
        self.device = device

        self.text_alignment = TextPromptAlignmentMetrics(device)
        if sex_model_path:
            self.text_alignment.load_sex_model(sex_model_path)

        self.similarity = RealSyntheticSimilarityMetrics(device)
        self.diversity = IntraPromptDiversityMetrics(device)

    def compute_all_metrics(
        self,
        real_images: torch.Tensor,
        synthetic_images: torch.Tensor,
        disease_labels: torch.Tensor,
        sex_labels: torch.Tensor,
        race_labels: torch.Tensor,
        age_labels: torch.Tensor,
        images_per_prompt: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute all validation metrics.

        Args:
            real_images: Real images [N, C, H, W]
            synthetic_images: Synthetic images [N, C, H, W]
            disease_labels: Disease labels [N, 5] for target diseases
            sex_labels: Sex labels [N]
            race_labels: Race labels [N]
            age_labels: Age labels [N]
            images_per_prompt: Optional list for diversity metrics

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}

        # 1. Text Prompt Alignment Metrics
        print("Computing text prompt alignment metrics...")

        disease_aurocs = self.text_alignment.compute_disease_auroc(
            synthetic_images, disease_labels
        )
        metrics.update(disease_aurocs)

        metrics["sex_accuracy"] = self.text_alignment.compute_sex_accuracy(
            synthetic_images, sex_labels
        )

        metrics["race_accuracy"] = self.text_alignment.compute_race_accuracy(
            synthetic_images, race_labels
        )

        metrics["age_rmse"] = self.text_alignment.compute_age_rmse(
            synthetic_images, age_labels
        )

        # 2. Real-Synthetic Image Similarity Metrics
        print("Computing real-synthetic similarity metrics...")

        metrics["fid"] = self.similarity.compute_fid(
            real_images, synthetic_images
        )

        metrics["biovil_similarity"] = self.similarity.compute_biovil_similarity(
            real_images, synthetic_images
        )

        metrics["ms_ssim"] = self.similarity.compute_ms_ssim(
            real_images, synthetic_images
        )

        # 3. Intra-Prompt Diversity Metrics (if provided)
        if images_per_prompt is not None:
            print("Computing intra-prompt diversity metrics...")

            metrics["intra_prompt_ms_ssim"] = self.diversity.compute_intra_prompt_ms_ssim(
                images_per_prompt
            )

            metrics["intra_prompt_biovil_similarity"] = self.diversity.compute_intra_prompt_biovil_similarity(
                images_per_prompt
            )

        return metrics

    def print_metrics_summary(self, metrics: Dict[str, float]):
        """
        Print formatted summary of validation metrics.

        Args:
            metrics: Dictionary of computed metrics
        """
        print("\n" + "="*60)
        print("VALIDATION METRICS SUMMARY")
        print("="*60)

        print("\n1. TEXT PROMPT ALIGNMENT")
        print("-" * 60)
        print(f"  Disease Classification (AUROC):")
        for disease in ["Atelectasis", "Cardiomegaly", "Edema", "Pneumothorax", "Pleural Effusion"]:
            if disease in metrics:
                print(f"    {disease:20s}: {metrics[disease]:.4f}")
        if "mean_auroc" in metrics:
            print(f"    {'Mean AUROC':20s}: {metrics['mean_auroc']:.4f}")

        print(f"\n  Demographic Attributes:")
        if "sex_accuracy" in metrics:
            print(f"    Sex Accuracy:   {metrics['sex_accuracy']:.4f}")
        if "race_accuracy" in metrics:
            print(f"    Race Accuracy:  {metrics['race_accuracy']:.4f}")
        if "age_rmse" in metrics:
            print(f"    Age RMSE:       {metrics['age_rmse']:.2f} years")

        print("\n2. REAL-SYNTHETIC IMAGE SIMILARITY")
        print("-" * 60)
        if "fid" in metrics:
            print(f"  FID Score:               {metrics['fid']:.2f} (lower is better)")
        if "biovil_similarity" in metrics:
            print(f"  BioViL Cosine Similarity: {metrics['biovil_similarity']:.4f} (1.0 = perfect)")
        if "ms_ssim" in metrics:
            print(f"  MS-SSIM:                 {metrics['ms_ssim']:.4f} (1.0 = identical)")

        print("\n3. INTRA-PROMPT DIVERSITY")
        print("-" * 60)
        if "intra_prompt_ms_ssim" in metrics:
            print(f"  Intra-Prompt MS-SSIM:     {metrics['intra_prompt_ms_ssim']:.4f} (lower = more diverse)")
        if "intra_prompt_biovil_similarity" in metrics:
            print(f"  Intra-Prompt BioViL Sim:  {metrics['intra_prompt_biovil_similarity']:.4f} (lower = more diverse)")

        print("\n" + "="*60)
