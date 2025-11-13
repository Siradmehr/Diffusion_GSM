"""
Forward-Hard Anisotropic Diffusion for MNIST
Implementation of Section 9 from the paper
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ============================================================================
# 1. Block-Mean Projection Module
# ============================================================================

class BlockMeanProjection:
    """
    Implements R_ell and P_ell projections for images via block averaging.

    For MNIST (28x28):
    - P_ell: Project to coarse (e.g., 7x7 block means)
    - R_ell: Residuals (within-block deviations)
    """
    def __init__(self, image_size=28, block_size=4):
        self.image_size = image_size
        self.block_size = block_size
        self.coarse_size = image_size // block_size

        # Dimensions
        self.d = image_size * image_size  # Total dimension
        self.d_P = self.coarse_size * self.coarse_size  # Coarse dimension
        self.d_R = self.d - self.d_P  # Residual dimension (approximately)

        print(f"Block-Mean Projection initialized:")
        print(f"  Image size: {image_size}x{image_size} (d={self.d})")
        print(f"  Block size: {block_size}x{block_size}")
        print(f"  Coarse size: {self.coarse_size}x{self.coarse_size} (d_P={self.d_P})")
        print(f"  Residual dim: d_R â‰ˆ {self.d_R}")

    def project_coarse(self, x):
        """
        P_ell: Project to coarse subspace via block averaging.

        Args:
            x: Tensor of shape (batch, 1, H, W) or (batch, H*W)
        Returns:
            x_coarse: Coarse component (batch, 1, H//block, W//block)
        """
        if len(x.shape) == 2:  # Flatten format
            batch_size = x.shape[0]
            x = x.view(batch_size, 1, self.image_size, self.image_size)

        # Average pooling to get block means
        x_coarse = F.avg_pool2d(x, kernel_size=self.block_size, stride=self.block_size)

        return x_coarse

    def project_residual(self, x):
        """
        R_ell: Residual = x - P_ell(x) (within-block deviations).

        Args:
            x: Tensor of shape (batch, 1, H, W)
        Returns:
            x_residual: Residual component
        """
        # Get coarse component
        x_coarse = self.project_coarse(x)

        # Upsample coarse back to original size
        x_coarse_upsampled = F.interpolate(
            x_coarse, 
            size=(self.image_size, self.image_size), 
            mode='nearest'
        )

        # Residual = original - coarse
        x_residual = x - x_coarse_upsampled

        return x_residual

    def decompose(self, x):
        """
        Decompose x = P_ell(x) + R_ell(x).

        Returns:
            x_coarse, x_residual
        """
        x_coarse = self.project_coarse(x)
        x_residual = self.project_residual(x)
        return x_coarse, x_residual

# ============================================================================
# 2. Forward-Hard Diffusion Process
# ============================================================================

class ForwardHardDiffusion:
    """
    Anisotropic forward diffusion with different rates for coarse/residual.

    Forward SDE:
        dX_t = (Î»_R R + Î»_P P) X_t dt + sqrt(2) [Î²_R R R^T + Î²_P P P^T]^{1/2} dW_t
    """
    def __init__(self, projection, beta_R=0.1, beta_P=1.0, lambda_R=3.0, lambda_P=0.5, T=1.0):
        self.projection = projection
        self.beta_R = beta_R
        self.beta_P = beta_P
        self.lambda_R = lambda_R
        self.lambda_P = lambda_P
        self.T = T

        # Hardness ratio
        self.rho_ell = lambda_R / beta_R

        print(f"\nForward-Hard Diffusion initialized:")
        print(f"  Î²_R = {beta_R} (residual noise - SMALL)")
        print(f"  Î²_P = {beta_P} (coarse noise - LARGE)")
        print(f"  Î»_R = {lambda_R} (residual drift - FAST)")
        print(f"  Î»_P = {lambda_P} (coarse drift - SLOW)")
        print(f"  Ï_â„“ = {self.rho_ell:.1f} (hardness ratio)")
        print(f"  T = {T} (total diffusion time)")

    def q_sample(self, x_0, t):
        """
        Sample from q(x_t | x_0) - the forward diffusion distribution.

        For OU process: X_t = exp(-Î»t) X_0 + N(0, Î²(1-exp(-2Î»t))/Î»)

        Args:
            x_0: Initial image (batch, 1, H, W)
            t: Time (batch,) or scalar
        Returns:
            x_t: Noised image
        """
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x_0.device)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        # Reshape t for broadcasting
        t = t.view(-1, 1, 1, 1)

        # Decompose into coarse and residual
        x_coarse, x_residual = self.projection.decompose(x_0)

        # Mean: exponential decay
        mean_coarse = torch.exp(-self.lambda_P * t) * x_coarse
        mean_residual = torch.exp(-self.lambda_R * t) * x_residual

        # Variance: depends on diffusion rate
        var_coarse = self.beta_P * (1 - torch.exp(-2 * self.lambda_P * t)) / self.lambda_P
        var_residual = self.beta_R * (1 - torch.exp(-2 * self.lambda_R * t)) / self.lambda_R

        # Sample noise
        noise_coarse = torch.randn_like(x_coarse) * torch.sqrt(var_coarse)
        noise_residual = torch.randn_like(x_residual) * torch.sqrt(var_residual)

        # Upsample coarse components
        x_coarse_t = F.interpolate(
            mean_coarse + noise_coarse,
            size=(self.projection.image_size, self.projection.image_size),
            mode='nearest'
        )

        x_residual_t = mean_residual + noise_residual

        # Combine
        x_t = x_coarse_t + x_residual_t

        return x_t, noise_coarse, noise_residual

# ============================================================================
# 3. Score Network (U-Net with Separate Coarse/Residual Branches)
# ============================================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.relu(self.norm1(x)))

        # Add time embedding
        h = h + self.time_mlp(F.relu(t_emb))[:, :, None, None]

        h = self.conv2(F.relu(self.norm2(h)))
        return h + self.shortcut(x)

class AnisotropicScoreNetwork(nn.Module):
    """
    Score network with separate branches for coarse and residual.

    Architecture:
        - Encoder: Separate branches for coarse and residual
        - Bottleneck: Shared representation
        - Decoder: Separate branches producing s^(R) and s^(P)
    """
    def __init__(self, projection, time_dim=64, base_channels=64):
        super().__init__()
        self.projection = projection
        self.time_dim = time_dim

        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)

        # Coarse branch encoder (operates on low-res)
        self.coarse_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            ResidualBlock(base_channels, base_channels, time_dim),
            ResidualBlock(base_channels, base_channels*2, time_dim),
        )

        # Residual branch encoder (operates on full-res)
        self.residual_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            ResidualBlock(base_channels, base_channels, time_dim),
            nn.AvgPool2d(2),
            ResidualBlock(base_channels, base_channels*2, time_dim),
        )

        # Shared bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels*4, base_channels*4, time_dim),
            ResidualBlock(base_channels*4, base_channels*4, time_dim),
        )

        # Coarse branch decoder
        self.coarse_decoder = nn.Sequential(
            ResidualBlock(base_channels*4, base_channels*2, time_dim),
            ResidualBlock(base_channels*2, base_channels, time_dim),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

        # Residual branch decoder
        self.residual_decoder = nn.Sequential(
            ResidualBlock(base_channels*4, base_channels*2, time_dim),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(base_channels*2, base_channels, time_dim),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(self, x_t, t):
        """
        Predict score s_Î¸(x_t, t) = R s^(R) + P s^(P)

        Args:
            x_t: Noised image (batch, 1, H, W)
            t: Time (batch,)
        Returns:
            s_coarse, s_residual: Predicted scores
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Decompose input
        x_coarse, x_residual = self.projection.decompose(x_t)

        # Encode separately
        h_coarse = self.coarse_encoder[0](x_coarse)
        for layer in self.coarse_encoder[1:]:
            h_coarse = layer(h_coarse, t_emb)

        h_residual = self.residual_encoder[0](x_residual)
        for i, layer in enumerate(self.residual_encoder[1:]):
            if isinstance(layer, ResidualBlock):
                h_residual = layer(h_residual, t_emb)
            else:
                h_residual = layer(h_residual)

        # Concatenate and process in bottleneck
        h = torch.cat([h_coarse, h_residual], dim=1)
        for layer in self.bottleneck:
            h = layer(h, t_emb)

        # Decode separately
        s_coarse = h
        for layer in self.coarse_decoder:
            if isinstance(layer, ResidualBlock):
                s_coarse = layer(s_coarse, t_emb)
            else:
                s_coarse = layer(s_coarse)

        s_residual = h
        for layer in self.residual_decoder:
            if isinstance(layer, ResidualBlock):
                s_residual = layer(s_residual, t_emb)
            else:
                s_residual = layer(s_residual)

        return s_coarse, s_residual

# ============================================================================
# 4. Training
# ============================================================================

class ForwardHardDiffusionTrainer:
    """Trainer for forward-hard anisotropic diffusion"""

    def __init__(self, model, diffusion, device='cuda'):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
        self.projection = diffusion.projection

    def compute_loss(self, x_0):
        """
        Score matching loss with anisotropic weighting:
        L = Î²_R ||s^(R) - s*^(R)||Â² + Î²_P ||s^(P) - s*^(P)||Â²
        """
        batch_size = x_0.shape[0]

        # Sample random time
        t = torch.rand(batch_size, device=self.device) * self.diffusion.T

        # Forward diffusion
        x_t, noise_coarse, noise_residual = self.diffusion.q_sample(x_0, t)

        # Predict score
        s_coarse_pred, s_residual_pred = self.model(x_t, t)

        # Compute target scores (negative of scaled noise)
        # For OU process: s(x,t) = -noise / variance
        var_coarse = self.diffusion.beta_P * (1 - torch.exp(-2 * self.diffusion.lambda_P * t[:, None, None, None])) / self.diffusion.lambda_P
        var_residual = self.diffusion.beta_R * (1 - torch.exp(-2 * self.diffusion.lambda_R * t[:, None, None, None])) / self.diffusion.lambda_R

        s_coarse_target = -noise_coarse / torch.sqrt(var_coarse + 1e-8)
        s_residual_target = -noise_residual / torch.sqrt(var_residual + 1e-8)

        # Anisotropic weighted loss
        loss_coarse = self.diffusion.beta_P * F.mse_loss(s_coarse_pred, s_coarse_target)
        loss_residual = self.diffusion.beta_R * F.mse_loss(s_residual_pred, s_residual_target)

        loss = loss_coarse + loss_residual

        return loss, loss_coarse, loss_residual

    def train(self, train_loader, num_epochs=10, lr=1e-4):
        """Train the model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            total_loss = 0
            total_loss_coarse = 0
            total_loss_residual = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_idx, (x, _) in enumerate(pbar):
                x = x.to(self.device)

                # Forward pass
                loss, loss_coarse, loss_residual = self.compute_loss(x)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                total_loss_coarse += loss_coarse.item()
                total_loss_residual += loss_residual.item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'L_P': loss_coarse.item(),
                    'L_R': loss_residual.item()
                })

            avg_loss = total_loss / len(train_loader)
            avg_loss_coarse = total_loss_coarse / len(train_loader)
            avg_loss_residual = total_loss_residual / len(train_loader)

            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, L_coarse={avg_loss_coarse:.4f}, L_residual={avg_loss_residual:.4f}')

        return self.model

# ============================================================================
# 5. Sampling (Reverse Diffusion)
# ============================================================================

@torch.no_grad()
def sample(model, diffusion, num_samples=64, num_steps=100, device='cuda'):
    """
    Generate samples via reverse diffusion.

    Reverse SDE:
        dY_t = [drift + Î£ s(Y_t, t)] dt + Î£^{1/2} dW_t
    """
    model.eval()

    # Start from noise
    x_t = torch.randn(num_samples, 1, 28, 28, device=device)

    dt = diffusion.T / num_steps

    for step in tqdm(range(num_steps), desc='Sampling'):
        t = diffusion.T - step * dt
        t_batch = torch.full((num_samples,), t, device=device)

        # Predict score
        s_coarse, s_residual = model(x_t, t_batch)

        # Decompose current state
        x_coarse, x_residual = diffusion.projection.decompose(x_t)

        # Reverse drift: +Î» X + 2Î² s
        drift_coarse = (diffusion.lambda_P * x_coarse + 2 * diffusion.beta_P * s_coarse) * dt
        drift_residual = (diffusion.lambda_R * x_residual + 2 * diffusion.beta_R * s_residual) * dt

        # Upsample coarse drift
        drift_coarse = F.interpolate(drift_coarse, size=(28, 28), mode='nearest')

        # Diffusion term (same as forward)
        noise_coarse = torch.randn_like(x_coarse) * np.sqrt(2 * diffusion.beta_P * dt)
        noise_residual = torch.randn_like(x_residual) * np.sqrt(2 * diffusion.beta_R * dt)

        noise_coarse = F.interpolate(noise_coarse, size=(28, 28), mode='nearest')

        # Update
        x_t = x_t + drift_coarse + drift_residual + noise_coarse + noise_residual

    return x_t

# ============================================================================
# 6. Main Training Script
# ============================================================================

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=4
    )

    print(f'Loaded MNIST: {len(train_dataset)} training samples')

    # Initialize projection
    projection = BlockMeanProjection(image_size=28, block_size=4)

    # Initialize forward-hard diffusion
    diffusion = ForwardHardDiffusion(
        projection,
        beta_R=0.1,   # Small noise on residuals
        beta_P=1.0,   # Large noise on coarse
        lambda_R=3.0, # Fast drift on residuals
        lambda_P=0.5, # Slow drift on coarse
        T=1.0
    )

    # Initialize score network
    model = AnisotropicScoreNetwork(projection, time_dim=64, base_channels=64)
    print(f'\nModel parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # Train
    trainer = ForwardHardDiffusionTrainer(model, diffusion, device=device)
    print('\nStarting training...')
    model = trainer.train(train_loader, num_epochs=10, lr=1e-4)

    # Save model
    torch.save(model.state_dict(), 'forward_hard_mnist.pth')
    print('\nModel saved to forward_hard_mnist.pth')

    # Generate samples
    print('\nGenerating samples...')
    samples = sample(model, diffusion, num_samples=64, num_steps=100, device=device)

    # Visualize
    samples = samples.cpu()
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('forward_hard_samples.png', dpi=150)
    print('Samples saved to forward_hard_samples.png')

if __name__ == '__main__':
    main()