# ==============================================================================
# LOSSES.PY - Loss Functions
# ==============================================================================

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss untuk data seismik yang sparse.
    
    EXACT COPY dari amatrice_st_gat.py yang terbukti bekerja.
    
    Memberikan bobot lebih tinggi untuk:
    1. Node dengan aktivitas seismik (non-zero targets)
    2. Feature tertentu (misalnya Log Energy lebih penting)
    """
    
    def __init__(self, active_weight=5.0, feature_weights=None):
        super().__init__()
        self.active_weight = active_weight
        
        # Default feature weights: [count, max_mw, log_energy, avg_depth]
        if feature_weights is None:
            self.feature_weights = torch.tensor([1.0, 1.0, 2.0, 1.0])
        elif isinstance(feature_weights, dict):
            feature_order = ['count', 'max_mw', 'log_energy', 'avg_depth']
            weights = [feature_weights.get(f, 1.0) for f in feature_order]
            self.feature_weights = torch.tensor(weights)
        else:
            self.feature_weights = torch.tensor(feature_weights)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, H, N, F) or (B, N, F)
            target: Targets (B, H, N, F) or (B, N, F)
        
        Returns:
            Weighted MSE loss
        """
        device = pred.device
        
        # Ensure feature weights on correct device
        if self.feature_weights.device != device:
            self.feature_weights = self.feature_weights.to(device)
        
        # Adjust feature weights shape to match input
        n_features = pred.shape[-1]
        if len(self.feature_weights) > n_features:
            fw = self.feature_weights[:n_features]
        elif len(self.feature_weights) < n_features:
            # Pad with 1.0 for missing features
            fw = torch.cat([self.feature_weights, 
                           torch.ones(n_features - len(self.feature_weights), device=device)])
        else:
            fw = self.feature_weights
        
        # Base MSE
        mse = (pred - target).pow(2)
        
        # ==========================================
        # WEIGHT 1: ACTIVE NODE WEIGHTING
        # ==========================================
        # Identifikasi node dengan aktivitas seismik (ANY non-zero feature)
        # EXACTLY like original: sum of abs values > 0
        if pred.dim() == 4:  # (B, H, N, F)
            active_mask = (target.abs().sum(dim=-1, keepdim=True) > 0).float()  # (B, H, N, 1)
            feature_weights_expanded = fw.view(1, 1, 1, -1)  # (1, 1, 1, F)
        else:  # (B, N, F)
            active_mask = (target.abs().sum(dim=-1, keepdim=True) > 0).float()  # (B, N, 1)
            feature_weights_expanded = fw.view(1, 1, -1)  # (1, 1, F)
        
        # Weight formula (EXACT from original):
        # - Inactive nodes: weight = 1.0 (standard)
        # - Active nodes: weight = 5.0 (penalize errors 5x lebih keras)
        active_weights = 1.0 + (self.active_weight - 1.0) * active_mask
        
        # ==========================================
        # APPLY WEIGHTS
        # ==========================================
        weighted_mse = mse * active_weights * feature_weights_expanded
        
        return weighted_mse.mean()


class FocalMSELoss(nn.Module):
    """
    Focal-style MSE Loss untuk fokus pada hard examples (sparse seismic data).
    
    Focal loss memodifikasi standard loss dengan down-weighting easy examples
    dan fokus pada hard examples. Untuk regresi seismik:
    - Down-weight predictions yang sudah akurat (small error)
    - Up-weight predictions dengan error besar (gempa besar yang missed)
    
    Args:
        gamma: Fokus parameter (default 2.0). Higher = more focus on hard examples
        active_weight: Weight multiplier for non-zero targets
        feature_weights: Per-feature importance weights
    """
    
    def __init__(self, gamma=2.0, active_weight=10.0, feature_weights=None):
        super().__init__()
        self.gamma = gamma
        self.active_weight = active_weight
        
        # Default feature weights: [count, max_mw, log_energy, avg_depth]
        if feature_weights is None:
            self.feature_weights = torch.tensor([1.0, 5.0, 2.0, 1.0])
        elif isinstance(feature_weights, dict):
            feature_order = ['count', 'max_mw', 'log_energy', 'avg_depth']
            weights = [feature_weights.get(f, 1.0) for f in feature_order]
            self.feature_weights = torch.tensor(weights)
        else:
            self.feature_weights = torch.tensor(feature_weights)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, H, N, F)
            target: Targets (B, H, N, F)
        
        Returns:
            Focal MSE loss
        """
        device = pred.device
        
        # Ensure feature weights on correct device
        if self.feature_weights.device != device:
            self.feature_weights = self.feature_weights.to(device)
        
        # Adjust feature weights shape
        n_features = pred.shape[-1]
        if len(self.feature_weights) > n_features:
            fw = self.feature_weights[:n_features]
        elif len(self.feature_weights) < n_features:
            fw = torch.cat([self.feature_weights, 
                           torch.ones(n_features - len(self.feature_weights), device=device)])
        else:
            fw = self.feature_weights
        
        # Per-element squared error
        mse = (pred - target).pow(2)
        
        # Focal modulation: down-weight easy examples (small mse)
        # Weight = (1 - exp(-mse))^gamma
        # When mse is small, weight is small; when mse is large, weight approaches 1
        focal_weight = (1 - torch.exp(-mse)).pow(self.gamma)
        
        # Active sample weighting
        active_mask = (target.abs().sum(dim=-1, keepdim=True) > 0).float()
        sample_weight = 1.0 + (self.active_weight - 1.0) * active_mask
        
        # Combine all weights
        total_weight = focal_weight * sample_weight * fw
        
        # Weighted loss
        weighted_loss = total_weight * mse
        
        return weighted_loss.mean()


class AsymmetricMSELoss(nn.Module):
    """
    Asymmetric MSE Loss untuk max_mw.
    
    Memberikan penalty lebih besar ketika model UNDERPREDICT nilai tinggi.
    Ini penting untuk gempa besar yang jarang terjadi.
    
    Args:
        alpha: Weight untuk underprediction (default 0.8 = 80% weight ke underpredict)
               alpha > 0.5 berarti underpredict lebih dihukum
        magnitude_idx: Index fitur max_mw dalam tensor (default 1)
        active_weight: Extra weight untuk active nodes
        feature_weights: Dict of feature weights
    """
    
    def __init__(self, alpha=0.8, magnitude_idx=1, active_weight=5.0, feature_weights=None):
        super().__init__()
        self.alpha = alpha
        self.magnitude_idx = magnitude_idx
        self.active_weight = active_weight
        
        # Default feature weights
        if feature_weights is None:
            self.feature_weights = torch.tensor([1.0, 5.0, 2.0, 1.0])  # Higher for max_mw
        elif isinstance(feature_weights, dict):
            feature_order = ['count', 'max_mw', 'log_energy', 'avg_depth']
            weights = [feature_weights.get(f, 1.0) for f in feature_order]
            self.feature_weights = torch.tensor(weights)
        else:
            self.feature_weights = torch.tensor(feature_weights)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, H, N, F) or (B, N, F)
            target: Targets (B, H, N, F) or (B, N, F)
        """
        device = pred.device
        
        if self.feature_weights.device != device:
            self.feature_weights = self.feature_weights.to(device)
        
        # Adjust feature weights
        n_features = pred.shape[-1]
        if len(self.feature_weights) > n_features:
            fw = self.feature_weights[:n_features]
        elif len(self.feature_weights) < n_features:
            fw = torch.cat([self.feature_weights, 
                           torch.ones(n_features - len(self.feature_weights), device=device)])
        else:
            fw = self.feature_weights
        
        # Calculate residual: positive = underpredict, negative = overpredict
        residual = target - pred
        
        # Asymmetric weighting: higher penalty for underprediction
        # When residual > 0 (underpredict): use alpha weight
        # When residual < 0 (overpredict): use (1-alpha) weight
        asymmetric_weight = torch.where(
            residual > 0,
            self.alpha * torch.ones_like(residual),
            (1 - self.alpha) * torch.ones_like(residual)
        )
        
        # Apply asymmetric weight only to magnitude feature (index magnitude_idx)
        # For other features, use symmetric weight (0.5)
        if self.magnitude_idx < n_features:
            symmetric_weight = 0.5 * torch.ones_like(residual)
            
            if pred.dim() == 4:  # (B, H, N, F)
                symmetric_weight[..., self.magnitude_idx] = asymmetric_weight[..., self.magnitude_idx]
            else:  # (B, N, F)
                symmetric_weight[..., self.magnitude_idx] = asymmetric_weight[..., self.magnitude_idx]
            
            asymmetric_weight = symmetric_weight
        
        # Base MSE with asymmetric weighting
        mse = residual.pow(2)
        weighted_mse = mse * asymmetric_weight * 2  # *2 to normalize (alpha + (1-alpha) = 1)
        
        # Active node weighting
        if pred.dim() == 4:
            active_mask = (target.abs().sum(dim=-1, keepdim=True) > 0).float()
            feature_weights_expanded = fw.view(1, 1, 1, -1)
        else:
            active_mask = (target.abs().sum(dim=-1, keepdim=True) > 0).float()
            feature_weights_expanded = fw.view(1, 1, -1)
        
        active_weights = 1.0 + (self.active_weight - 1.0) * active_mask
        
        # Final weighted loss
        final_loss = weighted_mse * active_weights * feature_weights_expanded
        
        return final_loss.mean()


class QuantileLoss(nn.Module):
    """
    Quantile Loss untuk prediksi extreme values.
    
    Berguna untuk memprediksi persentil tinggi (misal 90th percentile)
    dari distribusi magnitude.
    """
    
    def __init__(self, quantile=0.9):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, pred, target):
        residual = target - pred
        loss = torch.where(
            residual > 0,
            self.quantile * residual.abs(),
            (1 - self.quantile) * residual.abs()
        )
        return loss.mean()


class ActiveOnlyMSELoss(nn.Module):
    """
    MSE Loss yang HANYA menghitung loss pada sampel aktif.
    
    Sampel dengan target = 0 (tidak ada gempa) sepenuhnya diabaikan.
    Ini memaksa model untuk fokus pada memprediksi kejadian gempa,
    bukan mengoptimalkan prediksi 0.
    
    Args:
        feature_weights: Bobot per fitur
        min_active_ratio: Minimal ratio sampel aktif untuk fallback ke weighted MSE
    """
    
    def __init__(self, feature_weights=None, min_active_ratio=0.01):
        super().__init__()
        self.min_active_ratio = min_active_ratio
        
        if feature_weights is None:
            self.feature_weights = torch.tensor([1.0, 5.0, 2.0, 1.0])
        elif isinstance(feature_weights, dict):
            feature_order = ['count', 'max_mw', 'log_energy', 'avg_depth']
            weights = [feature_weights.get(f, 1.0) for f in feature_order]
            self.feature_weights = torch.tensor(weights)
        else:
            self.feature_weights = torch.tensor(feature_weights)
    
    def forward(self, pred, target):
        device = pred.device
        
        if self.feature_weights.device != device:
            self.feature_weights = self.feature_weights.to(device)
        
        # Adjust feature weights
        n_features = pred.shape[-1]
        if len(self.feature_weights) > n_features:
            fw = self.feature_weights[:n_features]
        elif len(self.feature_weights) < n_features:
            fw = torch.cat([self.feature_weights, 
                           torch.ones(n_features - len(self.feature_weights), device=device)])
        else:
            fw = self.feature_weights
        
        # Identify active samples (any feature > threshold, using count as proxy)
        # Flatten spatial dimensions for mask
        if pred.dim() == 4:  # (B, H, N, F)
            # Active if ANY feature is non-zero (use first feature as activity indicator)
            active_mask = target[..., 0].abs() > 0.01  # (B, H, N)
            feature_weights_expanded = fw.view(1, 1, 1, -1)
        else:  # (B, N, F)
            active_mask = target[..., 0].abs() > 0.01  # (B, N)
            feature_weights_expanded = fw.view(1, 1, -1)
        
        # Count active samples
        n_active = active_mask.sum().item()
        n_total = active_mask.numel()
        active_ratio = n_active / max(n_total, 1)
        
        # MSE
        mse = (pred - target).pow(2)
        weighted_mse = mse * feature_weights_expanded
        
        # If too few active samples, fallback to weighted MSE
        if active_ratio < self.min_active_ratio:
            # Weight active samples 10x more
            weights = torch.where(active_mask.unsqueeze(-1), 10.0, 1.0)
            return (weighted_mse * weights).mean()
        
        # Only compute loss on active samples
        active_mask_expanded = active_mask.unsqueeze(-1).expand_as(weighted_mse)
        active_loss = weighted_mse[active_mask_expanded]
        
        return active_loss.mean()


class SparseAwareLoss(nn.Module):
    """
    Loss function yang adaptif terhadap sparsity data.
    
    Fitur utama:
    1. Menghitung loss TERPISAH untuk active vs inactive samples
    2. Memberikan bobot dinamis berdasarkan ratio aktif/inaktif
    3. Menambah extra penalty untuk under-prediction pada active samples
    
    Args:
        active_loss_weight: Base weight untuk active samples (dinaikkan sesuai imbalance)
        underpredict_penalty: Extra penalty untuk underprediction magnitude
        feature_weights: Bobot per fitur
    """
    
    def __init__(self, active_loss_weight=10.0, underpredict_penalty=2.0, feature_weights=None):
        super().__init__()
        self.active_loss_weight = active_loss_weight
        self.underpredict_penalty = underpredict_penalty
        
        if feature_weights is None:
            self.feature_weights = torch.tensor([1.0, 5.0, 2.0, 1.0])
        elif isinstance(feature_weights, dict):
            feature_order = ['count', 'max_mw', 'log_energy', 'avg_depth']
            weights = [feature_weights.get(f, 1.0) for f in feature_order]
            self.feature_weights = torch.tensor(weights)
        else:
            self.feature_weights = torch.tensor(feature_weights)
    
    def forward(self, pred, target):
        device = pred.device
        
        if self.feature_weights.device != device:
            self.feature_weights = self.feature_weights.to(device)
        
        n_features = pred.shape[-1]
        if len(self.feature_weights) > n_features:
            fw = self.feature_weights[:n_features]
        elif len(self.feature_weights) < n_features:
            fw = torch.cat([self.feature_weights, 
                           torch.ones(n_features - len(self.feature_weights), device=device)])
        else:
            fw = self.feature_weights
        
        # Identify active samples
        if pred.dim() == 4:
            active_mask = target[..., 0].abs() > 0.01  # (B, H, N)
            feature_weights_expanded = fw.view(1, 1, 1, -1)
        else:
            active_mask = target[..., 0].abs() > 0.01  # (B, N)
            feature_weights_expanded = fw.view(1, 1, -1)
        
        # Calculate imbalance ratio for dynamic weighting
        n_active = active_mask.sum().item()
        n_inactive = (~active_mask).sum().item()
        imbalance_ratio = max(n_inactive / max(n_active, 1), 1.0)
        dynamic_active_weight = min(self.active_loss_weight * imbalance_ratio, 100.0)
        
        # Base MSE
        residual = pred - target
        mse = residual.pow(2)
        
        # Asymmetric penalty for underprediction on active samples
        # If target > pred (underprediction), multiply loss
        underpredict_mask = (target > pred) & active_mask.unsqueeze(-1)
        asymmetric_weight = torch.where(underpredict_mask, self.underpredict_penalty, 1.0)
        
        # Weight active samples
        active_weight = torch.where(active_mask.unsqueeze(-1), dynamic_active_weight, 1.0)
        
        # Combine weights
        weighted_mse = mse * feature_weights_expanded * active_weight * asymmetric_weight
        
        return weighted_mse.mean()


class HybridLoss(nn.Module):
    """
    Hybrid Loss untuk model dengan dual output:
    - Classification head (max_mw significance)
    - Regression head (count, log_energy, avg_depth)
    
    Combines:
    - Cross-Entropy Loss untuk classification (dengan class weights untuk imbalance)
    - Weighted MSE Loss untuk regression
    
    Args:
        cls_weight: Weight untuk classification loss
        reg_weight: Weight untuk regression loss
        class_weights: Weights untuk class imbalance (optional)
        active_weight: Weight untuk active nodes
        magnitude_threshold: Threshold untuk significant earthquake
    """
    
    def __init__(self, cls_weight=1.0, reg_weight=1.0, 
                 class_weights=None, active_weight=5.0,
                 n_classes=4):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.active_weight = active_weight
        self.n_classes = n_classes
        
        # Class weights untuk handle imbalance
        # 4 Classes: M<1, M1-2, M2-3, M>=3
        # Based on diagnostic: 0.8%, 55.8%, 37.5%, 5.8%
        # Higher weights for rarer classes
        if class_weights is None:
            # Default weights inversely proportional to frequency
            self.class_weights = torch.tensor([10.0, 1.0, 2.0, 15.0])  # M<1 and M>=3 are rare
        else:
            self.class_weights = torch.tensor(class_weights)
        
        # Regression feature weights [count, log_energy, avg_depth]
        self.reg_feature_weights = torch.tensor([1.0, 2.0, 1.0])
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'regression' and 'classification' keys
                - regression: (B, H, N, n_reg_features)
                - classification: (B, H, N, n_classes) - logits
            targets: dict with 'regression' and 'classification' keys
                - regression: (B, H, N, n_reg_features) - continuous values
                - classification: (B, H, N) - class labels (0 or 1)
        
        Returns:
            Total loss (scalar)
        """
        device = outputs['regression'].device
        
        # ==================================================
        # CLASSIFICATION LOSS (Cross-Entropy)
        # ==================================================
        cls_logits = outputs['classification']  # (B, H, N, n_classes)
        cls_targets = targets['classification']  # (B, H, N) - integer labels
        
        # Reshape for cross-entropy: (B*H*N, n_classes) and (B*H*N,)
        B, H, N, C = cls_logits.shape
        cls_logits_flat = cls_logits.reshape(-1, C)
        cls_targets_flat = cls_targets.reshape(-1).long()
        
        # Apply class weights
        if self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)
        
        cls_loss = nn.functional.cross_entropy(
            cls_logits_flat, 
            cls_targets_flat,
            weight=self.class_weights
        )
        
        # ==================================================
        # REGRESSION LOSS (Weighted MSE)
        # ==================================================
        reg_pred = outputs['regression']     # (B, H, N, n_reg)
        reg_target = targets['regression']   # (B, H, N, n_reg)
        
        # Feature weights
        if self.reg_feature_weights.device != device:
            self.reg_feature_weights = self.reg_feature_weights.to(device)
        
        n_reg = reg_pred.shape[-1]
        if len(self.reg_feature_weights) >= n_reg:
            fw = self.reg_feature_weights[:n_reg]
        else:
            fw = torch.cat([
                self.reg_feature_weights,
                torch.ones(n_reg - len(self.reg_feature_weights), device=device)
            ])
        
        # MSE with feature weights
        mse = (reg_pred - reg_target).pow(2)
        fw_expanded = fw.view(1, 1, 1, -1)
        
        # Active node weighting
        active_mask = (reg_target.abs().sum(dim=-1, keepdim=True) > 0).float()
        active_weights = 1.0 + (self.active_weight - 1.0) * active_mask
        
        weighted_mse = mse * fw_expanded * active_weights
        reg_loss = weighted_mse.mean()
        
        # ==================================================
        # COMBINED LOSS
        # ==================================================
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return total_loss
    
    def get_component_losses(self, outputs, targets):
        """Get individual loss components for logging."""
        device = outputs['regression'].device
        
        # Classification
        cls_logits = outputs['classification']
        cls_targets = targets['classification']
        B, H, N, C = cls_logits.shape
        cls_logits_flat = cls_logits.view(-1, C)
        cls_targets_flat = cls_targets.view(-1).long()
        
        if self.class_weights.device != device:
            self.class_weights = self.class_weights.to(device)
        
        cls_loss = nn.functional.cross_entropy(
            cls_logits_flat, cls_targets_flat, weight=self.class_weights
        )
        
        # Regression
        reg_pred = outputs['regression']
        reg_target = targets['regression']
        reg_loss = (reg_pred - reg_target).pow(2).mean()
        
        return {
            'classification_loss': cls_loss.item(),
            'regression_loss': reg_loss.item(),
            'total_loss': (self.cls_weight * cls_loss + self.reg_weight * reg_loss).item()
        }


class MultiScaleLoss(nn.Module):
    """
    Multi-Scale Loss for earthquake prediction at multiple time horizons.
    
    Evaluates predictions at different temporal scales (e.g., 6h, 12h, 24h)
    by computing max over different horizon windows.
    
    For magnitude prediction (max_mw), this answers:
    - "What's the max magnitude in next 6 hours?"
    - "What's the max magnitude in next 12 hours?"
    - "What's the max magnitude in next 24 hours?"
    
    Args:
        scales: List of horizon steps for each scale, e.g., [1, 2, 4] for 6h, 12h, 24h
        scale_weights: Weights for each scale loss
        magnitude_idx: Index of max_mw in feature dimension
        base_loss: Base loss type ('mse', 'focal', 'weighted')
        active_weight: Weight for active (non-zero) samples
    """
    
    def __init__(self, scales=[1, 2, 4], scale_weights=None, magnitude_idx=1,
                 base_loss='mse', active_weight=10.0, feature_weights=None):
        super().__init__()
        self.scales = scales
        self.scale_weights = scale_weights if scale_weights else [1.0] * len(scales)
        self.magnitude_idx = magnitude_idx
        self.base_loss = base_loss
        self.active_weight = active_weight
        
        # Default feature weights
        if feature_weights is None:
            self.feature_weights = torch.tensor([1.0, 5.0, 2.0, 1.0])  # Emphasize magnitude
        elif isinstance(feature_weights, dict):
            feature_order = ['count', 'max_mw', 'log_energy', 'avg_depth']
            weights = [feature_weights.get(f, 1.0) for f in feature_order]
            self.feature_weights = torch.tensor(weights)
        else:
            self.feature_weights = torch.tensor(feature_weights)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, H, N, F) where H = max horizon (e.g., 4 for 24h)
            target: Targets (B, H, N, F)
        
        Returns:
            Combined multi-scale loss
        """
        device = pred.device
        batch_size, horizon, nodes, features = pred.shape
        
        # Ensure feature weights on correct device
        if self.feature_weights.device != device:
            self.feature_weights = self.feature_weights.to(device)
        
        # Adjust feature weights
        n_features = features
        if len(self.feature_weights) > n_features:
            fw = self.feature_weights[:n_features]
        elif len(self.feature_weights) < n_features:
            fw = torch.cat([self.feature_weights, 
                           torch.ones(n_features - len(self.feature_weights), device=device)])
        else:
            fw = self.feature_weights
        
        total_loss = 0.0
        scale_losses = {}
        
        # Standard per-timestep loss for all features
        mse_all = (pred - target).pow(2)
        active_mask = (target.abs().sum(dim=-1, keepdim=True) > 0).float()
        sample_weight = 1.0 + (self.active_weight - 1.0) * active_mask
        weighted_mse = mse_all * sample_weight * fw
        base_loss_val = weighted_mse.mean()
        total_loss = base_loss_val
        scale_losses['base'] = base_loss_val.item()
        
        # Multi-scale loss for magnitude only
        mw_idx = self.magnitude_idx
        if mw_idx < features:
            pred_mw = pred[..., mw_idx]  # (B, H, N)
            target_mw = target[..., mw_idx]  # (B, H, N)
            
            for scale_idx, (scale_h, scale_w) in enumerate(zip(self.scales, self.scale_weights)):
                if scale_h > horizon:
                    scale_h = horizon  # Cap at available horizon
                
                # Compute max over the scale window
                # pred_max_scale[t] = max(pred[0:scale_h])
                pred_max_scale = pred_mw[:, :scale_h, :].max(dim=1)[0]  # (B, N)
                target_max_scale = target_mw[:, :scale_h, :].max(dim=1)[0]  # (B, N)
                
                # Scale-specific loss
                scale_mse = (pred_max_scale - target_max_scale).pow(2)
                
                # Weight by active samples
                scale_active = (target_max_scale.abs() > 0).float()
                scale_weight = 1.0 + (self.active_weight - 1.0) * scale_active
                scale_loss = (scale_mse * scale_weight).mean()
                
                total_loss = total_loss + scale_w * scale_loss
                scale_losses[f'scale_{scale_h}h'] = scale_loss.item()
        
        # Store for logging
        self._last_scale_losses = scale_losses
        
        return total_loss
    
    def get_scale_losses(self):
        """Get individual scale losses from last forward pass."""
        return getattr(self, '_last_scale_losses', {})
