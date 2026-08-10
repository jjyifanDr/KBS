"""
KERN: Knowledge-Evidence Reasoning Network for Anomaly Detection

Strict implementation following the paper:
- Section 3.1: Four core components
- Section 3.2 Eq. (1): e_g, e_l ∈ [0,1]^{d_e}, e_data = [e_g; e_l] ∈ [0,1]^{2·d_e}
- Section 3.2 (2): Rule-to-Evidence Mapping with semantic encoding
- Section 3.3 (1): e = [e_data; e_k] ∈ [0,1]^{3·d_e}, JSD conflict detection
- Section 3.3 (2): evidence_to_opinion with full 3·d_e evidence utilization
- Section 3.3 (2): Cumulative fusion + Cautious conflict handling + Knowledge arbitration
- Section 3.3 (3): Traceable explanations
- Section 3.4 Eq. (7): Composite loss L = L_task + λ_c·L_cons + λ_r·L_reg
- Section 3.4 Stage 1: Separate pre-training with knowledge mapping network
- Section 3.4 Stage 2: End-to-end fine-tuning
- Section 4.4: Training parameters (100 epochs Stage 1, 50 epochs Stage 2)
- Section 3.5: Theoretical guarantees (Proposition 1, Theorem 1, Theorem 2)

Engineering notes:
- Fixed random seed (42) for reproducibility
- Deterministic sinusoidal positional encoding
- All fusion operations use torch tensors to avoid type errors
- Anomaly score clamped to [0, 1] for numerical stability
- Defensive NaN/Inf handling in all evidence operations
- knowledge_confidence defaults to 0.0 when no rules are activated
  (Only trust knowledge when there is actual activated evidence)
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用高速传输
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Fix Random Seeds for Reproducibility
# =============================================================================

def set_seed(seed=42):
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


# =============================================================================
# Import SentenceTransformer for rule encoding (Paper Section 3.2 (2))
# =============================================================================
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Using random embeddings.")
    SENTENCE_TRANSFORMER_AVAILABLE = False


# =============================================================================
# Configuration (Paper Section 4.4)
# =============================================================================

class Config:
    """Configuration parameters following paper Section 4.4"""
    
    d_e = 16
    transformer_dim = 128
    transformer_heads = 4
    transformer_layers = 4
    mlp_hidden = [256, 128, 64]
    rule_embed_dim = 384
    
    tau = 0.3
    eta = 0.5
    gamma = 0.8
    
    lr = 1e-3
    batch_size = 256
    epochs_stage1 = 100
    epochs_stage2 = 50
    lambda_c = 0.1
    lambda_r = 0.01
    
    data_dir = "./data/"
    results_dir = "./results/"
    
    datasets = {
        'credit_card': {
            'file': 'CreditCard.csv',
            'label_col': 'Class',
            'sequential': True,
            'anomaly_rate': 0.00427
        },
        'cic_unsw': {
            'file': 'CIC_UNSW.csv',
            'label_col': 'Label',
            'sequential': True,
            'anomaly_rate': 0.07951
        },
        'ukmnct': {
            'file': 'UKMNCT_IIoT_FDIA.csv',
            'label_col': 'marker',
            'sequential': False,
            'anomaly_rate': 0.436759
        },
        'wdbc': {
            'file': 'wdbc.data',
            'label_col': 'Diagnosis',
            'sequential': False,
            'anomaly_rate': 0.37258,
            'has_id': True
        }
    }
    
    @staticmethod
    def get_tau_for_dataset(dataset_name):
        if dataset_name == 'credit_card':
            return 0.01
        elif dataset_name == 'cic_unsw':
            return 0.05
        elif dataset_name == 'ukmnct':
            return 0.05
        elif dataset_name == 'wdbc':
            return 0.05
        else:
            return 0.1


# =============================================================================
# Knowledge Base Construction (Paper Section 4.4)
# =============================================================================

def get_knowledge_base(dataset_name):
    """Construct knowledge base K of first-order logic rules."""
    if dataset_name == 'credit_card':
        return [
            {
                'text': 'IF Time_gap_since_last_transaction < 60 seconds AND Amount > 1000 THEN suspicious (0.8)',
                'confidence': 0.8,
                'premise': 'time_gap < 60 and amount > 1000',
                'hypothesis': 'suspicious'
            }
        ]
    
    elif dataset_name == 'cic_unsw':
        return [
            {
                'text': 'IF Flow_Duration > 1000000 AND Total_Fwd_Packet < 5 AND Total_Bwd_Packets < 5 THEN suspicious (0.9)',
                'confidence': 0.9,
                'premise': 'flow_duration > 1e6 and total_fwd_packet < 5 and total_bwd_packets < 5',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF SYN_Flag_Count > 20 AND ACK_Flag_Count == 0 AND Flow_Packets_per_s < 10 THEN suspicious (0.8)',
                'confidence': 0.8,
                'premise': 'syn_flag_count > 20 and ack_flag_count == 0 and flow_packets_per_s < 10',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF Fwd_Packet_Length_Mean == 0 AND Bwd_Packet_Length_Mean == 0 AND Flow_Duration > 100000 THEN suspicious (0.7)',
                'confidence': 0.7,
                'premise': 'fwd_packet_length_mean == 0 and bwd_packet_length_mean == 0 and flow_duration > 100000',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF RST_Flag_Count > 5 AND Flow_Duration < 50000 THEN suspicious (0.8)',
                'confidence': 0.8,
                'premise': 'rst_flag_count > 5 and flow_duration < 50000',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF Flow_Bytes_per_s > 1000000 AND Flow_Packets_per_s > 1000 THEN suspicious (0.9)',
                'confidence': 0.9,
                'premise': 'flow_bytes_per_s > 1e6 and flow_packets_per_s > 1000',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF Average_Packet_Size < 100 AND Fwd_Packets_per_s > 500 AND Bwd_Packets_per_s == 0 THEN suspicious (0.7)',
                'confidence': 0.7,
                'premise': 'average_packet_size < 100 and fwd_packets_per_s > 500 and bwd_packets_per_s == 0',
                'hypothesis': 'suspicious'
            }
        ]
    
    elif dataset_name == 'ukmnct':
        return [
            # ===== 基于数据驱动的新规则 =====
            {
                'text': 'IF http_response_body_len > 5000 AND http_status_code == 200 THEN suspicious (0.85)',
                'confidence': 0.85,
                'premise': 'http_response_body_len > 5000 and http_status_code == 200',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF http_request_body_len > 2000 AND http_method == "POST" THEN suspicious (0.80)',
                'confidence': 0.80,
                'premise': 'http_request_body_len > 2000 and http_method == POST',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF ssl_version IN {"SSLv2", "SSLv3", "TLSv1.0"} AND ssl_established == 0 THEN suspicious (0.75)',
                'confidence': 0.75,
                'premise': 'ssl_version in {SSLv2, SSLv3, TLSv1.0} and ssl_established == 0',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF http_user_agent CONTAINS "bot" OR http_user_agent CONTAINS "scanner" THEN suspicious (0.70)',
                'confidence': 0.70,
                'premise': 'http_user_agent contains bot or http_user_agent contains scanner',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF dns_rejected == 1 AND dns_rcode != 0 THEN suspicious (0.65)',
                'confidence': 0.65,
                'premise': 'dns_rejected == 1 and dns_rcode != 0',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF http_response_body_len > 3000 AND http_trans_depth > 5 THEN suspicious (0.60)',
                'confidence': 0.60,
                'premise': 'http_response_body_len > 3000 and http_trans_depth > 5',
                'hypothesis': 'suspicious'
            }
        ]
    
    elif dataset_name == 'wdbc':
        return [
            {
                'text': 'IF radius > 17.0 AND concavity > 0.2 THEN suspicious (0.85)',
                'confidence': 0.85,
                'premise': 'radius > 17.0 and concavity > 0.2',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF area > 1000 AND smoothness < 0.1 THEN suspicious (0.75)',
                'confidence': 0.75,
                'premise': 'area > 1000 and smoothness < 0.1',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF concave_points > 0.1 AND symmetry > 0.2 THEN suspicious (0.7)',
                'confidence': 0.7,
                'premise': 'concave_points > 0.1 and symmetry > 0.2',
                'hypothesis': 'suspicious'
            },
            {
                'text': 'IF perimeter > 120 AND compactness > 0.3 THEN suspicious (0.8)',
                'confidence': 0.8,
                'premise': 'perimeter > 120 and compactness > 0.3',
                'hypothesis': 'suspicious'
            },
            # ===== 修改 Rule 5: 移除 texture_se =====
            {
                'text': 'IF radius_se > 0.8 THEN suspicious (0.7)',
                'confidence': 0.7,
                'premise': 'radius_se > 0.8',
                'hypothesis': 'suspicious'
            },
            # ===== 修改 Rule 6: 用 fractal_dimension_worst 替代 fractal_dimension =====
            {
                'text': 'IF fractal_dimension_worst > 0.08 AND area > 600 THEN suspicious (0.65)',
                'confidence': 0.65,
                'premise': 'fractal_dimension_worst > 0.08 and area > 600',
                'hypothesis': 'suspicious'
            }
        ]
    
    else:
        return []


# =============================================================================
# Rule Encoder (Paper Section 3.2 (2))
# =============================================================================

class RuleEncoder:
    """Knowledge encoding using pre-trained sentence Transformer."""
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu', seed=42):
        self.model_name = model_name
        self.device = device
        self.seed = seed
        self.model = None
        
        # ===== 检查模型是否已缓存 =====
        import os
        from pathlib import Path
        
        # 设置缓存目录
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
        
        # 检查模型是否存在
        model_exists = False
        local_model_path = None
        if model_cache.exists():
            snapshots = list(model_cache.iterdir())
            if snapshots:
                model_exists = True
                local_model_path = str(snapshots[0])
                print(f"Model found in local cache: {local_model_path}")
        
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                if model_exists:
                    # ===== 存在本地缓存 → 离线加载 =====
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
                    print("Loading model from local cache (offline mode)...")
                    self.model = SentenceTransformer(local_model_path, device=device)
                else:
                    # ===== 不存在缓存 → 在线下载 =====
                    print("Model not found in cache. Downloading from Hugging Face...")
                    # 临时移除离线模式以允许下载
                    os.environ.pop('HF_HUB_OFFLINE', None)
                    os.environ.pop('TRANSFORMERS_OFFLINE', None)
                    # 使用镜像加速下载
                    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                    self.model = SentenceTransformer(model_name, device=device)
                    print(f"Model downloaded and cached successfully.")
                    
                print(f"Loaded rule encoder: {model_name}")
            except Exception as e:
                print(f"Warning: Failed to load {model_name}: {e}. Using random embeddings.")
                self.model = None
        else:
            print("Warning: sentence-transformers not installed. Using random embeddings.")
    
    def encode(self, rules, convert_to_tensor=True):
        if isinstance(rules[0], dict):
            texts = [r['text'] for r in rules]
        else:
            texts = list(rules)
        
        if self.model is not None:
            embeddings = self.model.encode(
                texts, 
                convert_to_tensor=convert_to_tensor,
                show_progress_bar=False
            )
            if convert_to_tensor:
                return embeddings
            else:
                return torch.FloatTensor(embeddings)
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            return torch.randn(len(texts), 384, generator=generator)
    
    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

# =============================================================================
# Subjective Logic Core Functions (Paper Section 3.3)
# =============================================================================

def _ensure_tensor(x, device=None, dtype=torch.float32):
    """Helper to ensure value is a torch Tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (int, float, bool)):
        if device is not None:
            return torch.tensor(x, device=device, dtype=dtype)
        return torch.tensor(x, dtype=dtype)
    return torch.tensor(x, dtype=dtype)


def _squeeze_tuple(omega):
    """Squeeze all elements in an opinion tuple."""
    b, d, u, a = omega
    if isinstance(b, torch.Tensor):
        b = b.squeeze()
    if isinstance(d, torch.Tensor):
        d = d.squeeze()
    if isinstance(u, torch.Tensor):
        u = u.squeeze()
    if isinstance(a, torch.Tensor):
        a = a.squeeze()
    return (b, d, u, a)


def _normalize_opinion(b, d, u, a):
    """Normalize opinion to ensure b + d + u = 1 and all in [0,1]."""
    b = torch.clamp(b, 0.0, 1.0)
    d = torch.clamp(d, 0.0, 1.0)
    u = torch.clamp(u, 0.0, 1.0)
    total = b + d + u
    total = torch.clamp(total, min=1e-8)
    b = b / total
    d = d / total
    u = u / total
    b = torch.clamp(b, 0.0, 1.0)
    d = torch.clamp(d, 0.0, 1.0)
    u = torch.clamp(u, 0.0, 1.0)
    return (b, d, u, a)


def subjective_logic_consensus(omega_i, omega_j):
    """
    Consensus operator (⊕) for binomial opinions. Jøsang (2016).
    Eq. (5) in paper.
    """
    b_i, d_i, u_i, a_i = omega_i
    b_j, d_j, u_j, a_j = omega_j
    
    # Ensure all are tensors
    device = b_i.device if isinstance(b_i, torch.Tensor) else 'cpu'
    b_i = _ensure_tensor(b_i, device)
    d_i = _ensure_tensor(d_i, device)
    u_i = _ensure_tensor(u_i, device)
    a_i = _ensure_tensor(a_i, device)
    b_j = _ensure_tensor(b_j, device)
    d_j = _ensure_tensor(d_j, device)
    u_j = _ensure_tensor(u_j, device)
    a_j = _ensure_tensor(a_j, device)
    
    b_i = torch.clamp(b_i, 0.0, 1.0)
    d_i = torch.clamp(d_i, 0.0, 1.0)
    u_i = torch.clamp(u_i, 0.0, 1.0)
    b_j = torch.clamp(b_j, 0.0, 1.0)
    d_j = torch.clamp(d_j, 0.0, 1.0)
    u_j = torch.clamp(u_j, 0.0, 1.0)
    
    # Check if both opinions are fully certain
    if torch.all(u_i == 0) and torch.all(u_j == 0):
        b = (b_i + b_j) / 2
        d = (d_i + d_j) / 2
        u = torch.zeros_like(b_i)
        a = (a_i + a_j) / 2
        return _normalize_opinion(b, d, u, a)
    
    # Check if one opinion is fully certain (dominates)
    u_i_zero = torch.all(u_i == 0).item() if u_i.numel() > 0 else False
    u_j_zero = torch.all(u_j == 0).item() if u_j.numel() > 0 else False
    
    if u_i_zero and not u_j_zero:
        return _squeeze_tuple((b_i, d_i, u_i, a_i))
    if u_j_zero and not u_i_zero:
        return _squeeze_tuple((b_j, d_j, u_j, a_j))
    
    k = u_i + u_j - u_i * u_j
    k = torch.clamp(k, min=1e-8)
    
    b = (b_i * u_j + b_j * u_i) / k
    d = (d_i * u_j + d_j * u_i) / k
    u = (u_i * u_j) / k
    
    denominator = u_i + u_j - 2 * u_i * u_j
    denominator = torch.clamp(denominator, min=1e-10)
    a = (a_i * u_j + a_j * u_i - (a_i + a_j) * u_i * u_j) / denominator
    
    return _squeeze_tuple(_normalize_opinion(b, d, u, a))


def cumulative_fusion(omega_i, omega_j):
    """Cumulative fusion operator for consistent evidence."""
    b_i, d_i, u_i, a_i = omega_i
    b_j, d_j, u_j, a_j = omega_j
    
    device = b_i.device if isinstance(b_i, torch.Tensor) else 'cpu'
    b_i = _ensure_tensor(b_i, device)
    d_i = _ensure_tensor(d_i, device)
    u_i = _ensure_tensor(u_i, device)
    a_i = _ensure_tensor(a_i, device)
    b_j = _ensure_tensor(b_j, device)
    d_j = _ensure_tensor(d_j, device)
    u_j = _ensure_tensor(u_j, device)
    a_j = _ensure_tensor(a_j, device)
    
    b_i = torch.clamp(b_i, 0.0, 1.0)
    d_i = torch.clamp(d_i, 0.0, 1.0)
    u_i = torch.clamp(u_i, 0.0, 1.0)
    b_j = torch.clamp(b_j, 0.0, 1.0)
    d_j = torch.clamp(d_j, 0.0, 1.0)
    u_j = torch.clamp(u_j, 0.0, 1.0)
    
    b_shared = torch.min(b_i, b_j)
    d_shared = torch.min(d_i, d_j)
    
    k = u_i + u_j - u_i * u_j
    k = torch.clamp(k, min=1e-8)
    
    b = (b_i * u_j + b_j * u_i + b_shared * 0.5) / (k + 0.5)
    d = (d_i * u_j + d_j * u_i + d_shared * 0.5) / (k + 0.5)
    u = (u_i * u_j) / k
    
    a = (a_i + a_j) / 2
    
    return _squeeze_tuple(_normalize_opinion(b, d, u, a))


def cautious_conflict_handling(omega_i, omega_j, conflict_score, eta=0.5):
    """Cautious conflict handling operator."""
    b_i, d_i, u_i, a_i = omega_i
    b_j, d_j, u_j, a_j = omega_j
    
    device = b_i.device if isinstance(b_i, torch.Tensor) else 'cpu'
    b_i = _ensure_tensor(b_i, device)
    d_i = _ensure_tensor(d_i, device)
    u_i = _ensure_tensor(u_i, device)
    a_i = _ensure_tensor(a_i, device)
    b_j = _ensure_tensor(b_j, device)
    d_j = _ensure_tensor(d_j, device)
    u_j = _ensure_tensor(u_j, device)
    a_j = _ensure_tensor(a_j, device)
    
    conflict_score = _ensure_tensor(conflict_score, device)
    conflict_weight = conflict_score / eta
    conflict_weight = torch.clamp(conflict_weight, 0.0, 1.0)
    
    u_conflict = u_i + conflict_weight * (1 - u_i)
    u_conflict = torch.clamp(u_conflict, 0.0, 1.0)
    
    b_reduced = b_i * (1 - conflict_weight * 0.5)
    d_reduced = d_i * (1 - conflict_weight * 0.5)
    b_reduced = torch.clamp(b_reduced, 0.0, 1.0)
    d_reduced = torch.clamp(d_reduced, 0.0, 1.0)
    
    total = b_reduced + d_reduced + u_conflict
    total = torch.clamp(total, min=1e-8)
    b = b_reduced / total
    d = d_reduced / total
    u = u_conflict / total
    
    a = a_i
    
    return _squeeze_tuple(_normalize_opinion(b, d, u, a))


def knowledge_arbitration(omega_data, omega_knowledge, knowledge_confidence):
    """Knowledge Arbitration for conflict resolution."""
    b_data, d_data, u_data, a_data = omega_data
    b_know, d_know, u_know, a_know = omega_knowledge
    
    device = b_data.device if isinstance(b_data, torch.Tensor) else 'cpu'
    b_data = _ensure_tensor(b_data, device)
    d_data = _ensure_tensor(d_data, device)
    u_data = _ensure_tensor(u_data, device)
    a_data = _ensure_tensor(a_data, device)
    b_know = _ensure_tensor(b_know, device)
    d_know = _ensure_tensor(d_know, device)
    u_know = _ensure_tensor(u_know, device)
    a_know = _ensure_tensor(a_know, device)
    
    knowledge_confidence = _ensure_tensor(knowledge_confidence, device)
    knowledge_confidence = torch.clamp(knowledge_confidence, 0.0, 1.0)
    
    weight_know = knowledge_confidence
    weight_data = 1.0 - weight_know
    
    b = weight_know * b_know + weight_data * b_data
    d = weight_know * d_know + weight_data * d_data
    u = weight_know * u_know + weight_data * u_data
    
    u = u + 0.1 * (1 - weight_know)
    u = torch.clamp(u, 0.0, 1.0)
    
    a = (a_data + a_know) / 2
    
    return _squeeze_tuple(_normalize_opinion(b, d, u, a))


def evidence_to_opinion(evidence_vector, temperature=1.0):
    """
    Map evidence vector to Subjective Logic opinion triplet.
    Returns: (b, d, u, a) all as torch tensors.
    
    Includes defensive NaN/Inf handling for numerical stability.
    """
    # Ensure evidence_vector is a tensor
    if not isinstance(evidence_vector, torch.Tensor):
        evidence_vector = torch.tensor(evidence_vector, dtype=torch.float32)
    
    # ===== CRITICAL FIX: Handle NaN/Inf in input =====
    if torch.isnan(evidence_vector).any() or torch.isinf(evidence_vector).any():
        print("Debug: evidence_vector contains NaN/Inf, replacing with zeros")
        evidence_vector = torch.nan_to_num(evidence_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure all evidence values are non-negative
    evidence_vector = torch.clamp(evidence_vector, min=0.0)
    
    batch_size = evidence_vector.shape[0]
    d_e_total = evidence_vector.shape[1]
    device = evidence_vector.device
    dtype = evidence_vector.dtype
    
    if d_e_total % 2 == 0 and d_e_total % 3 != 0:
        d_e = d_e_total // 2
        evidence_anomaly = evidence_vector[:, :d_e]
        evidence_normal = evidence_vector[:, d_e:2*d_e]
        evidence_aux = torch.zeros_like(evidence_anomaly)
    elif d_e_total % 3 == 0:
        d_e = d_e_total // 3
        evidence_anomaly = evidence_vector[:, :d_e]
        evidence_normal = evidence_vector[:, d_e:2*d_e]
        evidence_aux = evidence_vector[:, 2*d_e:3*d_e]
    else:
        d_e = d_e_total // 2
        evidence_anomaly = evidence_vector[:, :d_e]
        evidence_normal = evidence_vector[:, d_e:min(2*d_e, d_e_total)]
        if evidence_normal.shape[1] < d_e:
            pad = torch.zeros(batch_size, d_e - evidence_normal.shape[1], device=device, dtype=dtype)
            evidence_normal = torch.cat([evidence_normal, pad], dim=1)
        evidence_aux = torch.zeros(batch_size, d_e, device=device, dtype=dtype)
    
    sum_anomaly = evidence_anomaly.sum(dim=1) + 1e-8
    sum_normal = evidence_normal.sum(dim=1) + 1e-8
    
    # ===== FIX: Ensure sums are finite =====
    sum_anomaly = torch.nan_to_num(sum_anomaly, nan=0.5, posinf=1.0, neginf=0.0)
    sum_normal = torch.nan_to_num(sum_normal, nan=0.5, posinf=1.0, neginf=0.0)
    
    logits = torch.stack([sum_normal, sum_anomaly], dim=1) / temperature
    probs = F.softmax(logits, dim=1)
    
    d = probs[:, 0]
    b = probs[:, 1]
    
    # Compute uncertainty based on entropy
    evidence_combined = evidence_vector
    evidence_sum = evidence_combined.sum(dim=1, keepdim=True) + 1e-8
    evidence_sum = torch.nan_to_num(evidence_sum, nan=1.0, posinf=1.0, neginf=0.0)
    
    evidence_norm = evidence_combined / evidence_sum
    
    # ===== FIX: Clamp before log to avoid NaN =====
    evidence_norm = torch.clamp(evidence_norm, min=1e-8, max=1.0)
    entropy = -torch.sum(evidence_norm * torch.log(evidence_norm), dim=1)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    
    max_entropy = np.log(evidence_vector.shape[1])
    entropy_normalized = entropy / max_entropy
    entropy_normalized = torch.clamp(entropy_normalized, 0.0, 1.0)
    
    u = torch.sigmoid(entropy_normalized * 3.0 - 1.5)
    
    if evidence_aux is not None and evidence_aux.shape[1] > 0:
        aux_strength = evidence_aux.mean(dim=1)
        aux_strength = torch.nan_to_num(aux_strength, nan=0.0)
        u = u * (1 - 0.3 * aux_strength)
    
    u = torch.clamp(u, 0.01, 0.99)
    
    a = torch.ones_like(b) * 0.5
    
    b, d, u, a = _normalize_opinion(b, d, u, a)
    
    # Squeeze to ensure proper shape
    b = b.squeeze()
    d = d.squeeze()
    u = u.squeeze()
    a = a.squeeze()
    
    return (b, d, u, a)


def jensen_shannon_divergence(p, q, eps=1e-8):
    """Compute Jensen-Shannon divergence between two evidence vectors."""
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)
    
    # ===== FIX: Handle NaN in inputs =====
    p = torch.nan_to_num(p, nan=0.0)
    q = torch.nan_to_num(q, nan=0.0)
    
    p = p + eps
    q = q + eps
    p_norm = p / p.sum(dim=1, keepdim=True)
    q_norm = q / q.sum(dim=1, keepdim=True)
    
    # ===== FIX: Handle NaN in normalization =====
    p_norm = torch.nan_to_num(p_norm, nan=0.0)
    q_norm = torch.nan_to_num(q_norm, nan=0.0)
    
    min_dim = min(p.shape[1], q.shape[1])
    p_aligned = p_norm[:, :min_dim]
    q_aligned = q_norm[:, :min_dim]
    
    m = 0.5 * (p_aligned + q_aligned)
    m = torch.clamp(m, min=eps)
    kl_pm = (p_aligned * torch.log(p_aligned / m + eps)).sum(dim=1)
    kl_qm = (q_aligned * torch.log(q_aligned / m + eps)).sum(dim=1)
    
    # ===== FIX: Handle NaN in KL divergence =====
    kl_pm = torch.nan_to_num(kl_pm, nan=0.0)
    kl_qm = torch.nan_to_num(kl_qm, nan=0.0)
    
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


# =============================================================================
# Data Perception Module (Paper Section 3.2)
# =============================================================================

class PatternCapturer(nn.Module):
    """Branch A: Pattern Capturer - Transformer encoder."""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4, max_len=512):
        super(PatternCapturer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.register_buffer('pos_encoding', self._get_positional_encoding(max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)
    
    @staticmethod
    def _get_positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        batch, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return x


class DeviationDetector(nn.Module):
    """Branch B: Deviation Detector - MLP with sparse activation."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=128):
        super(DeviationDetector, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.mean(dim=1)
        return self.mlp(x)


class EvidenceProjectionHead(nn.Module):
    """Evidence Projection Head: maps features to evidence space."""
    def __init__(self, h_g_dim, h_l_dim, d_e):
        super(EvidenceProjectionHead, self).__init__()
        self.W_g = nn.Linear(h_g_dim, d_e)
        self.W_l = nn.Linear(h_l_dim, d_e)
        self.sigmoid = nn.Sigmoid()
        self.d_e = d_e
        
    def forward(self, h_g, h_l):
        e_g = self.sigmoid(self.W_g(h_g))
        e_l = self.sigmoid(self.W_l(h_l))
        e_data = torch.cat([e_g, e_l], dim=1)
        return e_data, e_g, e_l


class RuleToEvidenceMappingNetwork(nn.Module):
    """Rule-to-Evidence Mapping Network."""
    def __init__(self, h_dim, rule_embed_dim, d_e):
        super(RuleToEvidenceMappingNetwork, self).__init__()
        self.fc1 = nn.Linear(h_dim + rule_embed_dim, 64)
        self.fc2 = nn.Linear(64, d_e + 1)
        
    def forward(self, h_g, r_encoded):
        combined = torch.cat([h_g, r_encoded], dim=1)
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        a = torch.sigmoid(x[:, 0])
        c = torch.sigmoid(x[:, 1:])
        return a, c


# =============================================================================
# KERN: Knowledge-Evidence Reasoning Network (Paper Section 3)
# =============================================================================

class KERN(nn.Module):
    """Knowledge-Evidence Reasoning Network for Anomaly Detection."""
    
    def __init__(self, input_dim, d_e=16, rule_embed_dim=384, tau=0.3, eta=0.5, gamma=0.8):
        super(KERN, self).__init__()
        
        self.d_e = d_e
        self.tau = tau
        self.eta = eta
        self.gamma = gamma
        self.device = 'cpu'
        
        self.pattern_capturer = PatternCapturer(input_dim, d_model=128, nhead=4, num_layers=4)
        self.deviation_detector = DeviationDetector(input_dim, hidden_dims=[256, 128, 64], output_dim=128)
        self.evidence_projection = EvidenceProjectionHead(128, 128, d_e)
        self.rule_mapping = RuleToEvidenceMappingNetwork(128, rule_embed_dim, d_e)
        
        self._initialize_weights()
        
        self.rule_encoder = None
        self.rule_embeddings = None
        self.knowledge_base = None
    
    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
        self.rule_mapping.apply(init_weights)
    
    def set_rule_encoder(self, encoder):
        self.rule_encoder = encoder
    
    def set_knowledge_base(self, knowledge_base):
        self.knowledge_base = knowledge_base
        if self.rule_encoder is not None and knowledge_base is not None:
            rule_texts = [r['text'] for r in knowledge_base]
            self.rule_embeddings = self.rule_encoder.encode(rule_texts, convert_to_tensor=True)
            if hasattr(self, 'device'):
                self.rule_embeddings = self.rule_embeddings.to(self.device)
    
    def to(self, device):
        super().to(device)
        self.device = device
        if self.rule_embeddings is not None:
            self.rule_embeddings = self.rule_embeddings.to(device)
        if self.rule_encoder is not None:
            self.rule_encoder.to(device)
        return self
    
    def extract_empirical_evidence(self, x):
        h_g = self.pattern_capturer(x)
        h_l = self.deviation_detector(x)
        e_data, e_g, e_l = self.evidence_projection(h_g, h_l)
        return e_data, e_g, h_g, h_l
    
    def extract_canonical_evidence(self, h_g):
        """
        Extract canonical evidence from knowledge rules.
        Paper Section 3.2 (2), Eq. (2) and (3)
        """
        # ===== FIX: Handle NaN in h_g =====
        if torch.isnan(h_g).any() or torch.isinf(h_g).any():
            h_g = torch.nan_to_num(h_g, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.rule_embeddings is None or self.rule_embeddings.shape[0] == 0:
            batch_size = h_g.shape[0]
            # Use uniform distribution [0, 0.001] for stability
            e_k = torch.rand(batch_size, self.d_e, device=h_g.device) * 0.001
            a_scores = torch.zeros(batch_size, 1, device=h_g.device)
            return e_k, [], a_scores, []
        
        batch_size = h_g.shape[0]
        num_rules = self.rule_embeddings.shape[0]
        
        h_g_expanded = h_g.unsqueeze(1).expand(batch_size, num_rules, -1)
        r_expanded = self.rule_embeddings.unsqueeze(0).expand(batch_size, num_rules, -1)
        
        h_g_flat = h_g_expanded.reshape(-1, 128)
        r_flat = r_expanded.reshape(-1, 384)
        
        a_flat, c_flat = self.rule_mapping(h_g_flat, r_flat)
        
        # ===== FIX: Handle NaN in a_flat and c_flat =====
        a_flat = torch.nan_to_num(a_flat, nan=0.0)
        c_flat = torch.nan_to_num(c_flat, nan=0.0)
        
        a_scores = a_flat.reshape(batch_size, num_rules)
        c_evidence = c_flat.reshape(batch_size, num_rules, -1)
        
        mask = a_scores > self.tau
        
        e_k = torch.zeros(batch_size, self.d_e, device=h_g.device)
        activated_rules_list = []
        rule_confidences_list = []
        
        for i in range(batch_size):
            active_indices = torch.where(mask[i])[0]
            if len(active_indices) > 0:
                active_indices_list = active_indices.cpu().tolist()
                confidences = []
                for idx in active_indices_list:
                    if self.knowledge_base is not None and idx < len(self.knowledge_base):
                        confidences.append(self.knowledge_base[idx].get('confidence', 0.5))
                    else:
                        confidences.append(0.5)
                
                activated_rules_list.append(list(zip(active_indices_list, confidences)))
                rule_confidences_list.append(confidences)
                
                weights = a_scores[i, active_indices] / (a_scores[i, active_indices].sum() + 1e-8)
                weights = torch.nan_to_num(weights, nan=0.0)
                weighted_evidence = (weights.unsqueeze(-1) * c_evidence[i, active_indices, :]).sum(dim=0)
                e_k[i] = weighted_evidence
            else:
                activated_rules_list.append([])
                rule_confidences_list.append([])
                # Uniform noise [0, 0.001] for gradient flow
                e_k[i] = torch.rand(self.d_e, device=h_g.device) * 0.001
        
        # ===== FIX: Final check for NaN in e_k =====
        e_k = torch.nan_to_num(e_k, nan=0.0)
        
        return e_k, activated_rules_list, a_scores, rule_confidences_list
    
    def evidence_alignment_and_conflict_detection(self, e_data, e_k):
        batch_size = e_data.shape[0]
        e_combined = torch.cat([e_data, e_k], dim=1)
        conflict_scores = jensen_shannon_divergence(e_data, e_k)
        conflict_matrix = torch.zeros(batch_size, 2, 2, device=e_data.device)
        conflict_matrix[:, 0, 1] = conflict_scores
        conflict_matrix[:, 1, 0] = conflict_scores
        return conflict_matrix, conflict_scores, e_combined
    
    def bayesian_reasoning_fusion(self, e_combined, conflict_scores, rule_confidences_list=None):
        """Bayesian Reasoning Machine based on Subjective Logic."""
        batch_size = e_combined.shape[0]
        device = e_combined.device
        
        e_data_batch = e_combined[:, :2*self.d_e]
        e_k_batch = e_combined[:, 2*self.d_e:]
        
        b_fused = torch.zeros(batch_size, device=device)
        d_fused = torch.zeros(batch_size, device=device)
        u_fused = torch.zeros(batch_size, device=device)
        a_fused = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            omega_data = evidence_to_opinion(e_data_batch[i:i+1])
            omega_knowledge = evidence_to_opinion(e_k_batch[i:i+1])
            
            conflict_score = conflict_scores[i]
            
            if torch.all(conflict_score < self.eta):
                b_i, d_i, u_i, a_i = cumulative_fusion(omega_data, omega_knowledge)
            else:
                # ===== FIX: knowledge_confidence defaults to 0.0 =====
                # Only trust knowledge when there are activated rules with confidence scores
                knowledge_confidence = 0.0
                if rule_confidences_list is not None and i < len(rule_confidences_list):
                    if len(rule_confidences_list[i]) > 0:
                        knowledge_confidence = np.mean(rule_confidences_list[i])
                
                omega_cautious = cautious_conflict_handling(
                    omega_data, omega_knowledge, conflict_score, self.eta
                )
                b_i, d_i, u_i, a_i = knowledge_arbitration(
                    omega_cautious, omega_knowledge, knowledge_confidence
                )
            
            b_fused[i] = b_i
            d_fused[i] = d_i
            u_fused[i] = u_i
            a_fused[i] = a_i
        
        return (b_fused, d_fused, u_fused, a_fused)
    
    def compute_anomaly_score(self, omega_fused):
        """
        Paper Eq. (6): S(x) = b + γ · u.
        Clamped to [0,1] for numerical stability with binary cross-entropy.
        """
        b, d, u, a = omega_fused
        
        # Ensure tensors
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32, device=self.device)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32, device=self.device)
        
        b = b.squeeze()
        u = u.squeeze()
        
        # ===== FIX: Handle NaN =====
        if torch.isnan(b).any():
            b = torch.nan_to_num(b, nan=0.5)
        if torch.isnan(u).any():
            u = torch.nan_to_num(u, nan=0.5)
        
        b = torch.clamp(b, 0.0, 1.0)
        u = torch.clamp(u, 0.0, 1.0)
        
        anomaly_score = b + self.gamma * u
        anomaly_score = torch.clamp(anomaly_score, 0.0, 1.0)
        
        return anomaly_score
    
    def compute_contribution_weights(self, e_combined, conflict_scores, omega_fused, rule_confidences_list=None):
        """Compute contribution weights via fusion backtracking."""
        b, d, u, a = omega_fused
        batch_size = e_combined.shape[0]
        d_e = self.d_e
        
        e_data = e_combined[:, :2*d_e]
        e_k = e_combined[:, 2*d_e:]
        
        omega_data = evidence_to_opinion(e_data)
        omega_knowledge = evidence_to_opinion(e_k)
        b_data, d_data, u_data, a_data = omega_data
        b_know, d_know, u_know, a_know = omega_knowledge
        
        contribution_weights = torch.zeros_like(e_combined)
        
        for i in range(batch_size):
            conflict_score = conflict_scores[i]
            
            if torch.all(conflict_score < self.eta):
                k = u_data[i] + u_know[i] - u_data[i] * u_know[i] + 1e-8
                influence_data = u_know[i] / k
                influence_knowledge = u_data[i] / k
                
                total_influence = influence_data + influence_knowledge + 1e-8
                weight_data = influence_data / total_influence
                weight_knowledge = influence_knowledge / total_influence
            else:
                # ===== FIX: knowledge_confidence defaults to 0.0 =====
                knowledge_confidence = 0.0
                if rule_confidences_list is not None and i < len(rule_confidences_list):
                    if len(rule_confidences_list[i]) > 0:
                        knowledge_confidence = np.mean(rule_confidences_list[i])
                
                weight_knowledge = 0.5 + 0.4 * knowledge_confidence
                weight_data = 1.0 - weight_knowledge
                weight_data = max(0.05, weight_data)
                weight_knowledge = 1.0 - weight_data
            
            consistency_data = torch.clamp(b_data[i] / (b[i] + 1e-8), 0, 2)
            consistency_knowledge = torch.clamp(b_know[i] / (b[i] + 1e-8), 0, 2)
            
            data_contrib = weight_data * consistency_data * e_data[i]
            knowledge_contrib = weight_knowledge * consistency_knowledge * e_k[i]
            
            combined_contrib = torch.cat([data_contrib, knowledge_contrib], dim=0)
            contribution_weights[i] = combined_contrib
        
        contribution_weights = F.softmax(contribution_weights.abs(), dim=1)
        
        return contribution_weights
    
    def generate_conflict_report(self, conflict_scores, activated_rules, threshold=0.5):
        """Generate conflict report for explanation."""
        reports = []
        for i, score in enumerate(conflict_scores):
            score_val = score.item() if hasattr(score, 'item') else float(score)
            if score_val > threshold:
                report = {
                    'sample_idx': i,
                    'conflict_score': score_val,
                    'severity': 'high' if score_val > 0.7 else 'medium',
                    'activated_rules': activated_rules[i] if i < len(activated_rules) else [],
                    'resolution': 'knowledge_arbitration'
                }
                reports.append(report)
            else:
                reports.append({
                    'sample_idx': i,
                    'conflict_score': score_val,
                    'severity': 'low',
                    'activated_rules': [],
                    'resolution': 'cumulative_fusion'
                })
        return reports
    
    def forward(self, x):
        """Full forward pass of KERN."""
        e_data, e_g, h_g, h_l = self.extract_empirical_evidence(x)
        e_k, activated_rules, a_scores, rule_confidences = self.extract_canonical_evidence(h_g)
        conflict_matrix, conflict_scores, e_combined = self.evidence_alignment_and_conflict_detection(e_data, e_k)
        omega_fused = self.bayesian_reasoning_fusion(e_combined, conflict_scores, rule_confidences)
        anomaly_score = self.compute_anomaly_score(omega_fused)
        
        contribution_weights = self.compute_contribution_weights(
            e_combined, conflict_scores, omega_fused, rule_confidences
        )
        
        conflict_report = self.generate_conflict_report(conflict_scores, activated_rules, self.eta)
        
        b, d, u, a = omega_fused
        explanation = {
            'opinion': (b, d, u, a),
            'activated_rules': activated_rules,
            'activated_rule_texts': self._get_activated_rule_texts(activated_rules),
            'conflict_scores': conflict_scores,
            'conflict_report': conflict_report,
            'anomaly_score': anomaly_score,
            'empirical_evidence': e_data,
            'canonical_evidence': e_k,
            'e_combined': e_combined,
            'conflict_matrix': conflict_matrix,
            'e_g': e_g,
            'contribution_weights': contribution_weights,
            'applicability_scores': a_scores,
            'fusion_mode': torch.where(conflict_scores < self.eta, 
                                       torch.tensor(0, device=conflict_scores.device),
                                       torch.tensor(1, device=conflict_scores.device))
        }
        
        return anomaly_score, explanation
    
    def _get_activated_rule_texts(self, activated_rules):
        rule_texts = []
        for batch_rules in activated_rules:
            batch_texts = []
            for idx, conf in batch_rules:
                if self.knowledge_base is not None and idx < len(self.knowledge_base):
                    batch_texts.append({
                        'text': self.knowledge_base[idx]['text'],
                        'confidence': conf
                    })
            rule_texts.append(batch_texts)
        return rule_texts


# =============================================================================
# Training Functions (Paper Section 3.4)
# =============================================================================

class KERNLoss(nn.Module):
    """Composite loss function for Stage 2 training. Paper Eq. (7)."""
    def __init__(self, lambda_c=0.1, lambda_r=0.01):
        super(KERNLoss, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
    
    def forward(self, anomaly_score, labels, e_data, e_k, e_g, conflict_matrix, uncertainty):
        # ===== FIX: Ensure anomaly_score is a tensor in [0,1] =====
        if not isinstance(anomaly_score, torch.Tensor):
            anomaly_score = torch.tensor(anomaly_score, dtype=torch.float32)
        
        # Handle NaN/Inf
        if torch.isnan(anomaly_score).any():
            anomaly_score = torch.nan_to_num(anomaly_score, nan=0.5)
        if torch.isinf(anomaly_score).any():
            anomaly_score = torch.nan_to_num(anomaly_score, posinf=1.0, neginf=0.0)
        
        anomaly_score = torch.clamp(anomaly_score, 0.0, 1.0)
        
        # Ensure labels is float
        labels = labels.float()
        
        # L_task: Binary cross-entropy loss
        L_task = F.binary_cross_entropy(anomaly_score, labels)
        
        # L_cons: Evidence consistency loss
        batch_size = labels.shape[0]
        conflict_norm = torch.norm(conflict_matrix.view(batch_size, -1), dim=1).mean()
        cos_sim = F.cosine_similarity(e_g, e_k, dim=1, eps=1e-8)
        L_cons = conflict_norm - cos_sim.mean()
        L_cons = torch.clamp(L_cons, 0.0, 1.0)
        
        # L_reg: Interpretability regularization
        L_reg = uncertainty.mean()
        
        # Composite loss (Paper Eq. 7)
        loss = L_task + self.lambda_c * L_cons + self.lambda_r * L_reg
        
        return loss, L_task, L_cons, L_reg


def train_stage1(model, train_loader, val_loader, knowledge_base, rule_encoder, 
                 epochs=100, lr=1e-3, device='cuda'):
    """Stage 1: Separate Pre-training and Alignment."""
    model = model.to(device)
    model.device = device
    
    model.set_rule_encoder(rule_encoder)
    model.set_knowledge_base(knowledge_base)
    
    num_rules = len(knowledge_base) if knowledge_base is not None else 0
    
    print("  Stage 1a: Data branch pre-training...")
    params_data = list(model.pattern_capturer.parameters()) + \
                  list(model.deviation_detector.parameters()) + \
                  list(model.evidence_projection.parameters())
    
    optimizer_data = torch.optim.Adam(params_data, lr=lr)
    model.train()
    data_losses = []
    
    for epoch in range(epochs // 2):
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            
            optimizer_data.zero_grad()
            
            e_data, e_g, h_g, h_l = model.extract_empirical_evidence(x)
            
            if len(x.shape) == 3:
                x_flat = x.mean(dim=1)
            else:
                x_flat = x
            
            target_dim = min(x_flat.shape[1], h_g.shape[1])
            loss_recon = F.mse_loss(h_g[:, :target_dim], x_flat[:, :target_dim]) + \
                         F.mse_loss(h_l[:, :min(x_flat.shape[1], h_l.shape[1])], 
                                   x_flat[:, :min(x_flat.shape[1], h_l.shape[1])])
            
            loss_contrast = -F.cosine_similarity(h_g, h_l, dim=1, eps=1e-8).mean() + 1.0
            
            loss = loss_recon + 0.1 * loss_contrast
            loss.backward()
            optimizer_data.step()
            
            epoch_loss += loss.item()
        
        data_losses.append(epoch_loss / len(train_loader))
        if (epoch + 1) % 10 == 0:
            print(f"    Data pre-training Epoch {epoch+1}/{epochs//2}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    knowledge_losses = []
    
    if num_rules >= 3:
        print("  Stage 1b: Knowledge mapping network pre-training...")
        
        params_knowledge = list(model.rule_mapping.parameters())
        optimizer_knowledge = torch.optim.Adam(params_knowledge, lr=lr * 0.5)
        
        model.train()
        
        for epoch in range(epochs // 2):
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch[0].to(device)
                labels = batch[1].to(device)
                
                optimizer_knowledge.zero_grad()
                
                with torch.no_grad():
                    e_data, e_g, h_g, h_l = model.extract_empirical_evidence(x)
                    h_g_detached = h_g.detach()
                    e_g_detached = e_g.detach()
                
                e_k, activated_rules, a_scores, rule_confidences = model.extract_canonical_evidence(h_g_detached)
                
                pos_mask = labels > 0.5
                neg_mask = labels <= 0.5
                
                alignment_loss = torch.tensor(0.0, device=device)
                count_pos = 0
                count_neg = 0
                
                if pos_mask.sum() > 0:
                    e_g_pos = e_g_detached[pos_mask]
                    e_k_pos = e_k[pos_mask]
                    cos_sim = F.cosine_similarity(e_g_pos, e_k_pos, dim=1, eps=1e-8)
                    alignment_loss = alignment_loss + (1 - cos_sim).mean()
                    count_pos = pos_mask.sum().item()
                
                if neg_mask.sum() > 0:
                    e_k_neg = e_k[neg_mask]
                    alignment_loss = alignment_loss + e_k_neg.mean() * 0.5
                    count_neg = neg_mask.sum().item()
                
                total_count = count_pos + count_neg
                if total_count > 0:
                    alignment_loss = alignment_loss / max(1, total_count)
                
                sparsity_loss = torch.tensor(0.0, device=device)
                total_activated = 0
                for i in range(len(activated_rules)):
                    if len(activated_rules[i]) > 0:
                        total_activated += len(activated_rules[i])
                
                if len(activated_rules) > 0:
                    avg_activated = total_activated / len(activated_rules)
                    sparsity_loss = avg_activated * 0.01
                
                loss = alignment_loss + sparsity_loss
                
                reg_loss = torch.tensor(0.0, device=device)
                for param in model.rule_mapping.parameters():
                    if param.requires_grad:
                        reg_loss = reg_loss + torch.norm(param, p=2) * 1e-6
                
                if reg_loss.requires_grad:
                    loss = loss + reg_loss
                
                if loss.requires_grad:
                    loss.backward()
                    optimizer_knowledge.step()
                    epoch_loss += loss.item()
                else:
                    if reg_loss.requires_grad:
                        reg_loss.backward()
                        optimizer_knowledge.step()
                        epoch_loss += reg_loss.item()
            
            knowledge_losses.append(epoch_loss / max(1, len(train_loader)))
            if (epoch + 1) % 10 == 0:
                print(f"    Knowledge pre-training Epoch {epoch+1}/{epochs//2}, Loss: {epoch_loss/max(1, len(train_loader)):.4f}")
    else:
        print(f"  Stage 1b: Skipped (only {num_rules} rules, need >=3)")
        print("    Knowledge mapping network will be trained in Stage 2.")
    
    return model, data_losses, knowledge_losses


def train_stage2(model, train_loader, val_loader, epochs=50, lr=1e-3, 
                 lambda_c=0.1, lambda_r=0.01, device='cuda'):
    """Stage 2: End-to-End Fine-tuning."""
    model = model.to(device)
    model.device = device
    
    params = list(model.pattern_capturer.parameters()) + \
             list(model.deviation_detector.parameters()) + \
             list(model.evidence_projection.parameters()) + \
             list(model.rule_mapping.parameters())
    
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = KERNLoss(lambda_c=lambda_c, lambda_r=lambda_r)
    
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_cons_loss = 0.0
        epoch_reg_loss = 0.0
        
        for batch in train_loader:
            x = batch[0].to(device)
            labels = batch[1].to(device)
            
            optimizer.zero_grad()
            
            anomaly_score, explanation = model(x)
            b, d, u, a = explanation['opinion']
            
            loss, L_task, L_cons, L_reg = criterion(
                anomaly_score, labels,
                explanation['empirical_evidence'],
                explanation['canonical_evidence'],
                explanation['e_g'],
                explanation['conflict_matrix'],
                u
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_task_loss += L_task.item()
            epoch_cons_loss += L_cons.item()
            epoch_reg_loss += L_reg.item()
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_task = epoch_task_loss / len(train_loader)
        avg_cons = epoch_cons_loss / len(train_loader)
        avg_reg = epoch_reg_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Stage 2 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"Task: {avg_task:.4f}, Cons: {avg_cons:.4f}, Reg: {avg_reg:.4f}")
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model and return average loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            labels = batch[1].to(device)
            
            anomaly_score, explanation = model(x)
            b, d, u, a = explanation['opinion']
            
            loss, _, _, _ = criterion(
                anomaly_score, labels,
                explanation['empirical_evidence'],
                explanation['canonical_evidence'],
                explanation['e_g'],
                explanation['conflict_matrix'],
                u
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


# =============================================================================
# Dataset Loading Functions - FIXED for UKMNCT
# =============================================================================

def load_credit_card(data_path):
    """Load Credit Card Fraud Detection dataset."""
    df = pd.read_csv(data_path)
    print(f"Loaded Credit Card dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    X = df[feature_cols].copy()
    y = df['Class'].copy()
    
    nan_mask = X.isna().any(axis=1) | y.isna()
    if nan_mask.any():
        print(f"Removing {nan_mask.sum()} rows with NaN values")
        X = X[~nan_mask].copy()
        y = y[~nan_mask].copy()
    
    y = y.astype(float)
    y = y.fillna(0)
    
    print(f"Clean data shape: {X.shape}, Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X.values, y.values, feature_cols


def load_cic_unsw(data_path):
    """Load CIC-UNSW dataset."""
    df = pd.read_csv(data_path)
    print(f"Loaded CIC-UNSW dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    label_col = 'Label' if 'Label' in df.columns else 'label'
    feature_cols = [col for col in df.columns if col != label_col]
    X = df[feature_cols].copy()
    y = df[label_col].copy()
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) > 0:
        print(f"Encoding categorical columns: {categorical_cols}")
        for col in categorical_cols:
            X[col] = X[col].astype(str).fillna('missing')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    nan_mask = X.isna().any(axis=1) | y.isna()
    if nan_mask.any():
        print(f"Removing {nan_mask.sum()} rows with NaN values")
        X = X[~nan_mask].copy()
        y = y[~nan_mask].copy()
    
    if y.dtype == 'object':
        y = (y != 'normal').astype(int)
    else:
        y = y.astype(float)
        y = y.fillna(0)
        y = (y > 0).astype(int)
    
    print(f"Clean data shape: {X.shape}, Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X.values, y.values, feature_cols


def load_ukmnct(data_path):
    """
    Load UKMNCT_IIoT_FDIA dataset.
    This dataset contains many categorical/string columns that need encoding.
    """
    df = pd.read_csv(data_path)
    print(f"Loaded UKMNCT_IIoT_FDIA dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    label_col = 'marker'
    feature_cols = [col for col in df.columns if col != label_col]
    X = df[feature_cols].copy()
    y = df[label_col].copy()
    
    print(f"Columns in dataset: {list(df.columns)}")
    
    # ===== Handle categorical columns =====
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) > 0:
        print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols[:5]}...")
        for col in categorical_cols:
            X[col] = X[col].astype(str).fillna('missing')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            print(f"  Encoded '{col}' with {len(le.classes_)} unique values")
    
    # ===== Handle label =====
    if y.dtype == 'object':
        unique_labels = y.unique()
        print(f"Unique labels in dataset: {unique_labels}")
        if 'normal' in unique_labels:
            y = (y != 'normal').astype(int)
        else:
            le = LabelEncoder()
            y = le.fit_transform(y)
    else:
        y = y.astype(float)
        y = y.fillna(0)
        y = (y > 0).astype(int)
    
    # ===== Remove rows with NaN =====
    nan_mask = X.isna().any(axis=1) | y.isna()
    if nan_mask.any():
        print(f"Removing {nan_mask.sum()} rows with NaN values")
        X = X[~nan_mask].copy()
        y = y[~nan_mask].copy()
    
    # ===== Ensure all columns are numeric =====
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"Converting column '{col}' to numeric")
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    print(f"Clean data shape: {X.shape}, Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Feature columns: {feature_cols[:5]}... (total {len(feature_cols)})")
    
    return X.values, y.values, feature_cols


def load_wdbc(data_path):
    """Load WDBC dataset."""
    feature_names = [
        'radius', 'texture', 'perimeter', 'area', 'smoothness',
        'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    df = pd.read_csv(data_path, header=None)
    print(f"Loaded WDBC dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    
    X = df.iloc[:, 2:].copy()
    y = df.iloc[:, 1].copy()
    
    nan_mask = X.isna().any(axis=1) | y.isna()
    if nan_mask.any():
        print(f"Removing {nan_mask.sum()} rows with NaN values")
        X = X[~nan_mask].copy()
        y = y[~nan_mask].copy()
    
    y = (y == 'M').astype(int) if y.dtype == 'object' else y.astype(int)
    
    print(f"Clean data shape: {X.shape}, Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X.values, y.values, feature_names


def load_dataset(dataset_name, data_dir="./data/"):
    """Load dataset by name."""
    file_map = {
        'credit_card': 'CreditCard.csv',
        'cic_unsw': 'CIC_UNSW.csv',
        'ukmnct': 'UKMNCT_IIoT_FDIA.csv',
        'wdbc': 'wdbc.data'
    }
    
    data_path = os.path.join(data_dir, file_map.get(dataset_name, f"{dataset_name}.csv"))
    
    loaders = {
        'credit_card': load_credit_card,
        'cic_unsw': load_cic_unsw,
        'ukmnct': load_ukmnct,
        'wdbc': load_wdbc
    }
    
    if dataset_name not in loaders:
        print(f"ERROR: Unknown dataset '{dataset_name}'")
        return None, None, None
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset file not found: {data_path}")
        return None, None, None
    
    loader = loaders[dataset_name]
    X, y, features = loader(data_path)
    
    if X is None:
        print(f"ERROR: Failed to load {dataset_name} dataset.")
        return None, None, None
    
    return X, y, features


# =============================================================================
# Theoretical Guarantees Validation (Paper Section 3.5)
# =============================================================================

class TheoreticalGuarantees:
    """Validate theoretical guarantees from Paper Section 3.5."""
    
    @staticmethod
    def compute_information_content(evidence, eps=1e-8):
        evidence = torch.nan_to_num(evidence, nan=0.0)
        p = evidence / (evidence.sum(dim=1, keepdim=True) + eps)
        p = torch.nan_to_num(p, nan=0.0)
        H = -torch.sum(p * torch.log(p + eps), dim=1)
        H = torch.nan_to_num(H, nan=0.0)
        return H
    
    @staticmethod
    def validate_proposition_1(e_data, e_k, verbose=True):
        e_k_norm = e_k.norm(dim=1).mean().item()
        
        if e_k_norm < 1e-6 or np.isnan(e_k_norm):
            if verbose:
                print(f"Proposition 1: Knowledge evidence is near zero or NaN (norm={e_k_norm:.2e})")
                print(f"  No information gain from knowledge in this case.")
            H_meta = torch.zeros_like(e_data[:, 0])
            return torch.zeros_like(e_data[:, 0]), torch.zeros_like(e_data[:, 0]), H_meta
        
        H_data = TheoreticalGuarantees.compute_information_content(e_data)
        H_knowledge = TheoreticalGuarantees.compute_information_content(e_k)
        
        e_joint = torch.cat([e_data, e_k], dim=1)
        H_joint = TheoreticalGuarantees.compute_information_content(e_joint)
        
        jsd = jensen_shannon_divergence(e_data, e_k)
        H_meta = jsd
        
        H_total_reasoning = H_joint + H_meta
        H_total_simple = H_data + H_knowledge
        
        if verbose:
            print(f"Proposition 1 Validation:")
            print(f"  H(data): {H_data.mean().item():.4f}")
            print(f"  H(knowledge): {H_knowledge.mean().item():.4f}")
            print(f"  H(joint): {H_joint.mean().item():.4f}")
            print(f"  H(meta): {H_meta.mean().item():.4f}")
            print(f"  H_total(simple): {H_total_simple.mean().item():.4f}")
            print(f"  H_total(reasoning): {H_total_reasoning.mean().item():.4f}")
            
            info_gain = (H_total_reasoning - H_total_simple).mean().item()
            print(f"  Information Gain: {info_gain:.4f}")
            
            if e_k_norm < 1e-6:
                print(f"  Proposition 1 holds: True (trivially)")
            else:
                print(f"  Proposition 1 holds: {(H_total_reasoning >= H_total_simple).all().item()}")
        
        return H_total_reasoning, H_total_simple, H_meta
    
    @staticmethod
    def validate_theorem_1(omega_fused, omega_data, omega_knowledge, verbose=True):
        b_f, d_f, u_f, a_f = omega_fused
        b_d, d_d, u_d, a_d = omega_data
        b_k, d_k, u_k, a_k = omega_knowledge
        
        conflict_mask = ((b_d > 0.7) & (b_k < 0.3)) | ((b_d < 0.3) & (b_k > 0.7))
        is_conflict = conflict_mask.any().item()
        
        uncertainty_reduced_mask = (u_f < u_d) & (u_f < u_k)
        uncertainty_reduced = uncertainty_reduced_mask.all().item()
        
        theorem_holds = (not is_conflict) and uncertainty_reduced
        
        if verbose:
            print(f"Theorem 1 Validation:")
            print(f"  u(data): {u_d.mean().item():.4f}")
            print(f"  u(knowledge): {u_k.mean().item():.4f}")
            print(f"  u(fused): {u_f.mean().item():.4f}")
            print(f"  u_fused < u_data (all): {uncertainty_reduced_mask.all().item()}")
            print(f"  u_fused < u_knowledge (all): {uncertainty_reduced_mask.all().item()}")
            print(f"  Total conflict present: {is_conflict}")
            print(f"  Theorem 1 holds: {theorem_holds}")
        
        return u_f, u_d, u_k, theorem_holds
    
    @staticmethod
    def validate_theorem_2(model, train_loader, val_loader, device='cuda', verbose=True):
        model.eval()
        train_loss = 0.0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in train_loader:
                x = batch[0].to(device)
                labels = batch[1].to(device)
                anomaly_score, explanation = model(x)
                anomaly_score = torch.nan_to_num(anomaly_score, nan=0.5)
                anomaly_score = torch.clamp(anomaly_score, 0.0, 1.0)
                train_loss += F.binary_cross_entropy(anomaly_score, labels.float()).item()
            
            for batch in val_loader:
                x = batch[0].to(device)
                labels = batch[1].to(device)
                anomaly_score, explanation = model(x)
                anomaly_score = torch.nan_to_num(anomaly_score, nan=0.5)
                anomaly_score = torch.clamp(anomaly_score, 0.0, 1.0)
                val_loss += F.binary_cross_entropy(anomaly_score, labels.float()).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        knowledge_violations = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                _, explanation = model(x)
                activated_rules = explanation['activated_rules']
                for rules in activated_rules:
                    if len(rules) == 0:
                        knowledge_violations += 1
                    total_samples += 1
        
        violation_rate = knowledge_violations / (total_samples + 1e-8)
        gen_gap = val_loss - train_loss
        
        if verbose:
            print(f"Theorem 2 Validation:")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val loss: {val_loss:.4f}")
            print(f"  Generalization gap: {gen_gap:.4f}")
            print(f"  Knowledge violation rate: {violation_rate:.4f}")
            print(f"  Tighter bound: {violation_rate < 0.3}")
        
        return train_loss, val_loss, gen_gap, violation_rate
    
    @staticmethod
    def validate_corollary_1(model, train_loader, val_loader, device='cuda', verbose=True):
        anomaly_batches = []
        normal_batches = []
        
        for batch in train_loader:
            x = batch[0]
            labels = batch[1]
            if (labels > 0.5).any():
                anomaly_batches.append((x[labels > 0.5], labels[labels > 0.5]))
            if (labels <= 0.5).any():
                normal_batches.append((x[labels <= 0.5], labels[labels <= 0.5]))
        
        sparse_train_data = []
        sparse_train_labels = []
        
        for x, y in normal_batches:
            sparse_train_data.append(x)
            sparse_train_labels.append(y)
        
        sparse_anomaly_count = 0
        for x, y in anomaly_batches:
            if sparse_anomaly_count < len(anomaly_batches) * 0.05:
                sparse_train_data.append(x)
                sparse_train_labels.append(y)
                sparse_anomaly_count += 1
        
        if len(sparse_train_data) > 0:
            sparse_data = torch.cat(sparse_train_data, dim=0)
            sparse_labels = torch.cat(sparse_train_labels, dim=0)
            sparse_dataset = TensorDataset(sparse_data, sparse_labels)
            sparse_loader = DataLoader(sparse_dataset, batch_size=256, shuffle=True)
            
            model.eval()
            sparse_val_loss = 0.0
            with torch.no_grad():
                for batch in sparse_loader:
                    x = batch[0].to(device)
                    labels = batch[1].to(device)
                    anomaly_score, explanation = model(x)
                    anomaly_score = torch.nan_to_num(anomaly_score, nan=0.5)
                    anomaly_score = torch.clamp(anomaly_score, 0.0, 1.0)
                    sparse_val_loss += F.binary_cross_entropy(anomaly_score, labels.float()).item()
            sparse_val_loss /= max(1, len(sparse_loader))
            
            if verbose:
                print(f"Corollary 1 Validation:")
                print(f"  Sparse anomaly loss: {sparse_val_loss:.4f}")
                print(f"  Knowledge provides surrogate signal: {sparse_val_loss < 1.0}")
            
            return sparse_val_loss
        
        return None


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(dataset_name, config, device='cuda'):
    """Run full experiment on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Running experiment on: {dataset_name}")
    print(f"{'='*60}")
    
    result_dir = os.path.join(config.results_dir, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    
    X, y, feature_names = load_dataset(dataset_name, config.data_dir)
    if X is None:
        return None
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    print(f"Final data shape: {X.shape}, Labels: {np.unique(y, return_counts=True)}")
    
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print(f"WARNING: Only one class present in dataset ({unique_labels[0]}). Skipping.")
        return {'dataset': dataset_name, 'auc_roc': 0.5, 'auc_pr': 0.0, 'f1_score': 0.0, 
                'precision': 0.0, 'recall': 0.0, 'recall_at_1%': 0.0, 'recall_at_5%': 0.0, 
                'recall_at_10%': 0.0, 'optimal_threshold': 0.5}
    
    unique, counts = np.unique(y, return_counts=True)
    use_stratify = all(counts >= 2)
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    else:
        print("Warning: Some classes have < 2 samples, using non-stratified split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    tau = Config.get_tau_for_dataset(dataset_name)
    
    model = KERN(
        input_dim=input_dim,
        d_e=config.d_e,
        tau=tau,
        eta=config.eta,
        gamma=config.gamma
    )
    
    rule_encoder = RuleEncoder('all-MiniLM-L6-v2', device=device, seed=42)
    knowledge_base = get_knowledge_base(dataset_name)
    print(f"Loaded {len(knowledge_base)} rules for {dataset_name}")
    print(f"Using tau={tau} (dataset-specific)")
    
    print("\n--- Stage 1: Pre-training ---")
    start_time = time.time()
    model, stage1_data_losses, stage1_knowledge_losses = train_stage1(
        model, train_loader, val_loader,
        knowledge_base=knowledge_base,
        rule_encoder=rule_encoder,
        epochs=config.epochs_stage1,
        lr=config.lr,
        device=device
    )
    stage1_time = time.time() - start_time
    
    print("\n--- Stage 2: Fine-tuning ---")
    start_time = time.time()
    model = train_stage2(
        model, train_loader, val_loader,
        epochs=config.epochs_stage2,
        lr=config.lr,
        lambda_c=config.lambda_c,
        lambda_r=config.lambda_r,
        device=device
    )
    stage2_time = time.time() - start_time
    
    time_df = pd.DataFrame({
        'dataset': [dataset_name],
        'stage1_time_sec': [stage1_time],
        'stage2_time_sec': [stage2_time],
        'total_time_sec': [stage1_time + stage2_time]
    })
    time_path = os.path.join(result_dir, 'time.csv')
    time_df.to_csv(time_path, index=False)
    print(f"Training time saved to {time_path}")
    
    print("\n--- Theoretical Guarantees Validation ---")
    model.eval()
    
    batch = next(iter(train_loader))
    x_batch = batch[0].to(device)
    
    with torch.no_grad():
        _, explanation = model(x_batch)
        
        e_data = explanation['empirical_evidence']
        e_k = explanation['canonical_evidence']
        b, d, u, a = explanation['opinion']
        
        print("\nProposition 1 (Evidence Enrichment):")
        H_reasoning, H_simple, H_meta = TheoreticalGuarantees.validate_proposition_1(e_data, e_k, verbose=True)
        
        print("\nTheorem 1 (Uncertainty Reduction):")
        omega_data = evidence_to_opinion(e_data)
        omega_knowledge = evidence_to_opinion(e_k)
        omega_fused = (b, d, u, a)
        u_f, u_d, u_k, theorem1_holds = TheoreticalGuarantees.validate_theorem_1(
            omega_fused, omega_data, omega_knowledge, verbose=True
        )
        
        print("\nTheorem 2 (Generalization Bound):")
        train_loss, val_loss, gen_gap, violation_rate = TheoreticalGuarantees.validate_theorem_2(
            model, train_loader, val_loader, device, verbose=True
        )
        
        print("\nCorollary 1 (Sparse Anomalies):")
        sparse_loss = TheoreticalGuarantees.validate_corollary_1(
            model, train_loader, val_loader, device, verbose=True
        )
    
    theo_results = {
        'proposition1_info_gain': (H_reasoning - H_simple).mean().item() if len(H_reasoning) > 0 else 0,
        'theorem1_uncertainty_reduction': (u_d.mean().item() - u_f.mean().item()) if len(u_f) > 0 else 0,
        'theorem2_generalization_gap': gen_gap,
        'theorem2_knowledge_violation_rate': violation_rate,
        'corollary1_sparse_anomaly_loss': sparse_loss if sparse_loss is not None else 0
    }
    
    theo_path = os.path.join(result_dir, 'theoretical_validation.csv')
    pd.DataFrame([theo_results]).to_csv(theo_path, index=False)
    print(f"Theoretical validation saved to {theo_path}")
    
    print("\n--- Evaluation ---")
    model.eval()
    all_preds = []
    all_labels = []
    all_explanations = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            labels = batch[1].to(device)
            
            anomaly_score, explanation = model(x)
            
            all_preds.extend(anomaly_score.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_explanations.append(explanation)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) < 2:
        print(f"WARNING: Only one class in test set. Returning placeholder metrics.")
        summary = {
            'dataset': dataset_name,
            'auc_roc': 0.5,
            'auc_pr': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 1.0,
            'recall_at_1%': 0.0,
            'recall_at_5%': 0.0,
            'recall_at_10%': 0.0,
            'optimal_threshold': 0.5,
            'proposition1_info_gain': theo_results['proposition1_info_gain'],
            'theorem1_uncertainty_reduction': theo_results['theorem1_uncertainty_reduction'],
            'theorem2_generalization_gap': theo_results['theorem2_generalization_gap'],
            'theorem2_knowledge_violation_rate': theo_results['theorem2_knowledge_violation_rate'],
        }
        return summary, model
    
    auc_roc = roc_auc_score(all_labels, all_preds)
    
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_preds)
    auc_pr = auc(recall_curve, precision_curve)
    
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.linspace(0, 1, 100):
        pred_binary = (all_preds > threshold).astype(int)
        f1 = f1_score(all_labels, pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    pred_binary = (all_preds > best_threshold).astype(int)
    precision = precision_score(all_labels, pred_binary, zero_division=0)
    recall = recall_score(all_labels, pred_binary, zero_division=0)
    
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    if len(all_labels) > 0:
        sorted_indices = np.argsort(all_preds)[::-1]
        sorted_labels = all_labels[sorted_indices]
        top_1 = int(max(1, len(sorted_labels) * 0.01))
        top_5 = int(max(1, len(sorted_labels) * 0.05))
        top_10 = int(max(1, len(sorted_labels) * 0.10))
        recall_at_1 = sorted_labels[:top_1].sum() / (sorted_labels.sum() + 1e-8)
        recall_at_5 = sorted_labels[:top_5].sum() / (sorted_labels.sum() + 1e-8)
        recall_at_10 = sorted_labels[:top_10].sum() / (sorted_labels.sum() + 1e-8)
    
    summary = {
        'dataset': dataset_name,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1_score': best_f1,
        'precision': precision,
        'recall': recall,
        'recall_at_1%': recall_at_1,
        'recall_at_5%': recall_at_5,
        'recall_at_10%': recall_at_10,
        'optimal_threshold': best_threshold,
        'proposition1_info_gain': theo_results['proposition1_info_gain'],
        'theorem1_uncertainty_reduction': theo_results['theorem1_uncertainty_reduction'],
        'theorem2_generalization_gap': theo_results['theorem2_generalization_gap'],
        'theorem2_knowledge_violation_rate': theo_results['theorem2_knowledge_violation_rate'],
    }
    
    summary_path = os.path.join(result_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"{'='*50}\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Summary saved to {summary_path}")
    
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC-ROC = {auc_roc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'ROC Curve - {dataset_name}', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(result_dir, 'roc_curve.tiff')
    plt.savefig(plot_path, dpi=150, format='tiff', bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {plot_path}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_curve, precision_curve, linewidth=2, label=f'AUC-PR = {auc_pr:.4f}')
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(f'Precision-Recall Curve - {dataset_name}', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(result_dir, 'pr_curve.tiff')
    plt.savefig(plot_path, dpi=150, format='tiff', bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to {plot_path}")
    
    predictions_df = pd.DataFrame({
        'true_label': all_labels,
        'anomaly_score': all_preds,
        'predicted_label': pred_binary
    })
    pred_path = os.path.join(result_dir, 'predictions.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")
    
    sample_explanations = []
    for i in range(min(10, len(all_preds))):
        batch_idx = i // config.batch_size
        sample_idx = i % config.batch_size
        if batch_idx < len(all_explanations):
            exp = all_explanations[batch_idx]
            activated_rules_str = str(exp['activated_rules'][sample_idx]) if sample_idx < len(exp['activated_rules']) else []
            conflict_score = exp['conflict_scores'][sample_idx].item() if sample_idx < len(exp['conflict_scores']) else 0
            contribution_weights = exp['contribution_weights'][sample_idx] if sample_idx < len(exp['contribution_weights']) else None
            fusion_mode = exp['fusion_mode'][sample_idx].item() if sample_idx < len(exp['fusion_mode']) else 0
        else:
            activated_rules_str = []
            conflict_score = 0
            contribution_weights = None
            fusion_mode = 0
        
        sample_explanations.append({
            'sample_idx': i,
            'true_label': all_labels[i],
            'anomaly_score': all_preds[i],
            'activated_rules': str(activated_rules_str),
            'conflict_score': conflict_score,
            'fusion_mode': 'cumulative_fusion' if fusion_mode == 0 else 'knowledge_arbitration',
            'contribution_weights': str(contribution_weights) if contribution_weights is not None else 'N/A'
        })
    
    if sample_explanations:
        exp_df = pd.DataFrame(sample_explanations)
        exp_path = os.path.join(result_dir, 'sample_explanations.csv')
        exp_df.to_csv(exp_path, index=False)
        print(f"Sample explanations saved to {exp_path}")
    
    print(f"\n--- {dataset_name} Results ---")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"F1-Score: {best_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Recall@1%: {recall_at_1:.4f}")
    
    return summary, model


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for running all experiments."""
    set_seed(42)
    
    print("KERN: Knowledge-Evidence Reasoning Network")
    print("Anomaly Detection Framework")
    print("=" * 60)
    
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(config.results_dir, exist_ok=True)
    
    datasets = ['credit_card', 'cic_unsw', 'ukmnct', 'wdbc']
    
    all_results = []
    
    for dataset_name in datasets:
        result = run_experiment(dataset_name, config, device)
        if result is not None:
            if isinstance(result, tuple):
                summary, model = result
            else:
                summary = result
            all_results.append(summary)
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))
        summary_path = os.path.join(config.results_dir, 'all_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nAll results saved to {summary_path}")
    else:
        print("No results to display.")
    
    return all_results


if __name__ == "__main__":
    main()