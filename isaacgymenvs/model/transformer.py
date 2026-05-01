import torch
import torch.nn as nn
from isaacgymenvs.model.base_model import BaseModel
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from pointnet2_ops.pointnet2_modules import PointnetSAModule
import copy


def sample_points_fps(points, features, npoint):
    """
    Sample points from the point cloud using the farthest point sampling algorithm.
    Args:
        points: (B, N, D)
        features: (B, N, F)
        npoint: int
    Returns:
        (B, npoint, D)
    """
    assert points.shape[:2] == features.shape[:2], "Points and features must have the same batch size and number of points"
    features_flipped = features.transpose(1, 2).contiguous()
    npoint = min(npoint, features.shape[1])
    new_features = gather_operation(
        features_flipped, furthest_point_sample(points, npoint)
    ).transpose(1, 2).contiguous()
    return new_features


def strip_prefix_from_state_dict(state_dict, prefix=None):
    """
    Strips a common prefix from all state dict keys that might be added during torch.compile.
    
    Args:
        state_dict: The model state dictionary
        prefix: Optional specific prefix to strip. If None, detects from common prefixes.
    
    Returns:
        State dict with prefix removed from keys
    """
    if prefix is None:
        common_prefixes = ["_orig_mod.", "module._orig_mod."]
        for p in common_prefixes:
            if any(k.startswith(p) for k in state_dict.keys()):
                prefix = p
                break
    if prefix:
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


class PointNetEncoder(nn.Module):
    def __init__(self, output_dim, num_output_tokens, dropout=0):
        super().__init__()

        self.SA_module = PointnetSAModule(
            npoint=num_output_tokens,
            radius=0.1,
            nsample=64,
            mlp=[3, 64, 64, 64],
            bn=False,
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64, output_dim*2),
            nn.LayerNorm(output_dim*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(output_dim*2, output_dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = xyz.clone().transpose(1, 2).contiguous()
        xyz, features = self.SA_module(xyz, features)
        features = features.transpose(1, 2).contiguous()
        return self.fc_layer(features)

class MLPEncoder(nn.Module):
    def __init__(self, output_dim, num_output_tokens, dropout=0):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Linear(3, output_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim*2, output_dim)
        )

    def forward(self, points):
        if points.shape[1] > self.num_output_tokens:
            points = sample_points_fps(points, points, self.num_output_tokens)
        features = self.net(points)
        return features

class StateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PCDTransformer(BaseModel):
    def __init__(
        self,
        hidden_dim=256,
        type_dim=4,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        chunk_size=1,
        pcd_encoders_cfg=None,
        state_encoders_cfg=None,
        transformer_cfg=None,
        normalize_state=False,
        normalize_action=False,
        action_std=None,
        action_space="delta",
        action_dim=22,
        aux_weight=0.0,
        aux_prediction_mode="absolute",
        aux_delta_scale=0.01,
    ):
        super().__init__(
            normalize_state=normalize_state,
            normalize_action=normalize_action,
            action_std=action_std,
            action_space=action_space,
        )
        self.hidden_dim = hidden_dim
        self.type_dim = type_dim
        assert self.hidden_dim > self.type_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.pcd_encoders_cfg = pcd_encoders_cfg
        self.state_encoders_cfg = state_encoders_cfg
        self.transformer_cfg = transformer_cfg

        # update config for auxiliary object state prediction
        self.aux_prediction = (aux_weight > 0)
        self.aux_weight = aux_weight
        self.aux_memory_allowlist = ["local_pcd_t", "aux_object_state"]
        self.aux_prediction_mode = str(aux_prediction_mode).lower()
        if self.aux_prediction_mode not in ["absolute", "delta"]:
            raise ValueError(f"aux_prediction_mode must be 'absolute' or 'delta', got {aux_prediction_mode}")
        self.aux_delta_scale = float(aux_delta_scale)

        # Type embeddings and encodersfor different modalities
        self.type_embeddings = nn.ParameterDict()
        self.encoders = nn.ModuleDict()
        for key, cfg in pcd_encoders_cfg.items():
            if cfg["use_pcd"]:
                self.type_embeddings[key] = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, type_dim)))
                self.encoders[key] = self._initialize_pcd_encoder(self.hidden_dim-self.type_dim, cfg)
        for key, cfg in state_encoders_cfg.items():
            if cfg["use_state"]:
                self.type_embeddings[key] = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, type_dim)))
                self.encoders[key] = StateEncoder(cfg["input_dim"], cfg["hidden_dims"], self.hidden_dim-self.type_dim, cfg["dropout"])

        # Query tokens
        if self.aux_prediction:
            self.query_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(chunk_size+1, hidden_dim))) # assume aux first, then aciton tokens
        else:
            self.query_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(chunk_size, hidden_dim)))

        # Transformer
        if transformer_cfg["type"] == "encoder_decoder":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_cfg["encoder_heads"],
                dim_feedforward=transformer_cfg["encoder_dim_feedforward"],
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_cfg["encoder_layers"])

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=transformer_cfg["decoder_heads"],
                dim_feedforward=transformer_cfg["decoder_dim_feedforward"],
                dropout=dropout,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_cfg["decoder_layers"])
        else:
            raise NotImplementedError(f"Transformer type {transformer_cfg['type']} not implemented")

        # Output head
        self.action_head = nn.Linear(hidden_dim, action_dim)  # eef (6) + hand (16)
        if self.aux_prediction:
            self.aux_head = nn.Linear(hidden_dim, 3)  # auxiliary object state prediction head

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # Note: deal with torch.compile and DDP
        model_state_dict = strip_prefix_from_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
        else:
            epoch = checkpoint["episode"]
        eval_success_rate, val_loss = None, None
        if "eval_success_rate" in checkpoint:
            eval_success_rate = checkpoint["eval_success_rate"]
        if "val_loss" in checkpoint:
            val_loss = checkpoint["val_loss"]

        self.load_state_dict(model_state_dict)
        return epoch, eval_success_rate, val_loss

    def _initialize_pcd_encoder(self, hidden_dim, encoder_cfg):
        if encoder_cfg["type"] == "mlp":
            return MLPEncoder(hidden_dim, encoder_cfg["num_output_tokens"])
        elif encoder_cfg["type"] == "pointnet":
            return PointNetEncoder(hidden_dim, encoder_cfg["num_output_tokens"])
        else:
            raise ValueError(f"Unknown encoder type: {encoder_cfg['type']}")

    def _add_type_embeddings(self, tokens, token_type):
        B = tokens.shape[0]
        type_emb = self.type_embeddings[token_type].expand(B, tokens.shape[1], -1)
        return torch.cat([tokens, type_emb], dim=-1)

    def _postprocess_aux_prediction(self, aux_pred):
        if self.aux_prediction_mode == "delta":
            return torch.clamp(aux_pred, -1.0, 1.0)
        return aux_pred

    def _build_decoder_masks(self, query_tokens, token_ranges):
        num_queries = query_tokens.shape[1]
        num_memory_tokens = 0
        for token_slice in token_ranges.values():
            num_memory_tokens = max(num_memory_tokens, token_slice.stop)

        tgt_mask = torch.zeros((num_queries, num_queries), dtype=torch.bool, device=query_tokens.device)
        memory_mask = torch.zeros((num_queries, num_memory_tokens), dtype=torch.bool, device=query_tokens.device)

        # Aux query is index 0; action queries are index [1:].
        # For cross-attention, allow aux query to read only from the allowlist.
        memory_mask[0, :] = True
        found_allowed = False
        for key in self.aux_memory_allowlist:
            if key in token_ranges:
                memory_mask[0, token_ranges[key]] = False
                found_allowed = True

        if not found_allowed:
            raise ValueError(
                f"None of aux_memory_allowlist keys were found in encoder tokens: {self.aux_memory_allowlist}"
            )

        # For decoder self-attention, block aux query from reading action queries.
        if num_queries > 1:
            tgt_mask[0, 1:] = True

        return tgt_mask, memory_mask

    def forward(self, obs, target=None):
        # Get inputs
        obs_dict = copy.deepcopy(obs)
        B = obs_dict["q_hand"].shape[0]

        obs_tokens = []
        token_ranges = {}
        token_start_idx = 0

        for key in self.encoders.keys():
            tokens = self.encoders[key](obs_dict[key])
            if len(tokens.shape) == 2:
                tokens = tokens.unsqueeze(1)
            typed_tokens = self._add_type_embeddings(tokens, key)
            obs_tokens.append(typed_tokens)
            token_ranges[key] = slice(token_start_idx, token_start_idx + typed_tokens.shape[1])
            token_start_idx += typed_tokens.shape[1]
        obs_tokens = torch.cat(obs_tokens, dim=1)  # (B, N, H)

        memory = self.encoder(obs_tokens)  # (B, N, H)
        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, chunk_size/C+1, H)
        if self.aux_prediction:
            tgt_mask, memory_mask = self._build_decoder_masks(query_tokens, token_ranges)
            output = self.decoder(
                query_tokens,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )  # (B, chunk_size/C+1, H)
        else:
            output = self.decoder(query_tokens, memory)  # (B, chunk_size/C+1, H)

        pred = {}
        if self.aux_prediction:
            output_action = output[:, 1:, :]  # (B, chunk_size, H)
            output_aux = output[:, 0:1, :]  # (B, 1, H)
            pred["aux"] = self._postprocess_aux_prediction(self.aux_head(output_aux))  # (B, 1, 3)
        else:
            output_action = output  # (B, chunk_size, H)

        pred["action"] = self.action_head(output_action)  # (B, chunk_size, 7)

        # If target is provided, compute loss and return it
        if target is not None:
            loss = self.compute_loss(pred, target)
            return loss

        return pred

    def compute_loss(self, pred, target):
        assert len(target.shape) == 3

        loss = {}

        if self.aux_prediction:
            loss["action"] = torch.nn.functional.mse_loss(pred["action"], target[..., :-3])
            loss["aux"] = torch.nn.functional.mse_loss(pred["aux"], target[:, 0:1, -3:])
            loss["total"] = loss["action"] + self.aux_weight * loss["aux"]
        else:
            loss["total"] = torch.nn.functional.mse_loss(pred["action"], target)

        return loss

    def forward_pass(self, obs, target):
        # This method is kept for backward compatibility
        # but now just calls forward with the target
        return self.forward(obs, target)

    # Note this function is outdated, maybe update this later, coordinate with the inference scripts
    def get_action(self, obs):
        self.eval()
        current_angles = obs["current_angles"].clone()
        with torch.no_grad():
            pred = self.forward(obs)
        actions = self.decode_actions(current_angles, pred)
        self.train()
        return actions
