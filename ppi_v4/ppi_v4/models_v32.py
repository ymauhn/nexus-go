# ppi_v4/models_v32.py
import torch
import torch.nn as nn
from .models import build_image_backbone
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
except ImportError:
    pass

class FixedAlphaSPMM(nn.Module):
    def __init__(self, A_T: torch.Tensor, hops: int, alpha: float):
        super().__init__()
        self.hops = hops
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("A_T", A_T.to(torch.float32))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        H_work = H.to(torch.float32)
        for _ in range(self.hops):
            agg = torch.sparse.mm(self.A_T, H_work)
            H_work = self.alpha * H_work + (1.0 - self.alpha) * agg
        return H_work.to(H.dtype)

class UnifiedProteinModel(nn.Module):
    def __init__(self, cfg, H0, A_sparse=None, edge_index=None, edge_weight=None, n_classes=500):
        super().__init__()
        self.cfg = cfg  
        self.method = cfg["method"]
        self.classifier_type = cfg.get("classifier_type", "vision")
        
        # ==========================================
        # ESTÁGIO 1: Geração de Contexto (Grafos)
        # ==========================================
        if self.method in ["fixed", "hybrid"]:
            mp = FixedAlphaSPMM(A_sparse.transpose(0, 1).coalesce(), cfg["hops"], cfg["alpha_init"])
            H_fixed = mp(H0) if cfg["hops"] > 0 else H0
            self.register_buffer("H_cached", H_fixed)
            
            if self.method == "hybrid":
                gnn_emb = torch.load(cfg["gnn_emb_path"], map_location=H0.device)
                if cfg["hybrid_fusion"] == "concat":
                    H_fused = torch.cat([self.H_cached, gnn_emb], dim=1)
                elif cfg["hybrid_fusion"] == "add":
                    H_reshaped = self.H_cached.view(-1, 3, 1024)
                    H_fused = (H_reshaped + gnn_emb.unsqueeze(1)) / 2.0
                    H_fused = H_fused.view(-1, 3072)
                self.register_buffer("H_cached", H_fused)

        elif self.method == "load_gnn":
            gnn_emb = torch.load(cfg["gnn_emb_path"], map_location=H0.device)
            self.register_buffer("H_cached", gnn_emb)

        elif self.method == "gnn":
            hidden_dim = cfg.get("gnn_hidden_dim", 1024)
            self.num_layers = cfg.get("gnn_num_layers", 2) 
            self.heads = cfg.get("gnn_heads", 4)           
            
            self.gnn_layers = nn.ModuleList()
            in_channels = 3072
            
            for i in range(self.num_layers):
                is_last = (i == self.num_layers - 1)
                in_dim = in_channels if i == 0 else hidden_dim
                
                if cfg["gnn_type"] == "gcn":
                    self.gnn_layers.append(GCNConv(in_dim, hidden_dim))
                elif cfg["gnn_type"] == "gat":
                    if is_last:
                        self.gnn_layers.append(GATConv(in_dim, hidden_dim, heads=1, concat=False, edge_dim=1))
                    else:
                        self.gnn_layers.append(GATConv(in_dim, hidden_dim // self.heads, heads=self.heads, concat=True, edge_dim=1))
                elif cfg["gnn_type"] == "sage":
                    self.gnn_layers.append(SAGEConv(in_dim, hidden_dim))
            
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.register_buffer("x_all", H0)
            self.register_buffer("edge_index", edge_index)
            self.register_buffer("edge_weight", edge_weight)

        # ==========================================
        # ESTÁGIO 2: Classificador Downstream
        # ==========================================
        in_features = self.H_cached.shape[1] if hasattr(self, "H_cached") else cfg.get("gnn_hidden_dim", 1024)
        
        if self.classifier_type == "vision":
            self.backbone = build_image_backbone(cfg["arch"], n_classes, cfg["unfreeze"], bool(cfg["pretrained"]))
        else:
            # Construção Dinâmica da MLP via YAML
            mlp_hidden_dims = cfg.get("mlp_hidden_dims", [1024]) 
            mlp_dropout = cfg.get("mlp_dropout", 0.3)
            
            layers = []
            current_in_dim = in_features
            
            for hidden_dim in mlp_hidden_dims:
                layers.append(nn.Linear(current_in_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(mlp_dropout))
                current_in_dim = hidden_dim
                
            layers.append(nn.Linear(current_in_dim, n_classes))
            self.mlp = nn.Sequential(*layers)

    def _run_gnn_forward(self):
        """Motor centralizado da GNN (Elimina a duplicação)"""
        z = self.x_all
        e_attr = self.edge_weight.unsqueeze(-1) if self.edge_weight is not None else None
        
        for i, conv in enumerate(self.gnn_layers):
            if self.cfg.get("gnn_type") == "gcn":
                z = conv(z, self.edge_index, edge_weight=self.edge_weight)
            elif self.cfg.get("gnn_type") == "gat":
                z = conv(z, self.edge_index, edge_attr=e_attr)
            else: # sage
                z = conv(z, self.edge_index)
                
            if i < self.num_layers - 1:
                z = self.act(z)
                z = self.dropout(z)
        return z

    def extract_gnn_features(self):
        self.eval() 
        with torch.no_grad():
            if hasattr(self, 'gnn_layers'):
                return self._run_gnn_forward()
        return None

    def forward(self, idx: torch.Tensor):
        idx = idx.view(-1)
        
        if self.method in ["fixed", "hybrid", "load_gnn"]:
            x = self.H_cached.index_select(0, idx)
        else:
            z = self._run_gnn_forward()
            x = z[idx]

        if self.classifier_type == "vision":
            if x.shape[1] == 1024:
                x_img = x.clamp(-50.0, 50.0).sigmoid().view(-1, 1, 32, 32)
                x_img = x_img.expand(-1, 3, 32, 32).contiguous()
            else:
                x_img = x.clamp(-50.0, 50.0).sigmoid().view(-1, 3, 32, 32).contiguous()
            return self.backbone(x_img)
        else:
            return self.mlp(x)