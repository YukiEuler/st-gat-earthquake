# ==============================================================================
# ADJACENCY.PY - Graph Construction
# ==============================================================================

import numpy as np
from scipy import sparse as sp
import torch


class AdjacencyBuilder:
    """Builds distance-weighted adjacency matrix for spatial graph."""
    
    def __init__(self, config):
        self.config = config
        self.radius_km = config.get('radius_km', 50.0)
        self.sigma_km = config.get('sigma_km', 15.0)
        
    @staticmethod
    def haversine_km(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points."""
        R = 6371.0  # Earth radius in km
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    def build_distance_weighted_adj(self, num_nodes, node_info, grid_params, 
                                     use_distance_weighting=True):
        """
        Build sparse adjacency matrix with Gaussian distance weighting.
        
        Args:
            num_nodes: Number of nodes in graph
            node_info: Node mapping information
            grid_params: Grid parameters (lat/lon bounds, dimensions)
            use_distance_weighting: If False, use binary adjacency
        
        Returns:
            Normalized sparse adjacency matrix (scipy CSR format)
        """
        print(" Building adjacency matrix...")
        
        grid_size = self.config['grid_size']
        n_rows = grid_params['n_rows']
        n_cols = grid_params['n_cols']
        lat_min = grid_params['lat_min']
        lon_min = grid_params['lon_min']
        
        # Calculate node coordinates
        lat_centers = lat_min + (np.arange(n_rows) + 0.5) * grid_size
        lon_centers = lon_min + (np.arange(n_cols) + 0.5) * grid_size
        
        coords_lat = np.zeros(num_nodes, dtype=np.float32)
        coords_lon = np.zeros(num_nodes, dtype=np.float32)
        
        if node_info is not None and 'reduced_to_full' in node_info:
            for rid in range(num_nodes):
                full_id = node_info['reduced_to_full'][rid]
                r = full_id // n_cols
                c = full_id % n_cols
                coords_lat[rid] = lat_centers[r]
                coords_lon[rid] = lon_centers[c]
        else:
            for nid in range(num_nodes):
                r = nid // n_cols
                c = nid % n_cols
                coords_lat[nid] = lat_centers[r]
                coords_lon[nid] = lon_centers[c]
        
        # Build adjacency
        rows, cols, data = [], [], []
        
        for i in range(num_nodes):
            # Self-loop
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            
            # Calculate distances to all other nodes
            dists = self.haversine_km(coords_lat[i], coords_lon[i], coords_lat, coords_lon)
            
            # Find neighbors within radius
            neigh_mask = (dists <= self.radius_km) & (dists > 0)
            neigh_idx = np.where(neigh_mask)[0]
            
            if len(neigh_idx) == 0:
                continue
            
            if use_distance_weighting:
                # Gaussian decay
                weights = np.exp(-(dists[neigh_idx] / self.sigma_km)**2)
                # Apply threshold - filter both arrays together
                valid_mask = weights > 1e-3
                weights = weights[valid_mask]
                neigh_idx = neigh_idx[valid_mask]
            else:
                # Binary adjacency
                weights = np.ones(len(neigh_idx))
            
            rows.extend([i] * len(neigh_idx))
            cols.extend(neigh_idx.tolist())
            data.extend(weights.tolist())
        
        # Create sparse matrix
        adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), 
                           dtype=np.float32)
        
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        d_inv_sqrt = np.power(deg, -0.5)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        
        print(f"   Nodes: {num_nodes:,}")
        print(f"   Edges: {adj_norm.nnz:,}")
        print(f"   Density: {100 * adj_norm.nnz / (num_nodes**2):.2f}%")
        
        # Store coordinates for visualization
        self.coords = {'lat': coords_lat, 'lon': coords_lon}
        
        return adj_norm
    
    @staticmethod
    def scipy_to_torch_sparse(scipy_sparse, device=None):
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        coo = scipy_sparse.tocoo()
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        
        adj_torch = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
        adj_torch = adj_torch.coalesce()
        
        if device is not None:
            adj_torch = adj_torch.to(device)
        
        return adj_torch
