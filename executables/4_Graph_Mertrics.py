#!/usr/bin/env python3
import networkx as nx
import pandas as pd
import multiprocessing as mp
import pickle
import argparse
import os
from pathlib import Path
import logging
from datetime import datetime
import torch
import cupy as cp
import numpy as np
from tqdm import tqdm
import json
from enum import Enum, auto
from typing import Dict, Any, Optional, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Metrics(Enum):
    DEGREE = auto()
    WEIGHTED_DEGREE = auto()
    BETWEENNESS = auto()
    ALL = auto()

class CheckpointManager:
    def __init__(self, output_dir: str, graph_hash: str):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.graph_hash = graph_hash
        
    def get_checkpoint_path(self, metric: Metrics) -> Path:
        return self.checkpoint_dir / f"{self.graph_hash}_{metric.name.lower()}_checkpoint.pkl"
    
    def save_checkpoint(self, metric: Metrics, processed_nodes: Set[str], results: Dict):
        checkpoint_data = {
            'processed_nodes': processed_nodes,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.get_checkpoint_path(metric), 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
    def load_checkpoint(self, metric: Metrics) -> Optional[Dict]:
        checkpoint_path = self.get_checkpoint_path(metric)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None

def setup_output_directory(output_path: str) -> str:
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def load_graph(pickle_path: str) -> nx.Graph:
    """Load networkx graph from pickle file."""
    logger.info(f"Loading graph from {pickle_path}")
    try:
        with open(pickle_path, 'rb') as f:
            G = pickle.load(f)
        logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    except Exception as e:
        logger.error(f"Failed to load graph: {str(e)}")
        raise

def get_graph_hash(G: nx.Graph) -> str:
    """Generate a hash for the graph to identify checkpoints."""
    return f"{G.number_of_nodes()}_{G.number_of_edges()}_{hash(frozenset(G.nodes()))}"[:20]

def convert_to_simple_graph(G: nx.Graph) -> nx.Graph:
    """Convert multigraph to simple graph, summing edge weights."""
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        logger.info("Converting multigraph to simple graph...")
        simple_G = nx.Graph()
        simple_G.add_nodes_from(G.nodes(data=True))
        
        edge_weights = {}
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            if (u, v) in edge_weights:
                edge_weights[(u, v)] += weight
            else:
                edge_weights[(u, v)] = weight
        
        for (u, v), weight in edge_weights.items():
            simple_G.add_edge(u, v, weight=weight)
        
        return simple_G
    return G

def compute_betweenness_gpu(G: nx.Graph, nodes: list, device: torch.device) -> Dict:
    """Compute betweenness centrality using GPU acceleration with sparse matrices."""
    import scipy.sparse as sp
    
    # Create node to index mapping
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Convert to sparse matrix format using numeric indices
    adj_sparse = nx.to_scipy_sparse_array(G, nodelist=list(node_to_idx.keys()), format='csr')
    
    betweenness = {}
    batch_size = min(1000, len(nodes))  # Smaller batch size for memory efficiency
    
    # Create CUDA sparse tensor if available
    if device.type == 'cuda':  
        import cupyx.scipy.sparse as cusp
        adj_sparse_gpu = cusp.csr_matrix(adj_sparse)
    
    for i in tqdm(range(0, len(nodes), batch_size), desc="Computing betweenness"):
        batch_nodes = nodes[i:i + batch_size]
        batch_betweenness = {}
        
        for source_node in tqdm(batch_nodes, desc="Processing batch", leave=False):
            source_idx = node_to_idx[source_node]
            
            # Initialize distances
            distances = np.full(G.number_of_nodes(), np.inf)
            distances[source_idx] = 0
            
            if device.type == 'cuda':
                # Use CUDA sparse matrix multiplication
                distances_gpu = cp.asarray(distances)
                frontier = cp.zeros_like(distances_gpu)
                frontier[source_idx] = 1
                
                # BFS using sparse matrix multiplication
                level = 0
                while frontier.any():
                    level += 1
                    # Sparse matrix multiplication
                    next_frontier = adj_sparse_gpu @ frontier
                    mask = (distances_gpu == np.inf) & (next_frontier > 0)
                    distances_gpu[mask] = level
                    frontier = mask
                
                # Update betweenness scores
                batch_betweenness[source_node] = float(distances_gpu[distances_gpu != np.inf].sum())
                
            else:
                # CPU fallback using scipy sparse
                frontier = sp.csr_matrix((1, G.number_of_nodes()))
                frontier[0, source_idx] = 1
                
                level = 0
                while frontier.nnz > 0:
                    level += 1
                    next_frontier = frontier @ adj_sparse
                    mask = (distances == np.inf) & (next_frontier.toarray()[0] > 0)
                    distances[mask] = level
                    frontier = sp.csr_matrix((mask * 1.0).reshape(1, -1))
                
                batch_betweenness[source_node] = float(distances[distances != np.inf].sum())
        
        # Update results
        betweenness.update(batch_betweenness)
        
        # Save intermediate results
        checkpoint_path = f"checkpoints/betweenness_checkpoint_{i}.pkl"
        os.makedirs("checkpoints", exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'betweenness': betweenness,
                'node_mapping': node_to_idx,
                'processed_nodes': i + len(batch_nodes)
            }, f)
    
    # Normalize final results
    n = len(G)
    norm = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    betweenness = {k: v * norm for k, v in betweenness.items()}
    
    return betweenness

def compute_metrics(G: nx.Graph, 
                   metrics: Set[Metrics], 
                   checkpoint_mgr: CheckpointManager,
                   batch_size: int = 1000,
                   use_gpu: bool = False) -> pd.DataFrame:
    """Compute selected metrics with checkpointing and GPU support."""
    results = {}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    for metric in metrics:
        if metric == Metrics.ALL:
            continue
            
        # Load checkpoint if exists
        checkpoint = checkpoint_mgr.load_checkpoint(metric)
        if checkpoint:
            logger.info(f"Resuming {metric.name} from checkpoint")
            results[metric.name.lower()] = checkpoint['results']
            continue
            
        logger.info(f"Computing {metric.name}")
        if metric == Metrics.DEGREE:
            results['degree'] = dict(G.degree())
            
        elif metric == Metrics.WEIGHTED_DEGREE:
            results['weighted_degree'] = dict(G.degree(weight='weight'))
            
        elif metric == Metrics.BETWEENNESS:
            simple_G = convert_to_simple_graph(G)
            nodes = list(simple_G.nodes())
            
            if use_gpu:
                results['betweenness_centrality'] = compute_betweenness_gpu(
                    simple_G, nodes, device)
            else:
                results['betweenness_centrality'] = {}
                for i in tqdm(range(0, len(nodes), batch_size), 
                            desc="Computing betweenness"):
                    batch = nodes[i:i + batch_size]
                    batch_result = nx.betweenness_centrality_subset(
                        simple_G, batch, batch, normalized=True)
                    results['betweenness_centrality'].update(batch_result)
                    
                    # Save checkpoint after each batch
                    checkpoint_mgr.save_checkpoint(
                        metric,
                        set(nodes[:i + batch_size]),
                        results['betweenness_centrality']
                    )
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Calculate selected network metrics with GPU support.')
    parser.add_argument('-pkl', '--pickle', required=True, help='Path to pickled NetworkX graph')
    parser.add_argument('-out', '--output', required=True, help='Output directory for metrics')
    parser.add_argument('-m', '--metrics', nargs='+', default=['ALL'],
                       choices=[m.name for m in Metrics],
                       help='Metrics to compute')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing nodes')
    args = parser.parse_args()
    
    logger.info(f"Starting analysis{'with GPU' if args.gpu else ''}")
    
    # Create output directory
    output_dir = setup_output_directory(args.output)
    
    try:
        # Load graph
        G = load_graph(args.pickle)
        graph_hash = get_graph_hash(G)
        
        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(output_dir, graph_hash)
        
        # Determine metrics to compute
        selected_metrics = {Metrics[m] for m in args.metrics}
        if Metrics.ALL in selected_metrics:
            selected_metrics = {m for m in Metrics if m != Metrics.ALL}
        
        # Extract metrics
        metrics_df = compute_metrics(
            G, selected_metrics, checkpoint_mgr,
            batch_size=args.batch_size, use_gpu=args.gpu
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f"node_metrics_{timestamp}.parquet")
        metrics_df.to_parquet(output_path)
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()