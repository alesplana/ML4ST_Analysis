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
import time
import requests
from tqdm import tqdm
import cupy as cp
import cugraph
import rmm
import numpy as np
from typing import Union, Tuple, Dict, List, Set
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define available metrics
AVAILABLE_METRICS = {
    'betweenness': 'Betweenness centrality',
    'eigenvector': 'Eigenvector centrality',
    'in_degree': 'In-degree centrality',
    'out_degree': 'Out-degree centrality',
    'raw_degrees': 'Raw degree counts'
}

class StatusMonitor:
    """Monitor class to handle server status updates."""
    def __init__(self, server_url: str, interval_minutes: int):
        self.server_url = server_url
        self.interval = interval_minutes * 60
        self.running = False
        self.thread = None
        self.last_progress = 0
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def update_progress(self, progress: float):
        self.last_progress = progress
            
    def _monitor_loop(self):
        while self.running:
            try:
                status_data = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'running',
                    'progress': f"{self.last_progress:.1f}%"
                }
                requests.post(self.server_url, json=status_data)
                logger.debug(f"Status update sent: {status_data}")
            except Exception as e:
                logger.warning(f"Failed to send status update: {str(e)}")
            time.sleep(self.interval)

def setup_output_directory(output_path: str) -> str:
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def load_graph(pickle_path: str) -> nx.MultiDiGraph:
    """Load networkx multidigraph from pickle file."""
    logger.info(f"Loading graph from {pickle_path}")
    try:
        with open(pickle_path, 'rb') as f:
            G = pickle.load(f)
        if not isinstance(G, nx.MultiDiGraph):
            G = nx.MultiDiGraph(G)
        logger.info(f"Loaded multidigraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    except Exception as e:
        logger.error(f"Failed to load graph: {str(e)}")
        raise

def convert_to_simple_graph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Convert multidigraph to simple directed graph, summing edge weights."""
    logger.info("Converting multidigraph to simple directed graph...")
    simple_G = nx.DiGraph()
    simple_G.add_nodes_from(G.nodes(data=True))
    
    edge_weights = {}
    for u, v, key, data in tqdm(G.edges(data=True, keys=True), desc="Converting to simple graph"):
        weight = data.get('weight', 1.0)
        if (u, v) in edge_weights:
            edge_weights[(u, v)] += weight
        else:
            edge_weights[(u, v)] = weight
    
    for (u, v), weight in edge_weights.items():
        simple_G.add_edge(u, v, weight=weight)
    
    return simple_G

def networkx_to_cugraph(G: nx.DiGraph) -> cugraph.Graph:
    """Convert NetworkX graph to cuGraph format."""
    logger.info("Converting to cuGraph format...")
    
    edges = [(u, v, d.get('weight', 1.0)) for u, v, d in G.edges(data=True)]
    df = pd.DataFrame(edges, columns=['src', 'dst', 'weight'])
    
    # Create cuGraph graph
    cu_G = cugraph.Graph(directed=True)
    
    # Transpose the edge DataFrame for better eigenvector centrality performance
    df_transposed = df.rename(columns={'src': 'dst', 'dst': 'src'})
    df_combined = pd.concat([df, df_transposed], ignore_index=True)
    
    cu_G.from_pandas_edgelist(
        df_combined,  # Use combined DataFrame with both directions
        source='src',
        destination='dst',
        edge_attr='weight'
    )
    
    return cu_G

def compute_gpu_metrics(G: Union[nx.DiGraph, cugraph.Graph], 
                       metrics_to_compute: Set[str],
                       batch_size: int = 1000,
                       max_iterations: int = 1000,
                       tolerance: float = 1e-6) -> Dict:
    """Compute selected centrality metrics using GPU.
    
    The graph is preprocessed to improve eigenvector centrality convergence
    by including both forward and reverse edges."""
    logger.info("Computing centrality metrics on GPU...")
    
    if isinstance(G, nx.DiGraph):
        cu_G = networkx_to_cugraph(G)
    else:
        cu_G = G
    
    metrics = {}
    
    try:
        if 'betweenness' in metrics_to_compute:
            logger.info("Computing betweenness centrality...")
            betweenness = cugraph.betweenness_centrality(cu_G)
            metrics['betweenness'] = dict(zip(
                betweenness.index.values_host,
                betweenness.betweenness_centrality.values_host
            ))
        
        if 'eigenvector' in metrics_to_compute:
            logger.info("Computing eigenvector centrality...")
            try:
                eigenvector = cugraph.eigenvector_centrality(
                    cu_G,
                    max_iter=max_iterations,
                    tol=tolerance
                )
                metrics['eigenvector'] = dict(zip(
                    eigenvector.index.values_host,
                    eigenvector.eigenvector_centrality.values_host
                ))
            except Exception as e:
                logger.error(f"GPU eigenvector centrality failed: {str(e)}")
                raise RuntimeError("GPU computation failed - eigenvector centrality")
        
        if 'in_degree' in metrics_to_compute or 'out_degree' in metrics_to_compute:
            logger.info("Computing degree centrality...")
            if 'in_degree' in metrics_to_compute:
                in_degree = cugraph.degree_centrality(cu_G, type='in')
                metrics['in_degree_centrality'] = dict(zip(
                    in_degree.index.values_host,
                    in_degree.degree_centrality.values_host
                ))
            if 'out_degree' in metrics_to_compute:
                out_degree = cugraph.degree_centrality(cu_G, type='out')
                metrics['out_degree_centrality'] = dict(zip(
                    out_degree.index.values_host,
                    out_degree.degree_centrality.values_host
                ))
    
    except Exception as e:
        logger.error(f"GPU computation error: {str(e)}")
        raise RuntimeError(f"GPU computation failed - {str(e)}")
    
    return metrics

def compute_raw_degrees(G: nx.MultiDiGraph) -> Dict[str, Dict]:
    """Compute raw degree counts from original multidigraph."""
    logger.info("Computing raw degrees...")
    return {
        'in_degree': dict(G.in_degree(weight='weight')),
        'out_degree': dict(G.out_degree(weight='weight')),
        'total_degree': {
            node: G.in_degree(node, weight='weight') + G.out_degree(node, weight='weight')
            for node in G.nodes()
        }
    }

def extract_metrics(G: nx.MultiDiGraph, 
                   metrics_to_compute: Set[str],
                   use_gpu: bool,
                   batch_size: int,
                   max_iterations: int = 1000,
                   tolerance: float = 1e-6,
                   status_monitor: StatusMonitor = None) -> pd.DataFrame:
    """Extract selected node metrics using GPU."""
    if not metrics_to_compute:
        raise ValueError("No metrics selected for computation")
    
    results = {}
    total_steps = len(metrics_to_compute)
    current_step = 0
    
    # Always compute raw degrees if requested
    if 'raw_degrees' in metrics_to_compute:
        raw_degrees = compute_raw_degrees(G)
        results.update(raw_degrees)
        current_step += 1
        if status_monitor:
            status_monitor.update_progress((current_step / total_steps) * 100)
    
    # Compute centrality metrics if requested
    centrality_metrics = metrics_to_compute - {'raw_degrees'}
    if centrality_metrics:
        simple_G = convert_to_simple_graph(G)
        try:
            # Initialize CUDA memory manager
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=None,
                maximum_pool_size=None
            )
            
            metrics = compute_gpu_metrics(
                simple_G,
                centrality_metrics,
                batch_size,
                max_iterations,
                tolerance
            )
            results.update(metrics)
            
        except Exception as e:
            logger.error(f"GPU computation failed completely: {str(e)}")
            raise RuntimeError(f"GPU computation failed - {str(e)}")
    
    if status_monitor:
        status_monitor.update_progress(100.0)
    
    # Combine all computed metrics
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Calculate selected node metrics using GPU')
    parser.add_argument('-pkl', '--pickle', required=True, help='Path to pickled NetworkX graph')
    parser.add_argument('-out', '--output', required=True, help='Output directory for metrics')
    parser.add_argument('-m', '--metrics', nargs='+', choices=list(AVAILABLE_METRICS.keys()) + ['all'],
                       default=['all'], help='Metrics to compute (default: all)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for GPU processing (default: 1000)')
    parser.add_argument('--max-iterations', type=int, default=1000,
                       help='Maximum iterations for eigenvector centrality (default: 1000)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Convergence tolerance for eigenvector centrality (default: 1e-6)')
    parser.add_argument('--status-url', type=str, help='URL for status updates')
    parser.add_argument('--status-interval', type=int, default=5,
                       help='Status update interval in minutes (default: 5)')
    
    args = parser.parse_args()
    
    # Determine which metrics to compute
    metrics_to_compute = set(AVAILABLE_METRICS.keys()) if 'all' in args.metrics else set(args.metrics)
    
    logger.info(f"Starting analysis with metrics: {', '.join(metrics_to_compute)}")
    
    # Setup status monitoring if URL provided
    status_monitor = None
    if args.status_url:
        status_monitor = StatusMonitor(args.status_url, args.status_interval)
        status_monitor.start()
    
    try:
        # Create output directory
        output_dir = setup_output_directory(args.output)
        
        # Load graph
        G = load_graph(args.pickle)
        
        # Extract metrics
        metrics_df = extract_metrics(
            G,
            metrics_to_compute,
            use_gpu=True,
            batch_size=args.batch_size,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            status_monitor=status_monitor
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f"node_metrics_{timestamp}.parquet")
        metrics_df.to_parquet(output_path)
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
    finally:
        if status_monitor:
            status_monitor.stop()

if __name__ == "__main__":
    main()