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
from typing import Union, Tuple, Dict
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatusMonitor:
    """Monitor class to handle server status updates."""
    def __init__(self, server_url: str, interval_minutes: int):
        self.server_url = server_url
        self.interval = interval_minutes * 60  # Convert to seconds
        self.running = False
        self.thread = None
        self.last_progress = 0
        
    def start(self):
        """Start the monitoring thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join()
            
    def update_progress(self, progress: float):
        """Update the current progress."""
        self.last_progress = progress
            
    def _monitor_loop(self):
        """Main monitoring loop."""
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
    
    # Sum weights of parallel edges with progress bar
    edge_weights = {}
    for u, v, key, data in tqdm(G.edges(data=True, keys=True), desc="Converting to simple graph"):
        weight = data.get('weight', 1.0)
        if (u, v) in edge_weights:
            edge_weights[(u, v)] += weight
        else:
            edge_weights[(u, v)] = weight
    
    # Add consolidated edges
    for (u, v), weight in edge_weights.items():
        simple_G.add_edge(u, v, weight=weight)
    
    return simple_G

def networkx_to_cugraph(G: nx.DiGraph) -> cugraph.Graph:
    """Convert NetworkX graph to cuGraph format."""
    logger.info("Converting to cuGraph format...")
    
    # Extract edges with weights
    edges = [(u, v, d.get('weight', 1.0)) for u, v, d in G.edges(data=True)]
    
    # Create DataFrame
    df = pd.DataFrame(edges, columns=['src', 'dst', 'weight'])
    
    # Create cuGraph graph
    cu_G = cugraph.Graph(directed=True)
    cu_G.from_pandas_edgelist(
        df, 
        source='src',
        destination='dst',
        edge_attr='weight'
    )
    
    return cu_G

def compute_gpu_metrics(G: Union[nx.DiGraph, cugraph.Graph], batch_size: int = 1000) -> Dict:
    """Compute centrality metrics using GPU with memory-efficient batching."""
    logger.info("Computing centrality metrics on GPU...")
    
    # Convert to cuGraph if needed
    if isinstance(G, nx.DiGraph):
        cu_G = networkx_to_cugraph(G)
    else:
        cu_G = G
    
    metrics = {}
    
    # Compute betweenness centrality
    logger.info("Computing betweenness centrality...")
    betweenness = cugraph.betweenness_centrality(cu_G)
    metrics['betweenness'] = dict(zip(
        betweenness.index.values_host,
        betweenness.betweenness_centrality.values_host
    ))
    
    # Compute eigenvector centrality
    logger.info("Computing eigenvector centrality...")
    eigenvector = cugraph.eigenvector_centrality(cu_G)
    metrics['eigenvector'] = dict(zip(
        eigenvector.index.values_host,
        eigenvector.eigenvector_centrality.values_host
    ))
    
    # Compute degree centrality
    logger.info("Computing degree centrality...")
    in_degree = cugraph.degree_centrality(cu_G, type='in')
    out_degree = cugraph.degree_centrality(cu_G, type='out')
    
    metrics['in_degree_centrality'] = dict(zip(
        in_degree.index.values_host,
        in_degree.degree_centrality.values_host
    ))
    metrics['out_degree_centrality'] = dict(zip(
        out_degree.index.values_host,
        out_degree.degree_centrality.values_host
    ))
    
    return metrics

def compute_cpu_metrics(G: nx.DiGraph, n_jobs: int) -> Dict:
    """Compute centrality metrics using CPU with parallel processing where possible."""
    metrics = {}
    
    # Compute betweenness centrality (parallel)
    logger.info("Computing betweenness centrality...")
    metrics['betweenness'] = nx.betweenness_centrality(G, weight='weight')
    
    # Compute eigenvector centrality
    logger.info("Computing eigenvector centrality...")
    try:
        metrics['eigenvector'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality failed to converge, using alternative method...")
        metrics['eigenvector'] = nx.eigenvector_centrality_numpy(G, weight='weight')
    
    # Compute degree centrality
    logger.info("Computing degree centrality...")
    metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
    metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
    
    return metrics

def extract_metrics(G: nx.MultiDiGraph, use_gpu: bool, batch_size: int, 
                   n_jobs: int, status_monitor: StatusMonitor = None) -> pd.DataFrame:
    """Extract node metrics using either GPU or CPU."""
    total_steps = 4  # conversion + 3 centrality metrics
    current_step = 0
    
    # Convert to simple graph
    simple_G = convert_to_simple_graph(G)
    current_step += 1
    if status_monitor:
        status_monitor.update_progress((current_step / total_steps) * 100)
    
    # Calculate raw degrees from original multidigraph
    logger.info("Computing raw degrees...")
    in_degrees = dict(G.in_degree(weight='weight'))
    out_degrees = dict(G.out_degree(weight='weight'))
    total_degrees = {node: in_degrees.get(node, 0) + out_degrees.get(node, 0) 
                    for node in G.nodes()}
    
    # Compute centrality metrics
    if use_gpu:
        try:
            # Initialize CUDA memory manager
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=None,  # Default to 1/2 of available memory
                maximum_pool_size=None   # Default to all available memory
            )
            
            metrics = compute_gpu_metrics(simple_G, batch_size)
        except Exception as e:
            logger.warning(f"GPU computation failed: {str(e)}. Falling back to CPU...")
            use_gpu = False
    
    if not use_gpu:
        metrics = compute_cpu_metrics(simple_G, n_jobs)
    
    current_step = total_steps
    if status_monitor:
        status_monitor.update_progress(100.0)
    
    # Combine all metrics
    metrics_df = pd.DataFrame({
        'in_degree': in_degrees,
        'out_degree': out_degrees,
        'total_degree': total_degrees,
        'betweenness_centrality': metrics['betweenness'],
        'eigenvector_centrality': metrics['eigenvector'],
        'in_degree_centrality': metrics['in_degree_centrality'],
        'out_degree_centrality': metrics['out_degree_centrality']
    })
    
    return metrics_df

def main():
    parser = argparse.ArgumentParser(description='Calculate node metrics with GPU support')
    parser.add_argument('-pkl', '--pickle', required=True, help='Path to pickled NetworkX graph')
    parser.add_argument('-out', '--output', required=True, help='Output directory for metrics')
    parser.add_argument('-j', '--jobs', type=int, default=None, 
                       help='Number of CPU cores for fallback (default: number of CPU cores)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for GPU processing (default: 1000)')
    parser.add_argument('--status-url', type=str, help='URL for status updates')
    parser.add_argument('--status-interval', type=int, default=5,
                       help='Status update interval in minutes (default: 5)')
    args = parser.parse_args()
    
    if args.jobs is None:
        args.jobs = mp.cpu_count()
    
    logger.info(f"Starting analysis with GPU={args.gpu}, jobs={args.jobs}")
    
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
            use_gpu=args.gpu,
            batch_size=args.batch_size,
            n_jobs=args.jobs,
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