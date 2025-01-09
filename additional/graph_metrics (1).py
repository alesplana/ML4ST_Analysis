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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def convert_to_simple_graph(G: nx.Graph) -> nx.Graph:
    """Convert multigraph to simple graph, summing edge weights."""
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        logger.info("Converting multigraph to simple graph...")
        simple_G = nx.Graph()
        simple_G.add_nodes_from(G.nodes(data=True))
        
        # Sum weights of parallel edges
        edge_weights = {}
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            if (u, v) in edge_weights:
                edge_weights[(u, v)] += weight
            else:
                edge_weights[(u, v)] = weight
        
        # Add consolidated edges
        for (u, v), weight in edge_weights.items():
            simple_G.add_edge(u, v, weight=weight)
        
        return simple_G
    return G

def compute_betweenness_chunk(args):
    """Compute betweenness centrality for a chunk of nodes."""
    G, nodes = args
    return nx.betweenness_centrality_subset(G, nodes, nodes, normalized=True)

def extract_metrics(G: nx.Graph, n_jobs: int) -> pd.DataFrame:
    """Extract node degree and betweenness centrality."""
    # Calculate degrees (fast, no need for parallelization)
    logger.info("Computing node degrees...")
    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight='weight'))
    
    # Convert to simple graph for betweenness
    simple_G = convert_to_simple_graph(G)
    
    # Prepare for parallel betweenness computation
    logger.info("Computing betweenness centrality...")
    nodes = list(G.nodes())
    chunk_size = len(nodes) // n_jobs
    chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
    
    # Compute betweenness in parallel
    with mp.Pool(processes=n_jobs) as pool:
        betweenness_results = pool.map(compute_betweenness_chunk, 
                                     [(simple_G, chunk) for chunk in chunks])
    
    # Combine betweenness results
    betweenness = {}
    for result in betweenness_results:
        betweenness.update(result)
    
    # Combine all metrics into a DataFrame
    metrics_df = pd.DataFrame({
        'degree': degrees,
        'weighted_degree': weighted_degrees,
        'betweenness_centrality': betweenness
    })
    
    return metrics_df

def main():
    parser = argparse.ArgumentParser(description='Calculate node degrees and betweenness centrality.')
    parser.add_argument('-pkl', '--pickle', required=True, help='Path to pickled NetworkX graph')
    parser.add_argument('-out', '--output', required=True, help='Output directory for metrics')
    parser.add_argument('-j', '--jobs', type=int, default=None, 
                       help='Number of parallel jobs (default: number of CPU cores)')
    args = parser.parse_args()
    
    if args.jobs is None:
        args.jobs = mp.cpu_count()
    
    logger.info(f"Starting analysis with {args.jobs} processes")
    
    # Create output directory
    output_dir = setup_output_directory(args.output)
    
    try:
        # Load graph
        G = load_graph(args.pickle)
        
        # Extract metrics
        metrics_df = extract_metrics(G, args.jobs)
        
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