"""
Requirements:
brew install graphviz
pip install pygraphviz pandas networkx tqdm
"""

import networkx as nx
import pandas as pd
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Union, Any
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import math
import pickle
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

class MultidigraphAnalyzer:
    def __init__(self, G: nx.MultiDiGraph, use_notebook: bool = False):
        """
        Initialize with a multidigraph and optimize for M1 processing
        
        Parameters:
        -----------
        G : nx.MultiDiGraph
            The input graph
        use_notebook : bool
            Whether to use notebook-friendly progress bars
        """
        self.G = G
        self.n_cores = mp.cpu_count()
        self.tqdm = tqdm_notebook if use_notebook else tqdm
        
    def analyze_node_connections(self, 
                               target_node: Union[str, int],
                               include_edge_data: bool = True) -> pd.DataFrame:
        """
        Analyze connections using M1-optimized parallel processing with progress bars
        
        Parameters:
        -----------
        target_node : str or int
            The node to analyze
        include_edge_data : bool
            Whether to include edge attributes in output
        """
        connections = defaultdict(lambda: {
            'incoming_edges': 0,
            'outgoing_edges': 0,
            'bidirectional': False,
            'edge_keys': [],
            'edge_data': [],
            'node_type': '',  # Store node type (e.g., CUSTOMER, TRANSACTION)
            'edge_types': set()  # Store unique edge types
        })
        
        # Process outgoing edges
        try:
            out_edges = list(self.G.out_edges(target_node, keys=True, data=True))
        except nx.NetworkXError:
            print(f"Node {target_node} not found in graph")
            return pd.DataFrame()
            
        # Handle empty edge lists
        if not out_edges:
            chunks = []
        else:
            chunk_size = max(1, math.ceil(len(out_edges) / self.n_cores))
            chunks = [out_edges[i:i + chunk_size] 
                     for i in range(0, len(out_edges), chunk_size)]
        
        with self.tqdm(total=len(chunks), desc="Processing outgoing edges") as pbar:
            with mp.Pool(self.n_cores) as pool:
                results = []
                for result in pool.imap(partial(self._process_edge_chunk, 'out'), chunks):
                    results.append(result)
                    pbar.update(1)
        
        # Merge outgoing results
        with self.tqdm(total=len(results), desc="Merging outgoing results") as pbar:
            for chunk_dict in results:
                for node, data in chunk_dict.items():
                    connections[node]['outgoing_edges'] += data['outgoing_edges']
                    connections[node]['edge_keys'].extend(data['edge_keys'])
                    connections[node]['edge_types'].update(data.get('edge_types', set()))
                    if include_edge_data:
                        connections[node]['edge_data'].extend(data['edge_data'])
                pbar.update(1)
        
        # Process incoming edges
        in_edges = list(self.G.in_edges(target_node, keys=True, data=True))
        
        # Handle empty edge lists
        if not in_edges:
            chunks = []
        else:
            chunk_size = max(1, math.ceil(len(in_edges) / self.n_cores))
            chunks = [in_edges[i:i + chunk_size] 
                     for i in range(0, len(in_edges), chunk_size)]
        
        with self.tqdm(total=len(chunks), desc="Processing incoming edges") as pbar:
            with mp.Pool(self.n_cores) as pool:
                results = []
                for result in pool.imap(partial(self._process_edge_chunk, 'in'), chunks):
                    results.append(result)
                    pbar.update(1)
        
        # Merge incoming results
        with self.tqdm(total=len(results), desc="Merging incoming results") as pbar:
            for chunk_dict in results:
                for node, data in chunk_dict.items():
                    connections[node]['incoming_edges'] += data['incoming_edges']
                    connections[node]['edge_keys'].extend(data['edge_keys'])
                    connections[node]['edge_types'].update(data.get('edge_types', set()))
                    if include_edge_data:
                        connections[node]['edge_data'].extend(data['edge_data'])
                pbar.update(1)
        
        # Convert to DataFrame
        connection_data = []
        with self.tqdm(total=len(connections), desc="Creating DataFrame") as pbar:
            for node, data in connections.items():
                # Extract node type from ID (assuming format like 'CUSTOMER-123')
                node_type = node.split('-')[0] if isinstance(node, str) else 'UNKNOWN'
                
                data['bidirectional'] = data['incoming_edges'] > 0 and data['outgoing_edges'] > 0
                entry = {
                    'connected_node': node,
                    'node_type': node_type,
                    'total_connections': data['incoming_edges'] + data['outgoing_edges'],
                    'incoming_edges': data['incoming_edges'],
                    'outgoing_edges': data['outgoing_edges'],
                    'bidirectional': data['bidirectional'],
                    'edge_keys': data['edge_keys'],
                    'edge_types': list(data['edge_types'])  # Convert set to list for DataFrame
                }
                if include_edge_data:
                    entry['edge_data'] = data['edge_data']
                connection_data.append(entry)
                pbar.update(1)
        
        df = pd.DataFrame(connection_data)
        if not df.empty:
            df = df.sort_values('total_connections', ascending=False)
            df.index = range(1, len(df) + 1)
        
        return df
    
    @staticmethod
    def _process_edge_chunk(direction: str, edges: List[Tuple]) -> Dict:
        """Process a chunk of edges in parallel"""
        chunk_connections = defaultdict(lambda: {
            'incoming_edges': 0,
            'outgoing_edges': 0,
            'edge_keys': [],
            'edge_data': [],
            'edge_types': set()
        })
        
        for edge in edges:
            if direction == 'out':
                _, neighbor, key, data = edge
                chunk_connections[neighbor]['outgoing_edges'] += 1
            else:
                neighbor, _, key, data = edge
                chunk_connections[neighbor]['incoming_edges'] += 1
                
            chunk_connections[neighbor]['edge_keys'].append((direction, key))
            chunk_connections[neighbor]['edge_data'].append((direction, data))
            
            # Store edge type if available in data
            if 'type' in data:
                chunk_connections[neighbor]['edge_types'].add(data['type'])
            
        return dict(chunk_connections)
    
    def visualize_connections(self, 
                            target_node: Union[str, int],
                            max_nodes: int = 50,
                            output_file: str = 'graph.pdf',
                            figsize: Tuple[int, int] = (15, 10),
                            node_size_base: int = 1000,
                            with_labels: bool = True) -> None:
        """Create an M1-optimized visualization using pygraphviz"""
        print("Analyzing connections...")
        df = self.analyze_node_connections(target_node)
        
        if df.empty:
            print(f"No connections found for node {target_node}")
            return
        
        print("Creating visualization...")
        
        # Create a new directed graph for visualization
        G_viz = nx.DiGraph()
        
        # Color map for different node types
        color_map = {
            'CUSTOMER': '#ADD8E6',    # Light blue
            'TRANSACTION': '#90EE90',  # Light green
            'ACCOUNT': '#FFFFE0',     # Light yellow
            'DEFAULT': '#D3D3D3'      # Light gray
        }
        
        # Add nodes and edges
        nodes = []
        node_colors = []
        node_sizes = []
        labels = {}
        
        # Add target node first
        nodes.append(target_node)
        node_type = target_node.split('-')[0] if isinstance(target_node, str) else 'UNKNOWN'
        node_colors.append('#FFB6C1')  # Light pink for target node
        node_sizes.append(node_size_base * 1.5)  # Larger size for target node
        labels[target_node] = f"{node_type}\n{target_node.split('-')[1]}" if '-' in str(target_node) else str(target_node)
        
        # Add connected nodes
        with self.tqdm(total=min(len(df), max_nodes), desc="Adding nodes and edges") as pbar:
            for _, row in df.head(max_nodes).iterrows():
                node = row['connected_node']
                if node != target_node:
                    nodes.append(node)
                    node_type = row['node_type']
                    node_colors.append(color_map.get(node_type, color_map['DEFAULT']))
                    
                    # Size based on total connections
                    node_sizes.append(node_size_base * 
                                   (0.5 + row['total_connections'] / df['total_connections'].max()))
                    
                    # Create label
                    labels[node] = f"{node_type}\n{node.split('-')[1]}" if '-' in str(node) else str(node)
                    
                    # Add edges with appropriate direction
                    if row['incoming_edges'] > 0:
                        G_viz.add_edge(node, target_node)
                    if row['outgoing_edges'] > 0:
                        G_viz.add_edge(target_node, node)
                pbar.update(1)
        
        # Create plot
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G_viz, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G_viz, pos,
                             nodelist=nodes,
                             node_color=node_colors,
                             node_size=node_sizes)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G_viz, pos,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20,
                             width=1.5)
        
        # Add labels if requested
        if with_labels:
            nx.draw_networkx_labels(G_viz, pos, labels, font_size=8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Target Node',
                      markerfacecolor='#FFB6C1', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Customer',
                      markerfacecolor=color_map['CUSTOMER'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Transaction',
                      markerfacecolor=color_map['TRANSACTION'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Account',
                      markerfacecolor=color_map['ACCOUNT'], markersize=10),
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"Connections for Node: {target_node}")
        plt.axis('off')
        
        # Save plot
        print("Saving visualization...")
        plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"Visualization saved as {output_file}")

def print_connection_summary(analyzer: MultidigraphAnalyzer, target_node: Union[str, int]) -> None:
    """Print a human-readable summary of node connections"""
    df = analyzer.analyze_node_connections(target_node)
    
    if df.empty:
        print(f"No connections found for node {target_node}")
        return
    
    print(f"\nConnection Summary for Node {target_node}:")
    print(f"Total connected nodes: {len(df)}")
    print(f"Total bidirectional connections: {df['bidirectional'].sum()}")
    print(f"Total incoming edges: {df['incoming_edges'].sum()}")
    print(f"Total outgoing edges: {df['outgoing_edges'].sum()}")
    
    # Group by node type
    type_summary = df.groupby('node_type').agg({
        'connected_node': 'count',
        'incoming_edges': 'sum',
        'outgoing_edges': 'sum'
    }).round(2)
    
    print("\nConnections by node type:")
    print(type_summary)
    
    print("\nTop 10 most connected nodes:")
    for idx, row in df.head(10).iterrows():
        print(f"\nNode {row['connected_node']} ({row['node_type']}):")
        print(f"  - Total connections: {row['total_connections']}")
        print(f"  - Incoming edges: {row['incoming_edges']}")
        print(f"  - Outgoing edges: {row['outgoing_edges']}")
        print(f"  - Bidirectional: {'Yes' if row['bidirectional'] else 'No'}")
        if row['edge_types']:
            print(f"  - Edge types: {', '.join(row['edge_types'])}")

# Example usage
if __name__ == "__main__":
    print("Loading graph...")
    with open('../data/jp_morgan/pickled/graph_aml_final.pickle', 'rb') as f:
        G = pickle.load(f)
    
    # Create analyzer (use_notebook=True if in Jupyter)
    analyzer = MultidigraphAnalyzer(G, use_notebook=False)
    
    target_node = 'CASH'
    
    # Print summary
    print_connection_summary(analyzer, target_node)
    
    # Create visualization
    
    analyzer.visualize_connections(target_node, output_file='graph.png')
    
    # Get detailed DataFrame
    print("\nAnalyzing detailed connection information...")
    df = analyzer.analyze_node_connections(target_node)
    print("\nDetailed connection information:")
    print(df)