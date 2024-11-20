import platform
import psutil
import cpuinfo
import os
import subprocess
from typing import Dict, Any, List, Optional

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return (
        platform.system() == "Darwin" and 
        platform.machine().startswith("arm64")
    )

def get_mps_availability() -> Dict[str, bool]:
    """Check MPS (Metal Performance Shaders) availability for Apple Silicon."""
    mps_status = {
        'pytorch_mps': False,
        'tensorflow_metal': False
    }
    
    try:
        import torch
        mps_status['pytorch_mps'] = (
            is_apple_silicon() and 
            torch.backends.mps.is_available()
        )
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf_devices = tf.config.list_physical_devices()
        mps_status['tensorflow_metal'] = any(
            'GPU' in device.device_type 
            for device in tf_devices
        )
    except ImportError:
        pass
    
    return mps_status

def get_cpu_info() -> Dict[str, Any]:
    """Get CPU specifications."""
    if is_apple_silicon():
        # Special handling for Apple Silicon
        try:
            # Get chip info using sysctl
            chip_info = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string']
            ).decode().strip()
        except:
            chip_info = "Apple Silicon"
        
        try:
            # Get core count
            core_count = subprocess.check_output(
                ['sysctl', '-n', 'hw.physicalcpu']
            ).decode().strip()
            logical_count = subprocess.check_output(
                ['sysctl', '-n', 'hw.logicalcpu']
            ).decode().strip()
        except:
            core_count = psutil.cpu_count(logical=False)
            logical_count = psutil.cpu_count(logical=True)
            
        cpu_data = {
            'processor': chip_info,
            'architecture': platform.machine(),
            'cores_physical': core_count,
            'cores_logical': logical_count,
            'frequency': "Variable (Apple Silicon)",
            'note': "Apple Silicon processor with integrated Neural Engine"
        }
    else:
        # Regular CPU info gathering for non-Apple Silicon
        cpu_info = cpuinfo.get_cpu_info()
        cpu_data = {
            'processor': cpu_info['brand_raw'],
            'architecture': platform.machine(),
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'frequency_max': psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
            'cache_size': cpu_info.get('l3_cache_size', 'N/A')
        }
    return cpu_data

def get_memory_info() -> Dict[str, str]:
    """Get RAM specifications."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    memory_data = {
        'total_ram': f"{memory.total / (1024**3):.2f} GB",
        'available_ram': f"{memory.available / (1024**3):.2f} GB",
        'ram_usage': f"{memory.percent}%",
        'total_swap': f"{swap.total / (1024**3):.2f} GB",
        'swap_usage': f"{swap.percent}%"
    }
    return memory_data

def get_disk_info() -> List[Dict[str, str]]:
    """Get disk specifications."""
    disk_data = []
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_data.append({
                'device': partition.device,
                'mountpoint': partition.mountpoint,
                'filesystem': partition.fstype,
                'total': f"{usage.total / (1024**3):.2f} GB",
                'used': f"{usage.used / (1024**3):.2f} GB",
                'free': f"{usage.free / (1024**3):.2f} GB",
                'usage_percent': f"{usage.percent}%"
            })
        except PermissionError:
            continue
    return disk_data

def get_ml_acceleration_info() -> Dict[str, Any]:
    """Get ML acceleration capabilities (MPS/Metal/Neural Engine)."""
    ml_info = {
        'apple_silicon': is_apple_silicon(),
        'acceleration_available': False,
        'pytorch': {'available': False},
        'tensorflow': {'available': False}
    }
    
    try:
        import torch
        ml_info['pytorch']['available'] = True
        ml_info['pytorch']['version'] = torch.__version__
        
        if is_apple_silicon():
            ml_info['pytorch']['mps_available'] = torch.backends.mps.is_available()
            if torch.backends.mps.is_available():
                ml_info['acceleration_available'] = True
                ml_info['pytorch']['device'] = 'mps'
                # Test MPS device
                try:
                    test_tensor = torch.zeros(1).to('mps')
                    ml_info['pytorch']['mps_working'] = True
                except:
                    ml_info['pytorch']['mps_working'] = False
        else:
            ml_info['pytorch']['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                ml_info['acceleration_available'] = True
                ml_info['pytorch']['device'] = 'cuda'
                ml_info['pytorch']['cuda_devices'] = []
                for i in range(torch.cuda.device_count()):
                    ml_info['pytorch']['cuda_devices'].append({
                        'name': torch.cuda.get_device_name(i),
                        'memory_allocated': f"{torch.cuda.memory_allocated(i) / (1024**2):.2f} MB",
                        'memory_reserved': f"{torch.cuda.memory_reserved(i) / (1024**2):.2f} MB"
                    })
    except ImportError:
        pass

    try:
        import tensorflow as tf
        ml_info['tensorflow']['available'] = True
        ml_info['tensorflow']['version'] = tf.__version__
        
        physical_devices = tf.config.list_physical_devices()
        ml_info['tensorflow']['devices'] = [device.device_type for device in physical_devices]
        
        if is_apple_silicon():
            ml_info['tensorflow']['metal_available'] = any(
                'GPU' in device.device_type for device in physical_devices
            )
            if ml_info['tensorflow']['metal_available']:
                ml_info['acceleration_available'] = True
                ml_info['tensorflow']['device'] = 'metal'
        else:
            ml_info['tensorflow']['gpu_available'] = any(
                'GPU' in device.device_type for device in physical_devices
            )
            if ml_info['tensorflow']['gpu_available']:
                ml_info['acceleration_available'] = True
                ml_info['tensorflow']['device'] = 'gpu'
    except ImportError:
        pass

    return ml_info

def run_ml_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """Run ML benchmarks appropriate for the system."""
    benchmark_results = {
        'pytorch': None,
        'tensorflow': None
    }
    
    # PyTorch benchmark
    try:
        import torch
        import time
        
        if verbose:
            print("\nRunning PyTorch benchmark...")
        
        # Determine device
        if is_apple_silicon() and torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        # Adjust matrix size based on available memory
        matrix_size = 2000 if device.type != 'cpu' else 1000
        
        # Warmup
        if device.type in ['cuda', 'mps']:
            warmup_size = 100
            a = torch.randn(warmup_size, warmup_size, device=device)
            b = torch.randn(warmup_size, warmup_size, device=device)
            torch.matmul(a, b)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            del a, b
        
        # Benchmark
        start_time = time.time()
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
        c = torch.matmul(a, b)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
            
        end_time = time.time()
        
        benchmark_results['pytorch'] = {
            'device': str(device),
            'matrix_size': matrix_size,
            'execution_time': f"{end_time - start_time:.2f} seconds"
        }
    except ImportError:
        if verbose:
            print("PyTorch not available for benchmarking")
    
    # TensorFlow benchmark
    try:
        import tensorflow as tf
        import time
        
        if verbose:
            print("\nRunning TensorFlow benchmark...")
        
        # Determine device
        if is_apple_silicon():
            device_name = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
        else:
            device_name = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
        
        matrix_size = 2000 if 'GPU' in device_name else 1000
        
        # Warmup
        with tf.device(device_name):
            warmup_size = 100
            a = tf.random.normal((warmup_size, warmup_size))
            b = tf.random.normal((warmup_size, warmup_size))
            tf.matmul(a, b)
        
        # Benchmark
        start_time = time.time()
        with tf.device(device_name):
            a = tf.random.normal((matrix_size, matrix_size))
            b = tf.random.normal((matrix_size, matrix_size))
            c = tf.matmul(a, b)
        end_time = time.time()
        
        benchmark_results['tensorflow'] = {
            'device': device_name,
            'matrix_size': matrix_size,
            'execution_time': f"{end_time - start_time:.2f} seconds"
        }
    except ImportError:
        if verbose:
            print("TensorFlow not available for benchmarking")
    
    return benchmark_results

def print_system_info():
    """Print complete system information and ML capabilities."""
    print("\n=== System Overview ===")
    print(f"Operating System: {platform.system()} {platform.release()}")
    if is_apple_silicon():
        print("Hardware: Apple Silicon (M1/M2)")
    
    print("\n=== CPU Information ===")
    cpu_info = get_cpu_info()
    for key, value in cpu_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n=== Memory Information ===")
    memory_info = get_memory_info()
    for key, value in memory_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n=== Disk Information ===")
    disk_info = get_disk_info()
    for disk in disk_info:
        print(f"\nDevice: {disk['device']}")
        print(f"Mountpoint: {disk['mountpoint']}")
        print(f"Filesystem: {disk['filesystem']}")
        print(f"Total Space: {disk['total']}")
        print(f"Used Space: {disk['used']} ({disk['usage_percent']})")
        print(f"Free Space: {disk['free']}")
    
    print("\n=== ML Acceleration Information ===")
    ml_info = get_ml_acceleration_info()
    
    if ml_info['apple_silicon']:
        print("Apple Silicon Neural Engine: Available")
    
    if ml_info['pytorch']['available']:
        print("\nPyTorch:")
        print(f"Version: {ml_info['pytorch']['version']}")
        if is_apple_silicon():
            print(f"MPS (Metal) Support: {ml_info['pytorch'].get('mps_available', False)}")
            if ml_info['pytorch'].get('mps_available'):
                print(f"MPS Working: {ml_info['pytorch'].get('mps_working', False)}")
        else:
            print(f"CUDA Support: {ml_info['pytorch'].get('cuda_available', False)}")
            if ml_info['pytorch'].get('cuda_devices'):
                for i, device in enumerate(ml_info['pytorch']['cuda_devices']):
                    print(f"\nGPU {i}:")
                    print(f"Name: {device['name']}")
                    print(f"Memory Allocated: {device['memory_allocated']}")
                    print(f"Memory Reserved: {device['memory_reserved']}")
    
    if ml_info['tensorflow']['available']:
        print("\nTensorFlow:")
        print(f"Version: {ml_info['tensorflow']['version']}")
        print(f"Available Devices: {', '.join(ml_info['tensorflow']['devices'])}")
        if is_apple_silicon():
            print(f"Metal Support: {ml_info['tensorflow'].get('metal_available', False)}")
        else:
            print(f"GPU Support: {ml_info['tensorflow'].get('gpu_available', False)}")
    
    print("\n=== ML Performance Benchmark ===")
    benchmark_results = run_ml_benchmark()
    
    if benchmark_results['pytorch']:
        print("\nPyTorch Benchmark:")
        print(f"Device: {benchmark_results['pytorch']['device']}")
        print(f"Matrix Size: {benchmark_results['pytorch']['matrix_size']}x{benchmark_results['pytorch']['matrix_size']}")
        print(f"Execution Time: {benchmark_results['pytorch']['execution_time']}")
    
    if benchmark_results['tensorflow']:
        print("\nTensorFlow Benchmark:")
        print(f"Device: {benchmark_results['tensorflow']['device']}")
        print(f"Matrix Size: {benchmark_results['tensorflow']['matrix_size']}x{benchmark_results['tensorflow']['matrix_size']}")
        print(f"Execution Time: {benchmark_results['tensorflow']['execution_time']}")

if __name__ == "__main__":
    print_system_info()