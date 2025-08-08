"""
performance_optimizer.py
========================

High-performance optimization module for RTX 3090 hardware with:

- GPU memory management and pooling
- TF32 and mixed precision acceleration  
- CUDA stream optimization
- Memory allocation strategies
- Performance profiling and monitoring
- Automatic hardware detection and tuning
"""

from __future__ import annotations

import os
import gc
import time
import psutil
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Tuple
from contextlib import contextmanager

import numpy as np
import torch
import cv2

try:
    import pynvml  # type: ignore
    nvidia_ml_available = True
except ImportError:
    nvidia_ml_available = False


@dataclass
class HardwareInfo:
    """Hardware information and capabilities."""
    gpu_name: str = "Unknown"
    gpu_memory_total: int = 0  # MB
    gpu_memory_free: int = 0  # MB
    gpu_compute_capability: Tuple[int, int] = (0, 0)
    cpu_cores: int = 0
    ram_total: int = 0  # MB
    ram_available: int = 0  # MB
    cuda_available: bool = False
    tensorrt_available: bool = False
    opencv_cuda: bool = False


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    gpu_utilization: float = 0.0  # %
    gpu_memory_used: float = 0.0  # %
    gpu_temperature: float = 0.0  # Celsius
    cpu_utilization: float = 0.0  # %
    ram_usage: float = 0.0  # %
    processing_fps: float = 0.0
    inference_time_ms: float = 0.0
    memory_bandwidth: float = 0.0  # GB/s


class GPUMemoryManager:
    """Optimized GPU memory management for RTX 3090."""
    
    def __init__(self, target_utilization: float = 0.90):
        self.target_utilization = target_utilization
        self.memory_pools: Dict[str, List[torch.Tensor]] = {}
        self.allocation_stats = {"hits": 0, "misses": 0, "peak_usage": 0}
        self.lock = threading.Lock()
        
        if torch.cuda.is_available():
            self._initialize_cuda_memory()
    
    def _initialize_cuda_memory(self):
        """Initialize CUDA memory management."""
        # Enable memory pooling
        torch.cuda.empty_cache()
        
        # Set memory fraction for RTX 3090 (24GB)
        if torch.cuda.get_device_properties(0).total_memory > 20 * 1024**3:  # > 20GB
            torch.cuda.set_per_process_memory_fraction(self.target_utilization)
        
        # Enable memory allocation optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        logging.info(f"GPU memory manager initialized with {self.target_utilization*100:.0f}% utilization")
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                       pool_key: str = "default") -> torch.Tensor:
        """Allocate tensor with memory pooling."""
        with self.lock:
            # Check pool for reusable tensor
            if pool_key in self.memory_pools:
                for i, tensor in enumerate(self.memory_pools[pool_key]):
                    if (tensor.shape == shape and tensor.dtype == dtype and 
                        not tensor.is_cuda or tensor.device.index == 0):
                        # Reuse existing tensor
                        self.memory_pools[pool_key].pop(i)
                        self.allocation_stats["hits"] += 1
                        return tensor.zero_()
            
            # Allocate new tensor
            if torch.cuda.is_available():
                tensor = torch.zeros(shape, dtype=dtype, device='cuda')
            else:
                tensor = torch.zeros(shape, dtype=dtype)
            
            self.allocation_stats["misses"] += 1
            self._update_peak_usage()
            
            return tensor
    
    def deallocate_tensor(self, tensor: torch.Tensor, pool_key: str = "default"):
        """Return tensor to memory pool."""
        with self.lock:
            if pool_key not in self.memory_pools:
                self.memory_pools[pool_key] = []
            
            # Limit pool size to prevent memory leaks
            if len(self.memory_pools[pool_key]) < 10:
                self.memory_pools[pool_key].append(tensor)
            else:
                del tensor
    
    def _update_peak_usage(self):
        """Update peak memory usage statistics."""
        if torch.cuda.is_available():
            current_usage = torch.cuda.memory_allocated(0)
            self.allocation_stats["peak_usage"] = max(
                self.allocation_stats["peak_usage"], current_usage
            )
    
    def clear_pools(self):
        """Clear all memory pools."""
        with self.lock:
            for pool in self.memory_pools.values():
                pool.clear()
            self.memory_pools.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        stats = self.allocation_stats.copy()
        if torch.cuda.is_available():
            stats.update({
                "allocated_mb": torch.cuda.memory_allocated(0) / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved(0) / 1024**2,
                "max_allocated_mb": torch.cuda.max_memory_allocated(0) / 1024**2
            })
        return stats


class CUDAOptimizer:
    """CUDA optimization for maximum performance."""
    
    def __init__(self):
        self.streams: List[torch.cuda.Stream] = []
        self.optimization_enabled = False
        
        if torch.cuda.is_available():
            self._initialize_cuda_optimizations()
    
    def _initialize_cuda_optimizations(self):
        """Initialize CUDA optimizations."""
        # Enable TF32 for RTX 30 series
        if torch.cuda.get_device_capability(0)[0] >= 8:  # Ampere and newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("TF32 acceleration enabled for RTX 30 series")
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Create CUDA streams for parallel processing
        for i in range(4):
            self.streams.append(torch.cuda.Stream())
        
        # Set CUDA device properties for optimization
        device_props = torch.cuda.get_device_properties(0)
        if device_props.multi_processor_count >= 80:  # RTX 3090 has 82 SMs
            # Optimize for high SM count
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        
        self.optimization_enabled = True
        logging.info("CUDA optimizations initialized")
    
    def get_stream(self, stream_id: int = 0) -> Optional[torch.cuda.Stream]:
        """Get CUDA stream for parallel processing."""
        if self.streams and 0 <= stream_id < len(self.streams):
            return self.streams[stream_id]
        return None
    
    @contextmanager
    def cuda_stream(self, stream_id: int = 0):
        """Context manager for CUDA stream execution."""
        if self.optimization_enabled and stream_id < len(self.streams):
            with torch.cuda.stream(self.streams[stream_id]):
                yield self.streams[stream_id]
        else:
            yield None
    
    def synchronize_streams(self):
        """Synchronize all CUDA streams."""
        for stream in self.streams:
            stream.synchronize()
    
    def optimize_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference."""
        if not self.optimization_enabled:
            return model
        
        # Fuse operations
        if hasattr(model, 'fuse'):
            model.fuse()
        
        # Enable inference mode optimizations
        model.eval()
        
        # Compile with TorchScript for additional optimizations
        try:
            with torch.no_grad():
                # Create dummy input for tracing
                dummy_input = torch.randn(1, 3, 640, 640, device='cuda')
                traced_model = torch.jit.trace(model, dummy_input)
                traced_model = torch.jit.optimize_for_inference(traced_model)
                logging.info("Model optimized with TorchScript")
                return traced_model
        except Exception as e:
            logging.warning(f"TorchScript optimization failed: {e}")
            return model


class PerformanceProfiler:
    """Real-time performance profiling and monitoring."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics = PerformanceMetrics()
        self.callbacks: List[Callable[[PerformanceMetrics], None]] = []
        
        # Initialize NVIDIA ML if available
        if nvidia_ml_available:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logging.info("NVIDIA ML monitoring initialized")
            except:
                self.nvml_handle = None
        else:
            self.nvml_handle = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
        logging.info("Performance monitoring stopped")
    
    def add_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """Add callback for metrics updates."""
        self.callbacks.append(callback)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._update_metrics()
                
                # Call all callbacks
                for callback in self.callbacks:
                    try:
                        callback(self.metrics)
                    except Exception as e:
                        logging.error(f"Metrics callback error: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _update_metrics(self):
        """Update performance metrics."""
        # CPU and RAM metrics
        self.metrics.cpu_utilization = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        self.metrics.ram_usage = ram.percent
        
        # GPU metrics via NVIDIA ML
        if self.nvml_handle:
            try:
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                self.metrics.gpu_utilization = util.gpu
                
                # GPU memory
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                self.metrics.gpu_memory_used = (mem.used / mem.total) * 100
                
                # GPU temperature
                temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                self.metrics.gpu_temperature = temp
                
            except Exception as e:
                logging.debug(f"NVML metrics error: {e}")
        
        # PyTorch GPU metrics (fallback)
        elif torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            self.metrics.gpu_memory_used = (allocated / total) * 100
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics


class HardwareDetector:
    """Detect and analyze hardware capabilities."""
    
    @staticmethod
    def detect_hardware() -> HardwareInfo:
        """Detect current hardware configuration."""
        info = HardwareInfo()
        
        # CPU information
        info.cpu_cores = psutil.cpu_count(logical=False)
        
        # RAM information
        ram = psutil.virtual_memory()
        info.ram_total = int(ram.total / 1024**2)  # MB
        info.ram_available = int(ram.available / 1024**2)  # MB
        
        # CUDA availability
        info.cuda_available = torch.cuda.is_available()
        
        if info.cuda_available:
            # GPU information
            props = torch.cuda.get_device_properties(0)
            info.gpu_name = props.name
            info.gpu_memory_total = int(props.total_memory / 1024**2)  # MB
            info.gpu_compute_capability = props.major, props.minor
            
            # Available GPU memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                total = props.total_memory
                info.gpu_memory_free = int((total - allocated) / 1024**2)
        
        # TensorRT availability
        try:
            import tensorrt  # type: ignore
            info.tensorrt_available = True
        except ImportError:
            info.tensorrt_available = False
        
        # OpenCV CUDA support
        info.opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        return info
    
    @staticmethod
    def recommend_optimizations(hardware: HardwareInfo) -> Dict[str, Any]:
        """Recommend optimizations based on hardware."""
        recommendations = {
            "memory_fraction": 0.85,
            "batch_size": 1,
            "num_workers": 2,
            "precision": "fp32",
            "tensorrt_enabled": False,
            "cuda_streams": 2
        }
        
        # RTX 3090 specific optimizations
        if "RTX 3090" in hardware.gpu_name:
            recommendations.update({
                "memory_fraction": 0.95,
                "batch_size": 4,
                "num_workers": 4,
                "precision": "tf32",
                "tensorrt_enabled": hardware.tensorrt_available,
                "cuda_streams": 4
            })
        
        # High memory GPU optimizations
        elif hardware.gpu_memory_total > 16000:  # > 16GB
            recommendations.update({
                "memory_fraction": 0.90,
                "batch_size": 2,
                "num_workers": 3,
                "precision": "fp16",
                "cuda_streams": 3
            })
        
        # CPU fallback optimizations
        elif not hardware.cuda_available:
            recommendations.update({
                "batch_size": 1,
                "num_workers": min(hardware.cpu_cores, 4),
                "precision": "fp32",
                "cuda_streams": 0
            })
        
        return recommendations


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.hardware_info = HardwareDetector.detect_hardware()
        self.recommendations = HardwareDetector.recommend_optimizations(self.hardware_info)
        
        # Initialize optimizers
        self.memory_manager = GPUMemoryManager(
            target_utilization=self.recommendations["memory_fraction"]
        )
        self.cuda_optimizer = CUDAOptimizer()
        self.profiler = PerformanceProfiler()
        
        # Optimization state
        self.optimized_models: Dict[str, torch.nn.Module] = {}
        
        logging.info(f"Performance optimizer initialized for {self.hardware_info.gpu_name}")
    
    def optimize_model(self, model: torch.nn.Module, model_name: str = "default") -> torch.nn.Module:
        """Optimize model for current hardware."""
        if model_name in self.optimized_models:
            return self.optimized_models[model_name]
        
        optimized = self.cuda_optimizer.optimize_inference(model)
        self.optimized_models[model_name] = optimized
        
        logging.info(f"Model '{model_name}' optimized for {self.hardware_info.gpu_name}")
        return optimized
    
    def create_optimized_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create optimized tensor with memory pooling."""
        return self.memory_manager.allocate_tensor(shape, dtype)
    
    def release_tensor(self, tensor: torch.Tensor):
        """Release tensor back to memory pool."""
        self.memory_manager.deallocate_tensor(tensor)
    
    @contextmanager
    def optimized_inference(self, stream_id: int = 0):
        """Context manager for optimized inference."""
        with self.cuda_optimizer.cuda_stream(stream_id):
            yield
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.profiler.start_monitoring()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.profiler.stop_monitoring()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "hardware": self.hardware_info.__dict__,
            "recommendations": self.recommendations,
            "memory_stats": self.memory_manager.get_stats(),
            "current_metrics": self.profiler.get_metrics().__dict__
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_monitoring()
        self.memory_manager.clear_pools()
        self.cuda_optimizer.synchronize_streams()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Utility functions for performance testing
def benchmark_inference(model: torch.nn.Module, input_shape: Tuple[int, ...], 
                       num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark model inference performance."""
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = torch.randn(input_shape, device='cuda')
    else:
        dummy_input = torch.randn(input_shape)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    return {
        "total_time": total_time,
        "avg_inference_time": avg_time * 1000,  # ms
        "fps": fps,
        "iterations": num_iterations
    }


def test_memory_bandwidth() -> float:
    """Test GPU memory bandwidth."""
    if not torch.cuda.is_available():
        return 0.0
    
    size = 1024 * 1024 * 256  # 256M elements
    dtype = torch.float32
    
    # Allocate tensors
    a = torch.randn(size, dtype=dtype, device='cuda')
    b = torch.randn(size, dtype=dtype, device='cuda')
    
    # Warmup
    for _ in range(10):
        c = a + b
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    iterations = 100
    
    for _ in range(iterations):
        c = a + b
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate bandwidth
    bytes_per_operation = size * dtype.itemsize * 3  # 2 reads + 1 write
    total_bytes = bytes_per_operation * iterations
    total_time = end_time - start_time
    bandwidth_gbps = (total_bytes / total_time) / (1024**3)
    
    return bandwidth_gbps


if __name__ == "__main__":
    # Test performance optimization
    optimizer = PerformanceOptimizer()
    
    print("Hardware Information:")
    for key, value in optimizer.hardware_info.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\nRecommendations:")
    for key, value in optimizer.recommendations.items():
        print(f"  {key}: {value}")
    
    if torch.cuda.is_available():
        print(f"\nMemory bandwidth: {test_memory_bandwidth():.1f} GB/s")
    
    optimizer.cleanup()