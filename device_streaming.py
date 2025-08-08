"""
device_streaming.py
==================

Real-time device streaming module for low-latency funscript control.
Supports multiple device types with optimized streaming protocols:

- The Handy (official API)
- OSR2/SR6 (serial/bluetooth)  
- Buttplug.io compatible devices
- Device simulator for testing

Features:
- Target 50ms latency streaming
- Connection quality monitoring
- Automatic latency calibration
- Command buffering and smoothing
- Real-time metrics (commands/sec, latency, quality)
"""

from __future__ import annotations

import time
import json
import asyncio
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Any
from queue import Queue, Empty
from abc import ABC, abstractmethod

import requests
import numpy as np

try:
    import serial  # type: ignore
    import bluetooth  # type: ignore
except ImportError:
    serial = None
    bluetooth = None

try:
    import buttplug  # type: ignore
except ImportError:
    buttplug = None


@dataclass
class StreamingMetrics:
    """Real-time streaming performance metrics."""
    latency_ms: float = 0.0
    commands_per_sec: float = 0.0
    connection_quality: str = "Disconnected"
    bytes_sent: int = 0
    errors_count: int = 0
    uptime_seconds: float = 0.0


@dataclass
class DeviceCommand:
    """Device command with timing information."""
    position: int  # 0-100
    duration_ms: int  # Duration to reach position
    timestamp: float  # When command was created
    priority: int = 0  # Higher priority commands sent first


class DeviceInterface(ABC):
    """Abstract base class for device interfaces."""
    
    @abstractmethod
    async def connect(self, connection_string: str) -> bool:
        """Connect to device."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from device."""
        pass
    
    @abstractmethod
    async def send_command(self, command: DeviceCommand) -> bool:
        """Send command to device."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        pass


class HandyInterface(DeviceInterface):
    """The Handy device interface using official API."""
    
    def __init__(self):
        self.connection_key: Optional[str] = None
        self.base_url = "https://www.handyfeeling.com/api/handy/v2"
        self.session = requests.Session()
        self.connected = False
        self.device_info = {}
    
    async def connect(self, connection_string: str) -> bool:
        """Connect to The Handy using connection key."""
        self.connection_key = connection_string
        
        try:
            # Test connection
            response = self.session.get(
                f"{self.base_url}/connected",
                headers={"X-Connection-Key": self.connection_key},
                timeout=5.0
            )
            
            if response.status_code == 200:
                data = response.json()
                self.connected = data.get("connected", False)
                
                if self.connected:
                    # Get device info
                    info_response = self.session.get(
                        f"{self.base_url}/info",
                        headers={"X-Connection-Key": self.connection_key},
                        timeout=5.0
                    )
                    
                    if info_response.status_code == 200:
                        self.device_info = info_response.json()
                    
                    logging.info("Connected to The Handy successfully")
                    return True
            
        except Exception as e:
            logging.error(f"Failed to connect to The Handy: {e}")
        
        self.connected = False
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from The Handy."""
        self.connected = False
        self.connection_key = None
        self.device_info = {}
        logging.info("Disconnected from The Handy")
    
    async def send_command(self, command: DeviceCommand) -> bool:
        """Send position command to The Handy."""
        if not self.connected or not self.connection_key:
            return False
        
        try:
            # The Handy expects position and duration
            payload = {
                "position": command.position,
                "duration": command.duration_ms
            }
            
            response = self.session.put(
                f"{self.base_url}/slide",
                headers={
                    "X-Connection-Key": self.connection_key,
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=2.0
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logging.error(f"Failed to send command to The Handy: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to The Handy."""
        return self.connected
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get The Handy device information."""
        return self.device_info.copy()


class OSRInterface(DeviceInterface):
    """OSR2/SR6 device interface using serial communication."""
    
    def __init__(self):
        self.serial_port: Optional[serial.Serial] = None
        self.connected = False
        self.device_info = {"type": "OSR2/SR6", "port": None}
    
    async def connect(self, connection_string: str) -> bool:
        """Connect to OSR device via serial port."""
        if serial is None:
            logging.error("PySerial not available for OSR connection")
            return False
        
        try:
            # Parse connection string (e.g., "COM3:115200" or "/dev/ttyUSB0:115200")
            if ":" in connection_string:
                port, baudrate = connection_string.split(":")
                baudrate = int(baudrate)
            else:
                port = connection_string
                baudrate = 115200
            
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            # Test communication
            self.serial_port.write(b"D0\n")  # Stop command
            time.sleep(0.1)
            
            self.connected = True
            self.device_info["port"] = port
            self.device_info["baudrate"] = baudrate
            
            logging.info(f"Connected to OSR device on {port}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to OSR device: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from OSR device."""
        if self.serial_port:
            try:
                self.serial_port.write(b"D0\n")  # Stop command
                self.serial_port.close()
            except:
                pass
            self.serial_port = None
        
        self.connected = False
        logging.info("Disconnected from OSR device")
    
    async def send_command(self, command: DeviceCommand) -> bool:
        """Send command to OSR device."""
        if not self.connected or not self.serial_port:
            return False
        
        try:
            # OSR devices expect position commands in specific format
            # L0 command sets main axis position (0-9999)
            position_scaled = int((command.position / 100.0) * 9999)
            cmd_str = f"L0{position_scaled}\n"
            
            self.serial_port.write(cmd_str.encode())
            return True
            
        except Exception as e:
            logging.error(f"Failed to send command to OSR device: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to OSR device."""
        return self.connected and self.serial_port is not None
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get OSR device information."""
        return self.device_info.copy()


class ButtplugInterface(DeviceInterface):
    """Buttplug.io compatible device interface."""
    
    def __init__(self):
        self.client: Optional[buttplug.Client] = None
        self.devices: List[Any] = []
        self.connected = False
        self.device_info = {"type": "Buttplug.io", "devices": []}
    
    async def connect(self, connection_string: str) -> bool:
        """Connect to Buttplug.io server."""
        if buttplug is None:
            logging.error("Buttplug library not available")
            return False
        
        try:
            # Parse connection string (e.g., "ws://localhost:12345")
            self.client = buttplug.Client("FunGen VR")
            
            # Connect to server
            connector = buttplug.WebsocketConnector(connection_string)
            await self.client.connect(connector)
            
            # Start device scanning
            await self.client.start_scanning()
            await asyncio.sleep(2.0)  # Wait for devices
            await self.client.stop_scanning()
            
            self.devices = list(self.client.devices.values())
            self.connected = len(self.devices) > 0
            
            self.device_info["devices"] = [
                {"name": dev.name, "index": dev.index} 
                for dev in self.devices
            ]
            
            logging.info(f"Connected to Buttplug.io with {len(self.devices)} devices")
            return self.connected
            
        except Exception as e:
            logging.error(f"Failed to connect to Buttplug.io: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Buttplug.io."""
        if self.client:
            try:
                await self.client.disconnect()
            except:
                pass
            self.client = None
        
        self.devices = []
        self.connected = False
        logging.info("Disconnected from Buttplug.io")
    
    async def send_command(self, command: DeviceCommand) -> bool:
        """Send command to Buttplug.io devices."""
        if not self.connected or not self.devices:
            return False
        
        try:
            # Send linear position command to all devices that support it
            position_normalized = command.position / 100.0
            duration_sec = command.duration_ms / 1000.0
            
            for device in self.devices:
                if hasattr(device, 'linear'):
                    await device.linear(position_normalized, duration_sec)
                elif hasattr(device, 'vibrate'):
                    # Fallback to vibration for non-linear devices
                    await device.vibrate(position_normalized)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send command to Buttplug.io device: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Buttplug.io devices."""
        return self.connected
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get Buttplug.io device information."""
        return self.device_info.copy()


class DeviceSimulator(DeviceInterface):
    """Device simulator for testing without physical hardware."""
    
    def __init__(self):
        self.connected = False
        self.current_position = 0
        self.command_history: List[DeviceCommand] = []
        self.device_info = {
            "type": "Simulator",
            "version": "1.0",
            "capabilities": ["linear", "vibration"]
        }
    
    async def connect(self, connection_string: str) -> bool:
        """Connect to simulator (always succeeds)."""
        self.connected = True
        logging.info("Connected to device simulator")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from simulator."""
        self.connected = False
        self.command_history.clear()
        logging.info("Disconnected from device simulator")
    
    async def send_command(self, command: DeviceCommand) -> bool:
        """Simulate command execution."""
        if not self.connected:
            return False
        
        self.current_position = command.position
        self.command_history.append(command)
        
        # Keep only recent history
        if len(self.command_history) > 100:
            self.command_history.pop(0)
        
        # Simulate processing delay
        await asyncio.sleep(0.001)  # 1ms delay
        
        logging.debug(f"Simulator: Position {command.position}, Duration {command.duration_ms}ms")
        return True
    
    def is_connected(self) -> bool:
        """Check simulator connection (always True when connected)."""
        return self.connected
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get simulator information."""
        info = self.device_info.copy()
        info["current_position"] = self.current_position
        info["commands_processed"] = len(self.command_history)
        return info


class DeviceStreamer:
    """High-performance device streaming manager."""
    
    def __init__(self, target_latency_ms: float = 50.0):
        self.target_latency_ms = target_latency_ms
        self.device: Optional[DeviceInterface] = None
        self.command_queue: Queue[DeviceCommand] = Queue()
        self.metrics = StreamingMetrics()
        
        # Streaming control
        self.streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Performance monitoring
        self.command_times: List[float] = []
        self.last_command_time = 0.0
        self.start_time = 0.0
        
        # Callbacks
        self.metrics_callback: Optional[Callable[[StreamingMetrics], None]] = None
    
    def set_device(self, device_type: str) -> DeviceInterface:
        """Set device interface type."""
        device_map = {
            "The Handy": HandyInterface,
            "OSR2": OSRInterface,
            "SR6": OSRInterface,
            "Buttplug.io": ButtplugInterface,
            "Simulator": DeviceSimulator
        }
        
        device_class = device_map.get(device_type, DeviceSimulator)
        self.device = device_class()
        return self.device
    
    async def connect(self, device_type: str, connection_string: str) -> bool:
        """Connect to device."""
        if not self.device or not isinstance(self.device, type(self.set_device(device_type))):
            self.set_device(device_type)
        
        if self.device:
            success = await self.device.connect(connection_string)
            if success:
                self.metrics.connection_quality = "Connected"
                self.start_time = time.time()
            return success
        
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from device."""
        self.stop_streaming()
        
        if self.device:
            await self.device.disconnect()
            self.device = None
        
        self.metrics.connection_quality = "Disconnected"
    
    def start_streaming(self) -> bool:
        """Start real-time streaming thread."""
        if self.streaming or not self.device or not self.device.is_connected():
            return False
        
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
        self.metrics.connection_quality = "Streaming"
        logging.info("Started device streaming")
        return True
    
    def stop_streaming(self) -> None:
        """Stop streaming thread."""
        if not self.streaming:
            return
        
        self.streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
            self.stream_thread = None
        
        # Clear command queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except Empty:
                break
        
        self.metrics.connection_quality = "Connected" if self.device and self.device.is_connected() else "Disconnected"
        logging.info("Stopped device streaming")
    
    def send_position(self, position: int, duration_ms: int = None) -> bool:
        """Queue position command for streaming."""
        if not self.streaming:
            return False
        
        if duration_ms is None:
            duration_ms = int(self.target_latency_ms)
        
        command = DeviceCommand(
            position=max(0, min(100, position)),
            duration_ms=duration_ms,
            timestamp=time.time()
        )
        
        try:
            # Replace queued commands to maintain low latency
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    break
            
            self.command_queue.put(command, timeout=0.001)
            return True
            
        except:
            return False
    
    def send_funscript_data(self, actions: List[Dict], current_time_ms: int) -> bool:
        """Send funscript actions based on current playback time."""
        if not actions:
            return False
        
        # Find the closest action to current time
        closest_action = None
        min_time_diff = float('inf')
        
        for action in actions:
            time_diff = abs(action["at"] - current_time_ms)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_action = action
        
        if closest_action and min_time_diff < self.target_latency_ms:
            return self.send_position(closest_action["pos"])
        
        return False
    
    def _stream_worker(self) -> None:
        """Worker thread for streaming commands."""
        # Create event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._stream_loop())
        except Exception as e:
            logging.error(f"Streaming error: {e}")
        finally:
            self.loop.close()
            self.loop = None
    
    async def _stream_loop(self) -> None:
        """Main streaming loop."""
        last_metrics_update = time.time()
        commands_sent = 0
        
        while self.streaming:
            try:
                # Get command from queue with timeout
                try:
                    command = self.command_queue.get(timeout=0.01)
                except Empty:
                    await asyncio.sleep(0.001)
                    continue
                
                # Calculate latency
                current_time = time.time()
                latency = (current_time - command.timestamp) * 1000.0
                
                # Send command to device
                if self.device:
                    success = await self.device.send_command(command)
                    if success:
                        commands_sent += 1
                        self.command_times.append(current_time)
                        self.last_command_time = current_time
                    else:
                        self.metrics.errors_count += 1
                
                # Update metrics periodically
                if current_time - last_metrics_update >= 1.0:
                    self._update_metrics(commands_sent, latency)
                    commands_sent = 0
                    last_metrics_update = current_time
                    
                    # Call metrics callback
                    if self.metrics_callback:
                        self.metrics_callback(self.metrics)
                
                # Maintain target latency
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logging.error(f"Stream loop error: {e}")
                self.metrics.errors_count += 1
                await asyncio.sleep(0.01)
    
    def _update_metrics(self, commands_sent: int, latest_latency: float) -> None:
        """Update streaming metrics."""
        current_time = time.time()
        
        # Calculate commands per second
        self.metrics.commands_per_sec = commands_sent
        
        # Update latency
        self.metrics.latency_ms = latest_latency
        
        # Calculate uptime
        if self.start_time > 0:
            self.metrics.uptime_seconds = current_time - self.start_time
        
        # Keep command history for analysis
        cutoff_time = current_time - 10.0  # Keep last 10 seconds
        self.command_times = [t for t in self.command_times if t > cutoff_time]
    
    def get_metrics(self) -> StreamingMetrics:
        """Get current streaming metrics."""
        return self.metrics
    
    def set_metrics_callback(self, callback: Callable[[StreamingMetrics], None]) -> None:
        """Set callback for metrics updates."""
        self.metrics_callback = callback
    
    def calibrate_latency(self, target_ms: float) -> None:
        """Calibrate streaming latency."""
        self.target_latency_ms = max(10.0, min(500.0, target_ms))
        logging.info(f"Latency calibrated to {self.target_latency_ms}ms")
    
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self.streaming and self.device and self.device.is_connected()


# Test functions for development
async def test_device_interfaces():
    """Test all device interfaces."""
    print("Testing device interfaces...")
    
    # Test simulator
    simulator = DeviceSimulator()
    await simulator.connect("test")
    print(f"Simulator connected: {simulator.is_connected()}")
    
    command = DeviceCommand(position=50, duration_ms=100, timestamp=time.time())
    success = await simulator.send_command(command)
    print(f"Simulator command sent: {success}")
    
    info = simulator.get_device_info()
    print(f"Simulator info: {info}")
    
    await simulator.disconnect()
    print(f"Simulator disconnected: {not simulator.is_connected()}")


def test_streamer():
    """Test device streamer."""
    print("Testing device streamer...")
    
    async def run_test():
        streamer = DeviceStreamer(target_latency_ms=50.0)
        
        # Connect to simulator
        await streamer.connect("Simulator", "test")
        
        # Start streaming
        streamer.start_streaming()
        
        # Send some commands
        for i in range(10):
            streamer.send_position(i * 10, 100)
            await asyncio.sleep(0.05)
        
        # Stop streaming
        streamer.stop_streaming()
        
        # Disconnect
        await streamer.disconnect()
        
        print("Streamer test completed")
    
    asyncio.run(run_test())


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_device_interfaces())
    test_streamer()