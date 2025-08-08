"""
live_streaming.py
=================

Live streaming module for real-time device control and funscript streaming.
Supports The Handy, OSR2, SR6, and other interactive devices with
high-performance streaming optimized for low latency.
"""

from __future__ import annotations

import asyncio
import websockets
import json
import time
import threading
import queue
from typing import Dict, Any, List, Optional, Callable, Protocol
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
import numpy as np


class DeviceType(Enum):
    """Supported device types."""
    THE_HANDY = "the_handy"
    OSR2 = "osr2"
    SR6 = "sr6"
    BUTTPLUG = "buttplug"
    CUSTOM = "custom"


class ConnectionStatus(Enum):
    """Connection status states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class DeviceCommand:
    """Command to send to device."""
    timestamp: int  # milliseconds
    position: int   # 0-100
    duration: Optional[int] = None  # milliseconds for transition
    speed: Optional[int] = None     # 0-100


@dataclass
class StreamingMetrics:
    """Metrics for streaming performance."""
    latency_ms: float
    commands_sent: int
    commands_queued: int
    connection_quality: float  # 0-1
    bandwidth_usage: float     # bytes/sec
    error_count: int


class DeviceInterface(Protocol):
    """Protocol for device communication."""
    
    async def connect(self, connection_string: str) -> bool:
        """Connect to device."""
        ...
        
    async def disconnect(self) -> bool:
        """Disconnect from device."""
        ...
        
    async def send_command(self, command: DeviceCommand) -> bool:
        """Send command to device."""
        ...
        
    async def get_status(self) -> Dict[str, Any]:
        """Get device status."""
        ...


class TheHandyInterface:
    """Interface for The Handy device."""
    
    def __init__(self):
        self.api_key = ""
        self.base_url = "https://www.handyfeeling.com/api/handy/v2"
        self.session = requests.Session()
        self.connected = False
        
    async def connect(self, api_key: str) -> bool:
        """Connect to The Handy using API key."""
        self.api_key = api_key
        self.session.headers.update({"X-Connection-Key": api_key})
        
        try:
            # Test connection
            response = self.session.get(f"{self.base_url}/connected")
            if response.status_code == 200:
                data = response.json()
                self.connected = data.get("connected", False)
                return self.connected
        except Exception as e:
            print(f"Handy connection error: {e}")
            
        return False
        
    async def disconnect(self) -> bool:
        """Disconnect from The Handy."""
        self.connected = False
        return True
        
    async def send_command(self, command: DeviceCommand) -> bool:
        """Send position command to The Handy."""
        if not self.connected:
            return False
            
        try:
            # The Handy uses CSV format for scripts
            payload = {
                "at": command.timestamp,
                "pos": command.position
            }
            
            response = self.session.put(
                f"{self.base_url}/hdsp",
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Handy command error: {e}")
            return False
            
    async def get_status(self) -> Dict[str, Any]:
        """Get The Handy status."""
        if not self.connected:
            return {"connected": False}
            
        try:
            response = self.session.get(f"{self.base_url}/status")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Handy status error: {e}")
            
        return {"connected": False, "error": "Failed to get status"}


class ButtplugInterface:
    """Interface for Buttplug.io compatible devices."""
    
    def __init__(self):
        self.websocket = None
        self.connected = False
        self.device_list = []
        self.message_id = 0
        
    async def connect(self, address: str = "ws://localhost:12345") -> bool:
        """Connect to Buttplug server."""
        try:
            self.websocket = await websockets.connect(address)
            
            # Send handshake
            handshake = {
                "Id": self._get_message_id(),
                "RequestServerInfo": {
                    "ClientName": "FunGen Enhanced",
                    "MessageVersion": 3
                }
            }
            
            await self.websocket.send(json.dumps([handshake]))
            response = await self.websocket.recv()
            
            # Start scanning for devices
            await self._start_scanning()
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Buttplug connection error: {e}")
            return False
            
    async def disconnect(self) -> bool:
        """Disconnect from Buttplug server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
        self.connected = False
        return True
        
    async def _start_scanning(self):
        """Start scanning for devices."""
        scan_message = {
            "Id": self._get_message_id(),
            "StartScanning": {}
        }
        
        await self.websocket.send(json.dumps([scan_message]))
        
    async def send_command(self, command: DeviceCommand) -> bool:
        """Send command to Buttplug device."""
        if not self.connected or not self.device_list:
            return False
            
        try:
            # Use first available device
            device_id = self.device_list[0]["DeviceIndex"]
            
            # Send linear actuator command
            cmd_message = {
                "Id": self._get_message_id(),
                "DeviceMessage": {
                    "DeviceIndex": device_id,
                    "LinearCmd": {
                        "Vectors": [{
                            "Index": 0,
                            "Duration": command.duration or 100,
                            "Position": command.position / 100.0  # Buttplug uses 0-1 range
                        }]
                    }
                }
            }
            
            await self.websocket.send(json.dumps([cmd_message]))
            return True
            
        except Exception as e:
            print(f"Buttplug command error: {e}")
            return False
            
    async def get_status(self) -> Dict[str, Any]:
        """Get Buttplug status."""
        return {
            "connected": self.connected,
            "devices": len(self.device_list),
            "device_list": self.device_list
        }
        
    def _get_message_id(self) -> int:
        """Get next message ID."""
        self.message_id += 1
        return self.message_id


class OSR2Interface:
    """Interface for OSR2/SR6 devices via serial."""
    
    def __init__(self):
        self.serial_port = None
        self.connected = False
        
    async def connect(self, port: str, baud: int = 115200) -> bool:
        """Connect to OSR2/SR6 via serial."""
        try:
            import serial
            self.serial_port = serial.Serial(port, baud, timeout=1)
            self.connected = True
            return True
            
        except Exception as e:
            print(f"OSR2/SR6 connection error: {e}")
            return False
            
    async def disconnect(self) -> bool:
        """Disconnect from OSR2/SR6."""
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
            
        self.connected = False
        return True
        
    async def send_command(self, command: DeviceCommand) -> bool:
        """Send command to OSR2/SR6."""
        if not self.connected or not self.serial_port:
            return False
            
        try:
            # OSR2/SR6 uses TCode format
            tcode = f"L0{command.position:02d}I{command.duration or 100}\n"
            self.serial_port.write(tcode.encode())
            return True
            
        except Exception as e:
            print(f"OSR2/SR6 command error: {e}")
            return False
            
    async def get_status(self) -> Dict[str, Any]:
        """Get OSR2/SR6 status."""
        return {
            "connected": self.connected,
            "port": self.serial_port.port if self.serial_port else None
        }


class LiveStreamer(QObject):
    """High-performance live streaming manager."""
    
    statusChanged = pyqtSignal(str)  # ConnectionStatus
    metricsUpdated = pyqtSignal(dict)  # StreamingMetrics
    errorOccurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.device_type = DeviceType.THE_HANDY
        self.device_interface = None
        self.status = ConnectionStatus.DISCONNECTED
        self.command_queue = queue.Queue()
        self.streaming_thread = None
        self.is_streaming = False
        
        # Performance settings
        self.target_latency_ms = 50  # Target latency for real-time feel
        self.max_queue_size = 100
        self.command_buffer_ms = 200  # Buffer commands ahead
        
        # Metrics
        self.metrics = StreamingMetrics(0, 0, 0, 0, 0, 0)
        self.last_command_time = 0
        
        # Timer for metrics updates
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._update_metrics)
        self.metrics_timer.start(1000)  # Update every second
        
    def set_device_type(self, device_type: DeviceType):
        """Set the device type and create appropriate interface."""
        self.device_type = device_type
        
        if device_type == DeviceType.THE_HANDY:
            self.device_interface = TheHandyInterface()
        elif device_type == DeviceType.BUTTPLUG:
            self.device_interface = ButtplugInterface()
        elif device_type == DeviceType.OSR2 or device_type == DeviceType.SR6:
            self.device_interface = OSR2Interface()
        else:
            self.device_interface = None
            
    async def connect_device(self, connection_string: str) -> bool:
        """Connect to the specified device."""
        if not self.device_interface:
            self.errorOccurred.emit("No device interface configured")
            return False
            
        self._set_status(ConnectionStatus.CONNECTING)
        
        try:
            success = await self.device_interface.connect(connection_string)
            if success:
                self._set_status(ConnectionStatus.CONNECTED)
                return True
            else:
                self._set_status(ConnectionStatus.ERROR)
                self.errorOccurred.emit("Failed to connect to device")
                return False
                
        except Exception as e:
            self._set_status(ConnectionStatus.ERROR)
            self.errorOccurred.emit(f"Connection error: {str(e)}")
            return False
            
    async def disconnect_device(self) -> bool:
        """Disconnect from device."""
        self.stop_streaming()
        
        if self.device_interface:
            try:
                await self.device_interface.disconnect()
            except Exception as e:
                self.errorOccurred.emit(f"Disconnect error: {str(e)}")
                
        self._set_status(ConnectionStatus.DISCONNECTED)
        return True
        
    def start_streaming(self):
        """Start live streaming to device."""
        if self.status != ConnectionStatus.CONNECTED:
            self.errorOccurred.emit("Device not connected")
            return
            
        if self.is_streaming:
            return
            
        self.is_streaming = True
        self._set_status(ConnectionStatus.STREAMING)
        
        # Start streaming thread
        self.streaming_thread = threading.Thread(target=self._streaming_loop)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
    def stop_streaming(self):
        """Stop live streaming."""
        self.is_streaming = False
        
        if self.streaming_thread:
            self.streaming_thread.join(timeout=1.0)
            self.streaming_thread = None
            
        # Clear command queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
                
        if self.status == ConnectionStatus.STREAMING:
            self._set_status(ConnectionStatus.CONNECTED)
            
    def queue_command(self, position: int, timestamp: Optional[int] = None) -> bool:
        """Queue a command for streaming."""
        if not self.is_streaming:
            return False
            
        if self.command_queue.qsize() >= self.max_queue_size:
            # Remove oldest command to prevent overflow
            try:
                self.command_queue.get_nowait()
                self.metrics.error_count += 1
            except queue.Empty:
                pass
                
        # Create command with current timestamp if not provided
        if timestamp is None:
            timestamp = int(time.time() * 1000)
            
        command = DeviceCommand(
            timestamp=timestamp,
            position=max(0, min(100, position)),  # Clamp to valid range
            duration=50  # Quick transitions for responsiveness
        )
        
        try:
            self.command_queue.put_nowait(command)
            return True
        except queue.Full:
            self.metrics.error_count += 1
            return False
            
    def stream_funscript_data(self, positions: List[float], timestamps: List[float]):
        """Stream funscript data with timing."""
        current_time = time.time() * 1000
        
        for pos, ts in zip(positions, timestamps):
            # Convert relative timestamp to absolute
            abs_timestamp = int(current_time + ts)
            position = int(max(0, min(100, pos)))  # Convert to 0-100 range
            
            self.queue_command(position, abs_timestamp)
            
    def _streaming_loop(self):
        """Main streaming loop running in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_streaming_loop())
        except Exception as e:
            self.errorOccurred.emit(f"Streaming error: {str(e)}")
        finally:
            loop.close()
            
    async def _async_streaming_loop(self):
        """Async streaming loop."""
        while self.is_streaming:
            try:
                # Get command from queue with timeout
                try:
                    command = self.command_queue.get(timeout=0.01)
                except queue.Empty:
                    await asyncio.sleep(0.001)  # Short sleep to prevent busy waiting
                    continue
                    
                # Check if command is still relevant (not too old)
                current_time = int(time.time() * 1000)
                if command.timestamp < current_time - 500:  # 500ms old
                    # Skip old commands
                    self.metrics.error_count += 1
                    continue
                    
                # Wait until it's time to send command
                delay = (command.timestamp - current_time) / 1000.0
                if delay > 0:
                    await asyncio.sleep(min(delay, 0.1))  # Cap delay
                    
                # Send command to device
                start_time = time.time()
                success = await self.device_interface.send_command(command)
                end_time = time.time()
                
                if success:
                    self.metrics.commands_sent += 1
                    self.metrics.latency_ms = (end_time - start_time) * 1000
                else:
                    self.metrics.error_count += 1
                    
                self.last_command_time = time.time()
                
            except Exception as e:
                self.metrics.error_count += 1
                await asyncio.sleep(0.01)  # Brief pause on error
                
    def _set_status(self, status: ConnectionStatus):
        """Set connection status and emit signal."""
        if self.status != status:
            self.status = status
            self.statusChanged.emit(status.value)
            
    def _update_metrics(self):
        """Update and emit metrics."""
        # Update queue size
        self.metrics.commands_queued = self.command_queue.qsize()
        
        # Calculate connection quality based on errors and latency
        if self.metrics.commands_sent > 0:
            error_rate = self.metrics.error_count / max(1, self.metrics.commands_sent)
            latency_factor = max(0, 1 - (self.metrics.latency_ms / 200))  # Good if < 200ms
            self.metrics.connection_quality = max(0, min(1, (1 - error_rate) * latency_factor))
        else:
            self.metrics.connection_quality = 0
            
        # Emit metrics
        self.metricsUpdated.emit(asdict(self.metrics))
        
    def get_device_status(self) -> Dict[str, Any]:
        """Get current device status."""
        status = {
            "device_type": self.device_type.value,
            "connection_status": self.status.value,
            "is_streaming": self.is_streaming,
            "queue_size": self.command_queue.qsize(),
            "metrics": asdict(self.metrics)
        }
        
        if self.device_interface:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                device_status = loop.run_until_complete(self.device_interface.get_status())
                status["device_info"] = device_status
                loop.close()
            except:
                status["device_info"] = {"error": "Failed to get device info"}
                
        return status
        
    def calibrate_latency(self) -> float:
        """Calibrate streaming latency for optimal performance."""
        if not self.is_streaming:
            return 0
            
        # Send test commands and measure round-trip time
        test_commands = 10
        latencies = []
        
        for _ in range(test_commands):
            start_time = time.time()
            self.queue_command(50)  # Middle position
            time.sleep(0.1)  # Wait between commands
            
            # This is simplified - real implementation would measure actual device response
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
        avg_latency = np.mean(latencies) if latencies else 0
        self.target_latency_ms = max(20, avg_latency * 1.2)  # Add 20% buffer
        
        return avg_latency


class DeviceSimulator:
    """Device simulator for testing without physical devices."""
    
    def __init__(self):
        self.connected = False
        self.position = 0
        self.command_history = []
        
    async def connect(self, connection_string: str) -> bool:
        """Simulate device connection."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        return True
        
    async def disconnect(self) -> bool:
        """Simulate device disconnection."""
        self.connected = False
        return True
        
    async def send_command(self, command: DeviceCommand) -> bool:
        """Simulate command processing."""
        if not self.connected:
            return False
            
        self.position = command.position
        self.command_history.append({
            'timestamp': command.timestamp,
            'position': command.position,
            'received_at': int(time.time() * 1000)
        })
        
        # Keep only recent history
        if len(self.command_history) > 100:
            self.command_history.pop(0)
            
        await asyncio.sleep(0.001)  # Simulate processing time
        return True
        
    async def get_status(self) -> Dict[str, Any]:
        """Get simulator status."""
        return {
            "connected": self.connected,
            "current_position": self.position,
            "commands_received": len(self.command_history),
            "simulator": True
        }