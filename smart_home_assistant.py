#!/usr/bin/env python3
"""
Standalone ONNX Smart Home Assistant for Raspberry Pi 4
No dependencies on other modules - complete implementation
"""

# Standard library imports
import json
import time
import re
import argparse
import threading
import queue
import os
from datetime import datetime
from collections import deque, defaultdict
import statistics
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import numpy as np

# Try to import ONNX Runtime for inference
try:
    import onnxruntime as ort
except ImportError:
    print("⚠️ onnxruntime not found. Please install it: pip install onnxruntime")
    ort = None

# Try to import transformers for tokenization
try:
    from transformers import AutoTokenizer
except ImportError:
    print("⚠️ transformers not found. Please install it: pip install transformers")
    AutoTokenizer = None

# Try to import RPi.GPIO for Raspberry Pi GPIO control
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    print("⚠️  RPi.GPIO not found. Running in simulation mode.")
    GPIO_AVAILABLE = False

# Try to import Flask for web API
try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
except ImportError:
    print("⚠️ Flask not found. Please install it: pip install Flask Flask-Cors")
    Flask = None

# Enum for device types
class DeviceType(Enum):
    """Device types"""
    LIGHT = "light"
    FAN = "fan"
    HEATER = "heater"
    AC = "ac"
    DOOR_LOCK = "door_lock"
    OUTLET = "outlet"
    ALARM = "alarm"
    CAMERA = "camera"
    SMART_TV = "smart_tv"

# Device class to represent each smart device
class Device:
    """Device representation"""
    def __init__(self, name: str, pin: int, device_type: DeviceType,
                 aliases: List[str] = None, room: str = "general",
                 power_consumption: float = 0.0, dimmable: bool = False):
        self.name = name
        self.pin = pin
        self.device_type = device_type
        self.state = False
        self.aliases = aliases or []
        self.room = room
        self.power_consumption = power_consumption
        self.dimmable = dimmable
        self.dim_level = 100
        self.last_changed = datetime.now()

# Main assistant class
class ONNXSmartHomeAssistant:
    """
    Smart Home Assistant using ONNX Runtime for inference
    """

    def __init__(self, model_path: str):
        """Initialize the assistant with ONNX model"""
        print("🏠 Initializing ONNX Smart Home Assistant for Raspberry Pi 4...")
        print(f"📦 Model path: {model_path}")

        self._gpio_simulation_mode = not GPIO_AVAILABLE

        # Performance tracking
        self._setup_performance_tracking()

        # Initialize ONNX model and tokenizer
        self.ort_session = None
        self.tokenizer = None
        self._init_onnx_model(model_path)

        # Initialize devices and mappings
        self._init_devices()

        # Setup GPIO pins
        self._setup_gpio()

        # Initialize context and automation rules
        self._init_context()
        self._init_automation_rules()

        # Command processing queue and cache
        self.command_queue = queue.Queue()
        self.response_cache = {}

        # Start background threads for context updates
        self._start_background_tasks()

        print("✅ ONNX Smart Home Assistant initialized successfully!")

    def _init_onnx_model(self, model_path: str):
        """Initialize ONNX Runtime session and tokenizer"""
        if not ort or not AutoTokenizer:
            print("⚠️ ONNX/Transformers libraries not available. Running in rule-based mode only.")
            return
        try:
            model_dir = Path(model_path)
            if not model_dir.exists():
                raise ValueError(f"Model directory not found: {model_path}")

            # Find ONNX file (prefer quantized if available)
            onnx_files = list(model_dir.glob("*.onnx"))
            if not onnx_files:
                raise ValueError(f"No ONNX files found in {model_path}")

            onnx_file = next((f for f in onnx_files if "quantized" in f.name), onnx_files[0])

            print(f"📦 Loading ONNX model: {onnx_file.name}")

            # Create ONNX session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_cpu_mem_arena = True

            self.ort_session = ort.InferenceSession(
                str(onnx_file),
                sess_options,
                providers=['CPUExecutionProvider']
            )

            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print("✅ Tokenizer loaded")
            except Exception:
                print("⚠️  Could not load tokenizer, using simple tokenization")
                self.tokenizer = None

            # Get model input/output names
            self.input_names = [inp.name for inp in self.ort_session.get_inputs()]
            self.output_names = [out.name for out in self.ort_session.get_outputs()]

            print(f"✅ ONNX model loaded successfully")
            print(f"   Providers: {self.ort_session.get_providers()}")

        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}. Running in rule-based mode.")
            self.ort_session = None # Ensure session is None on failure

    def _init_devices(self):
        """Initialize device configuration and mappings"""
        self.devices = {
            # Living Room
            'living_room_light': Device(
                name='living_room_light', pin=17, device_type=DeviceType.LIGHT,
                aliases=['living room light', 'main light', 'lounge light'],
                room='living_room', power_consumption=60, dimmable=True
            ),
            'living_room_fan': Device(
                name='living_room_fan', pin=27, device_type=DeviceType.FAN,
                aliases=['living room fan', 'main fan', 'ceiling fan'],
                room='living_room', power_consumption=75
            ),
            'smart_tv': Device(
                name='smart_tv', pin=22, device_type=DeviceType.SMART_TV,
                aliases=['TV', 'television'],
                room='living_room', power_consumption=150
            ),

            # Bedroom
            'bedroom_light': Device(
                name='bedroom_light', pin=23, device_type=DeviceType.LIGHT,
                aliases=['bedroom light', 'bed light'],
                room='bedroom', power_consumption=40, dimmable=True
            ),
            'bedroom_ac': Device(
                name='bedroom_ac', pin=24, device_type=DeviceType.AC,
                aliases=['bedroom ac', 'air conditioner', 'AC'],
                room='bedroom', power_consumption=1200
            ),

            # Kitchen
            'kitchen_light': Device(
                name='kitchen_light', pin=5, device_type=DeviceType.LIGHT,
                aliases=['kitchen light', 'cooking light'],
                room='kitchen', power_consumption=80
            ),

            # Security
            'front_door_lock': Device(
                name='front_door_lock', pin=26, device_type=DeviceType.DOOR_LOCK,
                aliases=['front door', 'main door', 'door lock'],
                room='entrance', power_consumption=5
            ),

            # Outdoor
            'garden_light': Device(
                name='garden_light', pin=16, device_type=DeviceType.LIGHT,
                aliases=['garden', 'outdoor light', 'yard light'],
                room='outdoor', power_consumption=100
            ),
        }

        # Create mappings by room and type for quick lookup
        self.devices_by_room = defaultdict(list)
        for name, device in self.devices.items():
            self.devices_by_room[device.room].append(name)

        self.devices_by_type = defaultdict(list)
        for name, device in self.devices.items():
            self.devices_by_type[device.device_type].append(name)

    def _init_context(self):
        """Initialize context (environmental and system state)"""
        self.context = {
            'temperature': 72,
            'humidity': 45,
            'time_of_day': self._get_time_context(),
            'power_usage': 0.0,
            'sleep_mode': False,
            'eco_mode': False,
            'vacation_mode': False,
        }

    def _init_automation_rules(self):
        """Initialize scenes (predefined device states)"""
        self.scenes = {
            'movie_night': {
                'devices': {
                    'living_room_light': {'state': True, 'dim': 20},
                    'living_room_fan': False,
                    'smart_tv': True,
                },
                'description': 'Dims lights for movie watching'
            },
            'sleep': {
                'devices': {
                    'bedroom_light': False,
                    'bedroom_ac': True,
                    'living_room_light': False,
                    'kitchen_light': False,
                    'garden_light': False,
                },
                'description': 'Nighttime sleep mode'
            },
            'wake_up': {
                'devices': {
                    'bedroom_light': True,
                    'kitchen_light': True,
                },
                'description': 'Morning wake up'
            },
            'away': {
                'devices': {
                    'all_lights': False,
                    'front_door_lock': True,
                },
                'description': 'Security mode when away'
            },
        }

    def _setup_gpio(self):
        """Setup GPIO pins for device control (or simulation mode)"""
        if self._gpio_simulation_mode:
            print("🔧 GPIO running in simulation mode.")
            return

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            for device in self.devices.values():
                GPIO.setup(device.pin, GPIO.OUT, initial=GPIO.LOW)

            print("✅ GPIO pins initialized")
        except Exception as e:
            print(f"⚠️  GPIO initialization failed: {e}")
            self._gpio_simulation_mode = True

    def _setup_performance_tracking(self):
        """Initialize performance metrics tracking"""
        self.performance_stats = {
            'command_latencies': deque(maxlen=100),
            'inference_times': deque(maxlen=100),
            'tokens_per_second': deque(maxlen=100),
            'cache_hits': 0,
            'cache_misses': 0,
            'total_commands': 0,
        }

    def understand_command(self, user_input: str) -> Dict[str, Any]:
        """Process user command with ONNX inference or rules"""
        # Check cache first
        cache_key = user_input.lower().strip()
        if cache_key in self.response_cache:
            self.performance_stats['cache_hits'] += 1
            cached = self.response_cache[cache_key].copy()
            cached['from_cache'] = True
            return cached

        self.performance_stats['cache_misses'] += 1
        print("🤔 Processing command...")
        start_time = time.time()
        
        response = {'intent': 'unknown'}

        # 1. Try ONNX model first if available
        if self.ort_session and self.tokenizer:
            response = self._process_command_onnx(user_input)
        
        # 2. If ONNX result is unknown or model not available, use rule-based fallback
        if response.get('intent') == 'unknown':
            if self.ort_session and self.tokenizer:
                print("↪️ Falling back to rule-based processing.")
            response = self._process_command_rules(user_input)
        
        inference_time = time.time() - start_time
        response['inference_time'] = inference_time
        response['raw_input'] = user_input

        # Cache response
        self.response_cache[cache_key] = response
        self.performance_stats['inference_times'].append(inference_time)

        return response

    def _process_command_onnx(self, prompt: str, max_new_tokens: int = 12) -> Dict[str, Any]:
        """
        ONNX Inference - Always use FAST mode for ultra-low latency, and disable any autoregressive/story generation.
        Only generates actions for device control/status/scene.
        """
        # Ultra-fast: skip all heuristics, force fast mode for every prompt
        print(f"🔀 Forcing FAST mode for all prompts (ultra-low latency)")

        return self._process_command_onnx_hybrid(prompt, mode="fast", max_new_tokens=2)

    def _process_command_onnx_hybrid(self, prompt: str, mode: str = "fast", max_new_tokens: int = 2) -> Dict[str, Any]:
        """
        Ultra-fast: Only use single-step output for immediate control.
        Map output directly to actions (no multi-token, no fallback unless no action at all).
        """
        import time
        try:
            print("🧠 Processing command with ONNX model... [FAST mode, strict]")
            t0 = time.time()
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="np", padding=True, truncation=True)
            ort_inputs = {}
            for model_input in self.ort_session.get_inputs():
                n = model_input.name
                if n in inputs:
                    ort_inputs[n] = inputs[n]
                elif n == "position_ids":
                    seq_length = inputs["input_ids"].shape[1]
                    ort_inputs["position_ids"] = np.arange(seq_length, dtype=np.int64).reshape(1, seq_length)
            # Run model, only predict next token
            ort_outs = self.ort_session.run(self.output_names, ort_inputs)
            input_ids = inputs["input_ids"]
            next_token_logits = ort_outs[0][:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1)[0]
            generated_ids = np.concatenate([input_ids[0], [next_token_id]])
            generated_text = self.tokenizer.decode(generated_ids)  # Don't skip special tokens
            t1 = time.time()
            tokens_per_sec = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
            self.performance_stats['tokens_per_second'].append(tokens_per_sec)
            print(f"   🤖 [FAST] Model Generated: '{generated_text}'")
            print(f"   ⚡️ Tokens/Second: {tokens_per_sec:.2f}")
            print(f"   ⏱️ Response time: {(t1-t0)*1000:.1f} ms")
            sanitized_text = generated_text.strip().lower()

            # **Aggressive direct intent extraction**
            # Look for on/off commands and device in generated text
            ACTION_ON = ['on', 'enable', 'activate', 'open']
            ACTION_OFF = ['off', 'disable', 'deactivate', 'close']
            device_names = [d.replace('_', ' ') for d in self.devices.keys()]
            found_devices = [dev for dev in device_names if dev in sanitized_text]
            if found_devices:
                for dev in found_devices:
                    orig_dev = dev.replace(' ', '_')
                    if any(word in sanitized_text for word in ACTION_ON):
                        return {'intent': 'control', 'devices': [orig_dev], 'state': True, 'confidence': 1.0}
                    elif any(word in sanitized_text for word in ACTION_OFF):
                        return {'intent': 'control', 'devices': [orig_dev], 'state': False, 'confidence': 1.0}
            # Scenes
            scene_names = [scene.replace('_', ' ') for scene in self.scenes]
            for scene in scene_names:
                if scene in sanitized_text:
                    return {'intent': 'scene', 'scene': scene.replace(' ', '_'), 'confidence': 1.0}
            # Status
            if "status" in sanitized_text or "what is on" in sanitized_text or "report" in sanitized_text:
                return {'intent': 'status', 'confidence': 1.0}
            # If nothing actionable found, fall back to rule-based (still no stories)
            return self._process_command_rules(prompt)

        except Exception as e:
            print(f"❌ Error during ONNX inference: {e}")
            return self._process_command_rules(prompt)

    def _find_devices_in_prompt(self, prompt: str) -> List[str]:
        """Finds device names mentioned in a text prompt."""
        prompt_lower = prompt.lower()
        devices_found = []
        for device_name, device in self.devices.items():
            device_words = [device_name.replace('_', ' ')] + device.aliases
            if any(word in prompt_lower for word in device_words):
                devices_found.append(device_name)
        
        if 'all light' in prompt_lower or 'all the light' in prompt_lower:
            devices_found.extend(self.devices_by_type[DeviceType.LIGHT])
        elif 'everything' in prompt_lower or 'all device' in prompt_lower:
            devices_found.extend(list(self.devices.keys()))
            
        return list(set(devices_found))

    def _find_scene_in_prompt(self, prompt: str) -> Optional[str]:
        """Finds a scene name mentioned in a text prompt."""
        prompt_lower = prompt.lower()
        for scene_name in self.scenes:
            # Check for "movie_night" or "movie night"
            if scene_name.replace('_', ' ') in prompt_lower or scene_name in prompt_lower:
                return scene_name
        return None

    def _process_command_rules(self, prompt: str) -> Dict[str, Any]:
        """Rule-based command processing as a fallback."""
        prompt_lower = prompt.lower()
        
        # Power saving commands
        if any(word in prompt_lower for word in ['save power', 'reduce bill', 'save energy', 'eco mode']):
            devices_to_off = [name for name, device in self.devices.items() 
                              if device.state and device.device_type in [DeviceType.LIGHT, DeviceType.FAN, DeviceType.AC]]
            return {
                'intent': 'multi_action',
                'actions': [{'device': name, 'state': False} for name in devices_to_off],
                'reasoning': 'Turning off unnecessary devices to save power',
                'confidence': 0.9
            }

        # Temperature control
        elif any(word in prompt_lower for word in ['cold', 'freezing', 'chilly']):
            return {
                'intent': 'temperature',
                'value': 'increase',
                'devices': ['bedroom_ac'],
                'reasoning': 'User feels cold, adjusting temperature',
                'confidence': 0.85
            }
        elif any(word in prompt_lower for word in ['hot', 'warm', 'sweating']):
            return {
                'intent': 'temperature',
                'value': 'decrease',
                'devices': ['bedroom_ac', 'living_room_fan'],
                'reasoning': 'User feels hot, turning on cooling',
                'confidence': 0.85
            }

        # Scene activation
        scene = self._find_scene_in_prompt(prompt)
        if scene:
            return {
                'intent': 'scene',
                'scene': scene,
                'reasoning': f'Activating {scene} scene',
                'confidence': 0.95
            }

        # Device control
        devices_found = self._find_devices_in_prompt(prompt)
        action = None
        if any(word in prompt_lower for word in ['turn on', 'switch on', 'enable', 'activate']):
            action = 'on'
        elif any(word in prompt_lower for word in ['turn off', 'switch off', 'disable', 'deactivate']):
            action = 'off'
        
        if devices_found and action:
            return {
                'intent': 'control',
                'devices': devices_found,
                'state': action == 'on',
                'reasoning': f'Turning {action} {", ".join(devices_found)}',
                'confidence': 0.9
            }

        # Status check
        if any(word in prompt_lower for word in ['status', 'what is on', "what's on", 'report']):
            return {
                'intent': 'status',
                'reasoning': 'Checking device status',
                'confidence': 0.95
            }

        # Default fallback
        return {
            'intent': 'unknown',
            'reasoning': 'I didn\'t understand that command. Try: "turn on lights", "save power", or "activate movie night"',
            'confidence': 0.1
        }

    def execute_command(self, action: Dict[str, Any]) -> str:
        """Execute parsed command and return a response string"""
        if not action:
            return "I couldn't understand that command."

        intent = action.get('intent')
        self.performance_stats['total_commands'] += 1

        if intent == 'control':
            devices = action.get('devices', [])
            state = action.get('state', True)
            return self._control_devices(devices, state)

        elif intent == 'scene':
            scene_name = action.get('scene')
            return self._activate_scene(scene_name)

        elif intent == 'multi_action':
            actions = action.get('actions', [])
            return self._execute_multi_action(actions)

        elif intent == 'temperature':
            value = action.get('value')
            devices = action.get('devices', [])
            return self._adjust_temperature(value, devices)

        elif intent == 'status':
            return self._get_status_report()

        else:
            return action.get('reasoning', 'Command not recognized')

    def _control_devices(self, device_names: List[str], state: bool) -> str:
        """Control devices (turn on/off) and update context"""
        controlled = []
        power_change = 0

        for device_name in device_names:
            if device_name in self.devices:
                device = self.devices[device_name]

                if device.state != state:
                    if self._gpio_simulation_mode:
                        print(f"[SIM] {'Turning on' if state else 'Turning off'} {device_name} (Pin {device.pin})")
                    else:
                        GPIO.output(device.pin, GPIO.HIGH if state else GPIO.LOW)

                    # Update state
                    device.state = state
                    device.last_changed = datetime.now()

                    # Track power
                    power_change += device.power_consumption if state else -device.power_consumption
                    controlled.append(device_name.replace('_', ' '))

        if controlled:
            self.context['power_usage'] += power_change
            action_str = "on" if state else "off"
            response = f"✅ Turned {action_str}: {', '.join(controlled)}"
            if abs(power_change) > 1:
                response += f" (Power {'increased' if power_change > 0 else 'decreased'} by {abs(power_change):.0f}W)"
            return response

        return "No devices were changed"

    def _toggle_devices(self, device_names: List[str]) -> str:
        """Toggles the state of specified devices."""
        if not device_names:
            return "Which device would you like to toggle?"
        
        responses = []
        for device_name in device_names:
            if device_name in self.devices:
                device = self.devices[device_name]
                new_state = not device.state
                self._control_devices([device_name], new_state)
                responses.append(f"{device_name.replace('_', ' ')} is now {'ON' if new_state else 'OFF'}")
        
        return ". ".join(responses) if responses else "Could not find the specified devices."


    def _activate_scene(self, scene_name: str) -> str:
        """Activate a scene (set of device states)"""
        if scene_name not in self.scenes:
            available = ', '.join(self.scenes.keys())
            return f"Unknown scene. Available: {available}"

        scene = self.scenes[scene_name]
        
        for device_name, settings in scene['devices'].items():
            if device_name == 'all_lights':
                # Handle all lights
                state = settings if isinstance(settings, bool) else settings.get('state', False)
                self._control_devices(self.devices_by_type[DeviceType.LIGHT], state)
            elif device_name in self.devices:
                if isinstance(settings, bool):
                    self._control_devices([device_name], settings)
                elif isinstance(settings, dict):
                    state = settings.get('state', True)
                    self._control_devices([device_name], state)
                    if 'dim' in settings and self.devices[device_name].dimmable:
                        self.devices[device_name].dim_level = settings['dim']
                        print(f"💡 Dimming {device_name} to {settings['dim']}%")

        return f"✅ Scene '{scene_name}' activated: {scene['description']}"

    def _execute_multi_action(self, actions: List[Dict]) -> str:
        """Execute multiple device actions"""
        results = []
        
        for action in actions:
            device_name = action.get('device')
            state = action.get('state', False)
            
            if device_name in self.devices:
                self._control_devices([device_name], state)
                results.append(f"{device_name.replace('_', ' ')} {'on' if state else 'off'}")

        if results:
            response = f"✅ Executed: {', '.join(results)}"
            # Power calculation is handled in _control_devices, so we just return the text
            return response

        return "No actions executed"

    def _adjust_temperature(self, value: str, devices: List[str]) -> str:
        """Adjust temperature by controlling AC/fan devices"""
        if value == 'increase':
            self._control_devices(devices, False)  # Turn off AC
            return "🔥 Heating mode - AC turned off"
        else:
            self._control_devices(devices, True)  # Turn on AC/fans
            return "❄️ Cooling mode - AC and fans activated"

    def _get_status_report(self) -> str:
        """Get system status and active devices"""
        active_devices = [name for name, device in self.devices.items() if device.state]

        report = ["🏠 Smart Home Status:\n"]
        report.append(f"⚡ Power usage: {self.context['power_usage']:.0f}W")
        report.append(f"🌡️ Temperature: {self.context['temperature']}°F")
        report.append(f"🕐 Time: {self.context['time_of_day']}")

        if active_devices:
            report.append(f"\n✅ Active devices ({len(active_devices)}):")
            for device_name in active_devices:
                device = self.devices[device_name]
                report.append(f"  • {device_name.replace('_', ' ')} ({device.power_consumption}W)")
        else:
            report.append("\n❌ All devices are OFF")

        return '\n'.join(report)

    def _get_time_context(self) -> str:
        """Get time of day context (morning/afternoon/evening/night)"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _start_background_tasks(self):
        """Start background threads for context updates"""
        # Context updater
        context_thread = threading.Thread(target=self._context_updater, daemon=True)
        context_thread.start()

    def _context_updater(self):
        """Update context periodically (time, power usage)"""
        while True:
            try:
                # Update time
                self.context['time_of_day'] = self._get_time_context()

                # Update power usage
                total_power = sum(d.power_consumption for d in self.devices.values() if d.state)
                self.context['power_usage'] = total_power

                time.sleep(30)
            except Exception:
                # Avoid thread crashing on any error
                pass

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        avg_latency = 0
        if self.performance_stats['inference_times']:
            avg_latency = statistics.mean(self.performance_stats['inference_times']) * 1000

        avg_tps = 0
        if self.performance_stats['tokens_per_second']:
            avg_tps = statistics.mean(self.performance_stats['tokens_per_second'])

        cache_rate = 0
        total_lookups = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
        if total_lookups > 0:
            cache_rate = (self.performance_stats['cache_hits'] / total_lookups) * 100

        return {
            'avg_inference_ms': f"{avg_latency:.0f}",
            'avg_tokens_per_second': f"{avg_tps:.1f}",
            'cache_hit_rate': f"{cache_rate:.1f}%",
            'total_commands': self.performance_stats['total_commands'],
            'current_power': f"{self.context['power_usage']:.0f}W"
        }

    def cleanup(self):
        """Cleanup resources (turn off devices, cleanup GPIO)"""
        print("\n🔧 Shutting down...")
        if not self._gpio_simulation_mode:
            # Turn off all devices for safety
            for device in self.devices.values():
                if device.state:
                    GPIO.output(device.pin, GPIO.LOW)
                    device.state = False
            GPIO.cleanup()
            print("✅ GPIO cleanup complete")

# Web Dashboard HTML (for Flask UI)
DASHBOARD_HTML = """
<!-- HTML omitted for brevity, see original code for full dashboard markup -->
"""

# Create Flask app and endpoints
if Flask is None:
    print("❌ Flask is not installed. The web server cannot start.")
    # Allow script to exit gracefully if run without web dependencies
    assistant = None
else:
    app = Flask(__name__)
    CORS(app)
    assistant = None # Global assistant instance

@app.route('/')
def home():
    """Render the main dashboard"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/command', methods=['POST'])
def api_command():
    """Process commands from the web UI"""
    if not assistant:
        return jsonify({'error': 'Assistant not initialized'}), 500
    
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing command text'}), 400
    
    start_time = time.time()
    action = assistant.understand_command(data['text'])
    response = assistant.execute_command(action)
    elapsed_ms = (time.time() - start_time) * 1000
    
    return jsonify({
        'response': response,
        'elapsed_ms': f"{elapsed_ms:.0f}",
        'action_details': action
    })

@app.route('/api/status')
def api_status():
    """Get the current system status and performance metrics for the new UI"""
    if not assistant:
        return jsonify({'error': 'Assistant not initialized'}), 500
        
    # Organize devices by room for the UI
    rooms = {}
    for room, device_names in assistant.devices_by_room.items():
        rooms[room] = [
            {
                'name': name,
                'state': assistant.devices[name].state,
                'type': assistant.devices[name].device_type.value,
                'power': assistant.devices[name].power_consumption
            }
            for name in device_names
        ]
    
    # Format stats for the UI
    status_data = {
        'rooms': rooms,
        'stats': {
            'power_usage': f"{assistant.context['power_usage']:.0f}W",
            'temperature': f"{assistant.context['temperature']}°F",
            'active_devices': sum(1 for d in assistant.devices.values() if d.state),
            'total_devices': len(assistant.devices)
        },
        'scenes': list(assistant.scenes.keys()),
        'performance': assistant.get_performance_summary()
    }
    return jsonify(status_data)

@app.route('/api/device/toggle', methods=['POST'])
def api_toggle_device():
    """API endpoint to toggle a device from the UI"""
    if not assistant:
        return jsonify({'error': 'Assistant not initialized'}), 500
    
    data = request.json
    device_name = data.get('device')
    if not device_name or device_name not in assistant.devices:
        return jsonify({'error': 'Device not found'}), 404
        
    assistant._toggle_devices([device_name])
    new_state = assistant.devices[device_name].state
    return jsonify({'success': True, 'new_state': new_state})

@app.route('/api/scene/activate', methods=['POST'])
def api_activate_scene():
    """API endpoint to activate a scene from the UI"""
    if not assistant:
        return jsonify({'error': 'Assistant not initialized'}), 500
        
    data = request.json
    scene_name = data.get('scene')
    if not scene_name or scene_name not in assistant.scenes:
        return jsonify({'error': 'Scene not found'}), 404
        
    response = assistant._activate_scene(scene_name)
    return jsonify({'response': response})


def main():
    """Main function to parse arguments and run the assistant"""
    global assistant
    
    parser = argparse.ArgumentParser(description="ONNX Smart Home Assistant for Raspberry Pi")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model",
        help="Path to the directory containing the ONNX model and tokenizer."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the web server on."
    )
    args = parser.parse_args()
    
    # Check if required web libraries are installed before proceeding
    if Flask is None:
        print("Please install web dependencies to run the dashboard: pip install Flask Flask-Cors")
        return

    try:
        # Initialize the assistant
        assistant = ONNXSmartHomeAssistant(model_path=args.model_path)
        
        # Start the web server in a background thread
        api_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=args.port, debug=False),
            daemon=True
        )
        api_thread.start()
        print(f"🚀 Web server started at http://0.0.0.0:{args.port}")
        print("   Open a browser to this address to see the dashboard.")

        print("\n🎤 Smart Home Assistant Ready for Terminal Commands!")
        print("💬 Type a command like 'turn on the lights', 'status', or 'quit'\n")

        # Interactive loop for terminal commands
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            # Process command
            start_time = time.time()
            action = assistant.understand_command(user_input)
            response = assistant.execute_command(action)
            elapsed = (time.time() - start_time) * 1000

            # Print response
            print(f"\n🤖 Assistant: {response}")
            if action and action.get('reasoning'):
                print(f"   💭 Logic: {action['reasoning']}")
            print(f"   ⏱️  Response Time: {elapsed:.0f}ms")

            # Print performance summary
            perf = assistant.get_performance_summary()
            print("\n📊 Performance Summary:")
            for key, value in perf.items():
                print(f"  - {key.replace('_', ' ').title()}: {value}")
            print("-" * 20 + "\n")

    except KeyboardInterrupt:
        print("\nCaught interrupt, shutting down...")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if assistant:
            assistant.cleanup()

if __name__ == '__main__':
    main()
