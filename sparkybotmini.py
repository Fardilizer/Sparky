#!/usr/bin/env python3
# coding: utf-8
"""
SparkyBotMini Controller Library
Thread-safe serial communication library for controlling SparkyBotMini robot platforms
"""

import struct
import time
import serial
import threading
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import IntEnum


# ===== Protocol Constants =====
class Protocol:
    HEAD = 0xFF
    DEVICE_ID = 0xFC
    COMPLEMENT = 257 - DEVICE_ID
    CAR_ADJUST = 0x80


class FunctionCode(IntEnum):
    """Robot function codes"""
    AUTO_REPORT = 0x01
    BEEP = 0x02
    PWM_SERVO = 0x03
    PWM_SERVO_ALL = 0x04
    RGB = 0x05
    RGB_EFFECT = 0x06
    REPORT_SPEED = 0x0A
    REPORT_IMU_RAW = 0x0B
    REPORT_IMU_ATT = 0x0C
    REPORT_ENCODER = 0x0D
    MOTOR = 0x10
    CAR_RUN = 0x11
    MOTION = 0x12
    SET_MOTOR_PID = 0x13
    SET_YAW_PID = 0x14
    SET_CAR_TYPE = 0x15
    UART_SERVO = 0x20
    UART_SERVO_ID = 0x21
    UART_SERVO_TORQUE = 0x22
    ARM_CTRL = 0x23
    ARM_OFFSET = 0x24
    AKM_DEF_ANGLE = 0x30
    AKM_STEER_ANGLE = 0x31
    REQUEST_DATA = 0x50
    VERSION = 0x51
    RESET_FLASH = 0xA0


class CarType(IntEnum):
    """Supported car types"""
    X3 = 0x01
    X3_PLUS = 0x02
    X1 = 0x04
    R2 = 0x05


# ===== Conversion Constants =====
class Conversion:
    GYRO_RATIO = 1 / 3754.9
    ACCEL_RATIO = 1 / 1671.84
    MAG_RATIO = 1
    SERVO_PULSE_SCALE = 22.2
    SERVO_PULSE_OFFSET = 500
    RAD_TO_DEG = 57.2957795
    VOLTAGE_SCALE = 10.0
    SPEED_SCALE = 1000.0
    ATTITUDE_SCALE = 10000.0


@dataclass
class IMUData:
    """IMU sensor data"""
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class RobotState:
    """Complete robot state"""
    imu: IMUData = None
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    battery_voltage: int = 0
    encoder_m1: int = 0
    encoder_m2: int = 0
    encoder_m3: int = 0
    encoder_m4: int = 0
    version_h: int = 0
    version_l: int = 0
    
    def __post_init__(self):
        if self.imu is None:
            self.imu = IMUData()


class SparkyBotMini:
    """Main robot controller class with thread-safe operations"""
    
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, 
                 delay: float = 0.002, debug: bool = False):
        """
        Initialize SparkyBotMini controller
        
        Args:
            port: Serial port path
            baudrate: Communication baudrate
            delay: Command delay between transmissions
            debug: Enable debug output
        """
        self.port = port
        self.baudrate = baudrate
        self.delay = delay
        self.debug = debug
        
        # Serial connection
        self.ser: Optional[serial.Serial] = None
        
        # Thread control
        self._receive_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._state_lock = threading.Lock()
        
        # Robot state
        self._state = RobotState()
        
        # Servo control
        self._arm_ctrl_enabled = True
        self._read_servo_id = 0
        self._read_servo_value = 0
        self._servo_read_event = threading.Event()
        
        # Arm array read
        self._read_arm = [-1] * 6
        self._arm_read_event = threading.Event()
        
        # Offset calibration
        self._arm_offset_id = 0
        self._arm_offset_state = 0
        self._offset_event = threading.Event()
        
        # Version query
        self._version_event = threading.Event()
        
        # AKM servo
        self._akm_servo_id = 0x01
        self._akm_def_angle = 100
        self._akm_event = threading.Event()
        
        # PID parameters
        self._pid_index = 0
        self._kp = 0
        self._ki = 0
        self._kd = 0
    
    # ===== Connection Management =====
    
    def connect(self) -> bool:
        """
        Open serial connection and start receive thread
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            if self.ser.is_open:
                print(f"? SparkyBotMini Serial Opened! Port={self.port}, Baudrate={self.baudrate}")
                self._start_receive_thread()
                time.sleep(self.delay)
                return True
            else:
                print("? Serial Open Failed!")
                return False
        except Exception as e:
            print(f"? Connection Error: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection and stop receive thread"""
        self._stop_receive_thread()
        
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("? Serial port closed")
        except Exception as e:
            print(f"? Error closing serial: {e}")
    
    def _start_receive_thread(self):
        """Start background receive thread"""
        if not self._running.is_set():
            self._running.set()
            self._receive_thread = threading.Thread(
                target=self._receive_loop,
                name="SparkyBotMiniReceiver",
                daemon=True
            )
            self._receive_thread.start()
            if self.debug:
                print("? Receive thread started")
    
    def _stop_receive_thread(self):
        """Stop background receive thread"""
        if self._running.is_set():
            self._running.clear()
            if self._receive_thread:
                self._receive_thread.join(timeout=2)
                if self.debug:
                    print("? Receive thread stopped")
    
    # ===== Low-level Communication =====
    
    def _send_command(self, function: int, *data_bytes) -> bool:
        """
        Send command packet to robot
        
        Args:
            function: Function code
            *data_bytes: Variable data bytes
            
        Returns:
            True if sent successfully
        """
        try:
            cmd = [Protocol.HEAD, Protocol.DEVICE_ID, 0, function] + list(data_bytes)
            cmd[2] = len(cmd) - 1  # Length field
            checksum = (sum(cmd) + Protocol.COMPLEMENT) & 0xFF
            cmd.append(checksum)
            
            self.ser.write(bytes(cmd))
            
            if self.debug:
                print(f"TX [{function:02X}]: {cmd}")
            
            time.sleep(self.delay)
            return True
            
        except Exception as e:
            print(f"? Send error: {e}")
            return False
    
    def _receive_loop(self):
        """Background thread: read and parse incoming packets"""
        if self.debug:
            print("? Receive loop started")
        
        try:
            while self._running.is_set():
                try:
                    # Read header
                    raw = self.ser.read(1)
                    if not raw or len(raw) == 0:
                        continue
                    
                    if raw[0] != Protocol.HEAD:
                        continue
                    
                    # Read device ID
                    raw = self.ser.read(1)
                    if not raw or raw[0] != Protocol.DEVICE_ID - 1:
                        continue
                    
                    # Read length
                    raw = self.ser.read(1)
                    if not raw:
                        continue
                    ext_len = raw[0]
                    
                    # Read function code
                    raw = self.ser.read(1)
                    if not raw:
                        continue
                    ext_type = raw[0]
                    
                    # Read data
                    data_len = ext_len - 2
                    ext_data = self.ser.read(data_len)
                    
                    if len(ext_data) != data_len:
                        continue
                    
                    # Verify checksum
                    checksum = ext_len + ext_type + sum(ext_data[:-1])
                    if (checksum & 0xFF) == ext_data[-1]:
                        self._parse_packet(ext_type, ext_data[:-1])
                    elif self.debug:
                        print(f"? Checksum error: type={ext_type:02X}")
                
                except serial.SerialException as e:
                    if self._running.is_set():
                        print(f"? Serial exception: {e}")
                    break
                except Exception as e:
                    if self.debug:
                        print(f"? Parse error: {e}")
        
        finally:
            if self.debug:
                print("? Receive loop exited")
    
    def _parse_packet(self, func_code: int, data: bytes):
        """Parse received packet and update state"""
        try:
            if func_code == FunctionCode.REPORT_SPEED:
                with self._state_lock:
                    self._state.vx = struct.unpack('h', data[0:2])[0] / Conversion.SPEED_SCALE
                    self._state.vy = struct.unpack('h', data[2:4])[0] / Conversion.SPEED_SCALE
                    self._state.vz = struct.unpack('h', data[4:6])[0] / Conversion.SPEED_SCALE
                    self._state.battery_voltage = data[6]
            
            elif func_code == FunctionCode.REPORT_IMU_RAW:
                with self._state_lock:
                    self._state.imu.gx = struct.unpack('h', data[0:2])[0] * Conversion.GYRO_RATIO
                    self._state.imu.gy = struct.unpack('h', data[2:4])[0] * -Conversion.GYRO_RATIO
                    self._state.imu.gz = struct.unpack('h', data[4:6])[0] * -Conversion.GYRO_RATIO
                    self._state.imu.ax = struct.unpack('h', data[6:8])[0] * Conversion.ACCEL_RATIO
                    self._state.imu.ay = struct.unpack('h', data[8:10])[0] * Conversion.ACCEL_RATIO
                    self._state.imu.az = struct.unpack('h', data[10:12])[0] * Conversion.ACCEL_RATIO
                    self._state.imu.mx = struct.unpack('h', data[12:14])[0] * Conversion.MAG_RATIO
                    self._state.imu.my = struct.unpack('h', data[14:16])[0] * Conversion.MAG_RATIO
                    self._state.imu.mz = struct.unpack('h', data[16:18])[0] * Conversion.MAG_RATIO
            
            elif func_code == FunctionCode.REPORT_IMU_ATT:
                with self._state_lock:
                    self._state.imu.roll = struct.unpack('h', data[0:2])[0] / Conversion.ATTITUDE_SCALE
                    self._state.imu.pitch = struct.unpack('h', data[2:4])[0] / Conversion.ATTITUDE_SCALE
                    self._state.imu.yaw = struct.unpack('h', data[4:6])[0] / Conversion.ATTITUDE_SCALE
            
            elif func_code == FunctionCode.REPORT_ENCODER:
                with self._state_lock:
                    self._state.encoder_m1 = struct.unpack('i', data[0:4])[0]
                    self._state.encoder_m2 = struct.unpack('i', data[4:8])[0]
                    self._state.encoder_m3 = struct.unpack('i', data[8:12])[0]
                    self._state.encoder_m4 = struct.unpack('i', data[12:16])[0]
            
            elif func_code == FunctionCode.UART_SERVO:
                self._read_servo_id = data[0]
                self._read_servo_value = struct.unpack('h', data[1:3])[0]
                self._servo_read_event.set()
                if self.debug:
                    print(f"? Servo[{self._read_servo_id}] = {self._read_servo_value}")
            
            elif func_code == FunctionCode.ARM_CTRL:
                for i in range(6):
                    self._read_arm[i] = struct.unpack('h', data[2*i:2*i+2])[0]
                self._arm_read_event.set()
                if self.debug:
                    print(f"? Arm: {self._read_arm}")
            
            elif func_code == FunctionCode.VERSION:
                with self._state_lock:
                    self._state.version_h = data[0]
                    self._state.version_l = data[1]
                self._version_event.set()
                if self.debug:
                    print(f"? Version: {self._state.version_h}.{self._state.version_l}")
            
            elif func_code == FunctionCode.SET_MOTOR_PID:
                self._pid_index = data[0]
                self._kp = struct.unpack('h', data[1:3])[0]
                self._ki = struct.unpack('h', data[3:5])[0]
                self._kd = struct.unpack('h', data[5:7])[0]
                if self.debug:
                    print(f"? Motor PID[{self._pid_index}]: Kp={self._kp}, Ki={self._ki}, Kd={self._kd}")
            
            elif func_code == FunctionCode.SET_YAW_PID:
                self._pid_index = data[0]
                self._kp = struct.unpack('h', data[1:3])[0]
                self._ki = struct.unpack('h', data[3:5])[0]
                self._kd = struct.unpack('h', data[5:7])[0]
                if self.debug:
                    print(f"? Yaw PID[{self._pid_index}]: Kp={self._kp}, Ki={self._ki}, Kd={self._kd}")
            
            elif func_code == FunctionCode.ARM_OFFSET:
                self._arm_offset_id = data[0]
                self._arm_offset_state = data[1]
                self._offset_event.set()
                if self.debug:
                    print(f"? Offset[{self._arm_offset_id}]: state={self._arm_offset_state}")
            
            elif func_code == FunctionCode.AKM_DEF_ANGLE:
                self._akm_servo_id = data[0]
                self._akm_def_angle = data[1]
                self._akm_event.set()
                if self.debug:
                    print(f"? AKM[{self._akm_servo_id}]: angle={self._akm_def_angle}")
        
        except Exception as e:
            if self.debug:
                print(f"? Parse packet error [{func_code:02X}]: {e}")
    
    def _request_data(self, function: int, param: int = 0):
        """Request data from robot"""
        self._send_command(FunctionCode.REQUEST_DATA, function & 0xFF, param & 0xFF)
    
    # ===== Auto Report =====
    
    def set_auto_report(self, enable: bool, forever: bool = False):
        """
        Enable/disable automatic sensor data reporting
        
        Args:
            enable: True to enable, False to disable
            forever: True to persist setting in flash
        """
        state1 = 1 if enable else 0
        state2 = 0x5F if forever else 0
        self._send_command(FunctionCode.AUTO_REPORT, state1, state2)
    
    def clear_sensor_data(self):
        """Reset all sensor data to zero"""
        with self._state_lock:
            self._state = RobotState()
    
    # ===== Motor Control =====
    
    @staticmethod
    def _limit_motor(value: int) -> int:
        """Limit motor value to valid range"""
        if value == 127:
            return 127  # Emergency stop code
        return max(-100, min(100, int(value)))
    
    def set_motor(self, m1: int, m2: int, m3: int, m4: int):
        """
        Set DC motor speeds
        
        Args:
            m1-m4: Motor speeds (-100 to 100, or 127 for emergency stop)
        """
        speeds = [self._limit_motor(v) for v in (m1, m2, m3, m4)]
        packed = [struct.pack('b', s)[0] for s in speeds]
        self._send_command(FunctionCode.MOTOR, *packed)
    
    # ===== PWM Servo Control =====
    
    def set_pwm_servo(self, servo_id: int, angle: int):
        """
        Set single PWM servo angle
        
        Args:
            servo_id: Servo ID (1-4)
            angle: Angle in degrees (0-180)
        """
        if not 1 <= servo_id <= 4:
            print(f"? Invalid servo ID: {servo_id}")
            return
        
        angle = max(0, min(180, int(angle)))
        self._send_command(FunctionCode.PWM_SERVO, servo_id, angle)
    
    def set_pwm_servo_all(self, a1: int, a2: int, a3: int, a4: int):
        """
        Set all 4 PWM servos simultaneously
        
        Args:
            a1-a4: Angles in degrees (0-180, or 255 to skip)
        """
        angles = [a1, a2, a3, a4]
        for i in range(4):
            if not 0 <= angles[i] <= 180:
                angles[i] = 255  # Skip this servo
        
        self._send_command(FunctionCode.PWM_SERVO_ALL, *map(int, angles))
    
    # ===== UART Servo Control =====
    
    @staticmethod
    def _angle_to_pulse(angle: float) -> int:
        """Convert angle to servo pulse value"""
        return int(angle * Conversion.SERVO_PULSE_SCALE + Conversion.SERVO_PULSE_OFFSET)
    
    @staticmethod
    def _pulse_to_angle(pulse: int) -> float:
        """Convert servo pulse to angle"""
        return (pulse - Conversion.SERVO_PULSE_OFFSET) / Conversion.SERVO_PULSE_SCALE
    
    def set_uart_servo_pulse(self, servo_id: int, pulse: int, run_time: int = 500):
        """
        Set UART servo by pulse value (low-level)
        
        Args:
            servo_id: Servo ID (1-250)
            pulse: Pulse width (0-4000)
            run_time: Movement time in ms (0-2000)
        """
        if not self._arm_ctrl_enabled:
            return
        
        if not 1 <= servo_id <= 250:
            print(f"? Invalid servo ID: {servo_id}")
            return
        
        pulse = max(0, min(4000, pulse))
        run_time = max(0, min(2000, run_time))
        
        pulse_bytes = struct.pack('h', pulse)
        time_bytes = struct.pack('h', run_time)
        
        self._send_command(FunctionCode.UART_SERVO, 
                          servo_id, 
                          pulse_bytes[0], pulse_bytes[1],
                          time_bytes[0], time_bytes[1])
    
    def set_uart_servo_angle(self, servo_id: int, angle: float, run_time: int = 500):
        """
        Set UART servo by angle
        
        Args:
            servo_id: Servo ID (1-250)
            angle: Angle in degrees
            run_time: Movement time in ms (0-2000)
        """
        pulse = self._angle_to_pulse(angle)
        self.set_uart_servo_pulse(servo_id, pulse, run_time)
    
    def set_uart_servo_id(self, new_id: int):
        """
        Change UART servo ID
        
        Args:
            new_id: New servo ID (1-250)
        """
        if not 1 <= new_id <= 250:
            print(f"? Invalid servo ID: {new_id}")
            return
        
        self._send_command(FunctionCode.UART_SERVO_ID, new_id)
    
    def set_uart_servo_torque(self, enable: bool):
        """
        Enable/disable UART servo torque
        
        Args:
            enable: True to enable torque, False to disable
        """
        self._send_command(FunctionCode.UART_SERVO_TORQUE, 1 if enable else 0)
    
    def set_arm_control_enabled(self, enable: bool):
        """
        Enable/disable arm control commands
        
        Args:
            enable: True to enable, False to disable
        """
        self._arm_ctrl_enabled = enable
    
    def set_arm_angles(self, angles: List[float], run_time: int = 500):
        """
        Set all 6 arm servos simultaneously
        
        Args:
            angles: List of 6 angles in degrees
            run_time: Movement time in ms (0-2000)
        """
        if not self._arm_ctrl_enabled:
            return
        
        if len(angles) != 6:
            print(f"? Expected 6 angles, got {len(angles)}")
            return
        
        # Convert angles to pulses
        pulses = [self._angle_to_pulse(a) for a in angles]
        pulse_bytes = [struct.pack('h', p) for p in pulses]
        time_bytes = struct.pack('h', run_time)
        
        # Flatten pulse bytes
        data = [b for pulse in pulse_bytes for b in pulse]
        data.extend(time_bytes)
        
        self._send_command(FunctionCode.ARM_CTRL, *data)
    
    def set_uart_servo_offset(self, servo_id: int, timeout: float = 0.2) -> Optional[int]:
        """
        Calibrate UART servo zero offset
        
        Args:
            servo_id: Servo ID to calibrate
            timeout: Timeout in seconds
            
        Returns:
            Calibration state, or None on timeout
        """
        self._offset_event.clear()
        self._send_command(FunctionCode.ARM_OFFSET, servo_id)
        
        if self._offset_event.wait(timeout):
            if self._arm_offset_id == servo_id:
                return self._arm_offset_state
        
        return None
    
    # ===== LED Control =====
    
    def set_led(self, led_id: int, red: int, green: int, blue: int):
        """
        Set RGB LED color
        
        Args:
            led_id: LED ID
            red, green, blue: Color values (0-255)
        """
        self._send_command(FunctionCode.RGB,
                          led_id & 0xFF,
                          red & 0xFF,
                          green & 0xFF,
                          blue & 0xFF)
    
    def set_led_effect(self, effect: int, speed: int = 255, param: int = 255):
        """
        Start LED effect pattern
        
        Args:
            effect: Effect type
            speed: Effect speed (0-255)
            param: Effect parameter (0-255)
        """
        self._send_command(FunctionCode.RGB_EFFECT,
                          effect & 0xFF,
                          speed & 0xFF,
                          param & 0xFF)
    
    # ===== Audio =====
    
    def beep(self, duration_ms: int):
        """
        Trigger beep
        
        Args:
            duration_ms: Beep duration in milliseconds
        """
        if duration_ms < 0:
            print("? Invalid beep duration")
            return
        
        duration_bytes = struct.pack('h', int(duration_ms))
        self._send_command(FunctionCode.BEEP, duration_bytes[0], duration_bytes[1])
    
    # ===== Sensor Reading =====
    
    def get_imu_data(self) -> IMUData:
        """Get current IMU data (thread-safe)"""
        with self._state_lock:
            return IMUData(
                ax=self._state.imu.ax, ay=self._state.imu.ay, az=self._state.imu.az,
                gx=self._state.imu.gx, gy=self._state.imu.gy, gz=self._state.imu.gz,
                mx=self._state.imu.mx, my=self._state.imu.my, mz=self._state.imu.mz,
                roll=self._state.imu.roll, pitch=self._state.imu.pitch, yaw=self._state.imu.yaw
            )
    
    def get_attitude(self, degrees: bool = True) -> Tuple[float, float, float]:
        """
        Get roll, pitch, yaw
        
        Args:
            degrees: True for degrees, False for radians
            
        Returns:
            (roll, pitch, yaw) tuple
        """
        with self._state_lock:
            roll = self._state.imu.roll
            pitch = self._state.imu.pitch
            yaw = self._state.imu.yaw
        
        if degrees:
            return (roll * Conversion.RAD_TO_DEG,
                   pitch * Conversion.RAD_TO_DEG,
                   yaw * Conversion.RAD_TO_DEG)
        return (roll, pitch, yaw)
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Get velocity (vx, vy, vz) in m/s"""
        with self._state_lock:
            return (self._state.vx, self._state.vy, self._state.vz)
    
    def get_battery_voltage(self) -> float:
        """Get battery voltage in volts"""
        with self._state_lock:
            return self._state.battery_voltage / Conversion.VOLTAGE_SCALE
    
    def get_encoders(self) -> Tuple[int, int, int, int]:
        """Get motor encoder counts (m1, m2, m3, m4)"""
        with self._state_lock:
            return (self._state.encoder_m1, self._state.encoder_m2,
                   self._state.encoder_m3, self._state.encoder_m4)
    
    def get_uart_servo_value(self, servo_id: int, timeout: float = 0.03) -> Tuple[int, int]:
        """
        Read UART servo current value
        
        Args:
            servo_id: Servo ID to read
            timeout: Timeout in seconds
            
        Returns:
            (servo_id, pulse_value) tuple, or (-1, -1) on timeout
        """
        self._servo_read_event.clear()
        self._request_data(FunctionCode.UART_SERVO, servo_id)
        
        if self._servo_read_event.wait(timeout):
            return (self._read_servo_id, self._read_servo_value)
        
        return (-1, -1)
    
    def get_uart_servo_angle(self, servo_id: int, timeout: float = 0.03) -> float:
        """
        Read UART servo current angle
        
        Args:
            servo_id: Servo ID to read
            timeout: Timeout in seconds
            
        Returns:
            Current angle in degrees, or -1 on error
        """
        sid, pulse = self.get_uart_servo_value(servo_id, timeout)
        if sid < 0:
            return -1.0
        return round(self._pulse_to_angle(pulse), 1)
    
    def get_arm_angles(self, timeout: float = 0.03) -> List[float]:
        """
        Read all 6 arm servo angles
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            List of 6 angles, or [-1]*6 on timeout
        """
        self._arm_read_event.clear()
        self._request_data(FunctionCode.ARM_CTRL, 1)
        
        if self._arm_read_event.wait(timeout):
            return [round(self._pulse_to_angle(p), 1) for p in self._read_arm]
        
        return [-1.0] * 6
    
    def get_version(self, timeout: float = 0.02) -> float:
        """
        Get firmware version
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Version number (e.g., 2.1), or -1 on timeout
        """
        
        with self._state_lock:
            if self._state.version_h > 0:
                return self._state.version_h + self._state.version_l / 10.0
        
        self._version_event.clear()
        self._request_data(FunctionCode.VERSION)
        
        if self._version_event.wait(timeout):
            with self._state_lock:
                return self._state.version_h + self._state.version_l / 10.0
        
        return -1.0
    
    # ===== System Control =====
    
    def reset_flash(self):
        """Factory reset all flash parameters"""
        self._send_command(FunctionCode.RESET_FLASH, 0x5F)


# ===== Example Usage =====

if __name__ == "__main__":
    # Create robot instance
    robot = SparkyBotMini(port="/dev/ttyUSB0", debug=True)
    
    try:
        # Connect
        if not robot.connect():
            exit(1)
        
        # Enable auto-reporting
        robot.set_auto_report(True)
        time.sleep(0.5)
        
        # Get version
        version = robot.get_version()
        print(f"\n? SparkyBotMini Firmware: v{version}\n")
        
        # Read sensors
        roll, pitch, yaw = robot.get_attitude(degrees=True)
        print(f"? Attitude: Roll={roll:.1f}° Pitch={pitch:.1f}° Yaw={yaw:.1f}°")
        
        voltage = robot.get_battery_voltage()
        print(f"? Battery: {voltage:.1f}V")
        
        encoders = robot.get_encoders()
        print(f"??  Encoders: {encoders}")
        
        # Test beep
        print("\n? Beeping...")
        robot.beep(200)
        time.sleep(0.5)
        
        # Test LED
        print("? LED Rainbow...")
        robot.set_led(1, 255, 0, 0)  # Red
        time.sleep(0.3)
        robot.set_led(1, 0, 255, 0)  # Green
        time.sleep(0.3)
        robot.set_led(1, 0, 0, 255)  # Blue
        time.sleep(0.3)
        
        # Test motor (brief movement)
        print("? Motor test...")
        robot.set_motor(30, 30, 30, 30)
        time.sleep(0.5)
        robot.set_motor(0, 0, 0, 0)
        
        print("\n? All tests passed!")
        
    except KeyboardInterrupt:
        print("\n\n??  Interrupted by user")
    
    finally:
        # Cleanup
        robot.set_motor(0, 0, 0, 0)  # Stop motors
        robot.disconnect()
        print("? Goodbye!")
