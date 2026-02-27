# me203_cnn_pid_combined.py

import time
import sys

import numpy as np
import cv2
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# =====================================
# 공통 설정 (핀/속도/서보/PID 등)
# =====================================

# --- PID Gains (초음파 기반 벽 추종용) ---
Kp = 0.559
Ki = 0.0
Kd = 0.39

base_angle = 90

# GPIO pin locations
DIR_PIN   = 16
PWM_PIN   = 12
SERVO_PIN = 13

TRIG_RIGHT = 17
ECHO_RIGHT = 27
TRIG_LEFT = 5
ECHO_LEFT = 6

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# 속도 관련 (CNN + PID 공통)
SPEED_MIN = 10
SPEED_MAX = 15
EXTRA_FULL = 25      # 조향 각도 클수록 추가로 더 빠르게
EXTRA_POW  = 1.2
EXTRA_DEAD = 0.05    # 거의 직진일 땐 boost 없음

# 초음파 거리 필터링
MIN_CM, MAX_CM = 3.0, 90.0
ALPHA = 0.35

# CNN / TFLite 설정
IMG = 240
MODEL_PATH = "./model.tflite"
labels = ["Left", "Right", "green", "red", "straight"]

# 전역 변수
motor_pwm = None
servo_pwm = None


# =====================================
# GPIO / 모터 / 서보 유틸
# =====================================

def gpio_init():
    global motor_pwm, servo_pwm

    GPIO.setmode(GPIO.BCM)

    GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
    GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
    GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

    motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)

    motor_pwm.start(0)
    servo_pwm.start(0)

    set_servo_angle(base_angle)
    stop_motor()


def gpio_cleanup():
    global motor_pwm, servo_pwm
    try:
        stop_motor()
        if motor_pwm is not None:
            motor_pwm.stop()
        if servo_pwm is not None:
            servo_pwm.stop()
        GPIO.cleanup()
    except Exception:
        pass


def set_servo_angle(degree):
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.05)


def _clamp_duty(d):
    return max(0, min(100, d))


def speed_from_angle(angle,
                     amin=45, amid=90, amax=135,
                     vmin=SPEED_MIN, vmax=SPEED_MAX):
    # 조향 각도가 커질수록 속도 ↑ (라인 트레이싱/ PID 공통 사용)
    s = abs(angle - amid) / float(amax - amid)
    s = max(0.0, min(1.0, s))

    if s <= EXTRA_DEAD:
        boost = 0.0
    else:
        s_eff = (s - EXTRA_DEAD) / (1.0 - EXTRA_DEAD)
        boost = EXTRA_FULL * (s_eff ** EXTRA_POW)

    speed = vmax + boost
    return _clamp_duty(speed)


def move_forward(speed):
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(_clamp_duty(speed))


def move_backward(speed):
    GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(_clamp_duty(speed))


def stop_motor():
    motor_pwm.ChangeDutyCycle(0)


# =====================================
# 초음파 센서 / PID 자율주행 파트
# =====================================

def sample_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.001)
    GPIO.output(trig, False)

    t0 = time.time()
    while GPIO.input(echo) == 0:
        if time.time() - t0 > 0.02:
            return None
    start = time.time()

    while GPIO.input(echo) == 1:
        if time.time() - start > 0.02:
            return 8787
    end = time.time()

    dist = (end - start) * 34300 / 2.0
    dist = max(MIN_CM, min(dist, MAX_CM))
    return dist


def read_stable(trig, echo):
    val = sample_distance(trig, echo)
    time.sleep(0.001)
    return val


def smooth(prev_value, new_value, alpha=ALPHA):
    if new_value == 8787:
        return 150  # 매우 멀리 있다고 가정
    if new_value is None:
        return prev_value
    if prev_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * prev_value


def pid_autonomous_loop():
    """초음파 기반 PID 자율주행 모드 (green 이후 시작)"""
    print("[PID] Autonomous mode activated.")

    prev_error = 0.0
    integral = 0.0

    last_left = None
    last_right = None

    set_servo_angle(base_angle)

    while True:
        raw_left  = read_stable(TRIG_LEFT,  ECHO_LEFT)
        raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)

        left  = smooth(last_left,  raw_left)
        right = smooth(last_right, raw_right)

        last_left, last_right = left, right

        if left is None or right is None:
            continue

        error = left - right
        integral += error
        derivative = error - prev_error
        output = Kp * error + Ki * integral + Kd * derivative

        angle_raw = base_angle - output
        angle = max(45, min(135, angle_raw))

        motor_speed = speed_from_angle(angle)

        print(f"[PID] L: {left:.1f} R: {right:.1f} Err: {error:.1f} "
              f"Angle: {angle:.1f} Speed: {motor_speed:.0f}")

        if left <= 10:
            set_servo_angle(120)
        elif right <= 10:
            set_servo_angle(60)
        else:
            set_servo_angle(round(angle, 0))

        move_forward(motor_speed)

        prev_error = error


# =====================================
# TFLite / CNN 파트 (카메라 라인트레이싱)
# =====================================

def preprocess_frame(frame):
    """
    PiCamera2에서 받은 frame(BGR)을 (1,IMG,IMG,3) float32 [0,1]로 변환
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame, (IMG, IMG), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]


def create_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return interpreter, inp, out


def run_inference(interpreter, inp_detail, out_detail, x):
    interpreter.set_tensor(inp_detail["index"], x)
    t0 = time.time()
    interpreter.invoke()
    dt = (time.time() - t0) * 1e3
    probs = interpreter.get_tensor(out_detail["index"])[0]
    pred_id = int(np.argmax(probs))
    pred_label = labels[pred_id] if 0 <= pred_id < len(labels) else f"id_{pred_id}"
    return pred_label, probs, dt


def control_from_label_lane(label):
    """
    CNN 라벨에 따라 라인트레이싱(초기 단계) 제어
    (red_seen == False일 때만 호출됨)
    """
    l = label.lower()

    angle = base_angle
    speed = SPEED_MAX

    if l == "straight":
        angle = base_angle
        speed = SPEED_MAX

    elif l == "left":
        angle = 60
        speed = speed_from_angle(angle)

    elif l == "right":
        angle = 120
        speed = speed_from_angle(angle)

    elif l == "green":
        angle = base_angle
        speed = SPEED_MAX

    else:
        # red나 이상한 값이면 여기서는 그냥 멈춤
        stop_motor()
        return

    set_servo_angle(angle)
    move_forward(speed)


# =====================================
# 메인 루프: CNN → (red) 정지 → (green) PID 모드 전환
# =====================================

def main():
    picam2 = None
    try:
        print("[INFO] Initializing GPIO...")
        gpio_init()

        print(f"[INFO] Loading TFLite model from: {MODEL_PATH}")
        interpreter, inp, out = create_interpreter(MODEL_PATH)
        print("[INFO] TFLite model loaded.")

        print("[INFO] Initializing PiCamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.5)

        print("[INFO] Starting CNN lane-tracing phase. (Ctrl+C to exit)")
        red_seen = False  # red를 한 번이라도 본 이후인지

        # ---------- CNN 라인트레이싱 + 신호등 모드 전환 ----------
        while True:
            frame = picam2.capture_array()  # BGR

            x = preprocess_frame(frame)
            label, probs, dt = run_inference(interpreter, inp, out, x)

            print(f"[CNN] pred={label:9s}  probs={np.round(probs, 3)}  ({dt:.1f} ms)")

            l = label.lower()

            if not red_seen:
                # 아직 red를 본 적이 없음 → 자유롭게 라인트레이싱 주행
                if l == "red":
                    red_seen = True
                    stop_motor()
                    set_servo_angle(base_angle)
                    print("[CNN] RED detected. Stopping and waiting for GREEN...")
                else:
                    control_from_label_lane(label)
            else:
                # ✅ red 이후 조건:
                #    - green이 아닌 모든 라벨: 항상 완전 정지 (주행 절대 X)
                #    - green이 나오면 PID 모드로 전환
                if l == "green":
                    print("[CNN] GREEN detected after RED. Switching to PID mode...")
                    stop_motor()
                    set_servo_angle(base_angle)
                    break
                else:
                    stop_motor()
                    set_servo_angle(base_angle)

            time.sleep(0.02)

        # ---------- CNN phase 종료 / 카메라 완전 정리 ----------
        print("[INFO] Stopping camera and switching to PID autonomous mode...")
        try:
            picam2.stop()
        except Exception:
            pass
        picam2 = None  # 이후에는 카메라 절대 사용 X

        # ---------- 초음파 기반 PID 자율주행 모드 ----------
        pid_autonomous_loop()

    except KeyboardInterrupt:
        print("\n[INFO] Quit (KeyboardInterrupt).")
    finally:
        print("[INFO] Cleaning up GPIO and camera...")
        try:
            if picam2 is not None:
                picam2.stop()
        except Exception:
            pass
        gpio_cleanup()


if __name__ == "__main__":
    main()
