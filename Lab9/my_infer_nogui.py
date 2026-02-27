# my_infer_cli.py
#
# TFLite 추론 결과에 따라 모터/서보를 제어하는 코드 (PiCamera2 + CLI 전용)
# - GUI 창 없음 (cv2.imshow, waitKey, destroyAllWindows 제거)
# - 터미널에 로그만 출력, Ctrl+C 로 종료

import time
import sys
import signal

import numpy as np
import cv2
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# -----------------------------
# 모델 / 라벨 설정
# -----------------------------
IMG = 240
MODEL_PATH = "./model.tflite"

# ⚠️ CSV의 label_id 순서와 맞춰야 함
labels = ["Left", "Right", "green", "red", "straight"]

# -----------------------------
# GPIO / 모터 / 서보 설정
# -----------------------------
base_angle = 90  # 직진 기준 각도

DIR_PIN   = 16
PWM_PIN   = 12
SERVO_PIN = 13

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

SPEED_MIN = 10
SPEED_MAX = 15
EXTRA_FULL = 25
EXTRA_POW  = 1.2
EXTRA_DEAD = 0.05

motor_pwm = None
servo_pwm = None


def gpio_init():
    global motor_pwm, servo_pwm

    GPIO.setmode(GPIO.BCM)
    GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)

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


# -----------------------------
# TFLite 전처리 / 추론
# -----------------------------
def preprocess_frame(frame):
    """
    PiCamera2에서 받은 frame(RGB)을 (1,IMG,IMG,3) float32 [0,1]로 변환
    """
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


# -----------------------------
# CNN 결과 → 서보/모터 제어 정책
# -----------------------------
def control_from_label(label):
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

    elif l == "red":
        stop_motor()
        return

    else:
        stop_motor()
        return

    set_servo_angle(angle)
    move_forward(speed)


# -----------------------------
# 메인 루프 (PiCamera2 + CLI)
# -----------------------------
def main():
    def handler(sig, frame):
        print("\n[INFO] Signal received, stopping motor and cleaning up...")
        gpio_cleanup()
        try:
            picam2.stop()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print("[INFO] Initializing GPIO...")
    gpio_init()

    print(f"[INFO] Loading TFLite model from: {MODEL_PATH}")
    interpreter, inp, out = create_interpreter(MODEL_PATH)
    print("[INFO] TFLite model loaded.")

    print("[INFO] Initializing PiCamera2...")
    global picam2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    print("[INFO] Starting real-time inference (CLI). Press Ctrl+C to exit.")
    try:
        while True:
            frame = picam2.capture_array()  # RGB

            x = preprocess_frame(frame)
            label, probs, dt = run_inference(interpreter, inp, out, x)

            print(f"pred={label:9s}  probs={np.round(probs, 3)}  ({dt:.1f} ms)")

            control_from_label(label)

            time.sleep(0.02)

    finally:
        print("[INFO] Stopping...")
        try:
            picam2.stop()
        except Exception:
            pass
        gpio_cleanup()


if __name__ == "__main__":
    main()
