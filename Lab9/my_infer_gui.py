#!/usr/bin/env python3
# my_infer.py
#
# TFLite 추론 결과에 따라 모터/서보를 제어하는 코드 (PiCamera2 실시간 버전)
# - 핀/듀티/서보 세팅은 me203_Oct28_v2.py와 동일
# - PiCamera2에서 RGB 프레임을 받아와서 실시간 추론 + 모터 제어

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

# ⚠️ 학습할 때 사용한 CSV의 label_id 순서와 동일해야 함
# 예: 0: Left, 1: Right, 2: green, 3: red, 4: straight
labels = ["Left", "Right", "green", "red", "straight"]

# -----------------------------
# GPIO / 모터 / 서보 설정 (me203_Oct28_v2.py와 동일)
# -----------------------------
base_angle = 90  # 직진 기준 각도

# GPIO 핀
DIR_PIN   = 16
PWM_PIN   = 12
SERVO_PIN = 13

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# 속도 관련 (me203 코드 값 재사용)
SPEED_MIN = 10
SPEED_MAX = 15   # 직진 기준 베이스라인


EXTRA_FULL = 25  # 최대 조향 시 추가 듀티
EXTRA_POW  = 1.2 # 스티어링에 따른 boost 곡선
EXTRA_DEAD = 0.05

# 전역 PWM 객체
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

    # 초기에 직진 각도로 맞추고 정지
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
    """
    서보 각도 → 듀티로 변환 (me203 코드와 동일 공식)
    """
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    # 서보가 움직일 수 있도록 약간의 시간
    time.sleep(0.05)


def _clamp_duty(d):
    return max(0, min(100, d))


def speed_from_angle(angle,
                     amin=45, amid=90, amax=135,
                     vmin=SPEED_MIN, vmax=SPEED_MAX):
    """
    me203_Oct28_v2.py의 4WD policy 재사용:
    - 조향량 |angle-amid|가 커질수록 throttle 증가
    - straight 근처(EXTRA_DEAD 이하)는 boost 없음
    """
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
    PiCamera2에서 받은 frame(ndarray, RGB888)을
    (1,IMG,IMG,3) float32 [0,1] 로 변환
    """
    # frame: (H,W,3), RGB
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame, (IMG, IMG), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]  # (1,H,W,3)


def create_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return interpreter, inp, out


def run_inference(interpreter, inp_detail, out_detail, x):
    """
    x: (1,IMG,IMG,3) float32
    반환: (pred_label(str), probs(ndarray), dt_ms(float))
    """
    interpreter.set_tensor(inp_detail["index"], x)

    t0 = time.time()
    interpreter.invoke()
    dt = (time.time() - t0) * 1e3

    probs = interpreter.get_tensor(out_detail["index"])[0]  # (num_classes,)
    pred_id = int(np.argmax(probs))
    pred_label = labels[pred_id] if 0 <= pred_id < len(labels) else f"id_{pred_id}"
    return pred_label, probs, dt


# -----------------------------
# CNN 결과 → 서보/모터 제어 정책
# -----------------------------
def control_from_label(label):
    """
    CNN이 예측한 label에 따라 서보 각도와 속도를 결정하고 모터 제어.
    label 케이스(대소문자)는 모두 lower()로 처리.
    """
    l = label.lower()

    # 기본값: 직진
    angle = base_angle
    speed = SPEED_MAX

    if l == "straight":
        # 직선 주행
        angle = base_angle
        speed = SPEED_MAX

    elif l == "left":
        # 좌회전
        angle = 65   # 필요시 70~80 정도로 조정
        speed = speed_from_angle(angle)

    elif l == "right":
        # 우회전
        angle = 115  # 필요시 100~110 정도로 조정
        speed = speed_from_angle(angle)

    elif l == "green":
        # 예: "진행 가능" 신호 → 직진
        angle = base_angle
        speed = SPEED_MAX

    elif l == "red":
        # 정지 신호로 사용
        stop_motor()
        return

    else:
        # 알 수 없는 라벨이면 일단 정지 (안전)
        stop_motor()
        return

    set_servo_angle(angle)
    move_forward(speed)


# -----------------------------
# 메인 루프 (PiCamera2 실시간)
# -----------------------------
def main():
    # 안전한 종료를 위한 signal handler
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
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)  # 카메라 워밍업

    print("[INFO] Starting real-time inference. Press Ctrl+C in terminal to exit.")
    try:
        while True:
            # 1) 카메라에서 프레임 캡처 (RGB)
            frame = picam2.capture_array()  # (H,W,3), RGB888

            # 2) 전처리
            x = preprocess_frame(frame)

            # 3) TFLite 추론
            label, probs, dt = run_inference(interpreter, inp, out, x)

            # 4) 결과 출력
            print(f"pred={label:9s}  probs={np.round(probs, 3)}  ({dt:.1f} ms)")

            # 5) CNN 결과로 모터/서보 제어
            control_from_label(label)

            # 6) (옵션) 디버깅용으로 화면에 표시
            #    SSH만 쓰면 이 부분은 주석 처리하는 게 낫습니다.
            view = frame.copy()  # RGB
            # view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)  # OpenCV BGR
            cv2.putText(
                view,
                f"{label} ({dt:.1f} ms)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("camera", view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed in window - exiting loop.")
                break

            time.sleep(0.02)  # 너무 빠르게 돌지 않게 약간 쉼

    finally:
        print("[INFO] Stopping...")
        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        gpio_cleanup()


if __name__ == "__main__":
    main()
