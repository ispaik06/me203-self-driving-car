# my_infer.py
#
# TFLite 추론 결과에 따라 모터/서보를 제어하는 코드
# - 핀/듀티/서보 세팅은 me203_Oct28_v2.py와 동일
# - 현재는 이미지 파일("straight_line.png")을 계속 읽어서 추론
#   → Lab9 실습에서 PiCamera2 프레임으로 교체하면 실시간 주행 가능

import time
import sys
import signal

import numpy as np
import cv2
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite

# -----------------------------
# 모델 / 라벨 설정
# -----------------------------
IMG = 240
MODEL_PATH = "./model.tflite"

# ⚠️ 학습할 때 사용한 CSV의 label_id 순서와 동일해야 함
# 예: 0: forward, 1: green, 2: left, 3: red, 4: right
labels = ["Left", "Right", "green", "red", "straight"]

# -----------------------------
# GPIO / 모터 / 서보 설정 (me203_Oct28_v2.py와 동일)
# -----------------------------
# PID 관련 파라미터는 여기서는 사용하지 않음 (CNN 제어이므로)

base_angle = 90  # 직진 기준 각도

# GPIO 핀
DIR_PIN   = 16
PWM_PIN   = 12
SERVO_PIN = 13

# (초음파 센서는 Lab9 CNN 제어에선 사용하지 않으므로 생략해도 됨)

MOTOR_FREQ = 1000

SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# 속도 관련 (me203 코드 값 재사용)
SPEED_MIN = 35
SPEED_MAX = 70   # 직진 기준 베이스라인
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
    # 서보가 움직일 수 있도록 약간의 시간 (너무 크면 떨림)
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
def preprocess_image(path):
    """
    파일에서 이미지 읽어서 (1,IMG,IMG,3) float32 [0,1]로 반환
    Lab9에서 PiCamera2 프레임으로 바꾸려면
    - path 대신 frame(ndarray)을 받아오는 버전 하나 더 만들면 됨.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
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
    필요에 따라 각도/속도는 실험하며 조정하면 됨.
    """
    # 기본값: 직진
    angle = base_angle
    speed = SPEED_MAX

    if label == "forward":
        # 직선 주행
        angle = base_angle
        speed = SPEED_MAX

    elif label == "left":
        # 좌회전
        angle = 60   # 필요시 70~80 정도로 조정
        speed = speed_from_angle(angle)

    elif label == "right":
        # 우회전
        angle = 120  # 필요시 100~110 정도로 조정
        speed = speed_from_angle(angle)

    elif label == "green":
        # 예: "진행 가능" 신호 → 직진
        angle = base_angle
        speed = SPEED_MAX

    elif label == "red":
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
# 메인 루프
# -----------------------------
def main():
    # 안전한 종료를 위한 signal handler
    def handler(sig, frame):
        print("\n[INFO] Signal received, stopping motor and cleaning up...")
        gpio_cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print("[INFO] Initializing GPIO...")
    gpio_init()

    print(f"[INFO] Loading TFLite model from: {MODEL_PATH}")
    interpreter, inp, out = create_interpreter(MODEL_PATH)
    print("[INFO] TFLite model loaded.")

    # 테스트용 입력 이미지 (Lab9에서는 교재 예시 이미지나 라인 이미지 사용)
    test_image_path = "straight_line.png"

    print("[INFO] Starting inference loop. Press Ctrl+C to exit.")
    try:
        while True:
            # 1) 이미지 전처리
            x = preprocess_image(test_image_path)

            # 2) TFLite 추론
            label, probs, dt = run_inference(interpreter, inp, out, x)

            # 3) 결과 출력
            print(f"pred={label:8s}  probs={np.round(probs, 3)}  ({dt:.1f} ms)")

            # 4) CNN 결과로 모터/서보 제어
            control_from_label(label)

            # 5) (옵션) 디버깅용으로 입력 이미지 띄우기
            show = (x[0] * 255).astype(np.uint8)  # (H,W,3), RGB
            show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
            cv2.imshow("input", show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed - exiting loop.")
                break

            time.sleep(0.05)

    finally:
        cv2.destroyAllWindows()
        gpio_cleanup()


if __name__ == "__main__":
    main()
