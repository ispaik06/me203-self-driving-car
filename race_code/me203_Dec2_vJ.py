import RPi.GPIO as GPIO
import time

# PID Gains
Kp = 0.559  # Directly related to error value which is difference between left and right ultrasonic sensor
Ki = 0.0
Kd = 0.39

base_angle = 90
prev_error = 0
integral = 0

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

# Speed
SPEED_MIN = 35
SPEED_MAX = 50
MOTOR_SPEED = SPEED_MIN  # 초기값 (실제 주행 속도는 루프에서 계속 계산)

GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# Distance Clipping values (Values more than 1.5m are all limited to 1.5m)
MIN_CM, MAX_CM = 3.0, 90.0
ALPHA = 0.35

def sample_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.001)
    GPIO.output(trig, False)

    t0 = time.time()
    while GPIO.input(echo) == 0:
        if time.time() - t0 > 0.02:  # Echo pulse is not going (Wiring, system issues)
            return None
    start = time.time()

    while GPIO.input(echo) == 1:
        if time.time() - start > 0.02:  # Echo is not returning (No detections or objects are too far)
            return 8787
    end = time.time()

    dist = (end - start) * 34300 / 2.0  # Measuring distance with sound speed
    dist = max(MIN_CM, min(dist, MAX_CM))  # Clipping distance values
    return dist

def read_stable(trig, echo):
    val = sample_distance(trig, echo)
    time.sleep(0.001)
    return val

def smooth(prev_value, new_value, alpha=ALPHA):
    if new_value == 8787:
        return 150  # All values without echoes including noises are considered as very far
    if new_value is None:
        return prev_value
    if prev_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * prev_value

def set_servo_angle(degree):
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.1)

def move_forward(speed):
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(speed)

def move_backward(speed):
    GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(speed)

def stop_motor():
    motor_pwm.ChangeDutyCycle(0)

def speed_from_angle(angle, amin=45, amid=90, amax=135,
                     vmin=SPEED_MIN, vmax=SPEED_MAX):
    # (현재 코드는 거리 기반 속도를 쓰고 있어서 안 쓰지만, 남겨둠)
    if angle <= amid:
        t = (angle - amin) / (amid - amin)
        t = max(0.0, min(1.0, t))
        if t != 0:
            t = 1 / t * 3
        t = min(15, t)
        return vmin + (vmax - vmin) * t * 0.25
    else:
        t = (amax - angle) / (amax - amid)
        t = max(0.0, min(1.0, t))
        if t != 0:
            t = 1 / t * 3
        t = min(15, t)
        return vmin + (vmax - vmin) * t * 0.25

# === 거리 기반 보정 함수들 ===

def speed_scale_from_distance(d_cm):
    """left/right 중 가까운 쪽 거리로 감속 비율 결정"""
    if d_cm is None:
        return 1.0
    if d_cm <= 1.0:
        return 0.5   # 50% 속도
    elif d_cm <= 2.0:
        return 0.6
    elif d_cm <= 3.0:
        return 0.7
    elif d_cm <= 4.0:
        return 0.8
    elif d_cm <= 5.0:
        return 0.9
    else:
        return 1.0   # 감속 없음

def steering_offset_from_distance(d_cm):
    """각 센서 거리별 추가 스티어링 보정량 (절대값)"""
    if d_cm is None:
        return 0.0
    if d_cm <= 1.0:
        return 25.0
    elif d_cm <= 2.0:
        return 20.0
    elif d_cm <= 3.0:
        return 15.0
    elif d_cm <= 4.0:
        return 10.0
    elif d_cm <= 5.0:
        return 5.0
    else:
        return 0.0

# === 항상 PID 자율주행 모드 ===
try:
    print("PID Autonomous mode activated (no keyboard).")
    prev_error = 0
    integral = 0

    last_left = None
    last_right = None

    # 처음에 정면 각도로 맞추기
    set_servo_angle(base_angle)

    while True:
        # 초음파 센서 읽기
        raw_left  = read_stable(TRIG_LEFT,  ECHO_LEFT)
        raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)

        left  = smooth(last_left,  raw_left)
        right = smooth(last_right, raw_right)

        last_left, last_right = left, right

        if left is None or right is None:
            continue

        # --- PID 기본 스티어링 ---
        error = left - right
        integral += error
        derivative = error - prev_error
        output = Kp * error + Ki * integral + Kd * derivative  # Feedback control of angles

        # PID 기반 기본 조향 각
        angle_pid = max(45, min(135, base_angle - output))

        # --- 거리 기반 추가 스티어링 보정 ---
        # 왼쪽이 더 가까우면 오른쪽으로(+), 오른쪽이 더 가까우면 왼쪽으로(-)
        offset_left = steering_offset_from_distance(left)
        offset_right = steering_offset_from_distance(right)
        angle_offset = offset_left - offset_right

        angle = angle_pid + angle_offset
        angle = max(45, min(135, angle))    # 최종 각도 클리핑
        angle = round(angle, 0)

        # --- 거리 기반 속도 감속 ---
        nearest = min(left, right)
        scale = speed_scale_from_distance(nearest)
        BASE_SPEED = SPEED_MAX  # 기준 속도
        MOTOR_SPEED = BASE_SPEED * scale
        MOTOR_SPEED = max(0.0, min(100.0, MOTOR_SPEED))  # duty cycle 보호

        print(
            f"L: {left:.1f} R: {right:.1f} Err: {error:.1f} "
            f"AnglePID: {angle_pid:.1f} OffL: {offset_left:.1f} OffR: {offset_right:.1f} "
            f"Angle: {angle:.1f} "
            f"Nearest: {nearest:.1f}cm SpeedScale: {scale:.2f} Speed: {MOTOR_SPEED:.0f}"
        )

        # 가까운 벽 강제 회피 로직 (10cm 기준) 유지
        if left <= 10:
            set_servo_angle(120)
        elif right <= 10:
            set_servo_angle(60)
        else:
            set_servo_angle(angle)

        # 거리 기반 감속 속도로 전진
        move_forward(MOTOR_SPEED)

        prev_error = error

except KeyboardInterrupt:
    print("Quit (KeyboardInterrupt).")
finally:
    stop_motor()
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
