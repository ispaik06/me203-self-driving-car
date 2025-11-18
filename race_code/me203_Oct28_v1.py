import RPi.GPIO as GPIO
import time
import sys
import tty
import termios
import select

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
MOTOR_SPEED = SPEED_MIN #!


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
    return alpha*new_value + (1-alpha)*prev_value


def set_servo_angle(degree):
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.1) # ?

def move_forward(speed):
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(speed)

def move_backward():
    GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(MOTOR_SPEED)

def stop_motor():
    motor_pwm.ChangeDutyCycle(0)

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def speed_from_angle(angle, amin=45, amid=90, amax=135,
                     vmin=SPEED_MIN, vmax=SPEED_MAX):
    # Dividing cases if it is left or right
    if angle <= amid:
        t = (angle - amin) / (amid - amin)  # Smaller value calculated with bigger steering angle
        t = max(0.0, min(1.0, t))           # Normalization
        if t != 0:
            t = 1 / t * 3                   # New value t which has bigger value with bigger steering angle
        t = min(15, t)                      # Clipped with 15 to limit speed increase
        return vmin + (vmax - vmin) * t * 0.25  # Speed increased with bigger steering angle but tuned to avoid too much increase in speed
    else:  # Same as above except steering rotation
        t = (amax - angle) / (amax - amid)
        t = max(0.0, min(1.0, t))
        if t != 0:
            t = 1 / t * 3
        t = min(15, t)
        return vmin + (vmax - vmin) * t * 0.25


# Code Running
try:
    print("Press 'a' to enter PID autonomous mode, 'q' to quit.")
    while True:
        key = get_key()

        if key == 'a':
            print("PID Autonomous mode activated.")
            prev_error = 0
            integral = 0

            last_left = None
            last_right = None
            for _ in range(100000):

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
                output = Kp*error + Ki*integral + Kd*derivative  # Feedback control of angles with left and right ultrasonic sensor
                angle = max(45, min(135, base_angle - output))
                MOTOR_SPEED = speed_from_angle(angle)  # New speed of motor considering steering

                print(f"L: {left:.1f} R: {right:.1f} Err: {error:.1f} "
                      f"Angle: {angle:.1f} Speed: {MOTOR_SPEED:.0f}")

                angle1 = max(50, min(130, base_angle - output))  # Clipping angle value to avoid too much steering
                angle = round(angle1, 0)

                # Rule-based logics: if get close to wall, get far away (current threshold = 10cm)
                if left <= 10:
                    set_servo_angle(120)
                elif right <= 10:
                    set_servo_angle(60)
                else:
                    set_servo_angle(angle)

                move_forward(MOTOR_SPEED)  # Speed changes in related to steering angle
                # time.sleep(0.0001)

                prev_error = error

        elif key == 'r':
            set_servo_angle(135)
            MOTOR_SPEED = SPEED_MIN
            move_forward()
        elif key == 'l':
            set_servo_angle(45)
            MOTOR_SPEED = SPEED_MIN
            move_forward()
        elif key == 'f':
            set_servo_angle(90)
            MOTOR_SPEED = SPEED_MAX
            move_forward()
        elif key == 'b':
            set_servo_angle(90)
            MOTOR_SPEED = SPEED_MAX
            move_backward()
        elif key == 'u':
            for _ in range(10000):
                raw_left  = read_stable(TRIG_LEFT,  ECHO_LEFT)
                raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)

                left  = smooth(last_left,  raw_left)
                right = smooth(last_right, raw_right)
                
                last_left = None
                last_right = None
                
                last_left, last_right = left, right
                
                print(left, right)
        elif key == 's':
            stop_motor()
        elif key == 'q':
            print("Quit.")
            break
        else:
            print(f"Unknown key: {key}")

finally:
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()