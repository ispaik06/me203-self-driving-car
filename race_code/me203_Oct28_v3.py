import RPi.GPIO as GPIO
import time
import sys
import tty
import termios
import select

# PID Gains
Kp = 0.587  # Directly related to error value which is difference between left and right ultrasonic sensor
Ki = 0.0
Kd = 0.536

# Distance Clipping values (Values more than 1.5m are all limited to 1.5m)
MIN_CM, MAX_CM = 3.0, 80.0
ALPHA = 0.555 #! 

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
SPEED_MAX = 80   # straight-line baseline
EXTRA_FULL = 25  # max extra duty at full lock (s=1)
EXTRA_POW  = 1.2 # shaping exponent (>1: late surge, <1: early surge)
EXTRA_DEAD = 0.05 # below this steer magnitude, no extra
MOTOR_SPEED = SPEED_MAX  # start at baseline maximum


GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)



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


def _clamp_duty(d):
    return max(0, min(100, d))

def speed_from_angle(angle, amin=45, amid=90, amax=135,
                     vmin=SPEED_MIN, vmax=SPEED_MAX):
    """
    4WD policy: increase throttle as steering magnitude |angle-amid| grows.
    - Straight (s≈0): speed ≈ vmax
    - Full lock (s=1): speed ≈ vmax + EXTRA_FULL (capped to 100)
    EXTRA_POW controls curvature; EXTRA_DEAD creates a small deadzone around straight.
    """
    # steering magnitude 0 (straight) → 1 (full lock)
    s = abs(angle - amid) / float(amax - amid)
    s = max(0.0, min(1.0, s))

    if s <= EXTRA_DEAD:
        boost = 0.0
    else:
        # normalize after deadzone and apply shaping
        s_eff = (s - EXTRA_DEAD) / (1.0 - EXTRA_DEAD)
        boost = EXTRA_FULL * (s_eff ** EXTRA_POW)

    speed = vmax + boost
    return _clamp_duty(speed)


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