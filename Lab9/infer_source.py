#!/home/mecha/venvs/tflite/bin/python
# infer_source.py

import numpy as np, cv2, time
import tflite_runtime.interpreter as tflite

IMG = 240
model = "./model.tflite"
labels = ["Left", "Right", "green", "red", "straight"]

interpreter = tflite.Interpreter(model_path=model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]

while True:
    x = preprocess("straight_line.png")
    interpreter.set_tensor(inp["index"], x)

    t0 = time.time()
    interpreter.invoke()
    dt = (time.time() - t0) * 1e3

    probs = interpreter.get_tensor(out["index"])[0]
    pred_id = int(np.argmax(probs))
    print(f"pred={labels[pred_id]}  probs={probs}  ({dt:.1f} ms)")

    if labels[pred_id] == "straight":
        print("forward")
        # TODO: Write GPIO control code to move motors forward
        
    elif labels[pred_id] == "left":
        print("left")
        # TODO: Write GPIO control code to move motors left
        
    elif labels[pred_id] == "right":
        print("right")
        # TODO: Write GPIO control code to move motors right
        
    else:
        print("color?")

    time.sleep(0.1)
    
    show = (x[0] * 255).astype(np.uint8)        # [240,240,3], uint8 RGB
    show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
    cv2.imshow("input", show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.05)

cv2.destroyAllWindows()