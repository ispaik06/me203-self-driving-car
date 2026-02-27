#!/usr/bin/env python3
# train_source.py
import os, argparse, time, cv2, numpy as np, pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="annotations.csv: columns [filepath,label_id,label]")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--img_size", type=int, default=240)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tflite", default="./tflite_0915/model.tflite", help="output TFLite file path")
args = ap.parse_args()

IMG = args.img_size

# ---------- 데이터 ----------
df = pd.read_csv(args.csv)
if not {"filepath","label_id"}.issubset(df.columns):
    raise SystemExit("CSV must have columns: filepath,label_id[,label]")

filepaths = df["filepath"].tolist()
labels_id = df["label_id"].astype(int).tolist()
num_classes = int(df["label_id"].nunique())

def one_hot(ids, C):
    y = np.zeros((len(ids), C), dtype=np.float32)
    y[np.arange(len(ids)), ids] = 1.0
    return y

def load_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)       # BGR
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (img.shape[1], img.shape[0]) != (IMG, IMG):
        img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def batches(X_paths, y_ids, batch):
    idx = np.arange(len(X_paths))
    np.random.shuffle(idx)
    for i in range(0, len(idx), batch):
        j = idx[i:i+batch]
        X = np.stack([load_image_rgb(X_paths[k]) for k in j], axis=0)     # (B,H,W,3)
        Y = one_hot([y_ids[k] for k in j], num_classes)                   # (B,C)
        yield X, Y

def hms(sec):
    m, s = divmod(sec, 60); h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{s:06f}"

# ---------- TF1 그래프 ----------
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, IMG, IMG, 3], name="X")
Y = tf.placeholder(tf.float32, [None, num_classes], name="Y")

W1 = tf.Variable(tf.random.normal([3,3,3,32], stddev=0.01))
L1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME'))
L1 = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')   # /2

W2 = tf.Variable(tf.random.normal([3,3,32,64], stddev=0.01))
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME'))
L2 = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')   # /4

# Flatten
flat_dim = int(np.prod(L2.get_shape().as_list()[1:]))   # 12*12*64 = 9216
L2f = tf.reshape(L2, [-1, flat_dim])

W4 = tf.Variable(tf.random.normal([flat_dim, 512], stddev=0.01))
L4 = tf.nn.relu(tf.matmul(L2f, W4))

W5 = tf.Variable(tf.random.normal([512, num_classes], stddev=0.01))
logits = tf.matmul(L4, W5, name="logits")
probs  = tf.nn.softmax(logits, name="probs")

# Loss/Opt/Metric
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
opt  = tf.train.AdamOptimizer(args.lr).minimize(loss)
pred = tf.argmax(logits, axis=1)
true = tf.argmax(Y, axis=1)
acc  = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

# ---------- 학습 + TFLite 저장 ----------
tflite_abs = os.path.abspath(args.tflite)
os.makedirs(os.path.dirname(tflite_abs), exist_ok=True)
print(f"[INFO] TFLite output    : {tflite_abs}")

cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config=cfg) as sess:
    sess.run(tf.global_variables_initializer())

    t0 = time.time()
    print(f"[TIME] training start @ {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(1, args.epochs + 1):
        e0 = time.time()
        loss_sum, acc_sum, cnt = 0.0, 0.0, 0
        for Xb, Yb in batches(filepaths, labels_id, args.batch):
            _, lv, av = sess.run([opt, loss, acc], feed_dict={X:Xb, Y:Yb})
            loss_sum += lv; acc_sum += av; cnt += 1
        print(f"[epoch {epoch:03d}] loss={loss_sum/max(1,cnt):.4f} acc={acc_sum/max(1,cnt):.4f}  ({hms(time.time()-e0)})")

    # 학습 종료 후 TFLite만 저장
    t0_tfl = time.time()
    converter = tf.lite.TFLiteConverter.from_session(sess, [X], [probs])
    tflite_model = converter.convert()
    with open(tflite_abs, "wb") as f:
        f.write(tflite_model)
    print(f"[TFLITE] saved to {tflite_abs}  (convert {hms(time.time()-t0_tfl)})")

    print(f"[TIME] training end   @ {time.strftime('%Y-%m-%d %H:%M:%S')}  (total {hms(time.time()-t0)})")