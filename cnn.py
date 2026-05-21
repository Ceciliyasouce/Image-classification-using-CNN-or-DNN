import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_minority_classes_np(y, threshold=0.2):
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    minority_classes = classes[
        (counts / majority_count) < threshold
    ]

    return minority_classes.tolist()

def oversample_class(train_data_x, train_data_y,target_class, seed=42):
    rng = np.random.default_rng(seed)

    # Ensure channel dim
    if train_data_x.ndim == 3:
         train_data_x = train_data_x[..., None]  # (N,28,28,1)


    counts = np.bincount(train_data_y, minlength=10)
    max_count = counts.max()
    cls_count = counts[target_class]
    need = max_count - cls_count
    if need <= 0:
        return train_data_x, train_data_y

    idx_cls = np.where(train_data_y == target_class)[0]
    chosen = rng.choice(idx_cls, size=need, replace=True)

    X_extra = train_data_x[chosen]
    # Apply augmentation (expects float32)
   

    y_extra = np.full((need,), target_class, dtype=train_data_y.dtype)

    X_bal = np.concatenate([train_data_x, X_extra], axis=0)
    y_bal = np.concatenate([train_data_y, y_extra], axis=0)

    # Shuffle
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def main():
    
    if len(sys.argv) != 3:
        print("Usage: python3 cnn.py fashion_mnist.npz train/test")
        sys.exit(1)

    image_dir = sys.argv[1]
    choice = sys.argv[2] 

    #check with hardcoded
    #choice = 'test'

    #loading data
    #data = np.load("fashion_mnist.npz")
    data = np.load(image_dir)
    fashion_mnist_Arr = data["img_data"]
    label = data['img_lbl']
    #lenght of the uncorrupted data
    #size = len(fashion_mnist_Arr)
 

    #train test split
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(
        fashion_mnist_Arr, label,
        test_size=0.2,
        random_state=42,
        stratify=label,
    )

    minor = []
    minor= get_minority_classes_np(train_data_y)
    #print(minor)


    for i in minor:
        train_data_x, train_data_y = oversample_class(train_data_x, train_data_y,i)

    if choice == 'CNN_train':
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(28,28,1)))
        model.add(tf.keras.layers.Conv2D(6, 5)) #26,26,64
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPool2D()) #13,13,64
        model.add(tf.keras.layers.Conv2D(16, 5)) #26,26,64
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPool2D()) #13,13,64
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(120))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(84))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(len(set(label))))
        model.add(tf.keras.layers.Softmax())


        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.fit(
            train_data_x, train_data_y,
            epochs=10,
            batch_size=128,
            validation_split=0.1,
            verbose=1,
        )

        model.save_weights("cnn.weights.h5")

    if choice == 'CNN_test':
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(28,28,1)))
        model.add(tf.keras.layers.Conv2D(6, 5)) #26,26,64
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPool2D()) #13,13,64
        model.add(tf.keras.layers.Conv2D(16, 5)) #26,26,64
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPool2D()) #13,13,64
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(120))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(84))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(len(set(label))))
        model.add(tf.keras.layers.Softmax())

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.load_weights("cnn.weights.h5")
        #class_acc = model.evaluate(x=test_data_x, y=test_data_y)
        model.evaluate(x=test_data_x, y=test_data_y)

        #displaying confusion matrix
        y_pred = model.predict(test_data_x)
        y_pred_classes = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(test_data_y, y_pred_classes)

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()


    #train on DNN
    if choice == 'train':
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Reshape(target_shape = (784,)))
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Softmax())

        model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

        model.fit(train_data_x, train_data_y, epochs=20, batch_size=100)
        model.save_weights("dnn.weights.h5")

    if choice == 'test':
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Reshape(target_shape = (784,), input_shape = (28,28)))
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Softmax())

        model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

        model.load_weights("dnn.weights.h5")
        #class_acc = model.evaluate(x=test_data_x, y=test_data_y)
        model.evaluate(x=test_data_x, y=test_data_y)

        #displaying confusion matrix
        y_pred = model.predict(test_data_x)
        y_pred_classes = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(test_data_y, y_pred_classes)

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

if __name__ == "__main__":
    main()
