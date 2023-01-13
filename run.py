import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(gpus)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
import numpy as np
import matplotlib.pyplot as plt
from source.layers import PP_LCNet_model, CNN_model
import time

def draw_loss(history, label):
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(f'{label}.jpg')
    plt.clf()
    
    plt.plot(history.history["accuracy"], label="train_loss")
    plt.plot(history.history["val_accuracy"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.title("Train and Validation Acc Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(f'{label}_acc.jpg')
    plt.clf()
    
    start_time = time.time()
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {round(loss, 2)}")
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"{label} time : {time.time() - start_time}")

if __name__ == '__main__':
    # prepared dataset
    reduce_cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = reduce_cifar10.load_data()
    
    y_train = np.squeeze(np.eye(10)[y_train])
    y_test = np.squeeze(np.eye(10)[y_test])
    
    if len(X_train.shape) !=4 :
        X_train = np.concatenate([np.expand_dims(X_train, axis=-1) for i in range(3)], axis=-1)
        X_test = np.concatenate([np.expand_dims(X_test, axis=-1) for i in range(3)], axis=-1)
    
    print(X_train.shape, y_train.shape)

    #normalization
    X_train = X_train/255
    X_test = X_test/255

    # callbacks
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    cd = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=1e-2, first_decay_steps=5, t_mul=2.0, m_mul=0.9, alpha=1e-10)
    ls = tf.keras.callbacks.LearningRateScheduler(cd)
    # set hyper-parameters
    label_smoothing = 0.1
    batch_size = 128
    num_epochs = 20
    validation_split = 0.2
    
    model = PP_LCNet_model(X_train.shape[1:], y_train.shape[-1])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        optimizer='adam',#tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9, decay=3e-5),
        metrics=['accuracy'],
    )
    model.summary()
    history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=validation_split,
            callbacks=[es, ls],
        )
    draw_loss(history, 'PP_LCNet')

    cd = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=5, t_mul=2.0, m_mul=0.9, alpha=1e-5)
    ls = tf.keras.callbacks.LearningRateScheduler(cd)

    model = CNN_model(X_train.shape[1:], y_train.shape[-1])
    
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
#         loss='categorical_crossentropy',
        optimizer='adam',#tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=3e-5),
        metrics=['accuracy'],
    )
    model.summary()
    history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=validation_split,
            callbacks=[es, ls],
        )
    draw_loss(history, 'CNN')
