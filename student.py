import tensorflow as tf
import numpy as np
from numpy import load
from os import listdir

if __name__ == '__main__':
    # initialize testing data
    true_file = 'eeg_fpz_cz/' + 'SC4001E0.npz'
    true_data = load(true_file)
    x_train = true_data['x']
    predict_file = 'teacher_predict/' + 'pred_SC4001E0.npz'
    predict_data = load(predict_file)
    y_train = predict_data['y_pred']

    # get all training file names
    file_list = listdir('eeg_fpz_cz/')
    # go through files and get training data
    for item in file_list:
        if not 'npz' in item:
            continue
        if item == 'SC4001E0.npz':
            continue
        train_file = 'eeg_fpz_cz/' + item
        train_data = load(train_file)
        x_train = np.concatenate((x_train, train_data['x']))
        # use teacher predictions as soft targets
        pred_file = 'teacher_predict/' + 'pred_' + item
        pred_data = load(pred_file)
        y_train = np.concatenate((y_train, pred_data['y_pred']))

    print("x trainsing length")
    print(len(x_train))

    # get unseen testing data
    eval_test_file = 'eeg_no_pred/' + 'SC4201E0.npz'
    eval_test_data = load(eval_test_file)
    x_test = eval_test_data['x']
    y_test = eval_test_data['y']

    # create student model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    tf.random.set_seed(0)
    model.add(tf.keras.layers.Dropout(.05))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

    # train
    #opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer='adam', # 'adam'
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    # evaluate
    model.fit(x_train, y_train, epochs=4)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("loss on unseen data set:")
    print(val_loss)
    print("accuracy on unseen data set:")
    print(val_acc)

    # save model
    model.save('dumbish_student.model')
    conv = tf.lite.TFLiteConverter.from_saved_model('dumbish_student.model')
    lite_student_model = conv.convert()
    with open('student_model.tflite', 'wb') as lite_file:
        lite_file.write(lite_student_model)
