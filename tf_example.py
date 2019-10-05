from scipy.misc import imread
# from skimage.io import imread
from preprocess.normalize import preprocess_signature
import tensorflow as tf
import tf_signet
from tf_cnn_model import TF_CNNModel
import numpy as np
import os
import hypertools as hyp
import pandas as pd
from pathlib import Path
from sklearn import svm
import pickle
from augment import augment, resize_with_pad, scale_image, cut_block
import shutil
import cv2

canvas_size = (952, 1360)  # Maximum signature size

# Load the model
model_weight_path = 'models/signetf_lambda0.999.pkl'
model = TF_CNNModel(tf_signet, model_weight_path)

# Create a tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def get_feature_from_link(path):
    original = imread(path, flatten=1)
    processed = preprocess_signature(original, canvas_size)
    feature_vector = model.get_feature_vector(sess, processed)

    return feature_vector

def get_distance(reals, forgeries):
    real_features = [get_feature_from_link(e) for e in reals]
    
    mean = np.sum(real_features, axis=0)[0] /len(reals)

    real_distances = [np.linalg.norm(mean-e) for e in real_features]

    fake_features = [get_feature_from_link(e)[0] for e in forgeries]
    fake_distances = [np.linalg.norm(mean-e) for e in fake_features]

    print(np.amax(real_distances), np.amin(fake_distances))

    return max, min

def visualize( image_path_list, labels_list):
        X = []
        for e in image_path_list:
            newImg = get_feature_from_link(e)
            print(newImg.mean(), e)
            feature = newImg.reshape(2048)
            X.append(feature)
        X = np.asarray(X)

        labels_list = np.expand_dims(np.asarray(labels_list), axis=1)
        
        newData = np.concatenate((X, labels_list), axis=1)
        
        header = [str(i) + 'th dimension' for i in range(2048)] + ['label']
        df = pd.DataFrame(newData, columns=header)
        print(df.shape)
        class_labels = df["label"]
        hyp.plot(df, ".", group=class_labels, legend=class_labels.unique().tolist(), ndims=3)

def get_prepared_data(image_path_list):
    train_X = []
    train_y = []

    parent_path = Path(image_path_list[0]).parent
    scale_folder = os.path.join(parent_path, 'scale')
    augmented_folder = os.path.join(parent_path, 'padding', 'output')

    for index, image_path in enumerate(image_path_list):
        newImg = get_feature_from_link(image_path)
        feature = newImg.reshape(2048)
        train_X.append(feature)
        train_y.append('1')

        feature_cut = model.get_feature_vector(sess, cut_block(image_path))
        feature_cut = feature_cut.reshape(2048)
        train_X.append(feature_cut)
        train_y.append('0')

        if os.path.isdir(scale_folder):
            if Path(image_path).name not in os.listdir(scale_folder):
                resize_with_pad(image_path)
                scale_image(image_path)
        else:
            resize_with_pad(image_path)
            scale_image(image_path)

    if os.path.exists(augmented_folder):
        shutil.rmtree(augmented_folder)

    augment(Path(augmented_folder).parent)
    scaled_images = [os.path.join(scale_folder, e) for e in os.listdir(scale_folder)]
    augmented_images = [os.path.join(augmented_folder, e) for e in os.listdir(augmented_folder)]
    
    # for index, image_path in enumerate(scaled_images):
    #     newImg = get_feature_from_link(image_path)
    #     feature = newImg.reshape(2048)
    #     train_X.append(feature)
    #     train_y.append('0')

    for index, image_path in enumerate(augmented_images):
        newImg = get_feature_from_link(image_path)
        feature = newImg.reshape(2048)
        train_X.append(feature)
        train_y.append('0')

    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)

    return train_X, train_y


def train_model(image_path_list):
    if not image_path_list[0]:
        raise('invalid image_test set')
    parent_path = Path(image_path_list[0]).parent
    X, y = get_prepared_data(image_path_list)

    y2 = np.expand_dims(y, axis=1)
        
    newData = np.concatenate((X, y2), axis=1)
    
    header = [str(i) + 'th dimension' for i in range(2048)] + ['label']
    # print(header)
    df = pd.DataFrame(newData, columns=header)
    # print(df.head())
    # geo = hyp.l(df)
    class_labels = df["label"]
    # hyp.plot(df, ".", group=class_labels, legend=class_labels.unique().tolist())

    print('----------', (1 / (128 * X.var())))
    sample_weights = np.ones((len(X)), np.int16)
    sample_weights[len(image_path_list):] *= 100      
    clf = svm.SVC( kernel='rbf', probability=True, decision_function_shape='ovo', class_weight='balanced')
    clf.fit(X, y)
    print(clf.score(X,y))

    output = open(os.path.join(parent_path,'model_file.pkl'), 'wb')
    pickle.dump(clf, output)
    output.close()

def predict(image_path):
    parent_path = Path(image_path).parent
    pkl_file = open(os.path.join('data/thuy/model_file.pkl'), 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    image = get_feature_from_link(image_path)

    return data.predict(image)

def evaluate(path):
    objs = [os.path.join('')]

if __name__ == "__main__":
    path = 'data/thuy'
    forgery_path = 'data_forgery/thuyfake'
    imgList = [os.path.join(path, e) for e in os.listdir(path) if e != 'model_file.pkl' and os.path.isfile(os.path.join(path, e))]
    forgery_img = [os.path.join(forgery_path, e) for e in os.listdir(forgery_path) if e != 'output']
    train_model(imgList)

    for e in forgery_img:
        result = predict(e)
        print(result)
        # if result[0] == '1':
        #     cv2.imshow('test', cv2.imread(e))
        #     cv2.waitKey(0)

# if __name__ == "__main__":
#     path = 'data'
#     sub_folders = [os.path.join(path,e) for e in os.listdir(path)]
    
#     results = []
    
#     for folder in sub_folders:
#         image_list = []
#         reals = []
#         for folder_fake in sub_folders:
#             for image in os.listdir(folder_fake):
#                 if image != 'model_file.pkl' and os.path.isfile(os.path.join(folder_fake,image)):
#                     if folder == folder_fake:
#                         reals.append(os.path.join(folder_fake, image))
#                     else:
#                         image_list.append(os.path.join(folder_fake, image))
#         print(len(reals), len(image_list))
#         get_distance(reals, image_list)

# if __name__ == "__main__":
#     path = 'data'
#     sub_folders = [os.path.join(path,e) for e in os.listdir(path)]
#     image_list = []
#     labels_list = []
#     for folder in sub_folders:
#         for image in os.listdir(folder):
#             if image != 'model_file.pkl' and os.path.isfile(os.path.join(folder,image)):
#                 image_list.append(os.path.join(folder, image))
#                 labels_list.append(folder.split('/')[-1])

#     visualize(image_list, labels_list)