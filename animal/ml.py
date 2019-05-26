# coding:gbk

import os
import keras
import numpy as np
import shutil
from keras.models import Sequential

# 卷积神经网络
from keras.layers import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import random
import math
from .data_gen import  DataGenerate


class CatDog():
    def __init__(self, file_path):
        self.BASE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        self.file_path = file_path
        self.train = os.path.join(self.BASE_PATH, 'data', 'cat_dog', 'train') + os.path.sep
        # 目标训练地址
        self.target = self.train + 'arrange' + os.path.sep
        self.model_path = self.train + 'model'

    def init_cat_dog(self):
        train_list = os.listdir(self.train)

        dogs = [self.train + i for i in train_list if 'dog' in i]
        cats = [self.train + i for i in train_list if 'cat' in i]
        self.ensure_dir(self.target + 'train' + os.path.sep + 'dog' + os.path.sep)
        self.ensure_dir(self.target + 'train' + os.path.sep + 'cat' + os.path.sep)
        self.ensure_dir(self.target + 'validation' + os.path.sep + 'dog' + os.path.sep)
        self.ensure_dir(self.target + 'validation' + os.path.sep + 'cat' + os.path.sep)

        random.shuffle(dogs)
        random.shuffle(cats)

        # 训练数据
        # 复制狗图片
        for dog_file in dogs[:150]:
            shutil.copyfile(dog_file,
                            self.target + 'train' + os.path.sep + 'dog' + os.path.sep + os.path.basename(dog_file))
            # 复制猫图片
        for cat_file in dogs[:150]:
            shutil.copyfile(cat_file,
                            self.target + 'train' + os.path.sep + 'cat' + os.path.sep + os.path.basename(cat_file))

        # 验证数据
        # 复制狗图片
        for dog_file in dogs[150:200]:
            shutil.copyfile(dog_file,
                            self.target + 'validation' + os.path.sep + 'dog' + os.path.sep + os.path.basename(
                                dog_file))
        # 复制猫图片
        for cat_file in dogs[150:200]:
            shutil.copyfile(cat_file,
                            self.target + 'validation' + os.path.sep + 'cat' + os.path.sep + os.path.basename(
                                cat_file))

    # 读取训练和验证的数据
    def init_datas(self, data_type):
        train_datas = []
        data_path = self.target + data_type + os.path.sep
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            if os.path.isdir(file_path):
                for sub_file in os.listdir(file_path):
                    train_datas.append(os.path.join(file_path, sub_file))
        return train_datas

    # 删除目录
    def del_cat_dog_arrange_dir(self):
        target = self.target
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            print(f'文件目录{target}不存在')

    def ensure_dir(self, dir_path):
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)

            except OSError:
                pass

    # 构造模型
    # 卷积 + 全连接模型
    def ini_model(self):
        # 统一图像尺寸,
        img_width, img_height = 128, 128
        input_shape = (img_width, img_height, 3)
        model = Sequential([
            Convolution2D(32,(3,3),input_shape=input_shape,activation='relu'), #卷积层
            MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool1'),#池化层
            Convolution2D(64,(3,3),activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'),  # 池化层
            Flatten(),
            Dense(64,activation='relu'),
            Dropout(0.5),
            Dense(2,activation='sigmoid')
        ])

        #编译
        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        self.model = model
    #训练数据
    def train_cat_dog(self):
        train_datas = self.init_datas('train')
        train_generator = DataGenerate(train_datas,batch_size=32,shuffle=True)
        self.model.fit_generator(train_generator,epochs=2,max_queue_size=10,workers=1,verbose=1)

    # 验证数据
    def eval_cat_dog(self):
        eval_datas = self.init_datas('eval')
        eval_generator = DataGenerate(eval_datas, batch_size=32, shuffle=True)
        eval_res = self.model.fit_generator(eval_generator,max_queue_size=10,workers=1,verbose=1)
        return eval_res

    #单张图片的预测
    def pred_one_cat_dog(self):
        img = image.load_img(self.file_path,target_size=(128,128))
        x = image.img_to_array(img)
        x/=255
        x = np.expand_dims(x,axis=0)
        y = self.model.predict(x)
        cat_dog = np.argmax(y)
        if(cat_dog==0):
            return '猫'
        else:
            return "狗"
    #保存模型文件
    def save_my_model(self):
        self.ensure_dir(self.model_path)
        json_string = self.model.to_json()
        with open(os.path.join(self.model_path,'my_model_architecture.json'),'w')as f:
            print(666)
            f.write(json_string)
        #保存权重
        self.model.save_weights(os.path.join(self.model_path,'my_model_weights.h5'))
    #验证模型
    def load_my_model(self):
        model = model_from_json(open(os.path.join(self.model_path,'my_model_architecture.json')).read())
        model.load_weights(os.path.join(self.model_path,'my_model_weights.h5'))
        return model


