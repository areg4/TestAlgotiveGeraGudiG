import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow.keras as keras
from PIL import Image
from sklearn.model_selection import train_test_split
from random import randrange

def graph_and_sort_attr(df,order,save_as=None):
    """
        Función para graficar y seleccionar las clases dependiendo del 
        orden que se defina
        :param df: DataFrame principal
        :param order: Define el orden para elegir las clases
                0 = mayor número de presencia
                1 = menor número de presencia
                2 = random
        :param save_as: Nombre como se guardará la gráfica si así se desea.
        :return lista con las clases a utilizar
    """
    headers = list(df)[1:]
    dfs = df[headers][df[headers] > 0].sum()
    dfsort = dfs.sort_values()
    dfsort.plot(kind = "bar")
    plt.xlabel("Attributes")
    plt.ylabel("N. of images")
    if save_as:
        plt.savefig(os.path.join("./",save_as))
    plt.show()
    if order == 0:
        return list(dfsort.index[-4:])
    elif order == 1:
        return list(dfsort.index[:4])
    elif order == 2:
        heads = []
        rand_generated = []
        x = 0
        while x < 4:
            r_num = randrange(len(headers))
            if not r_num in rand_generated:
                rand_generated.append(r_num)
                heads.append(headers[r_num])
                x+=1

        return heads 
    else:
        return None

def group_data(df_attr,classes_selected,low_performance=False):
    """
        Función que selecciona y agrupa el nombre de la imagen con base en los 
        atributos definidos
        :param df_attr: DataFrame de los atributos
        :param classes_selected: nombre de las classes o atributos definidos
        :low_performance: bandera para definir si el dataset debe de ser recortado
                        a un límite de 8000 imágenes por clase.
                        Dependiendo de los recursos que se tengan.
        :return DataFrame: DataFrame con las classes y sus respectivos nombres de las imágenes 
                        que pertenecen a las mismas
    """
    df_grouped = df_attr.groupby(classes_selected)["image_id"].unique()
    img_list = []
    for idx, item in df_grouped.items():
        i = 0
        for ix in idx:
            if ix == 1:
                img_list.append([classes_selected[i],item])
                break
            i+=1

    if low_performance:
        print("Low Performance activado!")
        low_img_list=[]
        for c in classes_selected:
            obj = list()
            for il in img_list:
                if c == il[0]:
                    obj += list(il[1])
            low_img_list.append([c,obj])
        img_list = low_img_list[:]
        for i in range(len(img_list)):
            if len(img_list[i][1]) > 8000:
                print("Clase '{}' pasa de {} a 8000 imágenes".format(img_list[i][0],len(img_list[i][1])))
                img_list[i][1] = img_list[i][1][:8000]

    return pd.DataFrame(img_list,columns=["label","img_name"])
        

def train_model(dataset, epochs, batch_size, image_size=(64,64), test_split=0.2):
    """
        Esta función se encarga de entrenar el modelo.
        :param dataset: dataset con las imágenes definidas para trabajar.
        :param epoch: número de epocas a entrenar el modelo.
        :param batch_size: tamaño del batch ocupado para entrenar
        :param image_size: tamaño (w,h) de las imágenes para redimensionar
        :param test_split: relación de tamaño de train, test, val
        :return: modelo entrenado
    """
    dict_attr = {}
    for idx,row in dataset.iterrows():
        dict_attr.update({row['label']:0})
    
    # print("Tamaño total del dataset: {}".format(len(dataset)))
    INIT_LR = 1e-3 #valor inicial de learning rate
    labels = []
    images_x = []
    labels_y = [] 

    for cls_s in dataset['label'].unique():
        labels.append(cls_s)

    for idxL in range(len(labels)):
        print(idxL,labels[idxL])

    dataset_size = 0
    for idx,row in dataset.iterrows():
        dict_attr.update({row["label"]:dict_attr[row["label"]]+len(row["img_name"])})
        dataset_size += len(row["img_name"])

    for key in dict_attr.keys():
        print("{}: {}".format(key,dict_attr[key]))

    print("Tamaño total del dataset: ",dataset_size)
    print("Cargando las imágenes en memoria, esto puede tardar un tiempo dependiendo el tamaño del dataset...")
    
    for idx,row in dataset.iterrows():
        for img_name in row["img_name"]:
            labels_y.append(labels.index(row["label"]))
            image = Image.open(os.path.join("./celeba-dataset/img_align_celeba/",img_name))
            new_image = image.resize(image_size)
            images_x.append(np.array(new_image))
    
    labels_y = np.array(labels_y)
    images_x = np.array(images_x, dtype=np.uint8)

    print(len(labels_y))

    #dividir el dataset en train, val and test. Relación 80%-20%
    train_x,test_x,train_y,test_y = train_test_split(images_x,labels_y,test_size=test_split)
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x = train_x / 255.
    test_x = test_x / 255.

    train_y_one_hot = keras.utils.to_categorical(train_y)
    test_y_one_hot = keras.utils.to_categorical(test_y)

    train_x,val_x,train_label,val_label = train_test_split(train_x, train_y_one_hot, test_size=test_split, random_state=13)

    # Definiendo las capas para el entrenamiento
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(image_size[0],image_size[1],3)))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.MaxPooling2D((2, 2),padding='same'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='linear'))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dropout(0.5)) 
    model.add(keras.layers.Dense(len(labels), activation='softmax'))

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])
    train_model_out = model.fit(train_x, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(val_x, val_label))

    test_eval = model.evaluate(test_x, test_y_one_hot, verbose=1)
    
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    return train_model_out


def graph_model(model,value='accuracy',save_as=None):
    """
        Método para graficar los resultados del modelo entrenado
        ['accuracy','loss','val_loss','val_accuracy']
        :param model: modelo entrenado a graficar
        :param value: valor a graficar ['accuracy','loss']
        :param save_as: Nombre como se guardará la gráfica si así se desea.
    """
    plt.plot(model.history[value])
    plt.plot(model.history['val_{}'.format(value)])
    plt.title('Model {}'.format(value))
    plt.ylabel('{}'.format(value))
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save_as:
        plt.savefig(os.path.join("./",save_as))
    plt.show()


if __name__ == "__main__":
    #carga el dataframe de los atributos
    path_attr = "celeba-dataset/list_attr_celeba.csv"
    df_attr = pd.read_csv(path_attr)
    # Selecciona las clases a entrenar
    classes_selected = graph_and_sort_attr(df_attr,1,save_as="attr_graph.jpg")
    # selecciona y agrupa las imágenes dependiendo de las clases a entrenar en un nuevo DataFrame
    dataset = group_data(df_attr,classes_selected)
    # proceso para entrenar el modelo
    trained_model = train_model(dataset,250,64)
    # método para graficar los resultados del modelo
    graph_model(trained_model,save_as="val_graph.jpg")