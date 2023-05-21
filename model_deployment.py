#!/usr/bin/python

# Importación librerías
import pandas as pd
import numpy as np
import joblib
import os
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Descargar las stopwords y el lematizador si no están ya descargados
nltk.download('stopwords')
nltk.download('wordnet')

def predict_proba(plot):

    # Cargar el archivo pickle
    logistic_regression = joblib.load(os.path.dirname(__file__) + '/logistic_regression.pkl')

    plot_ = pd.DataFrame([plot], columns=['plot'])

    # Create features

    # Definición de variables predictoras (X)
    vect = TfidfVectorizer(max_features=60000, ngram_range=(1, 5), sublinear_tf = True)

    # Eliminar signos de puntuación
    translator = str.maketrans('', '', string.punctuation)
    plot_['plot'] = plot_['plot'].apply(lambda x: x.translate(translator))

    # Eliminar caracteres no alfabéticos
    plot_['plot'] = plot_['plot'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Convertir a minúsculas
    plot_['plot'] = plot_['plot'].apply(lambda x: x.lower())

    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    plot_['plot'] = plot_['plot'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Lematización
    lemmatizer = WordNetLemmatizer()
    plot_['plot'] = plot_['plot'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    X_dtm = vect.fit_transform(plot_['plot'])
    
    import numpy as np
    # Asegurarse de que la matriz de características tenga 60000 columnas
    if X_dtm.shape[1] < 60000:
        # Crear una matriz de ceros con las dimensiones adecuadas
        zeros = np.zeros((X_dtm.shape[0], 60000 - X_dtm.shape[1]))
        # Concatenar la matriz de características original con los ceros
        X_dtm = np.hstack((X_dtm.toarray(), zeros))

    # Make prediction
    p1 = logistic_regression.predict_proba(X_dtm)
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
            'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
            'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    res = pd.DataFrame(p1, index=plot_.index, columns=cols)

    dict_predic = {}
    for column in res.columns:
        dict_predic[column] = float(res[column])

    return dict_predic


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add a plot')
        
    else:

        plot = sys.argv[1]

        p1 = predict_proba(plot)
        
        print(url)
        print('Probability of Genders: ', p1)
        