from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def entrenar_modelo(df, target_column='Difficulty Level', test_size=0.3, random_state=None):
    #Entrenar el mmodelo para predecir el nivel de dificultad
    #Dividir en características (X) y columna objetivo (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    #Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    #Se crea el modelo Naive Bayes y se entrena el modelo
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, X_test, y_test

