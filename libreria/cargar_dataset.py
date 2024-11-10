import pandas as pd 

def cargar_dataset(filepath):
    #Cargar el dataset 
    df = pd.read_csv(filepath)
    #mostrar algunas filas para ver que se haya cargado el dataset
    print(f"Primeras filas del dataset cargadas:\n{df.head()}")
    #Columnas con datos categóricos
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['Name of Exercise'] = label_encoder.fit_transform(df['Name of Exercise'])
    df['Benefit'] = label_encoder.fit_transform(df['Benefit'])
    df['Target Muscle Group'] = label_encoder.fit_transform(df['Target Muscle Group'])
    df['Equipment Needed'] = label_encoder.fit_transform(df['Equipment Needed'])
    df['Difficulty Level'] = label_encoder.fit_transform(df['Difficulty Level'])
    return df 

