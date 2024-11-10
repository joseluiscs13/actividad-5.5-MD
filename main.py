from libreria.cargar_dataset import cargar_dataset
from libreria.modelos import entrenar_modelo
from libreria.evaluar_modelos import evaluar_modelo

def main(filepath, target_column='Difficulty Level', test_size=0.3, n_iter=5):
    #Función principal para cargar datos, entrenar y evaluar el modelo con varias iteraciones
    df = cargar_dataset(filepath)
    for i in range (n_iter):
        print(f"\nIteración {i+1}:")
        model, X_test, y_test = entrenar_modelo(df, target_column=target_column, test_size=test_size)
        evaluar_modelo(model, X_test, y_test)

if __name__ == "__main__":
    #valores de ejemplo
    filepath = './data/Top 50 Excerice for your body.csv'
    main(filepath, target_column='Difficulty Level', test_size=0.3, n_iter=5)
