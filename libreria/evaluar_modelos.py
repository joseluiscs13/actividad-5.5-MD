from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluar_modelo(model, X_test, y_test):
    #Se hacen las predicciones y se calcula la precisión
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo NB: {accuracy}")
    #Reporte de clasificación
    print("Reporte de clasificación:\n",classification_report(y_test, y_pred))
    #Matriz de confusión
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    return accuracy
