import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import OneHotEncoder

# Función para obtener la jugada ganadora
def obtener_ganadora(jugada):
    if jugada == 'piedra':
        return 'papel'  # Papel vence a piedra
    elif jugada == 'papel':
        return 'tijeras'  # Tijeras vence a papel
    else:
        return 'piedra'  # Piedra vence a tijeras

# Función para determinar el resultado del juego
def determinar_resultado(jugada_usuario, jugada_IA):
    if jugada_usuario == jugada_IA:
        return 'empate'
    elif obtener_ganadora(jugada_usuario) == jugada_IA:
        return 'IA_gana'
    else:
        return 'usuario_gana'

# Codificar las jugadas en One-Hot Encoding
def codificar_jugadas(jugadas):
    encoder = OneHotEncoder(sparse_output=False)
    jugadas_codificadas = np.array(jugadas).reshape(-1, 1)
    return encoder.fit_transform(jugadas_codificadas), encoder

# Descodificar las jugadas
def decodificar_jugada(codificada, encoder):
    return encoder.inverse_transform([codificada])[0][0]

# Entrenar el modelo basado en jugadas pasadas
def entrenar_modelo(modelo, jugadas_codificadas, secuencia):
    X, y = [], []
    for i in range(secuencia, len(jugadas_codificadas)):
        X.append(jugadas_codificadas[i-secuencia:i])
        y.append(jugadas_codificadas[i])

    X = np.array(X)
    y = np.array(y)

    # Asegurarse de que las dimensiones de X sean correctas
    X = X.reshape((X.shape[0], secuencia, 3))

    # Asegurar que y esté en formato one-hot
    y = y.reshape(-1, 3)

    # Entrenar el modelo con las jugadas
    modelo.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return modelo

# Crear un modelo inicial (fuera del bucle para no reiniciarlo)
def crear_modelo(secuencia):
    modelo = Sequential()
    modelo.add(LSTM(50, input_shape=(secuencia, 3)))
    modelo.add(Dense(3, activation='softmax'))  # 3 clases de salida

    # Compilar el modelo
    modelo.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
    return modelo

# Configuraciones del juego
jugadas_posibles = ['piedra', 'papel', 'tijeras']
secuencia = 5  # La IA verá las últimas 5 jugadas del oponente
jugadas = []

# Inicializar el modelo
modelo = crear_modelo(secuencia)
encoder = None

# Contadores de victorias
victorias_player = 0
victorias_IA = 0

# Bandera para verificar si el entrenamiento ha finalizado
fase_prueba_terminada = False

# Ciclo de juego interactivo
print("\n¡Bienvenido al juego de Piedra, Papel o Tijeras con IA!")
print("\nPrimero, iniciaremos un juego de práctica para que la IA aprenda tus patrones.")
print("\nElige tu jugada usando los números:\n1 - Piedra\n2 - Papel\n3 - Tijeras\nDeja vacío para salir.")

while True:
    # Solicitar jugada del usuario
    jugada_usuario = input("\nTu jugada (1-Piedra, 2-Papel, 3-Tijeras): ").strip()

    if jugada_usuario == "":
        print("\nGracias por jugar.")
        print(f"Resultados finales: Jugador {victorias_player} - {victorias_IA} IA")
        break

    # Convertir la elección del usuario en la jugada correspondiente
    if jugada_usuario == "1":
        jugada_usuario = 'piedra'
    elif jugada_usuario == "2":
        jugada_usuario = 'papel'
    elif jugada_usuario == "3":
        jugada_usuario = 'tijeras'
    else:
        print("Entrada no válida. Intenta de nuevo.")
        continue

    # Guardar la jugada del usuario
    jugadas.append(jugada_usuario)

    # Validar que se haya jugado al menos una vez cada jugada (piedra, papel, tijeras)
    jugadas_unicas = set(jugadas)
    if len(jugadas) >= secuencia and not fase_prueba_terminada:
        if not all(j in jugadas_unicas for j in jugadas_posibles):
            print("\nDebes jugar al menos una vez piedra, papel y tijeras antes de que la IA comience a predecir.")
            continue
        else:
            print("\nFase de práctica terminada, ¡ahora la IA está lista para predecir!\n")
            fase_prueba_terminada = True
            continue  # Después de este mensaje, no queremos predecir hasta la próxima jugada

    # Codificar las jugadas
    jugadas_codificadas, encoder = codificar_jugadas(jugadas)

    # Verificar que haya suficientes jugadas para la predicción
    if len(jugadas) >= secuencia and fase_prueba_terminada:
        # Entrenar incrementalmente el modelo con las jugadas acumuladas
        modelo = entrenar_modelo(modelo, jugadas_codificadas, secuencia)

        # Predecir la próxima jugada del usuario
        jugadas_nuevas_codificadas = jugadas_codificadas[-secuencia:].reshape(1, secuencia, 3)  # Forma correcta para la predicción
        prediccion = modelo.predict(jugadas_nuevas_codificadas)

        # Obtener la clase predicha como el índice con la mayor probabilidad
        clase_predicha = np.argmax(prediccion, axis=1)

        # Obtener la probabilidad más alta
        probabilidad_predicha = np.max(prediccion) * 100

        # Decodificar la predicción
        jugada_predicha = decodificar_jugada(np.eye(3)[clase_predicha][0], encoder)

        # La IA juega la jugada ganadora contra la predicción
        jugada_IA = obtener_ganadora(jugada_predicha)

        # Mostrar el resultado en el formato que indicaste
        print(f"\nSe predecía que jugarías: {jugada_predicha} con un {probabilidad_predicha:.2f}% de confianza.")
        print(f"La IA jugó: {jugada_IA}")

        # Determinar el resultado del juego correctamente
        resultado = determinar_resultado(jugada_usuario, jugada_IA)
        
        if resultado == 'IA_gana':
            print("La IA gana esta ronda.")
            victorias_IA += 1
        elif resultado == 'usuario_gana':
            print("¡Tú ganas esta ronda!")
            victorias_player += 1
        else:
            print("¡Es un empate!")

        # Mostrar resultados actuales
        print(f"Puntaje actual: Jugador {victorias_player} - {victorias_IA} IA\n")
    else:
        print(f"Estamos en la fase de práctica. Aún no tengo suficiente información para predecir. Sigue jugando.")
