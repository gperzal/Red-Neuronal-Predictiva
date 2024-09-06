# Red Neuronal Predictiva para el Juego de Piedra, Papel o Tijeras

Este proyecto implementa una Red Neuronal Predictiva utilizando una arquitectura LSTM (Long Short-Term Memory) para jugar y predecir las jugadas en el clásico juego de Piedra, Papel o Tijeras. La IA aprende los patrones de jugadas del jugador humano y utiliza esa información para predecir futuras jugadas, maximizando sus probabilidades de ganar.

## Descripción del Proyecto

El sistema funciona en dos fases:

1. **Fase de práctica**: Recopila información sobre las jugadas del usuario hasta que haya realizado al menos una jugada de cada tipo.
2. **Fase de predicción**: La IA predice la próxima jugada del jugador y elige su jugada óptima para vencer.

## Requisitos del Sistema

Para ejecutar este proyecto, necesitas las siguientes bibliotecas de Python:

- numpy
- tensorflow
- scikit-learn

Instala los requisitos con:

```bash
pip install numpy tensorflow scikit-learn
```

## Estructura del Proyecto

1. **Entrenamiento y predicción**: Utiliza una red neuronal LSTM para analizar y predecir las jugadas del jugador.
2. **Modo de juego**: El usuario ingresa su jugada y la IA predice y juega en consecuencia.
3. **Validación**: Asegura un conjunto diverso de datos antes de pasar a la fase de predicción.

## Uso

1. Ejecuta el script principal:

```bash
python game.py
```

2. Ingresa tus jugadas usando los números:
   - 1: Piedra
   - 2: Papel
   - 3: Tijeras

3. El sistema proporcionará retroalimentación de cada ronda, mostrando:
   - La jugada predicha por la IA
   - La confianza de la IA en su predicción
   - El resultado de la ronda
   - El puntaje acumulado

## Ejemplo de Juego

```
¡Bienvenido al juego de Piedra, Papel o Tijeras con IA!
Primero, iniciaremos un juego de práctica para que la IA aprenda tus patrones.
Elige tu jugada usando los números:
1 - Piedra
2 - Papel
3 - Tijeras

Tu jugada (1-Piedra, 2-Papel, 3-Tijeras): 1
Estamos en la fase de práctica. Aún no tengo suficiente información para predecir. Sigue jugando.

Tu jugada (1-Piedra, 2-Papel, 3-Tijeras): 3
Fase de práctica terminada, ¡ahora la IA está lista para predecir!

Tu jugada (1-Piedra, 2-Papel, 3-Tijeras): 2
Se predecía que jugarías: papel con un 36.27% de confianza.
La IA jugó: tijeras
¡La IA gana esta ronda!
Puntaje actual: Jugador 0 - 1 IA
```

## Probar el Proyecto en Google Colab

Puedes ejecutar este proyecto directamente en Google Colab sin necesidad de configuraciones adicionales en tu máquina local. Para acceder y probar el proyecto en Colab, sigue este [Enlance](https://colab.research.google.com/drive/1fc3aAfO8LXUrQhItCNAcDRkVDPxwd_se?usp=sharing).


## Estructura del Código

Funciones principales:

- `obtener_ganadora(jugada)`: Determina la jugada ganadora.
- `entrenar_modelo(modelo, jugadas_codificadas, secuencia)`: Entrena el modelo LSTM.
- `decodificar_jugada(codificada, encoder)`: Convierte una jugada codificada a su forma original.
- `determinar_resultado(jugada_usuario, jugada_IA)`: Evalúa el resultado del enfrentamiento.

## Mejoras Futuras

- Implementar diferentes niveles de dificultad.
- Guardar el historial de jugadas para futuras sesiones.
- Ampliar el número de rondas de práctica.

## Contribuciones

Las contribuciones son bienvenidas. Si tienes sugerencias o mejoras, por favor abre un pull request o crea un issue en el repositorio.
