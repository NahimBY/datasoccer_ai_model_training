import tensorflow as tf
import numpy as np
import pandas as pd
import json
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, LSTM, Dense, Bidirectional,
                                   Dropout, GlobalMaxPooling1D, BatchNormalization,
                                   LayerNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# DATASET EXPANDIDO Y MEJORADO
ejemplos_entrenamiento = [
    # GOLES - Más variaciones
    {"texto": "Jugador número 7 anotó un gol", "intencion": "gol", "entidades": {"jugador_num": 7}},
    {"texto": "El jugador número 7 metió un gol", "intencion": "gol", "entidades": {"jugador_num": 7}},
    {"texto": "El 7 metió gol", "intencion": "gol", "entidades": {"jugador_num": 7}},
    {"texto": "Gol del número 7", "intencion": "gol", "entidades": {"jugador_num": 7}},
    {"texto": "Gol para el 7", "intencion": "gol", "entidades": {"jugador_num": 7}},
    {"texto": "Anota el jugador 7", "intencion": "gol", "entidades": {"jugador_num": 7}},
    {"texto": "Marca gol el número 7", "intencion": "gol", "entidades": {"jugador_num": 7}},
    {"texto": "El siete marca", "intencion": "gol", "entidades": {"jugador_num": 7}},
    {"texto": "Golazo del 7", "intencion": "gol", "entidades": {"jugador_num": 7}},
    {"texto": "Omar Uicab metió gol", "intencion": "gol", "entidades": {"jugador_nombre": "Omar Uicab"}},
    {"texto": "Gol de Eduardo Casanova", "intencion": "gol", "entidades": {"jugador_nombre": "Eduardo Casanova"}},
    {"texto": "Anota Ricardo Balam", "intencion": "gol", "entidades": {"jugador_nombre": "Ricardo Balam"}},
    {"texto": "Jugador 7 del equipo rojo anota", "intencion": "gol", "entidades": {"jugador_num": 7, "equipo": "rojo"}},
    {"texto": "Gol del equipo azul, número 10", "intencion": "gol", "entidades": {"jugador_num": 10, "equipo": "azul"}},
    
    # FALTAS - Más variaciones
    {"texto": "Falta del número 10", "intencion": "falta", "entidades": {"jugador_num": 10}},
    {"texto": "El jugador 10 cometió falta", "intencion": "falta", "entidades": {"jugador_num": 10}},
    {"texto": "Falta cometida por el 10", "intencion": "falta", "entidades": {"jugador_num": 10}},
    {"texto": "El 10 hace falta", "intencion": "falta", "entidades": {"jugador_num": 10}},
    {"texto": "Infracción del jugador 10", "intencion": "falta", "entidades": {"jugador_num": 10}},
    {"texto": "Pedro Sánchez hizo falta", "intencion": "falta", "entidades": {"jugador_nombre": "Pedro Sánchez"}},
    {"texto": "Falta de Mizael Uc", "intencion": "falta", "entidades": {"jugador_nombre": "Mizael Uc"}},
    {"texto": "Falta contra el número 3", "intencion": "falta", "entidades": {"jugador_afectado_num": 3}},
    {"texto": "Falta sobre el 3", "intencion": "falta", "entidades": {"jugador_afectado_num": 3}},
    
    # TARJETAS AMARILLAS - Más variaciones
    {"texto": "Tarjeta amarilla para el 8", "intencion": "tarjeta_amarilla", "entidades": {"jugador_num": 8}},
    {"texto": "Amarilla para el número 8", "intencion": "tarjeta_amarilla", "entidades": {"jugador_num": 8}},
    {"texto": "El 8 recibe amarilla", "intencion": "tarjeta_amarilla", "entidades": {"jugador_num": 8}},
    {"texto": "Amonestación para el 8", "intencion": "tarjeta_amarilla", "entidades": {"jugador_num": 8}},
    {"texto": "Amarilla para Luis Fernández", "intencion": "tarjeta_amarilla", "entidades": {"jugador_nombre": "Luis Fernández"}},
    {"texto": "Luis Fernández ve la amarilla", "intencion": "tarjeta_amarilla", "entidades": {"jugador_nombre": "Luis Fernández"}},
    {"texto": "Tarjeta para Luis Fernández", "intencion": "tarjeta_amarilla", "entidades": {"jugador_nombre": "Luis Fernández"}},
    
    # TARJETAS ROJAS - Más variaciones
    {"texto": "Tarjeta roja para el jugador número 5", "intencion": "tarjeta_roja", "entidades": {"jugador_num": 5}},
    {"texto": "Roja para el 5", "intencion": "tarjeta_roja", "entidades": {"jugador_num": 5}},
    {"texto": "El 5 es expulsado", "intencion": "tarjeta_roja", "entidades": {"jugador_num": 5}},
    {"texto": "Expulsión del número 5", "intencion": "tarjeta_roja", "entidades": {"jugador_num": 5}},
    {"texto": "El jugador 5 ve la roja", "intencion": "tarjeta_roja", "entidades": {"jugador_num": 5}},
    {"texto": "Roja directa para Eduardo Casanova", "intencion": "tarjeta_roja", "entidades": {"jugador_nombre": "Eduardo Casanova"}},
    {"texto": "Fuera el 5", "intencion": "tarjeta_roja", "entidades": {"jugador_num": 5}},
    
    # PENALES - Nueva categoría
    {"texto": "Penal para el equipo azul", "intencion": "penal", "entidades": {"equipo": "azul"}},
    {"texto": "Penalti a favor del equipo rojo", "intencion": "penal", "entidades": {"equipo": "rojo"}},
    {"texto": "Penalty", "intencion": "penal", "entidades": {}},
    {"texto": "Tiro penal", "intencion": "penal", "entidades": {}},
    {"texto": "Falta dentro del área", "intencion": "penal", "entidades": {}},
    {"texto": "Penal cobrado por el 10", "intencion": "penal", "entidades": {"jugador_num": 10}},
    {"texto": "Penal claro para el equipo local", "intencion": "penal", "entidades": {"equipo": "local"}},
    {"texto": "El árbitro marca penal para el visitante", "intencion": "penal", "entidades": {"equipo": "visitante"}},
    {"texto": "Penalti cometido dentro del área", "intencion": "penal", "entidades": {}},
    {"texto": "Señalan penalti después de revisar el VAR", "intencion": "penal", "entidades": {}},
    {"texto": "El defensa toca el balón con la mano en el área", "intencion": "penal", "entidades": {}},
    {"texto": "Penal claro para el equipo local", "intencion": "penal", "entidades": {"equipo": "local"}},
    {"texto": "El árbitro marca penal para el visitante", "intencion": "penal", "entidades": {"equipo": "visitante"}},
    {"texto": "Penal fallado por el delantero", "intencion": "penal", "entidades": {"jugador_posicion": "delantero"}},
    
    # CORNERS - Nueva categoría
    {"texto": "Córner para el equipo azul", "intencion": "corner", "entidades": {"equipo": "azul"}},
    {"texto": "Saque de esquina", "intencion": "corner", "entidades": {}},
    {"texto": "Corner", "intencion": "corner", "entidades": {}},
    {"texto": "Tiro de esquina para el rojo", "intencion": "corner", "entidades": {"equipo": "rojo"}},
    
    # OFFSIDE - Nueva categoría
    {"texto": "Fuera de juego", "intencion": "offside", "entidades": {}},
    {"texto": "Offside", "intencion": "offside", "entidades": {}},
    {"texto": "El 9 estaba en fuera de juego", "intencion": "offside", "entidades": {"jugador_num": 9}},
    {"texto": "Posición adelantada", "intencion": "offside", "entidades": {}},
    
    # CAMBIOS/SUSTITUCIONES - Más variaciones
    {"texto": "Cambio del 9 por el 11", "intencion": "cambio", "entidades": {"jugador_sale_num": 9, "jugador_entra_num": 11}},
    {"texto": "El 9 sale y entra el 11", "intencion": "cambio", "entidades": {"jugador_sale_num": 9, "jugador_entra_num": 11}},
    {"texto": "Sustitución: 9 por 11", "intencion": "cambio", "entidades": {"jugador_sale_num": 9, "jugador_entra_num": 11}},
    {"texto": "Entra el 11 por el 9", "intencion": "cambio", "entidades": {"jugador_sale_num": 9, "jugador_entra_num": 11}},
    {"texto": "Sustitución: sale Carlos Vega y entra Juan Pérez", "intencion": "cambio", 
     "entidades": {"jugador_sale_nombre": "Carlos Vega", "jugador_entra_nombre": "Juan Pérez"}},
    {"texto": "Juan Pérez reemplaza a Carlos Vega", "intencion": "cambio", 
     "entidades": {"jugador_sale_nombre": "Carlos Vega", "jugador_entra_nombre": "Juan Pérez"}},
    
    # TIEMPOS/PERÍODOS - Más variaciones
    {"texto": "Comienza el partido", "intencion": "inicio_partido", "entidades": {}},
    {"texto": "Inicia el encuentro", "intencion": "inicio_partido", "entidades": {}},
    {"texto": "Pitazo inicial", "intencion": "inicio_partido", "entidades": {}},
    {"texto": "Arranca el partido", "intencion": "inicio_partido", "entidades": {}},
    {"texto": "Inicia el primer tiempo", "intencion": "inicio_primer_tiempo", "entidades": {}},
    {"texto": "Comienza la primera mitad", "intencion": "inicio_primer_tiempo", "entidades": {}},
    {"texto": "Comienza el segundo tiempo", "intencion": "inicio_segundo_tiempo", "entidades": {}},
    {"texto": "Inicia la segunda mitad", "intencion": "inicio_segundo_tiempo", "entidades": {}},
    {"texto": "Segundo tiempo en marcha", "intencion": "inicio_segundo_tiempo", "entidades": {}},
    {"texto": "Fin del primer tiempo", "intencion": "fin_primer_tiempo", "entidades": {}},
    {"texto": "Termina la primera mitad", "intencion": "fin_primer_tiempo", "entidades": {}},
    {"texto": "Descanso", "intencion": "fin_primer_tiempo", "entidades": {}},
    {"texto": "Final del partido", "intencion": "fin_partido", "entidades": {}},
    {"texto": "Termina el encuentro", "intencion": "fin_partido", "entidades": {}},
    {"texto": "Pitazo final", "intencion": "fin_partido", "entidades": {}},
    {"texto": "Fin del juego", "intencion": "fin_partido", "entidades": {}},
    
    # TIEMPO ADICIONAL - Nueva categoría
    {"texto": "Tiempo adicional", "intencion": "tiempo_adicional", "entidades": {}},
    {"texto": "Tiempo de descuento", "intencion": "tiempo_adicional", "entidades": {}},
    {"texto": "Minutos adicionales", "intencion": "tiempo_adicional", "entidades": {}},
    {"texto": "5 minutos de descuento", "intencion": "tiempo_adicional", "entidades": {"minutos": 5}},
    
    # SAQUES - Nueva categoría  
    {"texto": "Saque de banda", "intencion": "saque_banda", "entidades": {}},
    {"texto": "Lateral", "intencion": "saque_banda", "entidades": {}},
    {"texto": "Saque de puerta", "intencion": "saque_puerta", "entidades": {}},
    {"texto": "Saque del portero", "intencion": "saque_puerta", "entidades": {}},
    {"texto": "Tiro libre", "intencion": "tiro_libre", "entidades": {}},
    {"texto": "Falta directa", "intencion": "tiro_libre", "entidades": {}},
]

# TÉCNICAS DE AUMENTO DE DATOS MEJORADAS
def aumentar_datos_inteligente(ejemplos):
    """Función mejorada para balancear el dataset"""
    ejemplos_aumentados = ejemplos.copy()
    
    # Contar ejemplos por clase
    conteo_clases = {}
    for ejemplo in ejemplos:
        clase = ejemplo["intencion"]
        conteo_clases[clase] = conteo_clases.get(clase, 0) + 1
    
    # Encontrar la clase con más ejemplos
    max_ejemplos = max(conteo_clases.values())
    print(f"Clase con más ejemplos: {max_ejemplos}")
    
    # Balancear clases minoritarias
    for clase, count in conteo_clases.items():
        if count < max_ejemplos // 2:  # Si tiene menos de la mitad del máximo
            ejemplos_clase = [ej for ej in ejemplos if ej["intencion"] == clase]
            necesarios = min(max_ejemplos // 2 - count, count * 2)  # Duplicar máximo
            
            print(f"Aumentando clase '{clase}': {count} -> {count + necesarios}")
            
            # Crear variaciones
            for _ in range(necesarios):
                ejemplo_base = np.random.choice(ejemplos_clase)
                nuevo_ejemplo = ejemplo_base.copy()
                
                # Aplicar transformaciones aleatorias
                texto = nuevo_ejemplo["texto"]
                
                # Variación 1: Cambiar orden de palabras (para algunas frases)
                if len(texto.split()) <= 6 and np.random.random() > 0.5:
                    palabras = texto.split()
                    if len(palabras) >= 3:
                        # Intercambiar algunas palabras
                        if "el" in palabras and "numero" in palabras:
                            texto = texto.replace("el numero", "numero")
                        elif "numero" in texto and "el" not in texto:
                            texto = texto.replace("numero", "el numero")
                
                # Variación 2: Sinónimos básicos
                sinonimos = {
                    "gol": ["tanto", "anotacion"],
                    "metio": ["anoto", "marco"],
                    "jugador": ["futbolista"],
                    "falta": ["infraccion"],
                    "tarjeta": ["cartulina"],
                    "amarilla": ["amarilla"],
                    "roja": ["roja"],
                    "cambio": ["sustitucion", "reemplazo"],
                    "comienza": ["inicia", "arranca"],
                    "termina": ["finaliza", "acaba"]
                }
                
                for palabra, opciones in sinonimos.items():
                    if palabra in texto and np.random.random() > 0.7:
                        nuevo_sinonimo = np.random.choice(opciones)
                        texto = texto.replace(palabra, nuevo_sinonimo, 1)
                
                nuevo_ejemplo["texto"] = texto
                ejemplos_aumentados.append(nuevo_ejemplo)
    
    # Variaciones de números escritos vs dígitos
    numeros_texto = {
        "1": "uno", "2": "dos", "3": "tres", "4": "cuatro", "5": "cinco",
        "6": "seis", "7": "siete", "8": "ocho", "9": "nueve", "10": "diez",
        "11": "once", "12": "doce", "13": "trece", "14": "catorce", "15": "quince"
    }
    
    ejemplos_base = ejemplos_aumentados.copy()
    for ejemplo in ejemplos_base:
        for num, texto_num in numeros_texto.items():
            if f" {num} " in ejemplo["texto"] or ejemplo["texto"].endswith(f" {num}"):
                nuevo_texto = ejemplo["texto"].replace(num, texto_num)
                nuevo_ejemplo = ejemplo.copy()
                nuevo_ejemplo["texto"] = nuevo_texto
                ejemplos_aumentados.append(nuevo_ejemplo)
    
    return ejemplos_aumentados

# Aumentar el dataset de forma inteligente
ejemplos_entrenamiento = aumentar_datos_inteligente(ejemplos_entrenamiento)

# Preprocesamiento mejorado
def limpiar_texto(texto):
    """Función para limpiar y normalizar el texto"""
    texto = texto.lower()
    # Eliminar acentos básicos
    texto = texto.replace('á', 'a').replace('é', 'e').replace('í', 'i')
    texto = texto.replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    # Normalizar espacios
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Aplicar limpieza
for ejemplo in ejemplos_entrenamiento:
    ejemplo["texto"] = limpiar_texto(ejemplo["texto"])

# Extraer textos y etiquetas
textos = [ejemplo["texto"] for ejemplo in ejemplos_entrenamiento]
intenciones = [ejemplo["intencion"] for ejemplo in ejemplos_entrenamiento]

print(f"Total de ejemplos de entrenamiento: {len(textos)}")
print(f"Distribución de clases:")
for intencion in set(intenciones):
    count = intenciones.count(intencion)
    print(f"  {intencion}: {count} ejemplos")

# Crear mapeo de intenciones
intenciones_unicas = sorted(list(set(intenciones)))
intencion_a_indice = {intencion: i for i, intencion in enumerate(intenciones_unicas)}
indice_a_intencion = {i: intencion for intencion, i in intencion_a_indice.items()}

# Convertir a one-hot encoding
y_intencion = np.zeros((len(intenciones), len(intenciones_unicas)))
for i, intencion in enumerate(intenciones):
    y_intencion[i, intencion_a_indice[intencion]] = 1

# Tokenización mejorada
max_palabras = 2000  # Aumentado para mejor vocabulario
max_longitud = 25    # Aumentado para frases más largas

tokenizer = Tokenizer(
    num_words=max_palabras, 
    oov_token="<OOV>",
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',  # Mantener números
    lower=True
)
tokenizer.fit_on_texts(textos)
secuencias = tokenizer.texts_to_sequences(textos)
X = pad_sequences(secuencias, maxlen=max_longitud, padding='post')

# División de datos para validación con estratificación mejorada
from sklearn.model_selection import StratifiedShuffleSplit

# Verificar distribución de clases
print("Distribución original de clases:")
clase_counts = {}
for intencion in intenciones:
    clase_counts[intencion] = clase_counts.get(intencion, 0) + 1

print("Clases con pocos ejemplos (< 4):")
clases_pequenas = []
for clase, count in clase_counts.items():
    print(f"  {clase}: {count} ejemplos")
    if count < 4:
        clases_pequenas.append(clase)

# Para clases con muy pocos ejemplos, usar un enfoque diferente
if clases_pequenas:
    print(f"\nATENCIÓN: {len(clases_pequenas)} clases tienen menos de 4 ejemplos.")
    print("Estas clases pueden no aparecer en validación:")
    for clase in clases_pequenas:
        print(f"  - {clase}")

# Usar StratifiedShuffleSplit para mejor control
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

try:
    train_idx, val_idx = next(splitter.split(X, intenciones))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_intencion[train_idx], y_intencion[val_idx]
    
    print(f"\nDivisión exitosa:")
    print(f"Datos de entrenamiento: {X_train.shape[0]}")
    print(f"Datos de validación: {X_val.shape[0]}")
    
    # Verificar que todas las clases estén en validación
    y_val_classes = np.argmax(y_val, axis=1)
    clases_en_val = set(y_val_classes)
    clases_faltantes = set(range(len(intenciones_unicas))) - clases_en_val
    
    if clases_faltantes:
        print(f"ADVERTENCIA: {len(clases_faltantes)} clases no están en validación:")
        for clase_idx in clases_faltantes:
            print(f"  - {indice_a_intencion[clase_idx]}")
    
except ValueError as e:
    print(f"Error en estratificación: {e}")
    print("Usando división simple...")
    # Fallback a división simple si la estratificación falla
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_intencion, test_size=0.25, random_state=42
    )

# MODELO MEJORADO CON ARQUITECTURA MÁS SOFISTICADA
def crear_modelo_avanzado(vocab_size, embedding_dim=256, lstm_units=128):
    entrada = Input(shape=(max_longitud,))
    
    # Embedding layer mejorado
    x = Embedding(
        vocab_size, 
        embedding_dim, 
        input_length=max_longitud,
        embeddings_regularizer=l2(0.001)
    )(entrada)
    x = Dropout(0.2)(x)
    
    # Múltiples capas LSTM bidireccionales
    x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = LayerNormalization()(x)
    
    x = Bidirectional(LSTM(lstm_units//2, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = LayerNormalization()(x)
    
    # Pooling global mejorado
    max_pool = GlobalMaxPooling1D()(x)
    
    # Capas densas con regularización
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(max_pool)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Capa de salida
    salida = Dense(len(intenciones_unicas), activation='softmax')(x)
    
    modelo = Model(inputs=entrada, outputs=salida)
    
    # Optimizador mejorado
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    modelo.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return modelo


# Convertir y_true a formato numérico
y_true_classes = np.argmax(y_intencion, axis=1)

# Calcular pesos de clase
class_weights = compute_class_weight('balanced', classes=np.unique(y_true_classes), y=y_true_classes)
class_weights_dict = dict(enumerate(class_weights))

print("\nPesos de clase calculados:")
for idx, weight in class_weights_dict.items():
    print(f"Clase '{indice_a_intencion[idx]}': {weight:.4f}")


# Crear el modelo
vocab_size = min(max_palabras, len(tokenizer.word_index) + 1)
modelo_intencion = crear_modelo_avanzado(vocab_size)
modelo_intencion.summary()

# Callbacks mejorados
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------

# Entrenamiento con más épocas y mejores parámetros
print("Iniciando entrenamiento...")
history = modelo_intencion.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  # Más épocas con early stopping
    batch_size=8,  # Batch size más pequeño para mejor generalización
    callbacks=callbacks,
    verbose=1,
    class_weight=class_weights_dict
)

# Evaluación detallada
print("\nEvaluación en datos de validación:")
val_loss, val_accuracy, val_top_k = modelo_intencion.evaluate(X_val, y_val, verbose=0)
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Top-K Accuracy: {val_top_k:.4f}")

# Predicciones para reporte detallado
y_pred = modelo_intencion.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

print("\nReporte de clasificación:")
target_names = [indice_a_intencion[i] for i in range(len(intenciones_unicas))]

# Verificar qué clases están presentes en validación
clases_presentes = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
print(f"Clases en validación: {len(clases_presentes)} de {len(intenciones_unicas)} totales")

# Crear labels y target_names solo para las clases presentes
labels_presentes = sorted(clases_presentes)
target_names_presentes = [indice_a_intencion[i] for i in labels_presentes]

print(classification_report(
    y_true_classes, 
    y_pred_classes, 
    labels=labels_presentes,
    target_names=target_names_presentes,
    zero_division=0
))

# Mostrar matriz de confusión
print("\nMatriz de confusión:")
cm = confusion_matrix(y_true_classes, y_pred_classes, labels=labels_presentes)
print("Clases:", target_names_presentes)
print(cm)

# Análisis detallado por clase
print("\nAnálisis detallado por clase:")
print("-" * 50)
for i, clase in enumerate(intenciones_unicas):
    count_total = intenciones.count(clase)
    count_train = [intenciones[j] for j in range(len(intenciones)) if j in X_train[:, 0].tolist()].count(clase)
    count_val = [intenciones[j] for j in range(len(intenciones)) if j in X_val[:, 0].tolist()].count(clase)
    
    print(f"{clase}:")
    print(f"  Total: {count_total}, Train: aprox. {int(count_total * 0.8)}, Val: aprox. {int(count_total * 0.2)}")
    
    if i in clases_presentes:
        indices_clase = np.where(y_true_classes == i)[0]
        if len(indices_clase) > 0:
            precisiones_clase = y_pred[indices_clase, i]
            print(f"  Presente en validación: {len(indices_clase)} ejemplos")
            print(f"  Confianza promedio: {np.mean(precisiones_clase):.3f}")
        else:
            print(f"  No presente en set de validación")
    else:
        print(f"  No predicha correctamente en validación")



# Extractor de entidades mejorado
def extraer_entidades_mejorado(texto, intencion):
    """Extractor de entidades mejorado con más patrones"""
    entidades = {}
    texto_original = texto
    texto = texto.lower()
    
    # Mapeo de números escritos a dígitos
    numeros_texto = {
        "uno": "1", "dos": "2", "tres": "3", "cuatro": "4", "cinco": "5",
        "seis": "6", "siete": "7", "ocho": "8", "nueve": "9", "diez": "10",
        "once": "11", "doce": "12", "trece": "13", "catorce": "14", "quince": "15"
    }
    
    for palabra, numero in numeros_texto.items():
        texto = texto.replace(palabra, numero)
    
    # Patrones para números de jugadores
    if intencion in ["gol", "falta", "tarjeta_amarilla", "tarjeta_roja", "offside"]:
        patrones_jugador = [
            r'(?:jugador\s+(?:número\s+)?|número\s+|#|el\s+)(\d+)',
            r'(?:^|\s)(\d+)(?:\s|$)',  # Número solo
        ]
        
        for patron in patrones_jugador:
            match = re.search(patron, texto)
            if match:
                entidades["jugador_num"] = int(match.group(1))
                break
    
    # Para cambios
    if intencion == "cambio":
        patrones_cambio = [
            r'(\d+)\s+(?:por|y entra|reemplaza)\s+(?:el\s+)?(\d+)',
            r'(?:entra\s+(?:el\s+)?(\d+)\s+por\s+(?:el\s+)?(\d+))',
            r'(?:sale\s+(?:el\s+)?(\d+)\s+(?:y\s+)?entra\s+(?:el\s+)?(\d+))'
        ]
        
        for patron in patrones_cambio:
            match = re.search(patron, texto)
            if match:
                if "entra" in patron and "por" in patron:
                    entidades["jugador_entra_num"] = int(match.group(1))
                    entidades["jugador_sale_num"] = int(match.group(2))
                else:
                    entidades["jugador_sale_num"] = int(match.group(1))
                    entidades["jugador_entra_num"] = int(match.group(2))
                break
    
    # Equipos
    if "equipo" in intencion or intencion in ["gol", "penal", "corner"]:
        if "rojo" in texto:
            entidades["equipo"] = "rojo"
        elif "azul" in texto:
            entidades["equipo"] = "azul"
    
    # Tiempo adicional
    if intencion == "tiempo_adicional":
        match = re.search(r'(\d+)\s+minutos?', texto)
        if match:
            entidades["minutos"] = int(match.group(1))
    
    return entidades

# Función de procesamiento mejorada
def procesar_texto_mejorado(texto):
    texto_limpio = limpiar_texto(texto)
    secuencia = tokenizer.texts_to_sequences([texto_limpio])
    secuencia_padded = pad_sequences(secuencia, maxlen=max_longitud, padding='post')
    
    prediccion = modelo_intencion.predict(secuencia_padded, verbose=0)[0]
    intencion_idx = np.argmax(prediccion)
    intencion = indice_a_intencion[intencion_idx]
    confianza = prediccion[intencion_idx]
    
    # Top 3 predicciones para debugging
    top_3_idx = np.argsort(prediccion)[-3:][::-1]
    top_3 = [(indice_a_intencion[i], prediccion[i]) for i in top_3_idx]
    
    entidades = extraer_entidades_mejorado(texto, intencion)
    
    return {
        "intencion": intencion,
        "confianza": float(confianza),
        "entidades": entidades,
        "top_3_predicciones": top_3
    }

# Pruebas exhaustivas
ejemplos_prueba = [
    "Lateral para el equipo rojo",
    "Tiro libre directo",
    "Penalti claro para el azul",
    "Fuera de juego del 9",
    "Empieza el partido ahora",
    "Tres minutos de tiempo extra",
    "Omar Uicab metió un golazo",
    "Tarjeta amarilla para Román Pérez del equipo rojiblanco",
    "Cambio: sale el 8 y entra el 14",
    "Comienza el segundo tiempo",
    "Penal para el equipo azul",
    "Fuera de juego del número 9",
    "Córner para el rojo",
    "El siete anota un gol espectacular",
    "Roja directa para Eduardo Casanova",
    "Córner para el equipo rojo",
    "Tiro libre directo",
    "Penalti claro para el azul",
    "Fuera de juego del 9",
    "Empieza el partido ahora",
    "Tres minutos de tiempo extra",
    "Cinco minutos de tiempo adicional",
    "Tarjeta roja para Juan Escutia"
]

print("\n" + "="*50)
print("PRUEBAS DEL MODELO MEJORADO")
print("="*50)

for ejemplo in ejemplos_prueba:
    resultado = procesar_texto_mejorado(ejemplo)
    print(f"\nTexto: '{ejemplo}'")
    print(f"Intención: {resultado['intencion']} (confianza: {resultado['confianza']:.3f})")
    if resultado['entidades']:
        print(f"Entidades: {resultado['entidades']}")
    print(f"Top 3: {[(intent, f'{conf:.3f}') for intent, conf in resultado['top_3_predicciones']]}")

# Guardar archivos
modelo_intencion.save('intencion_model_mejorado.keras')
# modelo_intencion.export('intencion_model_mejorado_tflite')

with open('tokenizer_mejorado.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('intencion_mapeo_mejorado.json', 'w', encoding='utf-8') as f:
    json.dump({
        "intencion_a_indice": intencion_a_indice,
        "indice_a_intencion": indice_a_intencion,
        "max_longitud": max_longitud,
        "vocab_size": vocab_size
    }, f, ensure_ascii=False, indent=2)


# Convertir a TensorFlow Lite con soporte para operaciones de TF
def convertir_a_tflite_optimizado(keras_model_path, tflite_path):
    # Cargar el modelo Keras
    model = tf.keras.models.load_model(keras_model_path)

    # Configurar el convertidor desde el modelo Keras
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimización y cuantización
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # Para reducir tamaño

    # ✅ Permitir operaciones de TensorFlow (Select TF Ops)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Operaciones nativas de TFLite
        tf.lite.OpsSet.SELECT_TF_OPS    # Añadir operaciones de TensorFlow
    ]

    # ✅ Desactivar la bajada de TensorList (opcional)
    converter._experimental_lower_tensor_list_ops = False

    # Convertir
    tflite_model = converter.convert()
    
    # Guardar el modelo TFLite
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Modelo TFLite guardado en {tflite_path}")
    print(f"Tamaño del modelo: {len(tflite_model) / 1024:.2f} KB")

# Llamar a la función con el archivo .keras
convertir_a_tflite_optimizado('intencion_model_mejorado.keras', 'soccer_events_model_mejorado.tflite')


print(f"\nArchivos generados:")
print("- soccer_events_model_mejorado.tflite")
print("- tokenizer_mejorado.pickle") 
print("- intencion_mapeo_mejorado.json")
