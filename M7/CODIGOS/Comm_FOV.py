import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
from PIL import Image
import chardet
import os


# Parámetros de la Tierra
R_EARTH = 6371  # Radio de la Tierra en km
C = 3e8  # Velocidad de la luz en m/s
K = 1.38e-23  # Constante de Boltzmann en J/K

# Función para detectar codificación
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']
# Función para calcular el radio geográfico del FOV
def calculate_fov_radius(altitude_km, angle_of_view_deg):
    """
    Calcula el radio geográfico del FOV en la superficie de la Tierra
    basado en la altitud del satélite y el ángulo de visión.
    """
    angle_of_view_rad = np.radians(angle_of_view_deg)
    #print(f"Angle of view in radians: {angle_of_view_rad}")  # Imprimir el valor de angle_of_view_rad
    
    rho = np.arcsin(R_EARTH / (R_EARTH + altitude_km))  # Ángulo de la Tierra visto desde el satélite
    #print(f"Rho in radians: {rho}")  # Imprimir el valor de rho
    elevation_min_rad = np.arccos(np.sin(angle_of_view_rad/2) / np.sin(rho))  # Ángulo de elevación mínimo
    elevation_min_deg = np.degrees(elevation_min_rad)  # Convertir a grados
    #print(f"Elevacion min en grados: {elevation_min_deg}")  # Imprimir el valor de elevation_min_rad
    lambda_min_rad = (np.pi/2) - angle_of_view_rad/2 - elevation_min_rad  # Ángulo entre el borde del FOV y el centro de la Tierra
    lambda_min_deg = np.degrees(lambda_min_rad)  # Convertir a grados
    #print(f"Lambda min en grados: {lambda_min_deg}")  # Imprimir el valor de lambda_min
    r_fov = R_EARTH *lambda_min_rad  # Radio en km en la superficie
    radius_deg = np.degrees(r_fov / R_EARTH)    # Convertir a grados geográficos
    return radius_deg

# Función para calcular pérdidas por propagación libre
def calculate_free_space_loss(distance_km, frequency_mhz):
    distance_m = distance_km * 1e3  # Convertir km a metros
    frequency_hz = frequency_mhz * 1e6  # Convertir MHz a Hz
    loss_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) - 20 * np.log10(C / (4 * np.pi))
    return loss_db

# Función para calcular potencia recibida
def calculate_received_power(transmitter_power_dbm, transmitter_gain_db, receiver_gain_db, free_space_loss_db, other_losses_db=0):
    received_power_dbm = transmitter_power_dbm + transmitter_gain_db + receiver_gain_db - free_space_loss_db - other_losses_db
    return received_power_dbm

# Función para calcular SNR
def calculate_snr(received_power_dbm, bandwidth_hz, temperature_k=290):
    noise_power_dbm = 10 * np.log10(K * temperature_k * bandwidth_hz * 1000)  # Convertir ruido a dBm
    snr_db = received_power_dbm - noise_power_dbm
    return snr_db

# Convertir coordenadas cartesianas Earth-Fixed a latitud y longitud
def cartesian_to_geographic(x, y, z):
    lon = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, hyp))
    return lat, lon

sat_files_dir = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\Nodupla"

all_satellite_data = pd.DataFrame()

iot_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\IoT_Clean.txt"
encoding = detect_file_encoding(iot_file)

# Leer datos de todos los satélites (del 1 al 22)
for sat_num in range(1, 23):  # Cambiar el rango si tienes más o menos satélites
    sat_file = os.path.join(sat_files_dir, f"sat_{sat_num}_File.txt")
    
    # Verifica si el archivo existe
    if not os.path.exists(sat_file):
        print(f"Archivo no encontrado: {sat_file}")
        continue
    
    try:
        # Leer archivo del satélite
        sat_data = pd.read_csv(sat_file, sep='\\s+', encoding='latin-1', engine='python')
        # Agregar una columna para identificar el satélite
        sat_data['satellite_id'] = sat_num
        # Renombrar columnas relevantes
        sat_data.rename(columns={
            f'sat_{sat_num}.Earth.Latitude': 'Latitude',
            f'sat_{sat_num}.Earth.Longitude': 'Longitude'
        }, inplace=True)
        # Agregar al DataFrame general
        all_satellite_data = pd.concat([all_satellite_data, sat_data], ignore_index=True)
    except Exception as e:
        print(f"Error al leer el archivo {sat_file}: {e}")

# Verificar que se hayan cargado datos de satélites
if all_satellite_data.empty:
    print("No se cargaron datos de satélites. Verifica los archivos.")
    exit()

# Leer y procesar datos de dispositivos IoT

try:
    iot_data = pd.read_csv(iot_file, sep='\\s+', encoding=encoding, engine='python')
    print("Archivo IoT leído correctamente.")
except Exception as e:
    print(f"Error al leer el archivo IoT: {e}")
# Crear un DataFrame con las posiciones de los dispositivos
devices = []
for i in range(1, len(iot_data.columns) // 3 + 1):
    x_col = f'IoT_{i}.EarthFixed.X'
    y_col = f'IoT_{i}.EarthFixed.Y'
    z_col = f'IoT_{i}.EarthFixed.Z'
    
    if x_col in iot_data.columns and y_col in iot_data.columns and z_col in iot_data.columns:
        devices.append({
            'id': i,
            'x': iot_data[x_col].iloc[0],
            'y': iot_data[y_col].iloc[0],
            'z': iot_data[z_col].iloc[0]
        })

iot_df = pd.DataFrame(devices)



# Convertir posiciones a coordenadas geográficas
if not iot_df.empty:
    iot_df['lat'], iot_df['lon'] = zip(*iot_df.apply(lambda row: cartesian_to_geographic(row['x'], row['y'], row['z']), axis=1))
else:
    print("Error: No hay datos válidos en iot_df después de filtrar.")

# Agregar altitud y FOV predeterminados
all_satellite_data['alt'] = 850  # Altitud en km
all_satellite_data['fov_angle'] = 100  # Ángulo de visión en grados

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
plt.title("FOV Dinámico de Satélites")
plt.xlabel("Longitud")
plt.ylabel("Latitud")

# Mantener referencia a la figura para dibujar círculos
background = plt.imread(r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\Atlantis.png")
ax.imshow(background, extent=[-180, 180, -90, 90], aspect='auto')

# Variable para guardar el círculo actual
# Variable para guardar el círculo actual
fov_fill = None
fov_border = None

# Parámetros de enlace
transmitter_power_dbm = 14  # Potencia del transmisor en dBm
transmitter_gain_db = 2  # Ganancia de la antena transmisora en dB
receiver_gain_db = 20  # Ganancia de la antena receptora en dB
bandwidth_hz = 125e3  # Ancho de banda en Hz
frequency_mhz = 868  # Frecuencia en MHz

# Crear listas para almacenar los círculos FOV
fov_fills = []  # Lista para los rellenos de los FOV
fov_borders = []  # Lista para los bordes de los FOV
# Crear un DataFrame para guardar los resultados
resultados = []

# Animar el movimiento dinámico
for index, sat in all_satellite_data.iterrows():
    lat, lon, alt, fov_angle = sat['Latitude'], sat['Longitude'], sat['alt'], sat['fov_angle']
    radius_deg = calculate_fov_radius(alt, fov_angle)  # Calcular radio en grados geográficos
    
    # Generar puntos del círculo
    num_points = 100
    angles = np.linspace(0, 2 * np.pi, num_points)
    circle_points_lon = lon + radius_deg * np.cos(angles)
    circle_points_lat = lat + radius_deg * np.sin(angles)

    

    # Dibujar el círculo con relleno (rosa) y borde (negro)
    fov_fill = ax.fill(circle_points_lon, circle_points_lat, color='pink', alpha=0.5)[0]  # Relleno rosa
    fov_border, = ax.plot(circle_points_lon, circle_points_lat, color='black', linewidth=1)  # Borde negro
    
    # Verificar dispositivos dentro del FOV
    dispositivos_vistos = []  # Lista para dispositivos dentro del FOV
    for _, device in iot_df.iterrows():
        device_point = Point(device['lon'], device['lat'])
        fov_polygon = Polygon(zip(circle_points_lon, circle_points_lat))
        if fov_polygon.contains(device_point):
            distance_km = np.sqrt((R_EARTH + alt)**2 - R_EARTH**2)
            free_space_loss_db = calculate_free_space_loss(distance_km, frequency_mhz)
            received_power_dbm = calculate_received_power(
                transmitter_power_dbm,
                transmitter_gain_db,
                receiver_gain_db,
                free_space_loss_db
            )
            snr_db = calculate_snr(received_power_dbm, bandwidth_hz)
            # Agregar los resultados al DataFrame
            resultados.append({
                "Satélite": sat['satellite_id'],
                "Dispositivo IoT": device['id'],
                "Distancia (km)": distance_km,
                "SNR (dB)": snr_db,
                "Potencia Recibida (dBm)": received_power_dbm
            })
    
    # Imprimir resultados solo si hay dispositivos dentro del FOV
    if dispositivos_vistos:
        print(f"Satélite {sat['satellite_id']} ha visto los siguientes dispositivos:")
        for dispositivo in dispositivos_vistos:
            print(dispositivo)
    # Pausar para mostrar el movimiento
    plt.pause(0.01)  # Cambia el tiempo de pausa según sea necesario
# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame(resultados)

# Guardar los resultados en un archivo Excel
output_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\resultados_satélites.xlsx"
df_resultados.to_excel(output_file, index=False)

print(f"Los resultados se han guardado en {output_file}")
plt.show()