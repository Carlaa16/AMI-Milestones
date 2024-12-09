import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.geometry import Point, Polygon
import pandas as pd
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
    angle_of_view_rad = np.radians(angle_of_view_deg)
    rho = np.arcsin(R_EARTH / (R_EARTH + altitude_km))  # Ángulo de la Tierra visto desde el satélite
    elevation_min_rad = np.arccos(np.sin(angle_of_view_rad/2) / np.sin(rho))  # Ángulo de elevación mínimo
    lambda_min_rad = (np.pi/2) - angle_of_view_rad/2 - elevation_min_rad  # Ángulo entre el borde del FOV y el centro de la Tierra
    r_fov = R_EARTH * lambda_min_rad  # Radio en km
    radius_deg = np.degrees(r_fov / R_EARTH)  # Convertir a grados geográficos
    return radius_deg

# Convertir coordenadas cartesianas Earth-Fixed a latitud y longitud
def cartesian_to_geographic(x, y, z):
    lon = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, hyp))
    return lat, lon

# Función para calcular el radio de cobertura de dispositivos IoT
def calculate_device_coverage_radius(beamwidth_deg):
    beamwidth_rad = np.radians(beamwidth_deg)
    radius_km = R_EARTH * np.tan(beamwidth_rad / 2)
    return radius_km

# Función para generar un círculo de cobertura de dispositivos IoT
def generate_device_coverage_circle(lat, lon, radius_km, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    circle_lat = lat + (radius_km / R_EARTH) * (180 / np.pi) * np.sin(angles)
    circle_lon = lon + (radius_km / R_EARTH) * (180 / np.pi) * np.cos(angles) / np.cos(np.radians(lat))
    return circle_lat, circle_lon

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


# Parámetro del beamwidth para dispositivos IoT
beamwidth = 10  # Ajusta este valor según sea necesario


# Función para cargar y procesar los datos de satélites
def load_satellite_data(sat_files_dir, num_sats):
    all_satellite_data = pd.DataFrame()
    for sat_num in range(num_sats):
        sat_file = os.path.join(sat_files_dir, f"cleaned_sat_{sat_num}_ReportFile.txt")
        if not os.path.exists(sat_file):
            print(f"Archivo no encontrado: {sat_file}")
            continue
        try:
            sat_data = pd.read_csv(sat_file, sep='\\s+', encoding='latin-1', engine='python')
            sat_data['satellite_id'] = sat_num
            sat_data.rename(columns={
                f'sat_{sat_num}.Earth.Latitude': 'Latitude',
                f'sat_{sat_num}.Earth.Longitude': 'Longitude'
            }, inplace=True)
            all_satellite_data = pd.concat([all_satellite_data, sat_data], ignore_index=True)
        except Exception as e:
            print(f"Error al leer el archivo {sat_file}: {e}")
    return all_satellite_data

# Función para cargar los datos de IoT
def load_iot_data(iot_file):
    encoding = detect_file_encoding(iot_file)
    iot_data = pd.read_csv(iot_file, sep='\\s+', encoding=encoding, engine='python')
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
    if not iot_df.empty:
        iot_df['lat'], iot_df['lon'] = zip(*iot_df.apply(lambda row: cartesian_to_geographic(row['x'], row['y'], row['z']), axis=1))
    return iot_df

# Directorios y archivos
sat_files_dir = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\CleanedReports"
iot_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\CleanedReports\cleaned_IoT_ReportFile.txt"
background_image_path = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\Atlantis.png"

# Cargar datos
num_sats = 5
all_satellite_data = load_satellite_data(sat_files_dir, num_sats)
iot_df = load_iot_data(iot_file)

if all_satellite_data.empty:
    print("No se cargaron datos de satélites. Verifica los archivos.")
    exit()

if iot_df.empty:
    print("No se cargaron datos de dispositivos IoT. Verifica los archivos.")
    exit()

# Agregar altitud y FOV predeterminados
all_satellite_data['alt'] = 850  # Altitud en km
all_satellite_data['fov_angle'] = 100  # Ángulo de visión en grados


# Parámetros de enlace
transmitter_power_dbm = 14  # Potencia del transmisor en dBm
transmitter_gain_db = 2  # Ganancia de la antena transmisora en dB
receiver_gain_db = 20  # Ganancia de la antena receptora en dB
bandwidth_hz = 125e3  # Ancho de banda en Hz
frequency_mhz = 868  # Frecuencia en MHz

resultados = []

# Crear un mapa de colores
num_sats = len(all_satellite_data['satellite_id'].unique())  # Número de satélites únicos
cmap = cm.get_cmap('tab10', num_sats)  # Usar una paleta predefinida con 'num_sats' colores

# Configuración: elegir si los satélites se muestran de golpe o animados
animated = False  # Cambia a True para mostrar los satélites uno por uno

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
plt.title("FOV de Satélites e IoT")
plt.xlabel("Longitud")
plt.ylabel("Latitud")

# Dibujar fondo
background = plt.imread(background_image_path)
ax.imshow(background, extent=[-180, 180, -90, 90], aspect='auto')

# Dibujar las coberturas IoT con círculos más definidos (independiente del modo)
for _, device in iot_df.iterrows():
    radius_km = calculate_device_coverage_radius(beamwidth)
    circle_lat, circle_lon = generate_device_coverage_circle(device['lat'], device['lon'], radius_km)
    ax.plot(circle_lon, circle_lat, color='red', linewidth=1.5, label=f'Cobertura IoT {device["id"]}')  # Borde del círculo
    ax.fill(circle_lon, circle_lat, color='red', alpha=0.2)  # Relleno del círculo

# Animar el movimiento dinámico o procesar todo de golpe según el modo
if animated:
    print("Modo animado: los satélites se mostrarán uno por uno.")
    for index, sat in all_satellite_data.iterrows():
        lat, lon, alt, fov_angle = sat['Latitude'], sat['Longitude'], sat['alt'], sat['fov_angle']
        radius_deg = calculate_fov_radius(alt, fov_angle)  # Calcular radio en grados geográficos
        
        # Generar puntos del círculo
        num_points = 100
        angles = np.linspace(0, 2 * np.pi, num_points)
        circle_points_lon = lon + radius_deg * np.cos(angles)
        circle_points_lat = lat + radius_deg * np.sin(angles)

        # Asignar un color único basado en el ID del satélite
        color = cmap(sat['satellite_id'] % num_sats)

        # Dibujar el círculo con relleno y borde
        fov_fill = ax.fill(circle_points_lon, circle_points_lat, color=color, alpha=0.5)[0]  # Relleno
        fov_border, = ax.plot(circle_points_lon, circle_points_lat, color='black', linewidth=1)  # Borde negro

        # Pausar para mostrar el movimiento
        plt.pause(0.05)  # Cambia el tiempo de pausa según sea necesario
else:
    print("Modo estático: todos los satélites se mostrarán de golpe.")
    for sat_id in all_satellite_data['satellite_id'].unique():
        sat_data = all_satellite_data[all_satellite_data['satellite_id'] == sat_id]
        color = cmap(sat_id % num_sats)  # Color único para cada satélite
        for _, sat in sat_data.iterrows():
            lat, lon, alt, fov_angle = sat['Latitude'], sat['Longitude'], sat['alt'], sat['fov_angle']
            radius_deg = calculate_fov_radius(alt, fov_angle)  # Calcular radio en grados geográficos

            # Generar puntos del círculo
            num_points = 100
            angles = np.linspace(0, 2 * np.pi, num_points)
            circle_points_lon = lon + radius_deg * np.cos(angles)
            circle_points_lat = lat + radius_deg * np.sin(angles)

            # Dibujar la cobertura del satélite
            ax.fill(circle_points_lon, circle_points_lat, color=color, alpha=0.3, label=f'Satélite {sat_id}')
            ax.plot(circle_points_lon, circle_points_lat, color='black', linewidth=1)

# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame(resultados)

# Guardar los resultados en un archivo Excel
output_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\resultados_satélites.xlsx"
df_resultados.to_excel(output_file, index=False)

print(f"Los resultados se han guardado en {output_file}")



# Mostrar la figura final
plt.show()