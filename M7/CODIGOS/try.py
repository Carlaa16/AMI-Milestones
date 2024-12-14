import numpy as np
from math import radians, sin, cos, sqrt, atan2
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

############################################################################################################
############################################################################################################
############################################################################################################

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

############################################################################################################
############################################################################################################
############################################################################################################

# Función para calcular pérdidas por propagación libre
def free_space_loss(frequency_hz, distance_km):
    distance_m = distance_km * 1e3
    return 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) - 20 * np.log10(C / (4 * np.pi))

# Función para ajustar la potencia del transmisor respetando el límite de EIRP
def adjust_tx_power(tx_power_dbw, use_amplifier, amplifier_gain_db, amplifier_losses_db, antenna_gain_db, cable_losses_db, max_eirp_dbw):
    if use_amplifier:
        adjusted_power = tx_power_dbw + amplifier_gain_db - amplifier_losses_db
    else:
        adjusted_power = tx_power_dbw

    # Calcular el EIRP
    eirp_dbw = adjusted_power + antenna_gain_db - cable_losses_db

    # Ajustar si el EIRP excede el límite
    if eirp_dbw > max_eirp_dbw:
        adjusted_power -= (eirp_dbw - max_eirp_dbw)

    print(f"Adjusted Tx Power: {adjusted_power:.2f} dBW, EIRP: {eirp_dbw:.2f} dBW")
    return adjusted_power, eirp_dbw

# Función para calcular Eb/No
def calculate_eb_no(eirp_dbw, losses_db, rx_gain_db, system_noise_temp_k, data_rate_bps):
    log_t_s = 10 * np.log10(system_noise_temp_k)
    log_r = 10 * np.log10(data_rate_bps)
    return eirp_dbw - losses_db + rx_gain_db + 228.6 - log_t_s - log_r


############################################################################################################
############################################################################################################
############################################################################################################

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
num_sats = 13
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

frequency_hz = 2.4e9  # Frecuencia en Hz
tx_power_dbw = 20 - 30  # Potencia inicial del transmisor en dBW
use_amplifier = False  # Usar amplificador (True o False)
amplifier_gain_db = 20  # Ganancia del amplificador en dB
amplifier_losses_db = 1  # Pérdidas del amplificador en dB
antenna_gain_db = 12.5  # Ganancia de la antena en dB
cable_losses_db = 2  # Pérdidas por cable en dB
max_eirp_dbw = 14 - 30  # EIRP máximo permitido
rx_gain_db = 6.5  # Ganancia del receptor en dB
system_noise_temp_k = 615  # Temperatura de ruido del sistema en Kelvin
data_rate_bps = 1200  # Tasa de datos en bps


resultados = []

# Crear un mapa de colores
num_sats = len(all_satellite_data['satellite_id'].unique())  # Número de satélites únicos
cmap = cm.get_cmap('tab10', num_sats)  # Usar una paleta predefinida con 'num_sats' colores

def calculate_3d_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Calcula la distancia 3D entre dos puntos considerando latitud, longitud y altitud.
    """
    # Convertir grados a radianes
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Diferencias entre las coordenadas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula de haversine para calcular la distancia en superficie
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    surface_distance_km = R_EARTH * c  # Distancia en km en la superficie

    # Incorporar altitud para obtener la distancia 3D
    total_distance_km = sqrt(surface_distance_km**2 + (alt1 - alt2)**2)
    return total_distance_km


# Configuración: elegir si los satélites se muestran de golpe o animados
# Configuración: elegir si los satélites se muestran de golpe o animados
animated = True  # Cambia a True para mostrar los satélites uno por uno

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

# Procesar los satélites (modo estático o animado)
print("Modo estático: todos los satélites se mostrarán de golpe.")
for sat_id in all_satellite_data['satellite_id'].unique():
    sat_data = all_satellite_data[all_satellite_data['satellite_id'] == sat_id]
    color = cmap(sat_id % num_sats)  # Color único para cada satélite

    for _, sat in sat_data.iterrows():
        lat, lon, alt, fov_angle = sat['Latitude'], sat['Longitude'], sat['alt'], sat['fov_angle']

        # Calcular el radio del FOV en kilómetros
        fov_radius_km = calculate_fov_radius(alt, fov_angle)
        # Generar puntos del círculo
        # Generar puntos del círculo del FOV
        num_points = 100
        angles = np.linspace(0, 2 * np.pi, num_points)
        circle_points_lat = lat + fov_radius_km * np.sin(angles)
        circle_points_lon = lon + fov_radius_km * np.cos(angles)

        # Dibujar la cobertura del satélite
        ax.fill(circle_points_lon, circle_points_lat, color=color, alpha=0.3, label=f'Satélite {sat_id}')
        ax.plot(circle_points_lon, circle_points_lat, color='black', linewidth=1)

        # Dibujar la cobertura del satélite
        ax.fill(circle_points_lon, circle_points_lat, color=color, alpha=0.3, label=f'Satélite {sat_id}')
        ax.plot(circle_points_lon, circle_points_lat, color='black', linewidth=1)

        # Verificar dispositivos IoT dentro del FOV
        fov_polygon = Polygon(zip(circle_points_lon, circle_points_lat))
        for _, device in iot_df.iterrows():
            device_point = Point(device['lon'], device['lat'])
            if fov_polygon.contains(device_point):
                distance_km = np.sqrt((R_EARTH + alt)**2 - R_EARTH**2)

                # Uplink: calcular parámetros y Eb/No
                uplink_tx_power_dbw, uplink_eirp = adjust_tx_power(
                    tx_power_dbw=20 - 30,  # Potencia inicial del transmisor en dBW
                    use_amplifier=False,
                    amplifier_gain_db=20,
                    amplifier_losses_db=1,
                    antenna_gain_db=12.5,
                    cable_losses_db=2,
                    max_eirp_dbw=20 - 30,
                )
                uplink_losses_db = free_space_loss(frequency_hz=2.4e9, distance_km=distance_km) + 2
                eb_no_uplink = calculate_eb_no(
                    eirp_dbw=uplink_eirp,
                    losses_db=uplink_losses_db,
                    rx_gain_db=6.5,
                    system_noise_temp_k=615,
                    data_rate_bps=600,
                )

                # Downlink: calcular parámetros y Eb/No
                downlink_tx_power_dbw, downlink_eirp = adjust_tx_power(
                    tx_power_dbw=12.5 - 30,  # Potencia inicial del transmisor en dBW
                    use_amplifier=False,
                    amplifier_gain_db=20,
                    amplifier_losses_db=2,
                    antenna_gain_db=6.5,
                    cable_losses_db=1,
                    max_eirp_dbw=20 - 30,
                )
                downlink_losses_db = free_space_loss(frequency_hz=2.5e9, distance_km=distance_km) + 2
                eb_no_downlink = calculate_eb_no(
                    eirp_dbw=downlink_eirp,
                    losses_db=downlink_losses_db,
                    rx_gain_db=12,
                    system_noise_temp_k=135,
                    data_rate_bps=100,
                )

                # Guardar resultados para uplink y downlink
                resultados.append({
                    "Satélite": sat_id,
                    "Dispositivo IoT": device['id'],
                    "Distancia (km)": distance_km,
                    "Uplink Eb/No (dB)": eb_no_uplink,
                    "Downlink Eb/No (dB)": eb_no_downlink,
                    "Uplink EIRP (dBW)": uplink_eirp,
                    "Downlink EIRP (dBW)": downlink_eirp,
                })

# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame(resultados)

# Guardar los resultados en un archivo Excel
output_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\resultados_satélites.xlsx"
df_resultados.to_excel(output_file, index=False)

print(f"Los resultados se han guardado en {output_file}")

# Mostrar la figura final
plt.show()
