import numpy as np
from math import radians, sin, cos, sqrt, atan2, acos, degrees
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
# Parámetro del beamwidth para dispositivos IoT
beamwidth = 10  # Ajusta este valor según sea necesario
frequency_hz = 2.4e9  # Frecuencia en Hz
tx_power_dbw = 20 - 30  # Potencia inicial del transmisor en dBW
use_amplifier = False  # Usar amplificador (True o False)
amplifier_gain_db = 20  # Ganancia del amplificador en dB
amplifier_losses_db = 1  # Pérdidas del amplificador en dB
antenna_gain_db = 12.5  # Ganancia de la antena en dB
cable_losses_db = 2  # Pérdidas por cable en dB
max_eirp_dbw = 14 - 30  # EIRP máximo permitido
rx_gain_db = 6.5  # Ganancia del receptor en dB

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

    return adjusted_power, eirp_dbw

# Función para calcular Eb/No
def calculate_eb_no(eirp_dbw, losses_db, rx_gain_db, system_noise_temp_k, data_rate_bps):
    log_t_s = 10 * np.log10(system_noise_temp_k)
    log_r = 10 * np.log10(data_rate_bps)
    return eirp_dbw - losses_db + rx_gain_db + 228.6 - log_t_s - log_r

def detect_entry_exit_points(satellite_positions, iot_lat, iot_lon, beamwidth_deg):
    """
    Detecta los puntos de entrada y salida al FOV del IoT.

    Parámetros:
    - satellite_positions: Lista de tuplas (lat, lon, alt) con las posiciones del satélite.
    - iot_lat, iot_lon: Latitud y longitud del IoT.
    - beamwidth_deg: Ancho de haz del IoT en grados.

    Retorna:
    - entry_angle: Ángulo de entrada al FOV en radianes.
    - exit_angle: Ángulo de salida del FOV en radianes.
    """
    beamwidth_rad = radians(beamwidth_deg) / 2  # Convertir a radianes
    entry_angle = None
    exit_angle = None

    def calculate_angle(lat1, lon1, lat2, lon2):
        """
        Calcula el ángulo entre dos puntos en la superficie terrestre.
        """
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        delta_lon = lon2 - lon1
        angle = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(delta_lon)
        angle = max(-1, min(1, angle))  # Limitar el rango del coseno
        return acos(angle)

    for i in range(1, len(satellite_positions)):
        # Puntos consecutivos del satélite
        prev_lat, prev_lon, _ = satellite_positions[i - 1]
        curr_lat, curr_lon, _ = satellite_positions[i]

        # Calcular ángulos respecto al IoT
        prev_angle = calculate_angle(iot_lat, iot_lon, prev_lat, prev_lon)
        curr_angle = calculate_angle(iot_lat, iot_lon, curr_lat, curr_lon)

        # Detectar entrada
        if prev_angle > beamwidth_rad and curr_angle <= beamwidth_rad:
            entry_angle = curr_angle  # Guardar ángulo de entrada
        # Detectar salida
        elif prev_angle <= beamwidth_rad and curr_angle > beamwidth_rad:
            exit_angle = prev_angle  # Guardar ángulo de salida

        # Si ya tenemos entrada y salida, podemos salir del bucle
        if entry_angle is not None and exit_angle is not None:
            break

    return entry_angle, exit_angle

def calculate_visibility_time(beamwidth_deg, alt, sat_velocity_kmps, entry_angle, exit_angle):
    """
    Calcula el tiempo de visibilidad del satélite usando la velocidad orbital y los ángulos de entrada/salida.

    Parámetros:
    - beamwidth_deg: Ancho de haz del IoT en grados.
    - sat_velocity_kmps: Velocidad del satélite en km/s.
    - entry_angle: Ángulo de entrada al FOV en radianes.
    - exit_angle: Ángulo de salida del FOV en radianes.

    Retorna:
    - Tiempo de visibilidad en segundos.
    """
    if entry_angle is None or exit_angle is None:
        return 0  # No hay visibilidad si no hay entrada o salida

    # Ángulo cubierto dentro del FOV (en radianes)
    delta_theta = abs(exit_angle - entry_angle)

    # Calcular el radio de la órbita
    altitude_km = alt  # Altitud del satélite (puedes parametrizarlo si varía)
    orbit_radius_km = R_EARTH + altitude_km

    # Velocidad angular del satélite
    omega_orbita = sat_velocity_kmps / orbit_radius_km  # rad/s

    # Tiempo de visibilidad
    visibility_time = delta_theta / omega_orbita  # s
    return visibility_time

def calculate_orbital_velocity(altitude_km):
    """
    Calcula la velocidad orbital teórica de un satélite en km/s para una órbita circular.

    Parámetros:
    - altitude_km: Altitud del satélite en km.

    Retorna:
    - Velocidad orbital en km/s.
    """
    MU_EARTH = 3.986e14  # Constante gravitacional de la Tierra en m^3/s^2
    R_EARTH = 6371  # Radio de la Tierra en km

    # Convertir a metros
    altitude_m = altitude_km * 1e3
    radius_m = (R_EARTH * 1e3) + altitude_m

    # Calcular la velocidad orbital
    velocity_mps = sqrt(MU_EARTH / radius_m)  # Velocidad en m/s
    velocity_kmps = velocity_mps / 1e3       # Convertir a km/s
    return velocity_kmps

def is_satellite_in_iot_fov(sat_lat, sat_lon, iot_lat, iot_lon, beamwidth_deg):
    """
    Verifica si el satélite está dentro del FOV del IoT.

    Parámetros:
    - sat_lat, sat_lon: Latitud y longitud del satélite.
    - iot_lat, iot_lon: Latitud y longitud del IoT.
    - beamwidth_deg: Beamwidth del IoT en grados.

    Retorna:
    - True si el satélite está dentro del FOV del IoT, False en caso contrario.
    """
    beamwidth_rad = radians(beamwidth_deg) / 2
    # Calcular el ángulo entre el IoT y el satélite
    delta_lon = radians(sat_lon - iot_lon)
    iot_lat_rad, iot_lon_rad = radians(iot_lat), radians(iot_lon)
    sat_lat_rad, sat_lon_rad = radians(sat_lat), radians(sat_lon)

    angle = acos(
        sin(iot_lat_rad) * sin(sat_lat_rad) +
        cos(iot_lat_rad) * cos(sat_lat_rad) * cos(delta_lon)
    )
    return angle <= beamwidth_rad


import numpy as np
from math import radians, sin, cos, sqrt, acos, degrees
import pandas as pd


############################################################################################################
############################################################################################################
############################################################################################################



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
num_sats = 1
all_satellite_data = load_satellite_data(sat_files_dir, num_sats)
iot_df = load_iot_data(iot_file)

if all_satellite_data.empty:
    print("No se cargaron datos de satélites. Verifica los archivos.")
    exit()

if iot_df.empty:
    print("No se cargaron datos de dispositivos IoT. Verifica los archivos.")
    exit()

# Agregar altitud y FOV predeterminados
all_satellite_data['alt'] = 800  # Altitud en km
all_satellite_data['fov_angle'] = 100  # Ángulo de visión en grados



resultados = []

# Crear un mapa de colores
num_sats = len(all_satellite_data['satellite_id'].unique())  # Número de satélites únicos
num_iot = len(iot_df)  # Número de dispositivos IoT
cmap = cm.get_cmap('tab10', num_sats)  # Usar una paleta predefinida con 'num_sats' colores
cmap_iot = cm.get_cmap('tab10', num_iot)  # Usar una paleta predefinida con 'num_iot' colores


# Diccionario para guardar los polígonos de los IoT
iot_fov_polygons = {}

# Configuración: elegir si los satélites se muestran de golpe o animados
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

# Dibujar las coberturas IoT con colores únicos
for i, (_, device) in enumerate(iot_df.iterrows()):
    radius_km = calculate_device_coverage_radius(beamwidth)
    circle_lat, circle_lon = generate_device_coverage_circle(device['lat'], device['lon'], radius_km)

     # Crear el polígono del FOV del IoT
    iot_fov_polygon = Polygon(zip(circle_lon, circle_lat))
    iot_fov_polygons[device['id']] = iot_fov_polygon

    # Dibujar en el mapa
    device_color = cmap_iot(i)
    ax.plot(circle_lon, circle_lat, color=device_color, linewidth=1.5, label=f'IoT {device["id"]}')
    ax.fill(circle_lon, circle_lat, color=device_color, alpha=0.4)

print("Procesando FOV de los satélites y calculando enlaces...")

if animated:    # Procesar los satélites (modo estático o animado)
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
        
        # Extraer posiciones del satélite [(lat, lon, alt), ...]
        satellite_positions = list(zip(sat_data['Latitude'], sat_data['Longitude'], sat_data['alt']))
        altitude_km = sat_data['alt'].iloc[0]  # Altitud ya está definida

        # Calcular velocidad orbital
        sat_velocity_kmps = calculate_orbital_velocity(altitude_km)
        

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
            sat_fov_polygon = Polygon(zip(circle_points_lon, circle_points_lat))


            # Dibujar la cobertura del satélite
            ax.fill(circle_points_lon, circle_points_lat, color=color, alpha=0.3, label=f'Satélite {sat_id}')
            ax.plot(circle_points_lon, circle_points_lat, color='black', linewidth=1)
            
            # Calcular el enlace con cada IoT
        for iot_id, iot_polygon in iot_fov_polygons.items():
            # 1. Verificar intersección o contención
            intersects = sat_fov_polygon.intersects(iot_polygon)
            contains = sat_fov_polygon.contains(iot_polygon) or iot_polygon.contains(sat_fov_polygon)

            if not intersects and not contains:
                continue  # Si no hay interacción, pasar al siguiente IoT

            print(f"Procesando enlace entre Satélite {sat_id} y IoT {iot_id}...")

            # 2. Calcular Entry y Exit Angles para el Downlink
            entry_angle_downlink, exit_angle_downlink = detect_entry_exit_points(
                satellite_positions=list(zip(sat_data['Latitude'], sat_data['Longitude'], sat_data['alt'])),
                iot_lat=iot_df.loc[iot_df['id'] == iot_id, 'lat'].values[0],
                iot_lon=iot_df.loc[iot_df['id'] == iot_id, 'lon'].values[0],
                beamwidth_deg=fov_angle
            )

            # 3. Calcular Entry y Exit Angles para el Uplink
            entry_angle_uplink, exit_angle_uplink = detect_entry_exit_points(
                satellite_positions=list(zip(sat_data['Latitude'], sat_data['Longitude'], sat_data['alt'])),
                iot_lat=iot_df.loc[iot_df['id'] == iot_id, 'lat'].values[0],
                iot_lon=iot_df.loc[iot_df['id'] == iot_id, 'lon'].values[0],
                beamwidth_deg=beamwidth
            )

            # 4. Calcular tiempos de visibilidad
            visibility_time_downlink = 0
            visibility_time_uplink = 0

            if intersects or contains:
                # Considerar visibilidad completa si el FOV contiene completamente
                visibility_time_downlink = calculate_visibility_time(
                    beamwidth_deg=fov_angle,
                    alt=alt,
                    sat_velocity_kmps=calculate_orbital_velocity(alt),
                    entry_angle=entry_angle_downlink,
                    exit_angle=exit_angle_downlink
                ) if entry_angle_downlink and exit_angle_downlink else None

                visibility_time_uplink = calculate_visibility_time(
                    beamwidth_deg=beamwidth,
                    alt=alt,
                    sat_velocity_kmps=calculate_orbital_velocity(alt),
                    entry_angle=entry_angle_uplink,
                    exit_angle=exit_angle_uplink
                ) if entry_angle_uplink and exit_angle_uplink else None

            # 5. Calcular el balance de enlace (Downlink y Uplink)
            if visibility_time_downlink is not None:
                downlink_tx_power_dbw, downlink_eirp = adjust_tx_power(
                    tx_power_dbw=tx_power_dbw,
                    use_amplifier=use_amplifier,
                    amplifier_gain_db=amplifier_gain_db,
                    amplifier_losses_db=amplifier_losses_db,
                    antenna_gain_db=antenna_gain_db,
                    cable_losses_db=cable_losses_db,
                    max_eirp_dbw=max_eirp_dbw,
                )
                downlink_losses_db = free_space_loss(frequency_hz=2.5e9, distance_km=radius_km) + 2
                eb_no_downlink = calculate_eb_no(
                    eirp_dbw=downlink_eirp,
                    losses_db=downlink_losses_db,
                    rx_gain_db=rx_gain_db,
                    system_noise_temp_k=135,
                    data_rate_bps=100
                )
            else:
                eb_no_downlink = None

            if visibility_time_uplink is not None:
                uplink_tx_power_dbw, uplink_eirp = adjust_tx_power(
                    tx_power_dbw=tx_power_dbw,
                    use_amplifier=use_amplifier,
                    amplifier_gain_db=amplifier_gain_db,
                    amplifier_losses_db=amplifier_losses_db,
                    antenna_gain_db=antenna_gain_db,
                    cable_losses_db=cable_losses_db,
                    max_eirp_dbw=max_eirp_dbw,
                )
                uplink_losses_db = free_space_loss(frequency_hz=2.4e9, distance_km=radius_km) + 2
                eb_no_uplink = calculate_eb_no(
                    eirp_dbw=uplink_eirp,
                    losses_db=uplink_losses_db,
                    rx_gain_db=rx_gain_db,
                    system_noise_temp_k=615,
                    data_rate_bps=500
                )
            else:
                eb_no_uplink = None

            # 6. Guardar resultados
            resultados.append({
                "Satélite": sat_id,
                "Dispositivo IoT": iot_id,
                "Tiempo de Visibilidad (Downlink)": visibility_time_downlink,
                "Tiempo de Visibilidad (Uplink)": visibility_time_uplink,
                "Eb/No (Downlink)": eb_no_downlink,
                "Eb/No (Uplink)": eb_no_uplink,
            })
        

# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame(resultados)

# Guardar los resultados en un archivo Excel
output_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\res_sat.xlsx"
df_resultados.to_excel(output_file, index=False)

print(f"Los resultados se han guardado en {output_file}")

# Mostrar la figura final
plt.show()
