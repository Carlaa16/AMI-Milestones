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
    rho = np.arcsin(R_EARTH / (R_EARTH + altitude_km))
    elevation_min_rad = np.arccos(np.sin(angle_of_view_rad / 2) / np.sin(rho))
    lambda_min_rad = (np.pi / 2) - angle_of_view_rad / 2 - elevation_min_rad
    r_fov = R_EARTH * lambda_min_rad
    radius_deg = np.degrees(r_fov / R_EARTH)
    return radius_deg

# Convertir coordenadas cartesianas Earth-Fixed a latitud y longitud
def cartesian_to_geographic(x, y, z):
    lon = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, hyp))
    return lat, lon

# Función para calcular pérdidas por espacio libre
def free_space_loss(frequency_hz, distance_km):
    distance_m = distance_km * 1e3
    return 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) - 20 * np.log10(C / (4 * np.pi))

# Función para ajustar la potencia del transmisor respetando el límite de EIRP
def adjust_tx_power(tx_power_dbw, use_amplifier, amplifier_gain_db, amplifier_losses_db, antenna_gain_db, cable_losses_db, max_eirp_dbw):
    if use_amplifier:
        adjusted_power = tx_power_dbw + amplifier_gain_db - amplifier_losses_db
    else:
        adjusted_power = tx_power_dbw

    eirp_dbw = adjusted_power + antenna_gain_db - cable_losses_db

    if eirp_dbw > max_eirp_dbw:
        adjusted_power -= (eirp_dbw - max_eirp_dbw)

    return adjusted_power, eirp_dbw

# Función para calcular Eb/No
def calculate_eb_no(eirp_dbw, losses_db, rx_gain_db, system_noise_temp_k, data_rate_bps):
    log_t_s = 10 * np.log10(system_noise_temp_k)
    log_r = 10 * np.log10(data_rate_bps)
    return eirp_dbw - losses_db + rx_gain_db + 228.6 - log_t_s - log_r

# Función para calcular el balance de enlace
def calculate_link_balance(distance_km):
    # Uplink
    uplink_tx_power_dbw, uplink_eirp = adjust_tx_power(
        tx_power_dbw=14 - 30,  # Potencia inicial en dBW
        use_amplifier=False,
        amplifier_gain_db=20,
        amplifier_losses_db=1,
        antenna_gain_db=12.5,
        cable_losses_db=2,
        max_eirp_dbw=14 - 30
    )
    uplink_losses_db = free_space_loss(frequency_hz=2.4e9, distance_km=distance_km) + 2
    uplink_eb_no = calculate_eb_no(
        eirp_dbw=uplink_eirp,
        losses_db=uplink_losses_db,
        rx_gain_db=6.5,
        system_noise_temp_k=615,
        data_rate_bps=1200
    )

    # Downlink
    downlink_tx_power_dbw, downlink_eirp = adjust_tx_power(
        tx_power_dbw=12.5 - 30,
        use_amplifier=False,
        amplifier_gain_db=20,
        amplifier_losses_db=2,
        antenna_gain_db=6.5,
        cable_losses_db=1,
        max_eirp_dbw=12.5 - 30
    )
    downlink_losses_db = free_space_loss(frequency_hz=2.5e9, distance_km=distance_km) + 2
    downlink_eb_no = calculate_eb_no(
        eirp_dbw=downlink_eirp,
        losses_db=downlink_losses_db,
        rx_gain_db=12,
        system_noise_temp_k=135,
        data_rate_bps=100
    )

    return {
        "Uplink Eb/No (dB)": uplink_eb_no,
        "Downlink Eb/No (dB)": downlink_eb_no,
        "Uplink EIRP (dBW)": uplink_eirp,
        "Downlink EIRP (dBW)": downlink_eirp
    }

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
# Directorios y archivos
sat_files_dir = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\CleanedReports"
iot_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\CleanedReports\cleaned_IoT_ReportFile.txt"
background_image_path = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\Atlantis.png"

num_sats = 5
beamwidth = 10

# Cargar datos
all_satellite_data = load_satellite_data(sat_files_dir, num_sats)
iot_df = load_iot_data(iot_file)

# Agregar altitud y FOV predeterminados
all_satellite_data['alt'] = 850
all_satellite_data['fov_angle'] = 100

# Crear resultados
resultados = []

for _, sat in all_satellite_data.iterrows():
    lat, lon, alt, fov_angle = sat['Latitude'], sat['Longitude'], sat['alt'], sat['fov_angle']
    radius_deg = calculate_fov_radius(alt, fov_angle)
    num_points = 100
    angles = np.linspace(0, 2 * np.pi, num_points)
    circle_points_lon = lon + radius_deg * np.cos(angles)
    circle_points_lat = lat + radius_deg * np.sin(angles)

    fov_polygon = Polygon(zip(circle_points_lon, circle_points_lat))
    for _, device in iot_df.iterrows():
        device_point = Point(device['lon'], device['lat'])
        if fov_polygon.contains(device_point):
            distance_km = np.sqrt((R_EARTH + alt)**2 - R_EARTH**2)
            link_results = calculate_link_balance(distance_km)

            resultados.append({
                "Satélite": sat['satellite_id'],
                "Dispositivo IoT": device['id'],
                "Distancia (km)": distance_km,
                **link_results,  # Agregar los resultados de enlace al diccionario
            })

# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame(resultados)
output_file = "resultados_satélites.xlsx"
df_resultados.to_excel(output_file, index=False)

print(f"Resultados guardados en {output_file}")
