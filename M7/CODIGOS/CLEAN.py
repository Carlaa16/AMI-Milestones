import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
import chardet
import os
import json

####################################################################################################
###################################### Constantes y parámetros #####################################

# Parámetros de la Tierra
option = 'iot_fov_traces'  # Cambia a 'iot_fov_traces' para FOV de IoT y trazas de satélites / satellite_fov
R_EARTH = 6371  # Radio de la Tierra en km
MU = 3.986e14  # Constante gravitacional en m^3/s^2
C = 3e8  # Velocidad de la luz en m/s
K = 1.38e-23  # Constante de Boltzmann en J/K
num_sats = 8
altitude_km = 850  # Altitud en km
fov_angle = 60  # Ángulo de visión en grados
beamwidth = 80 # Ancho de haz de la antena en grados
frequency_hz_up = 2.4e9  # Frecuencia en Hz
frequency_hz_down = 2.4e9  # Frecuencia en Hz
tx_power_dbw_up = 20 - 30  # Potencia inicial del transmisor en dBW
tx_power_dbw_down = 12.5 - 30  # Potencia inicial del transmisor en dBW
use_amplifier_up = False  # Usar amplificador (True o False)
use_amplifier_down = False  # Usar amplificador (True o False)
amplifier_gain_db_up = 20  # Ganancia del amplificador en dB
amplifier_gain_db_down = 20  # Ganancia del amplificador en dB
amplifier_losses_db_up = 1  # Pérdidas del amplificador en dB
amplifier_losses_db_down = 1  # Pérdidas del amplificador en dB
antenna_gain_db_up = 12.5  # Ganancia de la antena iot en dB
antenna_gain_db_down = 9  # Ganancia de la antena sat en dB
cable_losses_db_up = 2  # Pérdidas por cable en dB
cable_losses_db_down = 2  # Pérdidas por cable en dB
max_eirp_dbw_up = 14 - 30  # EIRP máximo permitido
max_eirp_dbw_down = 14 - 30  # EIRP máximo permitido
rx_gain_db_up = antenna_gain_db_down  # Ganancia del receptor en dB
rx_gain_db_down = antenna_gain_db_up  # Ganancia del receptor en dB
system_noise_temp_k_up = 615  # Temperatura de ruido del sistema en Kelvin
system_noise_temp_k_down = 135  # Temperatura de ruido del sistema en Kelvin
# animated = False  # Cambia a True para mostrar los satélites uno por uno
uplink_data_size_bits = 100  # Tamaño total de datos
downlink_data_size_bits = 1000  # Tamaño total de datos

####################################################################################################
########################################## Geometría ###############################################

# Función para calcular el radio geográfico del FOV
def calculate_fov_radius(altitude_km, fov_angle):
    fov_angle_deg = float(fov_angle) 
    fov_angle_rads= np.radians(fov_angle_deg)
    rho = np.arcsin(R_EARTH / (R_EARTH + altitude_km))  # Ángulo de la Tierra visto desde el satélite
    elevation_min_rad = np.arccos(np.sin(fov_angle_rads/2) / np.sin(rho))  # Ángulo de elevación mínimo
    elevation_min_deg = np.degrees(elevation_min_rad)
    lambda_min_rad = (np.pi/2) - fov_angle_rads/2 - elevation_min_rad  # Ángulo entre el borde del FOV y el centro de la Tierra
    r_fov_km = R_EARTH * lambda_min_rad  # Radio en km
    radius_deg = np.degrees(r_fov_km / R_EARTH)  # Convertir a grados geográficos
    return radius_deg, r_fov_km, elevation_min_deg

# Convertir coordenadas cartesianas Earth-Fixed (X,Y,Z) a latitud y longitud
def cartesian_to_geographic(x, y, z):
    lon = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, hyp))
    return lat, lon

# Función para calcular el radio de cobertura de dispositivos IoT
def calculate_iot_radius(beamwidth):
    beamwidth_rad = np.radians(beamwidth)
    radius_km = R_EARTH * np.tan(beamwidth_rad / 2)
    return radius_km

# Función para generar un círculo de cobertura de dispositivos IoT
def generate_device_coverage_circle(lat, lon, radius_km, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    circle_lat = lat + (radius_km / R_EARTH) * (180 / np.pi) * np.sin(angles)
    circle_lon = lon + (radius_km / R_EARTH) * (180 / np.pi) * np.cos(angles) / np.cos(np.radians(lat))
    return circle_lat, circle_lon

# Función para calcular la distancia máxima de Downlink y Uplink
def calculate_max_distances(altitude_km, fov_angle, beamwidth):
    # Calcular la distancia máxima de Downlink (FOV del satélite)
    max_downlink_distance = (R_EARTH + altitude_km) / np.cos(np.radians(fov_angle/ 2))

    # Calcular la distancia máxima de Uplink (FOV/Beamwidth del IoT)
    max_uplink_distance = (R_EARTH + altitude_km) / np.cos(np.radians(beamwidth/ 2))

    return max_downlink_distance, max_uplink_distance

# calcular el ancho de haz mínimo necesario
def calculate_beamwidth_from_elevation_min(altitude_km, elevation_min_deg):
    """
    Calcula el ancho de haz mínimo necesario (beamwidth) dado epsilon mínima usando la fórmula:
    h^2 * cos^2(epsilon_min) = 2 * (R + h)^2 * sin^2((Theta/2) + epsilon_min) * (1 - cos(Theta)).
    """
    h = altitude_km  # Altitud del satélite en km
    elevation_min_rad = np.radians(elevation_min_deg)  # Convertir a radianes

    # Ecuación no lineal para resolver Theta (beamwidth)
    def equation(theta_rad):
        return (
            (h**2) * (np.cos(elevation_min_rad)**2) -
            2 * ((R_EARTH + h)**2) *
            (np.sin((theta_rad / 2) + elevation_min_rad)**2) *
            (1 - np.cos(theta_rad))
        )

    # Resolución numérica de Theta en radianes
    theta_rad_initial = np.radians(10)  # Valor inicial razonable
    theta_rad = fsolve(equation, theta_rad_initial)[0]

    # Convertir de radianes a grados
    theta_tierra_min = np.degrees(theta_rad)
    beamwidth_min =180 - 2*elevation_min_deg # Convertir a grados geográficos

    return beamwidth_min

# Calcular la velocidad orbital del satélite en km/s
def calculate_orbital_velocity(altitude_km):
    """
    Calcula la velocidad orbital teórica de un satélite en km/s para una órbita circular.

    Parámetros:
    - altitude_km: Altitud del satélite en km.

    Retorna:
    - Velocidad orbital en km/s.
    """

    # Convertir a metros
    altitude_m = altitude_km * 1e3
    radius_m = (R_EARTH * 1e3) + altitude_m

    # Calcular la velocidad orbital
    velocity_mps = sqrt(MU / radius_m)  # Velocidad en m/s
    velocity_kmps = velocity_mps / 1e3       # Convertir a km/s
    return velocity_kmps

####################################################################################################
####################################### Cálculos de enlace ##########################################

# Función para saber si el beamwidth de la antena es suficiente para la elevación mínima
def check_visibility(altitude_km, fov_angle, beamwidth):
    # Calcular la elevación mínima usando la función calculate_fov_radius
    _, _, elevation_min_deg = calculate_fov_radius(altitude_km, fov_angle)

    # Calcular el ancho de haz mínimo necesario
    beamwidth_min = calculate_beamwidth_from_elevation_min(altitude_km, elevation_min_deg)

    # Comparar el ancho de haz proporcionado con el mínimo necesario
    if beamwidth < beamwidth_min:
        print("Error: No hay visibilidad entre IoT y enlace. El ancho de haz proporcionado es menor que el mínimo necesario.")
        print(f"Ancho de haz mínimo necesario: {beamwidth_min:.2f} grados")
    else:
        print("Visibilidad entre IoT y enlace confirmada.")

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

def calculate_bitrates_from_visibility(consolidated_visibility, uplink_data_size_bits, downlink_data_size_bits):
    """
    Calcula la tasa de datos para transmitir datos en cada intervalo consolidado de visibilidad.

    Parámetros:
        consolidated_visibility (dict): {(sat_index, iot_index): [(start_time, end_time), ...]}
        uplink_data_size_bits (float): Tamaño total de datos a transmitir en el Uplink (bits).
        downlink_data_size_bits (float): Tamaño total de datos a transmitir en el Downlink (bits).

    Retorna:
        dict: {(sat_index, iot_index): {"uplink_bitrate": float, "downlink_bitrate": float}}
    """
    bitrate_results = {}

    for (sat_index, iot_index), intervals in consolidated_visibility.items():
        for start_time, end_time in intervals:
            effective_time = end_time - start_time
            if effective_time > 0:
                uplink_bitrate = uplink_data_size_bits / effective_time
                downlink_bitrate = downlink_data_size_bits / effective_time
            else:
                uplink_bitrate = None
                downlink_bitrate = None

            # Guardar las tasas de bits por intervalo
            bitrate_results.setdefault((sat_index, iot_index), []).append({
                "uplink_bitrate": uplink_bitrate,
                "downlink_bitrate": downlink_bitrate
            })

    return bitrate_results

# Función para verificar si un punto está dentro del FOV
def is_within_fov(lat1, lon1, lat2, lon2, radius_km):
    # Calcular la distancia entre los dos puntos en km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R_EARTH * c
    return distance_km <= radius_km

# Función para verificar la conexión entre satélites y dispositivos IoT
def check_connections(satellite_positions, iot_positions, altitude_km, fov_angle, beamwidth):
    """
    Verifica las conexiones entre satélites e IoT en función de su posición y cobertura.

    Parámetros:
        satellite_positions (dict): Diccionario {satellite_id: [(lat, lon), ...]}.
        iot_positions (list): Lista de posiciones [(lat, lon)] de los dispositivos IoT.
        altitude_km (float): Altitud de los satélites.
        fov_angle (float): Ángulo de visión del satélite en grados.
        beamwidth (float): Ancho de haz de los dispositivos IoT en grados.

    Retorna:
        list: Lista de conexiones [(sat_id, iot_id)].
    """
    # Calcular el FOV del satélite y del dispositivo IoT
    _, sat_fov_radius_km, _ = calculate_fov_radius(altitude_km, fov_angle)
    iot_fov_radius_km = calculate_iot_radius(beamwidth)

    connections = []

    # Iterar sobre cada satélite
    for sat_id, positions in satellite_positions.items():  # Corrección aquí
        for sat_lat, sat_lon in positions:  # Iterar sobre las posiciones del satélite
            for iot_index, (iot_lat, iot_lon) in enumerate(iot_positions):
                # Verificar si el satélite está dentro del FOV del IoT
                if is_within_fov(iot_lat, iot_lon, sat_lat, sat_lon, iot_fov_radius_km):
                    connections.append((sat_id, iot_index, "IoT -> Satélite"))

                # Verificar si el IoT está dentro del FOV del satélite
                if is_within_fov(sat_lat, sat_lon, iot_lat, iot_lon, sat_fov_radius_km):
                    connections.append((sat_id, iot_index, "Satélite -> IoT"))

    return connections

def is_continuous_connection(prev_pos, curr_pos, iot_pos, fov_radius):
    """
    Comprueba si la conexión entre satélite e IoT continúa en la siguiente posición.

    Parámetros:
        prev_pos (tuple): Posición anterior del satélite (lat, lon).
        curr_pos (tuple): Posición actual del satélite (lat, lon).
        iot_pos (tuple): Posición del dispositivo IoT (lat, lon).
        fov_radius (float): Radio del FOV del satélite en km.

    Retorna:
        bool: True si la conexión es continua, False si no lo es.
    """
    # Verificar si ambas posiciones (previa y actual) están dentro del FOV del IoT
    return (
        is_within_fov(iot_pos[0], iot_pos[1], prev_pos[0], prev_pos[1], fov_radius) and
        is_within_fov(iot_pos[0], iot_pos[1], curr_pos[0], curr_pos[1], fov_radius)
    )

# Función para rastrear la visibilidad a lo largo del tiempo
def track_visibility_over_time(satellite_positions, iot_positions, altitude_km, fov_angle, beamwidth):
    """
    Rastrea las conexiones entre satélites y dispositivos IoT a lo largo del tiempo.

    Parámetros:
        satellite_positions (dict): {satellite_id: [(lat, lon), ...]}.
        iot_positions (list): Lista de posiciones [(lat, lon)] de IoT.
        altitude_km (float): Altitud del satélite.
        fov_angle (float): Ángulo de visión del satélite.
        beamwidth (float): Ancho del haz del IoT.

    Retorna:
        dict: Diccionario {(satellite_id, iot_index): [time_steps]}.
    """
    visibility = {}
    iot_fov_radius = calculate_iot_radius(beamwidth)  # Radio del FOV del IoT
    sat_fov_radius = calculate_fov_radius(altitude_km, fov_angle)[1]  # Radio del FOV del satélite

    # Iterar por cada satélite y sus posiciones
    for sat_id, positions in satellite_positions.items():
        for time_index, (sat_lat, sat_lon) in enumerate(positions):
            # Iterar sobre cada IoT y verificar si hay conexión
            for iot_index, (iot_lat, iot_lon) in enumerate(iot_positions):
                # Verificar si el satélite está dentro del FOV del IoT
                if is_within_fov(sat_lat, sat_lon, iot_lat, iot_lon, iot_fov_radius):
                    visibility.setdefault((sat_id, iot_index), []).append(time_index)
                    
    return visibility

def consolidate_visibility_times(visibility, time_threshold):
    """
    Consolida los tiempos de visibilidad continuos en intervalos.

    Parámetros:
        visibility (dict): Diccionario con { (sat_index, iot_index): [time_steps] }.
        time_threshold (float): Umbral de tiempo para detectar ruptura en la conexión.

    Retorna:
        dict: { (sat_index, iot_index): [(start_time, end_time), ...] }.
    """
    consolidated_connections = {}

    for (sat_index, iot_index), time_steps in visibility.items():
        time_steps = sorted(time_steps)  # Ordenar los tiempos
        intervals = []

        start_time = time_steps[0]
        for i in range(1, len(time_steps)):
            if (time_steps[i] - time_steps[i - 1]) > time_threshold:
                end_time = time_steps[i - 1]
                intervals.append((start_time, end_time))
                start_time = time_steps[i]

        # Agregar el último intervalo
        end_time = time_steps[-1]
        intervals.append((start_time, end_time))

        consolidated_connections[(sat_index, iot_index)] = intervals

    return consolidated_connections

# Función para determinar los tiempos de conexión
def get_all_connection_times(visibility):
    """
    Obtiene todas las conexiones entre satélites e IoT consolidando intervalos continuos.

    Parámetros:
        visibility (dict): Diccionario con { (sat_index, iot_index): [time_steps] }.

    Retorna:
        dict: Diccionario con { (sat_index, iot_index): [(start_time, end_time), ...] }.
    """
    connection_times = {}

    for (sat_index, iot_index), time_steps in visibility.items():
        # Ordenar los tiempos por si no están en orden
        time_steps = sorted(time_steps)

        # Consolidar conexiones continuas
        consolidated_connections = []
        start_time = time_steps[0]
        for i in range(1, len(time_steps)):
            if abs(time_steps[i] - time_steps[i - 1]) > 1e-6:  # Detectar ruptura en continuidad
                end_time = time_steps[i - 1]
                consolidated_connections.append((start_time, end_time))
                start_time = time_steps[i]
        # Agregar última conexión
        consolidated_connections.append((start_time, time_steps[-1]))

        connection_times[(sat_index, iot_index)] = consolidated_connections

    return connection_times

# Funcion para obtener tiempo entre la primera y última conexión
def get_first_last_connection_difference(visibility):
    """
    Obtiene solo la diferencia entre el tiempo inicial de la primera conexión y el tiempo final de la última conexión.

    Parámetros:
        visibility (dict): Diccionario con { (sat_index, iot_index): [time_steps] }.

    Retorna:
        dict: Diccionario con { (sat_index, iot_index): (start_time, end_time) }.
    """
    connection_times = {}

    for (sat_index, iot_index), time_steps in visibility.items():
        # Ordenar los tiempos por si no están en orden
        time_steps = sorted(time_steps)

        if time_steps:
            start_time = time_steps[0]  # Primer tiempo
            end_time = time_steps[-1]  # Último tiempo
            connection_times[(sat_index, iot_index)] = (start_time, end_time)

    return connection_times

# Implementa funciones y hace el cálculo de enlace
def calculate_link_budget(consolidated_visibility, altitude_km, fov_angle, beamwidth, uplink_data_size_bits, downlink_data_size_bits):
    """
    Calcula el balance de enlace para Uplink y Downlink reutilizando funciones existentes.

    Parámetros:
        visibility (dict): Tiempos de visibilidad { (sat_index, iot_index): [time_steps] }.
        altitude_km (float): Altitud del satélite en km.
        fov_angle (float): Ángulo de visión del satélite.
        beamwidth (float): Ancho de haz del IoT.
        uplink_data_size_bits (float): Tamaño total de datos a transmitir en el Uplink (bits).
        downlink_data_size_bits (float): Tamaño total de datos a transmitir en el Downlink (bits).

    Retorna:
        dict: Resultados del balance de enlace para cada conexión.
    """
    # Validar parámetros antes de continuar
    validate_parameters(altitude_km, fov_angle, beamwidth)

    # Calcular distancias máximas
    max_downlink_distance, max_uplink_distance = calculate_max_distances(altitude_km, fov_angle, beamwidth)

    # Calcular tasas de datos
    bitrate_results = calculate_bitrates_from_visibility(consolidated_visibility, uplink_data_size_bits, downlink_data_size_bits)

    # Parámetros específicos para Uplink y Downlink
    uplink_params = {
        "frequency_hz": frequency_hz_up,
        "tx_power_dbw": tx_power_dbw_up,
        "use_amplifier": use_amplifier_up,
        "amplifier_gain_db": amplifier_gain_db_up,
        "amplifier_losses_db": amplifier_losses_db_up,
        "antenna_gain_db": antenna_gain_db_up,
        "cable_losses_db": cable_losses_db_up,
        "rx_gain_db": rx_gain_db_up,
        "system_noise_temp_k": system_noise_temp_k_up,
        "data_rate_bps": uplink_data_size_bits,
        "max_eirp_dbw": max_eirp_dbw_up
    }

    downlink_params = {
        "frequency_hz": frequency_hz_down,
        "tx_power_dbw": tx_power_dbw_down,
        "use_amplifier": use_amplifier_down,
        "amplifier_gain_db": amplifier_gain_db_down,
        "amplifier_losses_db": amplifier_losses_db_down,
        "antenna_gain_db": antenna_gain_db_down,
        "cable_losses_db": cable_losses_db_down,
        "rx_gain_db": rx_gain_db_down,
        "system_noise_temp_k": system_noise_temp_k_down,
        "data_rate_bps": downlink_data_size_bits,
        "max_eirp_dbw": max_eirp_dbw_down
    }

    results = {}

    for (sat_index, iot_index), rates_list in bitrate_results.items():
        for rates in rates_list:  # Iterar sobre la lista de diccionarios
            uplink_bitrate = rates["uplink_bitrate"]
            downlink_bitrate = rates["downlink_bitrate"]

            # --- Calcular Uplink ---
            uplink_results = calculate_link_parameters(
                distance_km=max_uplink_distance,
                params=uplink_params
            )

            # --- Calcular Downlink ---
            downlink_results = calculate_link_parameters(
                distance_km=max_downlink_distance,
                params=downlink_params
            )

            # Guardar resultados
            results.setdefault((sat_index, iot_index), []).append({
                "uplink": {
                    "bitrate": uplink_bitrate,
                    **uplink_results
                },
                "downlink": {
                    "bitrate": downlink_bitrate,
                    **downlink_results
                }
            })


    return results

# Función para calcular los parámetros del enlace y poder implementarlos
def calculate_link_parameters(distance_km, params):
    """
    Calcula los parámetros del enlace (pérdidas, EIRP, Eb/No) para una distancia y parámetros dados.

    Parámetros:
        distance_km (float): Distancia máxima del enlace en km.
        params (dict): Diccionario con los parámetros específicos del enlace.

    Retorna:
        dict: Resultados del cálculo (pérdidas, EIRP, Eb/No).
    """
    # Calcular pérdidas por propagación libre
    losses_db = free_space_loss(params["frequency_hz"], distance_km)

    # Calcular potencia ajustada y EIRP dinámicamente
    adjusted_power, eirp_dbw = adjust_tx_power(
        tx_power_dbw=params["tx_power_dbw"],
        use_amplifier=params["use_amplifier"],
        amplifier_gain_db=params["amplifier_gain_db"],
        amplifier_losses_db=params["amplifier_losses_db"],
        antenna_gain_db=params["antenna_gain_db"],
        cable_losses_db=params["cable_losses_db"],
        max_eirp_dbw=params["max_eirp_dbw"]  # Límite máximo permitido
    )

    # Calcular Eb/No
    eb_no_db = calculate_eb_no(
        eirp_dbw=eirp_dbw,
        losses_db=losses_db,
        rx_gain_db=params["rx_gain_db"],
        system_noise_temp_k=params["system_noise_temp_k"],
        data_rate_bps=params["data_rate_bps"]
    )

    return {
        "losses_db": losses_db,           # Pérdidas espaciales (dB)
        "adjusted_power_dbw": adjusted_power,  # Potencia ajustada del transmisor (dBW)
        "eirp_dbw": eirp_dbw,             # EIRP dinámicamente calculado (dBW)
        "eb_no_db": eb_no_db              # Relación Eb/No (dB)
    }


####################################################################################################
###################################### Archivos y directorios ######################################

# Función para detectar codificación
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Función para cargar datos de satélites desde múltiples archivos
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

# Función para cargar datos de dispositivos IoT desde un archivo
def load_iot_data(iot_file):
    iot_df = pd.read_csv(iot_file, sep='\\s+', engine='python')  # Ajusta el delimitador según tu archivo
    devices = []
    for i in range(len(iot_df.columns) // 3):
        x_col = f'IoT_{i}.EarthFixed.X'
        y_col = f'IoT_{i}.EarthFixed.Y'
        z_col = f'IoT_{i}.EarthFixed.Z'
        if x_col in iot_df.columns and y_col in iot_df.columns and z_col in iot_df.columns:
            lat, lon = cartesian_to_geographic(iot_df[x_col].iloc[0], iot_df[y_col].iloc[0], iot_df[z_col].iloc[0])
            devices.append({'id': i, 'lat': lat, 'lon': lon})
    return pd.DataFrame(devices)

# Calculo del tiempo acumulado entre posiciones consecutivas
def calculate_times_from_positions(df, altitude_km):
    """
    Calcula los tiempos acumulados entre posiciones consecutivas de un satélite.

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas 'Latitude' y 'Longitude'.
        altitude_km (float): Altitud del satélite en km.

    Retorna:
        list: Tiempos acumulados en segundos para cada posición.
    """
    # Obtener velocidad orbital
    velocity_kmps = calculate_orbital_velocity(altitude_km)

    # Convertir latitud y longitud a radianes
    latitudes = np.radians(df['Latitude'].values)
    longitudes = np.radians(df['Longitude'].values)

    # Calcular distancias entre posiciones consecutivas
    distances = []
    for i in range(1, len(latitudes)):
        phi1, phi2 = latitudes[i - 1], latitudes[i]
        lambda1, lambda2 = longitudes[i - 1], longitudes[i]
        # Distancia geodésica
        distance = R_EARTH * np.arccos(
            np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1)
        )
        distances.append(distance)

    # Calcular tiempos entre pasos (\Delta t = d / v)
    times = [0]  # Tiempo inicial en t=0
    for distance in distances:
        delta_t = distance / velocity_kmps
        times.append(times[-1] + delta_t)

    return times

# Función para agregar tiempos al DataFrame
def add_times_to_dataframe(df, altitude_km):
    """
    Agrega la columna 'time' al DataFrame basada en las posiciones consecutivas.

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas 'Latitude' y 'Longitude'.
        altitude_km (float): Altitud del satélite en km.

    Retorna:
        pd.DataFrame: DataFrame con columna 'time'.
    """
    times = calculate_times_from_positions(df, altitude_km)
    df['time'] = times
    return df

def organize_satellite_positions(df):
    """
    Organiza las posiciones de satélites desde un DataFrame en un diccionario estructurado por 'satellite_id' y tiempo.

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas 'satellite_id', 'Latitude', 'Longitude', y opcionalmente 'time'.

    Retorna:
        dict: Diccionario {satellite_id: [(lat, lon), ...]}.
    """
    # Verificar que las columnas necesarias existen
    required_columns = {'satellite_id', 'Latitude', 'Longitude'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas {required_columns}. Encontradas: {df.columns}")

    # Agrupar posiciones por 'satellite_id' y 'time' si existe
    satellite_positions = {}
    for sat_id, group in df.groupby('satellite_id'):
        if 'time' in df.columns:
            # Ordenar por tiempo y convertir a lista de tuplas (lat, lon)
            group = group.sort_values('time')
        positions = list(zip(group['Latitude'], group['Longitude']))
        satellite_positions[sat_id] = positions

    return satellite_positions


# Directorios y archivos
sat_files_dir = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\CleanedReports"
iot_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\CleanedReports\cleaned_IoT_ReportFile.txt"
background_image_path = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\Atlantis.png"

all_satellite_data = load_satellite_data(sat_files_dir, num_sats)
# Agregar tiempos al DataFrame
all_satellite_data = add_times_to_dataframe(all_satellite_data, altitude_km)
iot_df = load_iot_data(iot_file)
satellite_positions = all_satellite_data.groupby('satellite_id')[['Latitude', 'Longitude']].apply(list).to_dict()
# print(all_satellite_data.head())
# print(satellite_positions)

# Cargar la lista de diccionarios en un DataFrame
all_satellite_data_df = pd.DataFrame(all_satellite_data)

# Verificar las primeras filas
print(all_satellite_data_df.head())
print("Satélite IDs únicos:", all_satellite_data['satellite_id'].unique())

if all_satellite_data.empty:
    print("No se cargaron datos de satélites. Verifica los archivos.")
    exit()

if iot_df.empty:
    print("No se cargaron datos de dispositivos IoT. Verifica los archivos.")
    exit()



# Convertir DataFrames a listas de tuplas
# satellite_positions = list(zip(all_satellite_data['Latitude'], all_satellite_data['Longitude']))

iot_positions = list(zip(iot_df['lat'], iot_df['lon']))
# Agrupar posiciones por satellite_id en un diccionario con listas de tuplas
# Generar satellite_positions como un diccionario
satellite_positions = (
    all_satellite_data
    .groupby('satellite_id', group_keys=False)  # Evitar incluir columnas de agrupación automáticamente
    .apply(lambda group: list(zip(group['Latitude'], group['Longitude'])))
    .to_dict()
)

# Verificar el resultado
print("Satélites detectados en satellite_positions:", satellite_positions.keys())

# Guaradr resultados en excel
def save_results_to_excel(consolidated_visibility, link_budget_results, output_file):
    """
    Guarda los resultados de visibilidad consolidada y del balance de enlace en un archivo Excel.

    Parámetros:
        consolidated_visibility (dict): Visibilidad consolidada { (sat_id, iot_id): [(start, end), ...] }.
        link_budget_results (dict): Resultados del balance de enlace.
        output_file (str): Ruta del archivo Excel de salida.
    """
    try:
        # --- 1. Crear DataFrame para la visibilidad consolidada ---
        visibility_data = []
        for (sat_id, iot_id), intervals in consolidated_visibility.items():
            for start_time, end_time in intervals:
                visibility_data.append({
                    "Satellite_ID": sat_id,
                    "IoT_ID": iot_id,
                    "Duration (s)": end_time - start_time
                })
        visibility_df = pd.DataFrame(visibility_data)

        # --- 2. Crear DataFrame para el Link Budget ---
        link_budget_data = []
        for (sat_id, iot_id), results in link_budget_results.items():
            # Acceder al primer elemento si el resultado es una lista
            if isinstance(results, list):
                results = results[0]

            link_budget_data.append({
                "Satellite_ID": sat_id,
                "IoT_ID": iot_id,
                "Uplink_Losses_dB": results["uplink"]["losses_db"],
                "Uplink_EIRP_dBW": results["uplink"]["eirp_dbw"],
                "Uplink_Eb_No_dB": results["uplink"]["eb_no_db"],
                "Uplink_Bitrate_bps": results["uplink"]["bitrate"],
                "Downlink_Bitrate_bps": results["downlink"]["bitrate"],
                "Downlink_Losses_dB": results["downlink"]["losses_db"],
                "Downlink_EIRP_dBW": results["downlink"]["eirp_dbw"],
                "Downlink_Eb_No_dB": results["downlink"]["eb_no_db"]
                
                
                
            })
        link_budget_df = pd.DataFrame(link_budget_data)

        # --- 3. Guardar todo en un archivo Excel ---
        with pd.ExcelWriter(output_file) as writer:
            visibility_df.to_excel(writer, sheet_name="Visibility_Results", index=False)
            link_budget_df.to_excel(writer, sheet_name="Link_Budget_Results", index=False)

        print(f"Resultados guardados en {output_file}")

    except Exception as e:
        print(f"Error al generar el archivo Excel: {e}")

####################################################################################################
############################################### Checks #############################################

# Verificar visibilidad entre satélites y dispositivos IoT
check_visibility(altitude_km, fov_angle, beamwidth)

# Verificar conexiones
connections = check_connections(satellite_positions, iot_positions, altitude_km, fov_angle, beamwidth)

# Imprimir conexiones detectadas
# for connection in connections:
#     print(f"Conexión detectada: {connection}")

def validate_parameters(altitude_km, fov_angle, beamwidth):
    """
    Valida los parámetros de entrada para asegurar que son físicamente posibles.

    Parámetros:
        altitude_km (float): Altitud del satélite en km.
        fov_angle (float): Ángulo de visión del satélite en grados.
        beamwidth (float): Ancho de haz de la antena en grados.

    Retorna:
        None: Si los parámetros son válidos.

    Lanza:
        ValueError: Si algún parámetro está fuera de los límites físicos.
    """
    # Validar FOV (debe estar entre 0 y 180 grados)
    if not (0 < fov_angle <= 180):
        raise ValueError(f"Error: El FOV ({fov_angle}°) debe estar en el rango (0, 180] grados.")

    # Validar el ancho de haz (debe ser razonable, menor o igual que el FOV)
    if not (0 < beamwidth <= 180):
        raise ValueError(f"Error: El ancho de haz ({beamwidth}°) debe estar en el rango (0, 180] grados.")

    # Validar la altitud (debe ser mayor que 0)
    if altitude_km <= 0:
        raise ValueError(f"Error: La altitud ({altitude_km} km) debe ser mayor que 0.")

    
####################################################################################################
####################################### Visualización de datos #####################################


def plot_visualization(option, satellite_positions, iot_df, altitude_km, fov_angle, beamwidth, background_image_path):
    """
    Función para visualizar FOV de satélites o FOV de IoT con trazas de satélites.
    
    Parámetros:
        option (str): Modo de visualización ('satellite_fov' o 'iot_fov_traces').
        satellite_positions (dict): Posiciones de los satélites {sat_id: [(lat, lon), ...]}.
        iot_df (DataFrame): Posiciones de los dispositivos IoT (lat, lon).
        altitude_km (float): Altitud de los satélites.
        fov_angle (float): Ángulo de visión de los satélites.
        beamwidth (float): Ancho de haz de los IoT.
        background_image_path (str): Ruta de la imagen del mapa.
    """

    def generate_circle(lat, lon, radius_km, num_points=100):
        """Genera un círculo geográfico alrededor de un punto."""
        angles = np.linspace(0, 2 * np.pi, num_points)
        circle_lat = lat + (radius_km / R_EARTH) * (180 / np.pi) * np.sin(angles)
        circle_lon = lon + (radius_km / R_EARTH) * (180 / np.pi) * np.cos(angles) / np.cos(np.radians(lat))
        return circle_lat, circle_lon

    def plot_traces(ax, satellite_positions, color="green"):
        """Dibuja las trazas de los satélites manejando saltos de longitud."""
        for sat_index, positions in satellite_positions.items():
            lats, lons = zip(*positions)  # Extraer posiciones
            segments = []
            current_segment = [positions[0]]
            
            # Detectar saltos
            for i in range(1, len(positions)):
                if abs(lons[i] - lons[i - 1]) > 180:
                    segments.append(current_segment)
                    current_segment = []
                current_segment.append(positions[i])
            segments.append(current_segment)

            # Dibujar segmentos
            for segment in segments:
                if len(segment) > 1:
                    segment_lats, segment_lons = zip(*segment)
                    ax.plot(segment_lons, segment_lats, color=color, linewidth=1.2, label=f"Satélite {sat_index}")

    # Configurar la figura
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("Visualización de Satélites e IoT")

    # Cargar fondo
    background = plt.imread(background_image_path)
    ax.imshow(background, extent=[-180, 180, -90, 90], aspect='auto')

    # --- Modo 1: FOV de Satélites ---
    if option == 'satellite_fov':
        for sat_index, positions in satellite_positions.items():
            for lat, lon in positions:
                sat_radius = calculate_fov_radius(altitude_km, fov_angle)[1]
                circle_lat, circle_lon = generate_circle(lat, lon, sat_radius)
                ax.plot(circle_lon, circle_lat, color='red', linewidth=1.2)
                ax.fill(circle_lon, circle_lat, color='red', alpha=0.1)
        # Dibujar dispositivos IoT como puntos
        ax.scatter(iot_df['lon'], iot_df['lat'], color='black', s=20, label='Dispositivo IoT')
        plt.legend(["FOV Satélite", "Dispositivo IoT"])

    # --- Modo 2: FOV de IoT con trazas de satélites ---
    elif option == 'iot_fov_traces':
        # Dibujar FOV de dispositivos IoT
        for _, device in iot_df.iterrows():
            iot_radius = calculate_iot_radius(beamwidth)
            circle_lat, circle_lon = generate_circle(device['lat'], device['lon'], iot_radius)
            ax.plot(circle_lon, circle_lat, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
            ax.fill(circle_lon, circle_lat, color='green', alpha=0.2)
        plt.legend(["FOV IoT"])

        # Dibujar trazas de los satélites
        plot_traces(ax, satellite_positions, color="red")

    # Mostrar la figura
    plt.show()


# --- Llamada a la función ---
# Alternar entre modos:
# - 'satellite_fov': Solo muestra FOV de los satélites y puntos IoT.
# - 'iot_fov_traces': Muestra FOV de IoT y las trazas de satélites.

plot_visualization(option, satellite_positions, iot_df, altitude_km, fov_angle, beamwidth, background_image_path)



# Obtener VISIBILIDAD
visibility = track_visibility_over_time(satellite_positions, iot_positions, altitude_km, fov_angle, beamwidth)
# Consolidar los tiempos en intervalos continuos
consolidated_visibility = consolidate_visibility_times(visibility, time_threshold=90)

# Mostrar resultados consolidados

# TIEMPOS DE CONEXIÓN
# 1. Obtener todas las CONEXIONES
all_connections = get_all_connection_times(visibility)
# print("**Todas las conexiones detalladas**")
# for (sat_index, iot_index), connections in all_connections.items():
#     print(f"Satélite {sat_index} y IoT {iot_index} tienen las siguientes conexiones:")
#     for start_time, end_time in connections:
#         print(f"  Comienza en t={start_time}, finaliza en t={end_time}")

# 2. Obtener solo la diferencia entre la primera y última conexión
first_last_differences = get_first_last_connection_difference(visibility)
# Imprimir la diferencia entre primera y última conexión
# print("\n**Diferencia entre primera y última conexión**")
# for (sat_index, iot_index), (start_time, end_time) in first_last_differences.items():
#     print(f"Satélite {sat_index} y IoT {iot_index}: Comienza en t={start_time}, finaliza en t={end_time}")


# TASA DE DATOS
bitrate_results = calculate_bitrates_from_visibility(consolidated_visibility, uplink_data_size_bits, downlink_data_size_bits)

# bitrate_results = calculate_bitrates_from_visibility(visibility, uplink_data_size_bits, downlink_data_size_bits)
# Imprimir resultados ENLACE con manejo de None
# for (sat_index, iot_index), rates in bitrate_results.items():
#     uplink_bitrate = rates['uplink_bitrate']
#     downlink_bitrate = rates['downlink_bitrate']

#     print(f"Satélite {sat_index} ↔ IoT {iot_index}:")

#     if uplink_bitrate is not None:
#         print(f"  Tasa Uplink: {uplink_bitrate:.2f} bps")
#     else:
#         print("  Tasa Uplink: No se pudo calcular (sin tiempo de visibilidad)")

#     if downlink_bitrate is not None:
#         print(f"  Tasa Downlink: {downlink_bitrate:.2f} bps")
#     else:
#         print("  Tasa Downlink: No se pudo calcular (sin tiempo de visibilidad)")


results = calculate_link_budget(
    consolidated_visibility=consolidated_visibility,
    altitude_km=altitude_km,
    fov_angle=fov_angle,
    beamwidth=beamwidth,
    uplink_data_size_bits=uplink_data_size_bits, 
    downlink_data_size_bits=downlink_data_size_bits 
)
# Mostrar resultados enlace
# for (sat_index, iot_index), result in results.items():
#     uplink_bitrate = result['uplink']['bitrate']
#     downlink_bitrate = result['downlink']['bitrate']

#     # Si ambos bitrates son None, no mostramos nada
#     if uplink_bitrate is None and downlink_bitrate is None:
#         continue

#     print(f"Satélite {sat_index} ↔ IoT {iot_index}:")
    
#     # Uplink
#     if uplink_bitrate is not None:
#         print("  Uplink:")
#         print(f"    Tasa: {uplink_bitrate:.2f} bps")
#         print(f"    Pérdidas: {result['uplink']['losses_db']:.2f} dB")
#         print(f"    EIRP: {result['uplink']['eirp_dbw']:.2f} dBW")
#         print(f"    Eb/No: {result['uplink']['eb_no_db']:.2f} dB")
    
#     # Downlink
#     if downlink_bitrate is not None:
#         print("  Downlink:")
#         print(f"    Tasa: {downlink_bitrate:.2f} bps")
#         print(f"    Pérdidas: {result['downlink']['losses_db']:.2f} dB")
#         print(f"    EIRP: {result['downlink']['eirp_dbw']:.2f} dBW")
#         print(f"    Eb/No: {result['downlink']['eb_no_db']:.2f} dB")

output_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\Results.xlsx"


# Verificar y guardar resultados
save_results_to_excel(consolidated_visibility, results, output_file)


# Suponiendo que all_satellite_data es una lista o diccionario
all_satellite_data_dict = all_satellite_data.to_dict(orient='records')

output_file = r'C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\all_satellite_data.txt'

# Guardar los datos en un archivo de texto
with open(output_file, 'w') as file:
    file.write(json.dumps(all_satellite_data_dict, indent=4))

print(f"Datos guardados en {output_file}")
