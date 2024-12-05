import numpy as np
from scipy.optimize import fsolve

# Constantes para la Tierra
MU_TIERRA = 3.986e14  # Constante gravitacional estándar para la Tierra en m^3/s^2
R_TIERRA = 6371  # Radio de la Tierra en km

def calculate_satellite_fov(altitude_km, angle_of_view_deg):
    """
    Calcula el radio geográfico del FOV y la elevación mínima del satélite.
    """
    angle_of_view_rad = np.radians(angle_of_view_deg / 2)
    rho = np.arcsin(R_TIERRA / (R_TIERRA + altitude_km))
    elevation_min_rad = np.arccos(np.sin(angle_of_view_rad) / np.sin(rho))
    elevation_min_deg = np.degrees(elevation_min_rad)
    
    # Calcular el radio geográfico del FOV
    lambda_min_rad = (np.pi / 2) - angle_of_view_rad - elevation_min_rad
    fov_radius = R_TIERRA * lambda_min_rad  # Radio en km en la superficie
    
    return fov_radius, elevation_min_deg

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
            2 * ((R_TIERRA + h)**2) *
            (np.sin((theta_rad / 2) + elevation_min_rad)**2) *
            (1 - np.cos(theta_rad))
        )

    # Resolución numérica de Theta en radianes
    theta_rad_initial = np.radians(10)  # Valor inicial razonable
    theta_rad = fsolve(equation, theta_rad_initial)[0]

    # Convertir de radianes a grados
    beamwidth_deg_antena = np.degrees(theta_rad)

    return beamwidth_deg_antena


def calculate_visibility_time_and_fov(altitude_km, angle_of_view_deg):
    """
    Calcula el tiempo de visión del satélite y verifica el ancho de haz de la antena.
    """
    # Calcular el FOV del satélite
    fov_radius, sat_elevation_min_deg = calculate_satellite_fov(altitude_km, angle_of_view_deg)

    # Calcular el beamwidth mínimo requerido para la elevación mínima del satélite
    beamwidth_deg_antena = calculate_beamwidth_from_elevation_min(altitude_km, sat_elevation_min_deg)

    # Calcular la velocidad angular de la órbita
    h_orbit_m = altitude_km * 1e3  # Altitud en metros
    omega_orbita = np.sqrt(MU_TIERRA / ((R_TIERRA * 1e3 + h_orbit_m)**3))

    # Calcular el tiempo de visión
    theta_rad = np.radians(beamwidth_deg_antena / 2)
    t_visionado = 2 * theta_rad / omega_orbita

    return t_visionado, fov_radius, sat_elevation_min_deg, beamwidth_deg_antena

# Ejemplo de cálculo
altitude_km = 850  # Altitud del satélite en km
angle_of_view_deg = 100  # FOV del satélite en grados

# Realizar los cálculos
t_visionado, fov_radius, sat_elevation_min_deg, beamwidth_deg_antena = calculate_visibility_time_and_fov(
    altitude_km, angle_of_view_deg
)

# Mostrar resultados finales
print("\nResultados finales:")
print(f"Tiempo de visión: {t_visionado:.2f} segundos")
print(f"Radio geográfico del FOV: {fov_radius:.2f} km")
print(f"Elevación mínima del satélite: {sat_elevation_min_deg:.2f}°")
print(f"Beamwidth mínimo requerido para la antena: {beamwidth_deg_antena:.2f}°")

# Tamaño del paquete de datos a transmitir (en bits)
data_size_bits = 10 * 8 * 1024  # Ejemplo: 10 KB convertido a bits

# Tiempo de margen (en segundos)
margin_time = 30  # Ajusta según sea necesario, por ejemplo, 30 segundos

# Verificar que el margen no exceda el tiempo de visión
if margin_time >= t_visionado:
    raise ValueError("El tiempo de margen no puede ser mayor o igual al tiempo de visión.")

# Calcular el tiempo efectivo de visión
effective_time = t_visionado - margin_time

# Calcular la tasa de datos necesaria
bitrate_required = data_size_bits / effective_time

# Mostrar resultados
print(f"Tamaño del paquete de datos: {data_size_bits / 8 / 1024:.2f} KB")
print(f"Tiempo de visión total: {t_visionado:.2f} segundos")
print(f"Tiempo de margen: {margin_time:.2f} segundos")
print(f"Tiempo efectivo de visión: {effective_time:.2f} segundos")
print(f"Tasa de datos necesaria (bitrate): {bitrate_required:.2f} bps")
