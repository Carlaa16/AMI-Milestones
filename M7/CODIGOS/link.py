import numpy as np
import pandas as pd

# Constantes
C = 3e8  # Velocidad de la luz en m/s
K = 1.38e-23  # Constante de Boltzmann en J/K

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

# Uplink: Cálculo específico
uplink_tx_power_dbw, uplink_eirp = adjust_tx_power(
    tx_power_dbw=20 - 30,  # Potencia inicial del transmisor en dBW
    use_amplifier=False,
    amplifier_gain_db=20,
    amplifier_losses_db=1,
    antenna_gain_db=12.5,
    cable_losses_db=2,
    max_eirp_dbw=20 - 30,
)

uplink_losses_db = (
    free_space_loss(frequency_hz=2.4e9, distance_km=850) + 2  # Pérdidas adicionales
)

uplink_free_space_loss = free_space_loss(frequency_hz=2.4e9, distance_km=850)

uplink_eb_no = calculate_eb_no(
    eirp_dbw=uplink_eirp,
    losses_db=uplink_losses_db,
    rx_gain_db=6.5,
    system_noise_temp_k=615,
    data_rate_bps=1200,
)

# Downlink: Cálculo específico
downlink_tx_power_dbw, downlink_eirp = adjust_tx_power(
    tx_power_dbw=12.5 - 30,  # Potencia inicial del transmisor en dBW
    use_amplifier=False,
    amplifier_gain_db=20,
    amplifier_losses_db=2,
    antenna_gain_db=6.5,
    cable_losses_db=1,
    max_eirp_dbw=20 - 30,
)

downlink_losses_db = (
    free_space_loss(frequency_hz=2.5e9, distance_km=850) + 2  # Pérdidas adicionales
)

downlink_free_space_loss = free_space_loss(frequency_hz=2.5e9, distance_km=850)

downlink_eb_no = calculate_eb_no(
    eirp_dbw=downlink_eirp,
    losses_db=downlink_losses_db,
    rx_gain_db=12,
    system_noise_temp_k=135,
    data_rate_bps=100,
)

# Resultados
df_results = pd.DataFrame([
    {
        "Enlace": "Uplink",
        "Tx Power (dBW)": uplink_tx_power_dbw,
        "EIRP (dBW)": uplink_eirp,
        "Free Space Loss (dB)": uplink_free_space_loss,
        "Eb/No (dB)": uplink_eb_no,
    },
    {
        "Enlace": "Downlink",
        "Tx Power (dBW)": downlink_tx_power_dbw,
        "EIRP (dBW)": downlink_eirp,
        "Free Space Loss (dB)": downlink_free_space_loss,
        "Eb/No (dB)": downlink_eb_no,
    },
])

# Exportar resultados
output_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\uplink_downlink.xlsx"
df_results.to_excel(output_file, index=False)

print(f"Resultados del cálculo guardados en {output_file}")
