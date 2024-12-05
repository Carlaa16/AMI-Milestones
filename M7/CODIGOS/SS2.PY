import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt


# Archivo donde se guardará el script GMAT
output_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\ISG\CODIGOS\Atlantis.script"
# Ruta completa al archivo .shp
shapefile_path = r"C:\Users\carla\OneDrive\Documentos\MUSE\ISG\CODIGOS\ne_10m_geography_marine_polys\ne_10m_geography_marine_polys.shp"

# Cargar el mapa del mundo desde el archivo
world = gpd.read_file(shapefile_path)
# Filtrar el Atlántico (Norte y Sur)
oceans = world[world['name'].isin(['North Atlantic Ocean', 'South Atlantic Ocean'])]

# # Mostrar las columnas disponibles
# print(world.columns)
# print(world['featurecla'].unique())  # Valores únicos en la columna 'featurecla'
# print(world['name'].unique())        # Valores únicos en la columna 'name'

# oceans.plot()
# plt.show()

# Parámetros generales
raan_increment = 16.4  # Incremento de RAAN entre satélites, en grados
base_epoch = '01 Jan 2025 12:00:00.000'  # Fecha de inicio
sma = 7000  # Semi-major axis (en km)
ecc = 0.001  # Excentricidad
inc = 98.5  # Inclinación orbital (en grados)
aop = 0  # Argumento del periapsis
ta = 0  # Anomalía verdadera inicial
dry_mass = 850  # Masa seca en kg
orbit_color = "Red"  # Color de órbita en GMAT
sats = 22

# Elementos adicionales a incluir en el campo Add
additional_elements = ["Earth", "Sun"]



# Definir los límites específicos del océano Atlántico en latitud y longitud
lat_min, lat_max = -60, 60
lon_min, lon_max = -80, 20

# Número de estaciones terrestres
num_stations = 500

# Generar coordenadas dentro de los límites, pero asegurándonos que estén en el océano Atlántico
latitudes = []
longitudes = []

while len(latitudes) < num_stations:
    lat = np.random.uniform(lat_min, lat_max)
    lon = np.random.uniform(lon_min, lon_max)
    point = Point(lon, lat)
   
    # Verificar si el punto está en el océano
    if oceans.contains(point).any():
        latitudes.append(lat)
        longitudes.append(lon)
# while len(latitudes) < num_stations:
#     lat = np.random.uniform(lat_min, lat_max)
#     lon = np.random.uniform(lon_min, lon_max)

#     # Filtrar para asegurarnos que las coordenadas estén en el océano Atlántico
#     if (
#         (-60 <= lat <= 60) and          # Asegura que esté dentro del rango del Atlántico
#         (-70 <= lon <= 20) and          # Asegura que esté dentro del rango de longitud
#         not (-10 <= lat <= 30 and -20 <= lon <= 0)  # Excluye el Mediterráneo y regiones cercanas
#     ):
#         latitudes.append(lat)
#         longitudes.append(lon)



# Crear el script GMAT
with open(output_file, "w", encoding="utf-8") as file:
    # Escribir la cabecera
    file.write("% GMAT Script generado automaticamente\n\n")

    file.write("%----------------------------------------\n")
    file.write("%---------- Spacecraft\n")
    file.write("%----------------------------------------\n\n")

    # Crear la nave espacial
    for i in range(1, sats + 1):
        # Calcular el RAAN para el satélite actual
        raan = i * raan_increment

        file.write(f"Create Spacecraft sat_{i};\n")
        file.write(f"GMAT sat_{i}.DateFormat = UTCGregorian;\n")
        file.write(f"GMAT sat_{i}.Epoch = '{base_epoch}';\n")
        file.write(f"GMAT sat_{i}.CoordinateSystem = EarthMJ2000Eq;\n")
        file.write(f"GMAT sat_{i}.DisplayStateType = Keplerian;\n")
        file.write(f"GMAT sat_{i}.SMA ={sma};\n")
        file.write(f"GMAT sat_{i}.ECC = {ecc};\n")  
        file.write(f"GMAT sat_{i}.INC = {inc};\n")  # Longitud
        file.write(f"GMAT sat_{i}.RAAN = {raan:.4f};\n")  # Altitud
        file.write(f"GMAT sat_{i}.AOP = {aop};\n")
        file.write(f"GMAT sat_{i}.TA = {ta};\n")
        file.write(f"GMAT sat_{i}.DryMass = {dry_mass};\n")
        file.write(f"GMAT sat_{i}.Cd = 2.2;\n")
        file.write(f"GMAT sat_{i}.Cr = 1.8;\n")
        file.write(f"GMAT sat_{i}.DragArea = 15;\n")
        file.write(f"GMAT sat_{i}.SRPArea = 1;\n")
        file.write(f"GMAT sat_{i}.SPADDragScaleFactor = 1;\n")
        file.write(f"GMAT sat_{i}.SPADSRPScaleFactor = 1;\n")
        file.write(f"GMAT sat_{i}.AtmosDensityScaleFactor = 1;\n")
        file.write(f"GMAT sat_{i}.ExtendedMassPropertiesModel = 'None';\n")
        file.write(f"GMAT sat_{i}.NAIFId = -10000001;\n")
        file.write(f"GMAT sat_{i}.NAIFIdReferenceFrame = -9000001;\n")
        file.write(f"GMAT sat_{i}.OrbitColor = {orbit_color};\n")
        file.write(f"GMAT sat_{i}.TargetColor = Teal;\n")
        file.write(f"GMAT sat_{i}.OrbitErrorCovariance = [ 1e+70 0 0 0 0 0 ; 0 1e+70 0 0 0 0 ; 0 0 1e+70 0 0 0 ; 0 0 0 1e+70 0 0 ; 0 0 0 0 1e+70 0 ; 0 0 0 0 0 1e+70 ];\n")
        file.write(f"GMAT sat_{i}.CdSigma = 1e+70;\n")
        file.write(f"GMAT sat_{i}.CrSigma = 1e+70;\n")
        file.write(f"GMAT sat_{i}.Id = 'SatId';\n")
        file.write(f"GMAT sat_{i}.Attitude = CoordinateSystemFixed;\n")
        file.write(f"GMAT sat_{i}.SPADSRPInterpolationMethod = Bilinear;\n")
        file.write(f"GMAT sat_{i}.SPADSRPScaleFactorSigma = 1e+70;\n")
        file.write(f"GMAT sat_{i}.SPADDragInterpolationMethod = Bilinear;\n")
        file.write(f"GMAT sat_{i}.SPADDragScaleFactorSigma = 1e+70;\n")
        file.write(f"GMAT sat_{i}.AtmosDensityScaleFactorSigma = 1e+70;\n")
        file.write(f"GMAT sat_{i}.ModelFile = 'aura.3ds';\n")
        file.write(f"GMAT sat_{i}.ModelOffsetX = 0;\n")
        file.write(f"GMAT sat_{i}.ModelOffsetY = 0;\n")
        file.write(f"GMAT sat_{i}.ModelOffsetZ = 0;\n")
        file.write(f"GMAT sat_{i}.ModelRotationX = 0;\n")
        file.write(f"GMAT sat_{i}.ModelRotationY = 0;\n")
        file.write(f"GMAT sat_{i}.ModelRotationZ = 0;\n")
        file.write(f"GMAT sat_{i}.ModelScale = 1;\n")
        file.write(f"GMAT sat_{i}.AttitudeDisplayStateType = 'Quaternion';\n")
        file.write(f"GMAT sat_{i}.AttitudeRateDisplayStateType = 'AngularVelocity';\n")
        file.write(f"GMAT sat_{i}.AttitudeCoordinateSystem = EarthMJ2000Eq;\n")
        file.write(f"GMAT sat_{i}.EulerAngleSequence = '321';\n\n")


    file.write("%----------------------------------------\n")
    file.write("%---------- Ground Stations\n")
    file.write("%----------------------------------------\n\n")
    
    # Crear estaciones terrestres
    for i in range(1, num_stations + 1):
        file.write(f"Create GroundStation IoT_{i};\n")
        file.write(f"GMAT IoT_{i}.OrbitColor = Thistle;\n")
        file.write(f"GMAT IoT_{i}.TargetColor = DarkGray;\n")
        file.write(f"GMAT IoT_{i}.CentralBody = Earth;\n")
        file.write(f"GMAT IoT_{i}.StateType = Spherical;\n")
        file.write(f"GMAT IoT_{i}.HorizonReference = Sphere;\n")
        file.write(f"GMAT IoT_{i}.Location1 = {latitudes[i-1]:.4f};\n")  # Latitud
        file.write(f"GMAT IoT_{i}.Location2 = {longitudes[i-1]:.4f};\n")  # Longitud
        file.write(f"GMAT IoT_{i}.Location3 = 0;\n")  # Altitud
        file.write(f"GMAT IoT_{i}.Id = 'IoT_{i}';\n")
        file.write(f"GMAT IoT_{i}.IonosphereModel = 'None';\n")
        file.write(f"GMAT IoT_{i}.TroposphereModel = 'None';\n")
        file.write(f"GMAT IoT_{i}.DataSource = 'Constant';\n")
        file.write(f"GMAT IoT_{i}.Temperature = 295.1;\n")
        file.write(f"GMAT IoT_{i}.Pressure = 1013.5;\n")
        file.write(f"GMAT IoT_{i}.Humidity = 55;\n")
        file.write(f"GMAT IoT_{i}.MinimumElevationAngle = 5;\n\n")

    file.write("%----------------------------------------\n")
    file.write("%---------- FaceModels\n")
    file.write("%----------------------------------------\n\n")
        
    file.write(f"Create ForceModel DefaultProp_ForceModel;\n")
    file.write(f"GMAT DefaultProp_ForceModel.CentralBody = Earth;\n")
    file.write(f"GMAT DefaultProp_ForceModel.PrimaryBodies = {{Earth}};\n")
    file.write(f"GMAT DefaultProp_ForceModel.Drag = None;\n")
    file.write(f"GMAT DefaultProp_ForceModel.SRP = Off;\n")
    file.write(f"GMAT DefaultProp_ForceModel.RelativisticCorrection = Off;\n")
    file.write(f"GMAT DefaultProp_ForceModel.ErrorControl = RSSStep;\n")
    file.write(f"GMAT DefaultProp_ForceModel.GravityField.Earth.Degree = 4;\n")
    file.write(f"GMAT DefaultProp_ForceModel.GravityField.Earth.Order = 4;\n")
    file.write(f"GMAT DefaultProp_ForceModel.GravityField.Earth.StmLimit = 100;\n")
    file.write(f"GMAT DefaultProp_ForceModel.GravityField.Earth.PotentialFile = 'JGM2.cof';\n")
    file.write(f"GMAT DefaultProp_ForceModel.GravityField.Earth.TideModel = 'None';\n\n")

    file.write("%----------------------------------------\n")
    file.write("%---------- Propagators\n")
    file.write("%----------------------------------------\n\n")
        
    file.write(f"Create Propagator DefaultProp;\n")
    file.write(f"GMAT DefaultProp.FM = DefaultProp_ForceModel;\n")
    file.write(f"GMAT DefaultProp.Type = RungeKutta89;\n")
    file.write(f"GMAT DefaultProp.InitialStepSize = 60;\n")
    file.write(f"GMAT DefaultProp.Accuracy = 9.999999999999999e-12;\n")
    file.write(f"GMAT DefaultProp.MinStep = 0.001;\n")
    file.write(f"GMAT DefaultProp.MaxStep = 2700;\n")
    file.write(f"GMAT DefaultProp.MaxStepAttempts = 50;\n")
    file.write(f"GMAT DefaultProp.StopIfAccuracyIsViolated = true;\n\n")

    all_DefaultOrbitView = [f"sat_{i+1}" for i in range(sats)] + additional_elements
    all_DefaultGroundTrackPlot = [f"sat_{i+1}" for i in range(sats)] + [f"IoT_{i+1}" for i in range(num_stations)]
    all_coordsreport = [f"sat_{i+1}.Earth.Latitude" for i in range(sats)] + [f"sat_{i+1}.Earth.Longitude" for i in range(sats)]

    file.write("%----------------------------------------\n")
    file.write("%---------- Subscribers\n")
    file.write("%----------------------------------------\n\n")
        
    file.write(f"Create OrbitView DefaultOrbitView;\n")
    file.write(f"GMAT DefaultOrbitView.SolverIterations = Current;\n")
    file.write(f"GMAT DefaultOrbitView.UpperLeft = [ 0.05398967844382691 0 ];\n")
    file.write(f"GMAT DefaultOrbitView.Size = [ 0.1242556570067487 0.5118694362017804 ];\n")
    file.write(f"GMAT DefaultOrbitView.RelativeZOrder = 225;\n")
    file.write(f"GMAT DefaultOrbitView.Maximized = false;\n")
    file.write(f"GMAT DefaultOrbitView.Add = {{{', '.join(all_DefaultOrbitView)}}};\n")
    file.write(f"GMAT DefaultOrbitView.CoordinateSystem = EarthMJ2000Eq;\n")
    file.write(f"GMAT DefaultOrbitView.DrawObject = [ true true true true ];\n")
    file.write(f"GMAT DefaultOrbitView.DataCollectFrequency = 1;\n")
    file.write(f"GMAT DefaultOrbitView.UpdatePlotFrequency = 50;\n")
    file.write(f"GMAT DefaultOrbitView.NumPointsToRedraw = 0;\n")
    file.write(f"GMAT DefaultOrbitView.ShowPlot = true;\n")
    file.write(f"GMAT DefaultOrbitView.MaxPlotPoints = 20000;\n")
    file.write(f"GMAT DefaultOrbitView.ShowLabels = true;\n")
    file.write(f"GMAT DefaultOrbitView.ViewPointReference = Earth;\n")
    file.write(f"GMAT DefaultOrbitView.ViewPointVector = [ 30000 0 0 ];\n")
    file.write(f"GMAT DefaultOrbitView.ViewDirection = Earth;\n")
    file.write(f"GMAT DefaultOrbitView.ViewScaleFactor = 1;\n")
    file.write(f"GMAT DefaultOrbitView.ViewUpCoordinateSystem = EarthMJ2000Eq;\n")
    file.write(f"GMAT DefaultOrbitView.ViewUpAxis = Z;\n")
    file.write(f"GMAT DefaultOrbitView.EclipticPlane = Off;\n")
    file.write(f"GMAT DefaultOrbitView.XYPlane = On;\n")
    file.write(f"GMAT DefaultOrbitView.WireFrame = Off;\n")
    file.write(f"GMAT DefaultOrbitView.Axes = On;\n")
    file.write(f"GMAT DefaultOrbitView.Grid = Off;\n")
    file.write(f"GMAT DefaultOrbitView.SunLine = Off;\n")
    file.write(f"GMAT DefaultOrbitView.UseInitialView = On;\n")
    file.write(f"GMAT DefaultOrbitView.StarCount = 7000;\n")
    file.write(f"GMAT DefaultOrbitView.EnableStars = On;\n")
    file.write(f"GMAT DefaultOrbitView.EnableConstellations = On;\n\n")

    file.write(f"Create GroundTrackPlot DefaultGroundTrackPlot;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.SolverIterations = Current;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.UpperLeft = [ -0.01468836840015879 0.1609792284866469 ];\n")
    file.write(f"GMAT DefaultGroundTrackPlot.Size = [ 0.4493846764589123 0.6431750741839762 ];\n")
    file.write(f"GMAT DefaultGroundTrackPlot.RelativeZOrder = 242;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.Maximized = true;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.Add = {{{', '.join(all_DefaultGroundTrackPlot)}}};\n")
    file.write(f"GMAT DefaultGroundTrackPlot.DataCollectFrequency = 1;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.UpdatePlotFrequency = 50;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.NumPointsToRedraw = 0;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.ShowPlot = true;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.MaxPlotPoints = 20000;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.CentralBody = Earth;\n")
    file.write(f"GMAT DefaultGroundTrackPlot.TextureMap = 'ModifiedBlueMarble.jpg';\n\n")


##########################
    file.write(f"Create ReportFile DefaultReportFile;\n")
    file.write(f"GMAT DefaultReportFile.SolverIterations = Current;\n")
    file.write(f"GMAT DefaultReportFile.UpperLeft = [ 0 0 ];\n")
    file.write(f"GMAT DefaultReportFile.Size = [ 0 0 ];\n")
    file.write(f"GMAT DefaultReportFile.RelativeZOrder = 0;\n")
    file.write(f"GMAT DefaultReportFile.Maximized = false;\n")
    file.write(f"GMAT DefaultReportFile.Filename = 'DefaultReportFile.txt';\n")
    file.write(f"GMAT DefaultReportFile.Precision = 16;\n")
    file.write(f"GMAT DefaultReportFile.WriteHeaders = true;\n")
    file.write(f"GMAT DefaultReportFile.LeftJustify = On;\n")
    file.write(f"GMAT DefaultReportFile.ZeroFill = Off;\n")
    file.write(f"GMAT DefaultReportFile.FixedWidth = true;\n")
    file.write(f"GMAT DefaultReportFile.Delimiter = ' ';\n")
    file.write(f"GMAT DefaultReportFile.ColumnWidth = 23;\n")
    file.write(f"GMAT DefaultReportFile.WriteReport = true;\n")
    #file.write(f"GMAT DefaultReportFile.Filename = 'C:/Users/carla/OneDrive/Documentos/MUSE/ISG/Reports/DefaultReportFile.txt';\n\n")



    # file.write(f"Create ReportFile coords;\n")
    # file.write(f"GMAT coords.SolverIterations = Current;\n")
    # file.write(f"GMAT coords.UpperLeft = [ 0 0 ];\n")
    # file.write(f"GMAT coords.Size = [ 0 0 ];\n")
    # file.write(f"GMAT coords.RelativeZOrder = 0;\n")
    # file.write(f"GMAT coords.Maximized = false;\n")
    # file.write(f"GMAT coords.Filename = 'coords.txt';\n")
    # file.write(f"GMAT coords.Precision = 16;\n")
    # file.write(f"GMAT coords.Add = {all_coordsreport};\n")
    # file.write(f"GMAT coords.WriteHeaders = true;\n")
    # file.write(f"GMAT coords.LeftJustify = On;\n")
    # file.write(f"GMAT coords.ZeroFill = Off;\n")
    # file.write(f"GMAT coords.FixedWidth = true;\n")
    # file.write(f"GMAT coords.Delimiter = ' ';\n")
    # file.write(f"GMAT coords.ColumnWidth = 23;\n")
    # file.write(f"GMAT coords.WriteReport = true;\n\n")

    for i in range(1, sats + 1):  # Asumiendo que `sats` es el número total de satélites
        file.write(f"Create ReportFile sat_{i}_ReportFile;\n")
        file.write(f"GMAT sat_{i}_ReportFile.SolverIterations = Current;\n")
        file.write(f"GMAT sat_{i}_ReportFile.UpperLeft = [ 0.0 0.0 ];\n")
        file.write(f"GMAT sat_{i}_ReportFile.Size = [ 500 200 ];\n")
        file.write(f"GMAT sat_{i}_ReportFile.Filename = 'sat_{i}_ReportFile.txt';\n")
        file.write(f"GMAT sat_{i}_ReportFile.WriteHeaders = true;\n")
        file.write(f"GMAT sat_{i}_ReportFile.LeftJustify = true;\n")
        file.write(f"GMAT sat_{i}_ReportFile.ZeroFill = false;\n")
        file.write(f"GMAT sat_{i}_ReportFile.FixedWidth = true;\n")
        file.write(f"GMAT sat_{i}_ReportFile.Delimiter = ',';\n")
        file.write(f"GMAT sat_{i}_ReportFile.ColumnWidth = 15;\n")
        
        # Añadir individualmente los parámetros a registrar
        #file.write(f"GMAT sat_{i}_ReportFile.Add = {{sat_{i}.A1ModJulian}};\n")
        file.write(f"GMAT sat_{i}_ReportFile.Add = {{sat_{i}.Earth.Latitude}};\n")
        file.write(f"GMAT sat_{i}_ReportFile.Add = {{sat_{i}.Earth.Longitude}};\n")
        file.write(f"GMAT sat_{i}_ReportFile.WriteReport = true;\n")
        file.write(f"GMAT sat_{i}_ReportFile.Filename = 'C:/Users/carla/OneDrive/Documentos/MUSE/ISG/CODIGOS/Reports/sat_{i}_ReportFile.txt';\n\n")
    
    for i in range(1, num_stations + 1):  # Asumiendo que `sats` es el número total de satélites
        file.write(f"Create ReportFile IoT_ReportFile;\n")
        file.write(f"GMAT IoT_ReportFile.SolverIterations = Current;\n")
        file.write(f"GMAT IoT_ReportFile.UpperLeft = [ 0.0 0.0 ];\n")
        file.write(f"GMAT IoT_ReportFile.Size = [ 500 200 ];\n")
        file.write(f"GMAT IoT_ReportFile.Filename = 'IoT_ReportFile.txt';\n")
        file.write(f"GMAT IoT_ReportFile.WriteHeaders = true;\n")
        file.write(f"GMAT IoT_ReportFile.LeftJustify = true;\n")
        file.write(f"GMAT IoT_ReportFile.ZeroFill = false;\n")
        file.write(f"GMAT IoT_ReportFile.FixedWidth = true;\n")
        file.write(f"GMAT IoT_ReportFile.Delimiter = ',';\n")
        file.write(f"GMAT IoT_ReportFile.ColumnWidth = 15;\n")
        
        # Añadir individualmente los parámetros a registrar
        #file.write(f"GMAT sat_{i}_ReportFile.Add = {{sat_{i}.A1ModJulian}};\n")
        file.write(f"GMAT IoT_ReportFile.Add = {{IoT_{i}.EarthFixed.X}};\n")
        file.write(f"GMAT IoT_ReportFile.Add = {{IoT_{i}.EarthFixed.Y}};\n")
        file.write(f"GMAT IoT_ReportFile.Add = {{IoT_{i}.EarthFixed.Z}};\n")
        file.write(f"GMAT IoT_ReportFile.WriteReport = true;\n")
        file.write(f"GMAT IoT_ReportFile.Filename = 'C:/Users/carla/OneDrive/Documentos/MUSE/ISG/CODIGOS/Reports/IoT_ReportFile.txt';\n\n")



    # Escribir la secuencia de misión
# Parámetros de propagación
    propagation_duration_days = 0.08333  # 7200 segundos en días
    earth_ecc = 3600  # Parámetro ficticio como ejemplo


    file.write("%----------------------------------------\n")
    file.write("%---------- Mission Sequence\n")
    file.write("%----------------------------------------\n\n")
    file.write("BeginMissionSequence;\n")
    for i in range(sats):
        file.write(f"Propagate DefaultProp(sat_{i+1}) {{sat_{i+1}.ElapsedDays = {propagation_duration_days:.5f}}};\n")
        file.write(f"GMAT sat_{i+1}_ReportFile.WriteReport = true;\n")
    # for i in range(num_stations):
    #     file.write(f"GMAT IoT_{i+1}_ReportFile.WriteReport = true;\n")

print(f"Script GMAT con estaciones terrestres en el Atlántico generado: {output_file}")


# import pyautogui
# import time


# # Esperar a que GMAT termine de ejecutar
# time.sleep(60)  # Ajusta el tiempo según sea necesario

# # Capturar la pantalla completa
# screenshot = pyautogui.screenshot()

# # Recortar la región específica del gráfico
# # Usa las coordenadas (x, y, ancho, alto) específicas para tu pantalla
# # region = (100, 100, 800, 600)  # Ajusta estos valores
# # screenshot = screenshot.crop(region)

# # Guardar la captura
# screenshot.save(r"C:\Users\carla\OneDrive\Documentos\MUSE\ISG\Reports\GroundTrackPlot.png")
    
