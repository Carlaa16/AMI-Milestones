import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import os
import subprocess
import sys
from PyQt5.QtWidgets import QApplication
from GUI import GMATGUI
from datetime import datetime



def GMAT_Files(Inputs):
    workspace_dir = Inputs["Workspace:"]#os.path.dirname(os.path.abspath(__file__))
    # Archivo donde se guardará el script GMAT
    output_file = os.path.join(workspace_dir, "Atlantis.script")
    # Ruta completa al archivo .shp
    shapefile_path = os.path.join(workspace_dir, "ne_10m_geography_marine_polys", "ne_10m_geography_marine_polys.shp")
    # Cargar el mapa del mundo desde el archivo
    world = gpd.read_file(shapefile_path)
    # Filtrar el Atlántico (Norte y Sur)
    oceans = world[world['name'].isin([Inputs['Zona a estudiar:']])]

    init_RAAN = float(Inputs["RAAN inicial (deg):"])

    # Parámetros generales
    base_epoch = convert_to_utcgregorian(Inputs['Fecha de inicio:'])  # Fecha de inicio
    sma = Inputs['SMA (Km):']  # Semi-major axis (en km)
    ecc = Inputs['Excentricidad:']  # Excentricidad
    inc =  Inputs['Inclinacion (deg):'] # Inclinación orbital (en grados)
    aop = 0  # Argumento del periapsis
    dry_mass = 850  # Masa seca en kg
    orbit_color = "Red"  # Color de órbita en GMAT
    sats_por_plano = int(Inputs['Nº satelites por plano:'])
    n_plano = int(Inputs['Número de planos:'])
    n_sats = n_plano * sats_por_plano
    raan_increment = 360/n_plano # Incremento de RAAN entre satélites, en grados
    ta_increment = 360/sats_por_plano
    propagation_duration_days = Inputs["Duracion propagacion (dias):"]

    GMAT_GUI_flag = Inputs["Mostrar GMAT GUI"]

    # Elementos adicionales a incluir en el campo Add
    additional_elements = ["Earth", "Sun"]


    ###PODRIAMOS HACER QUE GENERE PUNTOS DONDE SEA DE LA TIERRA Y VEA SI ESTA DENTRO DE LO QUE NOSOTROS QUEREMOS ESTUDIAR
    # Definir los límites específicos del océano Atlántico en latitud y longitud
    lat_min, lat_max = -90, 90
    lon_min, lon_max = -180, 180

    # Número de estaciones terrestres
    num_stations = int(Inputs['Número de estaciones:'])

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



    # Crear el script GMAT
    with open(output_file, "w", encoding="utf-8") as file:
        # Escribir la cabecera
        file.write("% GMAT Script generado automaticamente\n\n")

        file.write("%----------------------------------------\n")
        file.write("%---------- Spacecraft\n")
        file.write("%----------------------------------------\n\n")

        # Crear la nave espacial
        n=0
        for i in range(n_plano):
            # Calcular el RAAN para el satélite actual
            raan = init_RAAN + i * raan_increment

            for j in range(sats_por_plano):
                ta = j * ta_increment

                file.write(f"Create Spacecraft sat_{n};\n")
                file.write(f"GMAT sat_{n}.DateFormat = UTCGregorian;\n")
                file.write(f"GMAT sat_{n}.Epoch = '{base_epoch}';\n")
                file.write(f"GMAT sat_{n}.CoordinateSystem = EarthMJ2000Eq;\n")
                file.write(f"GMAT sat_{n}.DisplayStateType = Keplerian;\n")
                file.write(f"GMAT sat_{n}.SMA ={sma};\n")
                file.write(f"GMAT sat_{n}.ECC = {ecc};\n")  
                file.write(f"GMAT sat_{n}.INC = {inc};\n")  # Longitud
                file.write(f"GMAT sat_{n}.RAAN = {raan:.4f};\n")  # Altitud
                file.write(f"GMAT sat_{n}.AOP = {aop};\n")
                file.write(f"GMAT sat_{n}.TA = {ta};\n")
                file.write(f"GMAT sat_{n}.DryMass = {dry_mass};\n")
                file.write(f"GMAT sat_{n}.Cd = 2.2;\n")
                file.write(f"GMAT sat_{n}.Cr = 1.8;\n")
                file.write(f"GMAT sat_{n}.DragArea = 15;\n")
                file.write(f"GMAT sat_{n}.SRPArea = 1;\n")
                file.write(f"GMAT sat_{n}.SPADDragScaleFactor = 1;\n")
                file.write(f"GMAT sat_{n}.SPADSRPScaleFactor = 1;\n")
                file.write(f"GMAT sat_{n}.AtmosDensityScaleFactor = 1;\n")
                file.write(f"GMAT sat_{n}.ExtendedMassPropertiesModel = 'None';\n")
                file.write(f"GMAT sat_{n}.NAIFId = -10000001;\n")
                file.write(f"GMAT sat_{n}.NAIFIdReferenceFrame = -9000001;\n")
                file.write(f"GMAT sat_{n}.OrbitColor = {orbit_color};\n")
                file.write(f"GMAT sat_{n}.TargetColor = Teal;\n")
                file.write(f"GMAT sat_{n}.OrbitErrorCovariance = [ 1e+70 0 0 0 0 0 ; 0 1e+70 0 0 0 0 ; 0 0 1e+70 0 0 0 ; 0 0 0 1e+70 0 0 ; 0 0 0 0 1e+70 0 ; 0 0 0 0 0 1e+70 ];\n")
                file.write(f"GMAT sat_{n}.CdSigma = 1e+70;\n")
                file.write(f"GMAT sat_{n}.CrSigma = 1e+70;\n")
                file.write(f"GMAT sat_{n}.Id = 'SatId';\n")
                file.write(f"GMAT sat_{n}.Attitude = CoordinateSystemFixed;\n")
                file.write(f"GMAT sat_{n}.SPADSRPInterpolationMethod = Bilinear;\n")
                file.write(f"GMAT sat_{n}.SPADSRPScaleFactorSigma = 1e+70;\n")
                file.write(f"GMAT sat_{n}.SPADDragInterpolationMethod = Bilinear;\n")
                file.write(f"GMAT sat_{n}.SPADDragScaleFactorSigma = 1e+70;\n")
                file.write(f"GMAT sat_{n}.AtmosDensityScaleFactorSigma = 1e+70;\n")
                file.write(f"GMAT sat_{n}.ModelFile = 'aura.3ds';\n")
                file.write(f"GMAT sat_{n}.ModelOffsetX = 0;\n")
                file.write(f"GMAT sat_{n}.ModelOffsetY = 0;\n")
                file.write(f"GMAT sat_{n}.ModelOffsetZ = 0;\n")
                file.write(f"GMAT sat_{n}.ModelRotationX = 0;\n")
                file.write(f"GMAT sat_{n}.ModelRotationY = 0;\n")
                file.write(f"GMAT sat_{n}.ModelRotationZ = 0;\n")
                file.write(f"GMAT sat_{n}.ModelScale = 1;\n")
                file.write(f"GMAT sat_{n}.AttitudeDisplayStateType = 'Quaternion';\n")
                file.write(f"GMAT sat_{n}.AttitudeRateDisplayStateType = 'AngularVelocity';\n")
                file.write(f"GMAT sat_{n}.AttitudeCoordinateSystem = EarthMJ2000Eq;\n")
                file.write(f"GMAT sat_{n}.EulerAngleSequence = '321';\n\n")
                n += 1

        file.write("%----------------------------------------\n")
        file.write("%---------- Ground Stations\n")
        file.write("%----------------------------------------\n\n")
        
        # Crear estaciones terrestres
        for i in range(num_stations):
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

        all_DefaultOrbitView = [f"sat_{i}" for i in range(n_sats)] + additional_elements
        all_DefaultGroundTrackPlot = [f"sat_{i}" for i in range(n_sats)] + [f"IoT_{i}" for i in range(num_stations)]
        all_coordsreport = [f"sat_{i}.Earth.Latitude" for i in range(n_sats)] + [f"sat_{i}.Earth.Longitude" for i in range(n_sats)]

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
        

        for i in range(n_sats):  # Asumiendo que `n_sats` es el número total de satélites
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
            file.write(f"GMAT sat_{i}_ReportFile.Filename = '{os.path.join(workspace_dir,"Reports",f"sat_{i}_ReportFile.txt")}';\n\n")
        
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

        for i in range(num_stations):  # Asumiendo que `n_sats` es el número total de satélites
            # Añadir individualmente los parámetros a registrar
            file.write(f"GMAT IoT_ReportFile.Add = {{IoT_{i}.EarthFixed.X}};\n")
            file.write(f"GMAT IoT_ReportFile.Add = {{IoT_{i}.EarthFixed.Y}};\n")
            file.write(f"GMAT IoT_ReportFile.Add = {{IoT_{i}.EarthFixed.Z}};\n")

        file.write(f"GMAT IoT_ReportFile.WriteReport = true;\n")
        file.write(f"GMAT IoT_ReportFile.Filename = '{os.path.join(workspace_dir,"Reports","IoT_ReportFile.txt")}';\n\n")



        # Escribir la secuencia de misión


        file.write("%----------------------------------------\n")
        file.write("%---------- Mission Sequence\n")
        file.write("%----------------------------------------\n\n")
        file.write("BeginMissionSequence;\n")
        sat_list=[]
        for i in range(n_sats):
            sat_list.append(f"sat_{i}")

        file.write(f"Propagate DefaultProp({', '.join(sat_list)}) {{sat_0.ElapsedDays = {propagation_duration_days}}};\n")
        for i in range(n_sats):
            file.write(f"GMAT sat_{i}_ReportFile.WriteReport = true;\n")
        

    print(f"Script GMAT con estaciones terrestres en el Atlántico generado: {output_file}")


    # Ruta al archivo .exe
    exe_path = Inputs["GMAT.exe path:"] #r"D:\Escritorio V2\aerocosas\MUSE\GMAT\bin\GMAT.exe"

    # Ruta al script
    script_path = os.path.join(workspace_dir, "Atlantis.script")

    if GMAT_GUI_flag:
        subprocess.run([exe_path, "--run", script_path])
    else:
        subprocess.run([exe_path, "--minimize", "--run", "--exit", script_path])

    print('END OF EXECUTION')

def convert_to_utcgregorian(date_str):
    """
    Convierte una fecha en formato 'dd-mm-yyyy' al formato '01 Jan 2024 00:00:00'.

    Args:
        date_str (str): Fecha en formato 'dd-mm-yyyy'.

    Returns:
        str: Fecha en formato UTCGregorian con espacios.
    """
    # Parsear la fecha de entrada
    date_object = datetime.strptime(date_str, "%d-%m-%Y")

    # Formatear al estilo '01 Jan 2024 00:00:00'
    utc_gregorian_date = date_object.strftime("%d %b %Y %H:%M:%S")

    return utc_gregorian_date

def on_confirm():
    inputs = window.get_parameters()
    print("\n".join([f"{key}: {value}" for key, value in inputs.items()]))
    # Aquí puedes llamar a funciones específicas para procesar los parámetros.
    GMAT_Files(inputs)

if __name__ == "__main__":

    app = QApplication(sys.argv)

    # Crear y mostrar la ventana principal
    window = GMATGUI()
    window.show()

    window.confirm_button.clicked.connect(on_confirm)

    sys.exit(app.exec_())



