import os
import glob

def replace_in_file(input_path, replacements, output_path):
    """
    Reemplaza texto en un archivo de texto y guarda el resultado.
    
    :param input_path: Ruta del archivo original
    :param replacements: Diccionario con reemplazos {texto_a_buscar: texto_a_reemplazar}
    :param output_path: Ruta del archivo de salida
    """
    if not os.path.isfile(input_path):
        print(f"El archivo {input_path} no existe.")
        return
    
    # Lee el contenido del archivo
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Realiza todos los reemplazos
    for search_text, replace_text in replacements.items():
        content = content.replace(search_text, replace_text)
    
    # Escribe el archivo modificado en la ruta de salida
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)
    
    print(f"Archivo procesado guardado en: {output_path}")


if __name__ == "__main__":
    # Carpeta donde están los archivos originales
    input_folder = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS\CleanClean"
    
    # Carpeta donde se guardarán los archivos procesados
    output_folder = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS"
    
    # Verifica si el directorio de salida existe, si no lo crea
    if not os.path.exists(output_folder):
        print(f"Creando carpeta de salida: {output_folder}")
        os.makedirs(output_folder)
    else:
        print(f"Carpeta de salida ya existe: {output_folder}")
    
    # Patrón para buscar archivos
    file_pattern = os.path.join(input_folder, "IoT_Clean.txt")
    # Lista de archivos que coinciden con el patrón
    files = glob.glob(file_pattern)
    print(f"Archivos encontrados: {files}")
    
    # Diccionario con los reemplazos que quieres realizar
    replacements = {
        ",,,": "    ",  # Cambiar espacios por puntos
        ",,,,": "    ",   
        ",,,,,": "    ",   # Cambiar comas por puntos
        ",,,,,,": "    ",
        "    ,": "    ",
        "        ,": "    ",
    }
    
    # Procesar cada archivo
    for input_file in files:
        # Define el nombre del archivo de salida en la carpeta de salida
        output_file = os.path.join(
            output_folder, 
            os.path.basename(input_file).replace("Clean", "Final")
        )
        
        # DEBUG: Muestra las rutas para confirmar
        print(f"Procesando archivo de entrada: {input_file}")
        print(f"Archivo procesado se guardará en: {output_file}")
        
        # Llama a la función con las rutas correctas
        replace_in_file(input_file, replacements, output_file)

