import os
import glob

def remove_duplicates_in_file(input_path, replacements, output_path):
    """
    Elimina líneas duplicadas y reemplaza texto en un archivo de texto.
    
    :param input_path: Ruta del archivo original
    :param replacements: Diccionario con reemplazos {texto_a_buscar: texto_a_reemplazar}
    :param output_path: Ruta del archivo de salida
    """
    if not os.path.isfile(input_path):
        print(f"El archivo {input_path} no existe.")
        return
    
    # Lee el contenido del archivo
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.readlines()
    
    # Realiza los reemplazos en cada línea
    for i in range(len(content)):
        for search_text, replace_text in replacements.items():
            content[i] = content[i].replace(search_text, replace_text)
    
    # Elimina las líneas duplicadas manteniendo el orden
    unique_lines = list(dict.fromkeys(content))
    
    # Escribe el archivo modificado sin duplicados
    with open(output_path, "w", encoding="utf-8") as file:
        file.writelines(unique_lines)
    
    print(f"Archivo procesado: {output_path}. Se eliminaron duplicados y aplicaron reemplazos.")


if __name__ == "__main__":
    # Carpeta donde están los archivos originales
    input_folder = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS"
    
    # Carpeta para guardar los archivos limpiados
    output_folder = r"C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\M7\CODIGOS"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Carpeta de salida creada/verificada: {output_folder}")
    
    # Patrón para buscar archivos
    file_pattern = os.path.join(input_folder, "IoT_Final.txt")
    
    # Lista de archivos que coinciden con el patrón
    files = glob.glob(file_pattern)
    print(f"Archivos encontrados: {files}")
    
    if not files:
        print("No se encontraron archivos que coincidan con el patrón.")
    else:
        # Diccionario con los reemplazos que quieres realizar
        replacements = {
            ",,,": "    ",
            ",,,,": "    ",
            ",,,,,": "    ",
            ",,,,,,": "    ",
            "    ,": "    ",
        }
        
        # Procesar cada archivo
        for input_file in files:
            # Define el nombre del archivo de salida en la carpeta nueva
            output_file = os.path.join(output_folder, os.path.basename(input_file).replace("Final", "Cleaned"))
            print(f"Procesando {input_file} -> {output_file}")
            
            # Llama a la función para procesar el archivo
            remove_duplicates_in_file(input_file, replacements, output_file)
