import os
import glob

def remove_duplicates_in_file(input_path, replacements, output_path):
    """
    Elimina líneas duplicadas y reemplaza texto en un archivo de texto.
    """
    if not os.path.isfile(input_path):
        print(f"El archivo {input_path} no existe.")
        return
    
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.readlines()
    
    for i in range(len(content)):
        for search_text, replace_text in replacements.items():
            content[i] = content[i].replace(search_text, replace_text)
    
    unique_lines = list(dict.fromkeys(content))
    
    with open(output_path, "w", encoding="utf-8") as file:
        file.writelines(unique_lines)
    
    print(f"Archivo procesado: {output_path}. Se eliminaron duplicados y aplicaron reemplazos.")


if __name__ == "__main__":
    # Carpeta donde están los archivos originales
    input_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\ISG\CODIGOS\CleanReports"
    
    # Carpeta para guardar los archivos limpiados
    output_file = r"C:\Users\carla\OneDrive\Documentos\MUSE\ISG\CODIGOS"
    os.makedirs(output_file, exist_ok=True)
    
    print(f"Archivo de salida creada/verificada: {output_file}")
    
    # Patrón para buscar archivos
    file_pattern = os.path.join(input_file, "IoT_Cleaned.txt")
    files = glob.glob(file_pattern)
    print(f"Archivos encontrados que coinciden con el patrón: {files}")
    
    if not files:
        print("No se encontraron archivos que coincidan con el patrón.")
    else:
        replacements = {
            ",,,": "    ",
            ",,,,": "    ",
            ",,,,,": "    ",
            ",,,,,,": "    ",
            "    ,": "    ",
        }
        
        for input_file in files:
            # Define el nombre del archivo de salida en la carpeta `CleanClean`
            output_file = os.path.join(output_file, os.path.basename(input_file).replace("Cleaned", "Clean"))
            print(f"Procesando {input_file} -> {output_file}")
            remove_duplicates_in_file(input_file, replacements, output_file)
