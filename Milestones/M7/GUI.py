import tkinter as tk
from tkinter import ttk, filedialog, messagebox  # Para widgets adicionales como Treeview, Combobox, etc.
import subprocess
# vamos a usar la libreria TKINTER

# Crear la ventana principal
root = tk.Tk()
root.title("Mi Primera GUI")
root.geometry("300x200")
root.iconbitmap(r'C:\Users\carla\OneDrive\Documentos\MUSE\AM1\AMI-Milestones\Milestones\M7\GMAT.ico')

# Crear una etiqueta
label = ttk.Label(root, text="¡Hola, Mundo!")
label.pack(pady=10)

# Función que se ejecuta al hacer clic en el botón
def on_button_click():
    label.config(text="¡Botón presionado!")

# Crear un botón
button = ttk.Button(root, text="Presióname", command=on_button_click)
button.pack(pady=10)

# Iniciar el bucle principal de la aplicación
root.mainloop()