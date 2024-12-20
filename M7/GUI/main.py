import sys
from PyQt5.QtWidgets import QApplication
from GUI import GMATGUI


def on_confirm():
    inputs = window.get_parameters()
    print("\n".join([f"{key}: {value}" for key, value in inputs.items()]))
    # Aquí puedes llamar a funciones específicas para procesar los parámetros.

if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    # Crear y mostrar la ventana principal
    window = GMATGUI()
    window.show()

    window.confirm_button.clicked.connect(on_confirm)

    sys.exit(app.exec_())