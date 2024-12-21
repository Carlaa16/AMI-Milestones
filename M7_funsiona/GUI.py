import PyQt5.QtWidgets as Widget
import PyQt5.QtGui as Gui
import PyQt5.QtCore as Core
import os

class GMATGUI(Widget.QMainWindow):
    def __init__(self):
        super().__init__()
        # Configuración de la ventana principal
        current_path = os.path.dirname(__file__)
        logo_path = os.path.join(current_path, "Header.png")
        self.setWindowTitle("GMAT Interface")
        self.setGeometry(100, 100, 500, 400)

        # Establecer un tamaño mínimo y máximo
        self.setMinimumSize(1000, 900)  # Tamaño mínimo para asegurar la visibilidad de todos los elementos
        self.setMaximumSize(1920, 1080)

        self.setWindowIcon(Gui.QIcon(logo_path)) 
        
        # Fondo y estilo general
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f6fa;  /* Color de fondo de toda la ventana */
            }
            QLabel {
                font-size: 14px;
                color: #2f3640;
            }
            QLineEdit {
                padding: 8px;
                font-size: 14px;
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #ffffff;
            }
            QComboBox {
                padding: 8px;
                font-size: 14px;
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #ffffff;
            }
            QSlider {
                padding: 8px;
                font-size: 14px;
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #ffffff;
            }
            QCheckBox {
                padding: 8px;
                font-size: 14px;
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 12px 20px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        # Título
        self.title_label = Widget.QLabel("Configuración de la misión")
        self.title_label.setFont(Gui.QFont("Arial", 22, Gui.QFont.Bold))
        self.title_label.setAlignment(Core.Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2d3436; margin-bottom: 20px;")

        # Logo (opcional)
        self.logo_label = Widget.QLabel(self)
        self.logo_pixmap = Gui.QPixmap(logo_path)

        if self.logo_pixmap.isNull():
            print(f"El logo no se pudo cargar. Verifica la ruta {logo_path}.")
        else:
            self.logo_label.setPixmap(self.logo_pixmap)
            self.logo_label.setAlignment(Core.Qt.AlignCenter)

        # Layout principal
        self.main_layout = Widget.QVBoxLayout()
        self.main_layout.addWidget(self.title_label)
        self.main_layout.addWidget(self.logo_label)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setSpacing(5)
        self.main_layout.setAlignment(Core.Qt.AlignTop)

        # Layout para los campos principales y las opciones adicionales (horizontal)
        self.fields_layout = Widget.QHBoxLayout()

        # Layout para parámetros orbitales (inicialmente oculto)
        self.orbits_layout = Widget.QVBoxLayout()
        self.orbits_layout.setContentsMargins(30, 10, 30, 10)

        # Botón de "Parámetros orbitales"
        self.orbits_button = Widget.QPushButton("Parámetros orbitales")
        self.orbits_button.clicked.connect(lambda: self.toggle_options(self.orbits_layout, self.orbits_button))

        self.comms_sat_layout = Widget.QVBoxLayout()
        self.comms_sat_layout.setContentsMargins(30, 10, 30, 10)

        self.comms_sat_button = Widget.QPushButton("Comunicaciones (satélite)")
        self.comms_sat_button.clicked.connect(lambda: self.toggle_options(self.comms_sat_layout, self.comms_sat_button))

        self.comms_iot_layout = Widget.QVBoxLayout()
        self.comms_iot_layout.setContentsMargins(30, 10, 30, 10)

        self.comms_iot_button = Widget.QPushButton("Comunicaciones (estación)")
        self.comms_iot_button.clicked.connect(lambda: self.toggle_options(self.comms_iot_layout, self.comms_iot_button))

        self.comms_zone_layout = Widget.QVBoxLayout()
        self.comms_zone_layout.setContentsMargins(30, 10, 30, 10)

        self.comms_zone_button = Widget.QPushButton("Área de estudio")
        self.comms_zone_button.clicked.connect(lambda: self.toggle_options(self.comms_zone_layout, self.comms_zone_button))

        self.more_paths_layout = Widget.QVBoxLayout()
        self.more_paths_layout.setContentsMargins(30, 10, 30, 10)

        self.more_paths_button = Widget.QPushButton("Configuración de directorios")
        self.more_paths_button.clicked.connect(lambda: self.toggle_options(self.more_paths_layout, self.more_paths_button))

        # Contenedor para los inputs
        self.popups = []
        self.inputs = {}
        self.option_widgets = {}
        self.layout_dict = {
            self.more_paths_button: (self.more_paths_layout, "Configuración de directorios", "Ocultar configuración de directorios"),
            self.orbits_button: (self.orbits_layout, "Parámetros orbitales", "Ocultar parámetros orbitales"),
            self.comms_sat_button: (self.comms_sat_layout, "Comunicaciones (satélite)", "Ocultar Comunicaciones (satélite)"),
            self.comms_iot_button: (self.comms_iot_layout, "Comunicaciones (estación)", "Ocultar Comunicaciones (estación)"),
            self.comms_zone_button: (self.comms_zone_layout, "Área de estudio", "Ocultar área de estudio")
                            }       

        self.main_layout.addWidget(self.orbits_button)

        self.add_input_date("Fecha de inicio:",self.orbits_layout)
        self.add_input_decimal("SMA (Km):",self.orbits_layout, initial_value="6878.0")
        self.add_input_decimal("Excentricidad:",self.orbits_layout, initial_value="0")
        self.add_input_decimal("RAAN inicial (deg):",self.orbits_layout, initial_value="0")
        self.add_input_slider("Inclinación (deg):", -90, 90, 10,self.orbits_layout)
        self.add_input_integer("Número de planos:",self.orbits_layout, initial_value="1")
        self.add_input_integer("Nº satélites por plano:",self.orbits_layout, initial_value="1")
        self.add_input_decimal("Duración propagación (días):",self.orbits_layout, initial_value="1")
        self.add_input_checkbox("Mostrar GMAT GUI",self.orbits_layout)

        self.main_layout.addWidget(self.comms_sat_button)

        self.add_input_slider("Ángulo de visión satélite (deg):", 0, 180, 18,self.comms_sat_layout)
        self.add_input_decimal("Potencia transmisor satélite (dBm):",self.comms_sat_layout, initial_value="12,5")
        self.add_input_checkbox("Usar amplificador satélite",self.comms_sat_layout)
        self.add_input_decimal("Ganancia amplificador satélite (dB):",self.comms_sat_layout, initial_value="20")
        self.add_input_decimal("Ganancia antena satélite (dB):",self.comms_sat_layout, initial_value="12,5")
        self.add_input_integer("Tamaño mensaje satélite:",self.comms_sat_layout, initial_value="100")

        self.main_layout.addWidget(self.comms_iot_button)

        self.add_input_slider("Ángulo de visión estación (deg):", 0, 180, 18,self.comms_iot_layout)
        self.add_input_decimal("Potencia transmisor estación (dBm):",self.comms_iot_layout, initial_value="12,5")
        self.add_input_checkbox("Usar amplificador estación",self.comms_iot_layout)
        self.add_input_decimal("Ganancia amplificador estación (dB):",self.comms_iot_layout, initial_value="20")
        self.add_input_decimal("Ganancia antena estación (dB):",self.comms_iot_layout, initial_value="14,5")
        self.add_input_integer("Tamaño mensaje estación:",self.comms_iot_layout, initial_value="10000000")

        self.main_layout.addWidget(self.comms_zone_button)
        
        self.add_input_integer("Número de estaciones:",self.comms_zone_layout, initial_value="1")
        self.add_input_list("Zona a estudiar:", \
                                ["Arctic Ocean", "North Atlantic Ocean", "South Atlantic Ocean", \
                                "North Pacific Ocean", "South Pacific Ocean", "INDIAN OCEAN", \
                                "Mediterranean Sea"],self.comms_zone_layout)

        self.main_layout.addWidget(self.more_paths_button)

        # Parametros más opciones
        self.add_input_path("Workspace:", self.more_paths_layout, folder_mode=True, initial_value=current_path)
        self.add_input_path("Output:", self.more_paths_layout, folder_mode=True, initial_value=current_path)
        self.add_input_path("GMAT.exe path:", self.more_paths_layout, folder_mode=False, initial_value=os.path.join(current_path,"GMAT.exe")) 

        # Añadir los widgets adicionales al layout principal
        self.fields_layout.addLayout(self.more_paths_layout)

        # Añadimos el layout de campos y el botón "Más opciones" al layout principal
        self.main_layout.addLayout(self.fields_layout)

        # Añadir un espacio expansible para empujar los botones hacia abajo
        spacer = Widget.QSpacerItem(20, 40, Widget.QSizePolicy.Minimum, Widget.QSizePolicy.Expanding)
        self.main_layout.addItem(spacer)

        # Botón de Confirmación
        self.confirm_button = Widget.QPushButton("Confirmar")
        self.confirm_button.setFont(Gui.QFont("Arial", 14))
        self.main_layout.addWidget(self.confirm_button)

        # Texto dinámico en la parte inferior
        self.status_label = Widget.QLabel("Estado: Esperando acciones...")
        self.status_label.setAlignment(Core.Qt.AlignCenter)  # Centrado horizontalmente
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #2f3640;
                margin-top: 10px;
            }
        """)

        # Layout para el texto dinámico
        status_layout = Widget.QHBoxLayout()
        status_layout.addWidget(self.status_label)

        # Añadir el layout al layout principal
        self.main_layout.addLayout(status_layout)

        # Crear el botón de cerrar
        self.close_button = Widget.QPushButton("Cerrar")
        self.close_button.setFont(Gui.QFont("Arial", 10))  # Hacer el botón más pequeño

        # Establecer estilo para hacerlo rojo
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;  /* Rojo */
                color: white;
                padding: 6px 12px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #c0392b;  /* Rojo más oscuro al pasar el ratón */
            }
            QPushButton:pressed {
                background-color: #e74c3c;  /* Rojo al presionar */
            }
        """)

        # Conectar el botón de cerrar al método close()
        self.close_button.clicked.connect(self.close)

        # Layout para colocar el botón en la esquina inferior derecha
        button_layout = Widget.QHBoxLayout()
        button_layout.addStretch()  # Añadir espacio vacío para empujar el botón a la derecha
        button_layout.addWidget(self.close_button)

        # Layout vertical principal
        self.main_layout.addLayout(button_layout)

        # Configuración del contenedor central
        container = Widget.QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)


    def toggle_options(self, options_layout, options_button):

        """
        Muestra el layout activo y oculta todos los demás layouts.
        Actualiza el texto de los botones correspondientes.
        """
        # Iterar sobre todos los layouts en self.all_layouts
        for button, (layout, inactive_text, active_text) in self.layout_dict.items():
            widgets_visible = any(widget.isVisible() for widget in self.option_widgets[layout])

            # Si están visibles, ocultamos los widgets
            if widgets_visible:
                for widget in self.option_widgets[layout]:
                    widget.setVisible(False)
                button.setText(inactive_text)

            else:
                # Si no están visibles, mostramos los widgets
                for widget in self.option_widgets[layout]:
                    widget.setVisible(layout == options_layout)
                    if button == options_button: button.setText(active_text)
            
            

    def open_browser(self, browser, input_field):
        if browser:
            path = Widget.QFileDialog.getExistingDirectory(self, "Seleccionar carpeta")
        else:
            path, _ = Widget.QFileDialog.getOpenFileName(self, "Seleccionar archivo")
        
        if path:
            input_field.setText(path)


    def add_in_layout(self, target_layout, field=None, label = None, value_label = None):

        row_layout = Widget.QHBoxLayout()
        row_layout.setAlignment(Core.Qt.AlignCenter)
 
        if label: 
            label.setAlignment(Core.Qt.AlignRight | Core.Qt.AlignVCenter)
            label.setMinimumWidth(250)
            row_layout.addWidget(label)

        if value_label:
            row_layout.addWidget(value_label)
            
        if field:
            row_layout.addWidget(field)

        row_layout.setAlignment(Core.Qt.AlignCenter)
        target_layout = target_layout or self.main_layout
        target_layout.setSpacing(3)
        target_layout.addLayout(row_layout)

    def list_more_options(self, layout, field=None, label=None, value_label=None):

        if layout != self.main_layout:

            if layout not in self.option_widgets:
                self.option_widgets[layout] = []  # Inicializar lista para este layout si no existe

            # Añadir widgets al diccionario y hacerlos inicialmente invisibles
            for widget in (field, label, value_label):
                if widget:
                    self.option_widgets[layout].append(widget)
                    widget.setVisible(False)


    def add_input_decimal(self, label_text, target_layout=None, initial_value=""):
        """Agregar un input para números decimales."""
        label = Widget.QLabel(label_text)
        label.setFont(Gui.QFont("Arial", 14))
        input_field = Widget.QLineEdit()
        # Configurar el validador para números decimales con punto
        validator = Gui.QRegularExpressionValidator(Core.QRegularExpression(r"^-?\d{0,10}(,\d{0,10})?$"))
        validator.setLocale(Core.QLocale(Core.QLocale.English))  # Forzar formato con punto decimal
        input_field.setValidator(validator)

        input_field.setPlaceholderText("Ingrese un número decimal")
        input_field.setText(str(initial_value).replace(",", "."))

        self.add_in_layout(target_layout, input_field, label)
        self.list_more_options(target_layout, input_field, label)

        self.inputs[label_text] = input_field


    def add_input_integer(self, label_text, target_layout=None, initial_value=""):
        """Agregar un input para números enteros."""
        label = Widget.QLabel(label_text)
        label.setFont(Gui.QFont("Arial", 14))
        input_field = Widget.QLineEdit()
        input_field.setValidator(Gui.QIntValidator(-100, 100))
        input_field.setPlaceholderText("Ingrese un número entero")
        input_field.setText(str(initial_value))

        self.add_in_layout(target_layout, input_field, label)
        self.list_more_options(target_layout, input_field, label)

        self.inputs[label_text] = input_field


    def add_input_dropdown(self, label_text, options, target_layout=None):
        """Agregar un desplegable con opciones."""
        label = Widget.QLabel(label_text)
        label.setFont(Gui.QFont("Arial", 14))
        dropdown = Widget.QComboBox()
        dropdown.addItems(options)
        dropdown.setFont(Gui.QFont("Arial", 14))
        dropdown.setSizePolicy(Widget.QSizePolicy.Expanding, Widget.QSizePolicy.Fixed)

        self.add_in_layout(target_layout, dropdown, label)
        self.list_more_options(target_layout, dropdown, label)
    
        self.inputs[label_text] = dropdown


    def add_input_slider(self, label_text, min_val, max_val, steps, target_layout=None):

    
        """Agregar un deslizador con una etiqueta para mostrar su valor junto al texto descriptivo."""

        # Etiqueta descriptiva del deslizador
        label = Widget.QLabel(label_text)
        label.setFont(Gui.QFont("Arial", 14))

        # Crear el deslizador
        slider = Widget.QSlider(Core.Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue((min_val + max_val) // 2)
        slider.setTickPosition(Widget.QSlider.TicksBelow)
        slider.setTickInterval((max_val - min_val) // steps)

        # Crear la etiqueta para mostrar el valor del deslizador
        value_label = Widget.QLabel(f"{slider.value()}")
        value_label.setFont(Gui.QFont("Arial", 12))
        value_label.setAlignment(Core.Qt.AlignCenter)
        value_label.setStyleSheet("background-color: #dfe4ea; border: 1px solid #ccc; padding: 2px;")
        value_label.setFixedWidth(40)  # Tamaño fijo para consistencia visual

        self.add_in_layout(target_layout, slider, label, value_label)
        self.list_more_options(target_layout, slider, label, value_label)

        # Función para actualizar el texto de la etiqueta con el valor actual del deslizador
        def update_label(value):
            value_label.setText(f"{value}")

        # Conectar el evento valueChanged del deslizador a la función de actualización
        slider.valueChanged.connect(update_label)

        # Guardar el deslizador en el diccionario
        self.inputs[label_text] = slider


    def add_input_list(self, label_text, options, target_layout=None):
        """
        Agregar un desplegable con buscador para seleccionar múltiples opciones.
        
        :param label_text: Texto descriptivo de la lista.
        :param options: Lista de opciones disponibles.
        :param target_layout: Layout donde agregar el widget.
        """

        # Etiqueta descriptiva
        label = Widget.QLabel(label_text)
        label.setFont(Gui.QFont("Arial", 14))

        # Widget principal con buscador y lista
        container = Widget.QFrame()
        container.setFrameShape(Widget.QFrame.StyledPanel)
        container.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 5px;")
        layout = Widget.QVBoxLayout(container)

        # Input de búsqueda
        search_box = Widget.QLineEdit()
        search_box.setPlaceholderText("Buscar opciones...")
        layout.addWidget(search_box)
    

        # Lista de opciones (con selección múltiple)
        list_widget = Widget.QListWidget()
        list_widget.addItems(options)
        list_widget.setSelectionMode(Widget.QAbstractItemView.MultiSelection)
        list_widget.setMaximumHeight(150)  # Altura máxima en píxeles para la lista
        list_widget.setSizePolicy(Widget.QSizePolicy.Expanding, Widget.QSizePolicy.Fixed)

        # **Límite de altura máxima para la lista**
        layout.addWidget(list_widget)

        # Filtro en tiempo real
        def filter_list(text):
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                item.setHidden(text.lower() not in item.text().lower())

        search_box.textChanged.connect(filter_list)

        # Autocompletar basado en las opciones
        completer = Widget.QCompleter(options)
        completer.setCaseSensitivity(Core.Qt.CaseInsensitive)
        search_box.setCompleter(completer)

        # Añadir al layout principal
        self.add_in_layout(target_layout, container, label)
        self.list_more_options(target_layout, container, label)

        # Guardar en el diccionario de inputs
        self.inputs[label_text] = list_widget

    def add_input_checkbox(self, label_text, target_layout=None):
        """Agregar una casilla de verificación."""
        checkbox = Widget.QCheckBox(label_text)
        checkbox.setFont(Gui.QFont("Arial", 14))

        self.add_in_layout(target_layout, checkbox)
        self.list_more_options(target_layout, checkbox)

        self.inputs[label_text] = checkbox
        
    def add_input_path(self, label_text, target_layout=None, folder_mode=False, initial_value=""):
        """
        Agregar un input para ingresar una ruta de archivo o carpeta con un botón de búsqueda.
        
        :param label_text: Texto descriptivo para el input.
        :param target_layout: Layout donde agregar el widget (opcional).
        :param folder_mode: Si es True, permite buscar carpetas en lugar de archivos.
        """
        # Crear la etiqueta y el cuadro de texto
        label = Widget.QLabel(label_text)
        label.setFont(Gui.QFont("Arial", 14))
        input_field = Widget.QLineEdit()
        input_field.setPlaceholderText("Ingrese la ruta o use el botón de búsqueda")
        input_field.setText(initial_value)

        # Crear el botón de búsqueda
        browse_button = Widget.QPushButton("Buscar")
        browse_button.setFixedSize(80, 35)

        # Función para abrir el diálogo de selección
        def open_dialog():
            if folder_mode:
                path = Widget.QFileDialog.getExistingDirectory(self, "Seleccionar carpeta")
            else:
                path, _ = Widget.QFileDialog.getOpenFileName(self, "Seleccionar archivo")
            
            if path:
                input_field.setText(path)

        # Conectar el botón de búsqueda a la función
        browse_button.clicked.connect(open_dialog)

        # Crear un layout horizontal para el campo y el botón
        path_layout = Widget.QHBoxLayout()
        label.setAlignment(Core.Qt.AlignRight | Core.Qt.AlignVCenter)
        label.setMinimumWidth(150)
        path_layout.addWidget(label)
        path_layout.addWidget(input_field)
        path_layout.addWidget(browse_button)

        target_layout.addLayout(path_layout)
        self.list_more_options(target_layout, input_field, label, browse_button)

        # Guardar el input en el diccionario
        self.inputs[label_text] = input_field


    def add_input_date(self, label_text, target_layout=None, default_date=None):
        """
        Agregar un input para seleccionar una fecha.
        
        :param label_text: Texto descriptivo para el input.
        :param target_layout: Layout donde agregar el widget (opcional).
        :param default_date: Fecha predeterminada (formato 'dd-MM-yyy', opcional).
        """
        # Crear la etiqueta descriptiva
        label = Widget.QLabel(label_text)
        label.setFont(Gui.QFont("Arial", 14))

        # Crear el widget de selección de fecha
        date_edit = Widget.QDateEdit()
        date_edit.setCalendarPopup(True)  # Habilitar el calendario desplegable
        date_edit.setDisplayFormat("dd-MM-yyyy")  # Formato de la fecha

        # Personalizar el calendario (QCalendarWidget)
        calendar = date_edit.calendarWidget()
        calendar.setStyleSheet("""
            QCalendarWidget {
                background-color: #e3f2fd; /* Fondo azul claro */
                border: 2px solid #64b5f6; /* Borde azul */
                color: #000000; /* Texto negro */
                border-radius: 8px;
            }
            QCalendarWidget QToolButton {
                background-color: #64b5f6; /* Fondo azul más fuerte en botones de navegación */
                color: #ffffff; /* Texto blanco */
                border: none;
                border-radius: 4px;
                padding: 5px;
            }
            QCalendarWidget QToolButton:hover {
                background-color: #42a5f5; /* Hover en los botones */
            }
            QCalendarWidget QToolButton:pressed {
                background-color: #1e88e5; /* Botón presionado */
            }
            QCalendarWidget QMenu {
                background-color: #ffffff; /* Fondo del menú desplegable */
                color: #000000; /* Texto negro */
                border: 1px solid #64b5f6; /* Borde azul */
            }
            QCalendarWidget QSpinBox {
                background-color: #ffffff; /* Fondo blanco */
                color: #000000; /* Texto negro */
                border: 1px solid #64b5f6; /* Borde azul */
                border-radius: 4px;
            }
            QCalendarWidget QAbstractItemView {
                background-color: #ffffff; /* Fondo blanco del calendario */
                selection-background-color: #64b5f6; /* Selección azul */
                selection-color: #ffffff; /* Texto blanco en la selección */
                color: #000000; /* Texto negro */
            }
        """)

        # Configurar la fecha predeterminada si se proporciona
        if default_date:
            date_edit.setDate(Core.QDate.fromString(default_date, "dd-MM-yyyy"))
        else:
            date_edit.setDate(Core.QDate.currentDate())  # Usar la fecha actual por defecto

        # Añadir al layout
        self.add_in_layout(target_layout, date_edit, label)
        self.list_more_options(target_layout, date_edit, label)

        # Guardar el widget en el diccionario de inputs
        self.inputs[label_text] = date_edit


    def get_parameters(self):
        """Devuelve los valores de los parámetros ingresados."""
        parameters = {}

        # Recoger los valores actualizados de cada widget
        for label_text, widget in self.inputs.items():
            if isinstance(widget, Widget.QLineEdit):
                parameters[label_text] = widget.text()
            elif isinstance(widget, Widget.QComboBox):
                parameters[label_text] = widget.currentText()
            elif isinstance(widget, Widget.QSlider):
                parameters[label_text] = widget.value()
            elif isinstance(widget, Widget.QCheckBox):
                parameters[label_text] = widget.isChecked()
            elif isinstance(widget, Widget.QDateEdit):
                parameters[label_text] = widget.date().toString("dd-MM-yyyy")
            elif isinstance(widget, Widget.QListWidget):
                selected_items = widget.selectedItems()
                parameters[label_text] = [item.text() for item in selected_items]

        return parameters

    
    def show_popup(self, message, title="Información"):
        """Muestra una ventana emergente con un mensaje."""
        popup = Widget.QMessageBox(self)
        popup.setWindowTitle(title)
        popup.setText(message)
        popup.setIcon(Widget.QMessageBox.Information)
        popup.setStandardButtons(Widget.QMessageBox.Ok)
        popup.exec_()
    

    def update_status(self, message):
        """
        Actualiza el texto dinámico en la parte inferior de la GUI.
        """
        self.status_label.setText(f"Estado: {message}")
        Widget.QApplication.processEvents()  # Forzar la actualización inmediata de la GUI

    def show_image_popup(self, image_path):
        """
        Muestra una ventana emergente con una imagen y un botón de cerrar.
        La ventana se ajusta automáticamente al tamaño de la imagen.
        Parámetros:
            image_path (str): Ruta de la imagen a mostrar.
        """
    def show_image_popup(self, image_path):
            """
            Muestra una ventana emergente con una imagen utilizando QMessageBox con un botón estándar.
            La ventana se ajusta automáticamente al tamaño de la imagen.
            """
            try:
                # Crear el cuadro de mensaje
                popup = Widget.QMessageBox(self)
                popup.setWindowTitle("Visualización de Imagen")
                popup.setIcon(Widget.QMessageBox.NoIcon)  # Sin icono

                # Crear QLabel para la imagen
                label = Widget.QLabel()
                pixmap = Gui.QPixmap(image_path)
                if pixmap.isNull():
                    raise FileNotFoundError(f"No se puede cargar la imagen: {image_path}")
                label.setPixmap(pixmap)
                label.setScaledContents(False)  # Mantener tamaño original
                label.setAlignment(Core.Qt.AlignCenter)

                # Añadir la imagen al layout del QMessageBox
                layout = popup.layout()
                layout.addWidget(label, 0, 1)  # Añadir el QLabel con la imagen

                # Configurar botones estándar
                popup.setStandardButtons(Widget.QMessageBox.Close)

                # Conectar el botón "Cerrar" para cerrar el popup
                popup.button(Widget.QMessageBox.Close).clicked.connect(lambda: popup.close())

                # Ajustar el tamaño de la ventana
                popup.setMinimumSize(pixmap.width() + 5, pixmap.height() + 5)

                self.popups.append(popup)  # Mantener la referencia para evitar recolección de basura
                popup.show()  # Mostrar ventana no bloqueante
            except Exception as e:
                self.show_popup(f"Error al mostrar la imagen: {e}")

    def _close_popup(self, popup):
        """
        Cierra una ventana emergente y la elimina de la lista de referencias.
        """
        if popup in self.popups:
            self.popups.remove(popup)
        popup.close()