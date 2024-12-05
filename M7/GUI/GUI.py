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
        self.setGeometry(100, 100, 600, 400)
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
        self.title_label = Widget.QLabel("Definir Variables para GMAT")
        self.title_label.setFont(Gui.QFont("Arial", 22, Gui.QFont.Bold))
        self.title_label.setAlignment(Core.Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2d3436; margin-bottom: 20px;")

        # Logo (opcional)
        self.logo_label = Widget.QLabel(self)
        self.logo_pixmap = Gui.QPixmap(logo_path)

        if self.logo_pixmap.isNull():
            print("El logo no se pudo cargar. Verifica la ruta.")
        else:
            self.logo_label.setPixmap(self.logo_pixmap)
            self.logo_label.setAlignment(Core.Qt.AlignCenter)

        # Layout principal
        self.main_layout = Widget.QVBoxLayout()
        self.main_layout.addWidget(self.title_label)
        self.main_layout.addWidget(self.logo_label)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

        # Layout para los campos principales y las opciones adicionales (horizontal)
        self.fields_layout = Widget.QHBoxLayout()

        # Layout para más opciones (inicialmente oculto)
        self.more_options_layout = Widget.QVBoxLayout()
        self.more_options_layout.setContentsMargins(30, 10, 30, 10)

        # Contenedor para los inputs
        self.inputs = {}
        self.more_options_widgets = []

        # Parametros
        self.add_input_date("Fecha de inicio:")
        self.add_input_decimal("SMA (Km):", initial_value="6878.0")
        self.add_input_decimal("Excentricidad:", initial_value="0")
        self.add_input_decimal("RAAN inicial (deg):", initial_value="0")
        self.add_input_slider("Inclinacion (deg):", -90, 90, 10)
        self.add_input_integer("Número de planos:", initial_value="1")
        self.add_input_integer("Nº satelites por plano:", initial_value="1")
        self.add_input_decimal("Duracion propagacion (dias):", initial_value="1")
        self.add_input_integer("Número de estaciones:", initial_value="0")
        self.add_input_dropdown("Zona a estudiar:", \
                                ["Arctic Ocean", "North Atlantic Ocean", "South Atlantic Ocean", \
                                "North Pacific Ocean", "South Pacific Ocean", "INDIAN OCEAN", \
                                "Mediterranean Sea"])
        self.add_input_checkbox("Mostrar GMAT GUI")
        #self.add_input_integer("Parámetro 3 (entero):")
        #self.add_input_slider("Parámetro 4 (deslizable):", 0, 100, 10)
        #self.add_input_checkbox("Activar parámetro 5")

        # Botón de "Más opciones"
        self.more_options_button = Widget.QPushButton("Más opciones")
        self.more_options_button.clicked.connect(self.toggle_more_options)
        self.main_layout.addWidget(self.more_options_button)

        # Parametros más opciones
        #self.add_input_decimal("Parámetro 6 (decimal):", self.more_options_layout)
        #self.add_input_slider("Parámetro 7 (deslizable):", 0, 100, 10, self.more_options_layout)
        # self.add_input_path("Ruta del archivo:", self.more_options_layout, folder_mode=False)  # Para seleccionar un archivo
        self.add_input_path("Workspace:", self.more_options_layout, folder_mode=True, initial_value=current_path)  # Para seleccionar una carpeta
        self.add_input_path("GMAT.exe path:", self.more_options_layout, folder_mode=False, initial_value=os.path.join(current_path,"GMAT.exe")) 

        # Añadir los widgets adicionales al layout principal
        self.fields_layout.addLayout(self.more_options_layout)
  
        # Añadimos el layout de campos y el botón "Más opciones" al layout principal
        self.main_layout.addLayout(self.fields_layout)

        # Botón de Confirmación
        self.confirm_button = Widget.QPushButton("Confirmar")
        self.confirm_button.setFont(Gui.QFont("Arial", 14))
        self.main_layout.addWidget(self.confirm_button)

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


    def toggle_more_options(self):
        """Mostrar u ocultar los widgets adicionales a la derecha."""
        # Comprobar si alguna opción está visible
        widgets_visible = any(widget.isVisible() for widget in self.more_options_widgets)

        # Si están visibles, ocultamos los widgets
        if widgets_visible:
            for widget in self.more_options_widgets:
                widget.setVisible(False)
            self.more_options_button.setText("Más opciones")  # Cambiar el texto del botón
        else:
            # Si no están visibles, mostramos los widgets
            for widget in self.more_options_widgets:
                widget.setVisible(True)
            self.more_options_button.setText("Ocultar opciones")   # Cambiar el texto del botón

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
            label.setMinimumWidth(150)
            row_layout.addWidget(label)

        if value_label:
            # value_label.setAlignment(Core.Qt.AlignCenter)  # Centrar la etiqueta de valor
            # value_label.setFixedWidth(40)  # Tamaño fijo para la etiqueta de valor
            row_layout.addWidget(value_label)
            
        if field:
            # field.setMaximumWidth(500)  # Establecer el ancho máximo
            # field.setMinimumWidth(150)  # Establecer el ancho mínimo
            row_layout.addWidget(field)

        # spacer = Widget.QSpacerItem(40, 20, Widget.QSizePolicy.Expanding, Widget.QSizePolicy.Minimum)
        # row_layout.addItem(spacer)
        row_layout.setAlignment(Core.Qt.AlignCenter)
        target_layout = target_layout or self.main_layout
        target_layout.addLayout(row_layout)

    def list_more_options(self, layout, field=None, label=None, value_label=None):

        if layout == self.more_options_layout:
            if field: self.more_options_widgets.append(field); field.setVisible(False)
            if label: self.more_options_widgets.append(label); label.setVisible(False)
            if value_label: self.more_options_widgets.append(value_label); value_label.setVisible(False)


    def add_input_decimal(self, label_text, target_layout=None, initial_value=""):
        """Agregar un input para números decimales."""
        label = Widget.QLabel(label_text)
        label.setFont(Gui.QFont("Arial", 14))
        input_field = Widget.QLineEdit()
        # Configurar el validador para números decimales con punto
        validator = Gui.QDoubleValidator(-1e10, 1e10, 9)
        validator.setLocale(Core.QLocale(Core.QLocale.English))  # Forzar formato con punto decimal
        input_field.setValidator(validator)

        input_field.setPlaceholderText("Ingrese un número decimal")
        input_field.setText(str(initial_value))

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
        path_layout.addWidget(input_field)
        path_layout.addWidget(browse_button)

        # Añadir la etiqueta y el layout horizontal al layout principal
        self.add_in_layout(target_layout, None, label)
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

        return parameters
