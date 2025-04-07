from .base_agent import KnifeStoreAgent, CONVERSATION_EXAMPLES
import cv2
import pytesseract
import re
import logging
import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class Cliente(Base):
    __tablename__ = 'clientes'
    
    id = Column(Integer, primary_key=True)
    nombre = Column(String, nullable=False, unique=True)
    cuit_cuil = Column(String, nullable=True)
    telefono = Column(String, nullable=True)

class PurchaseOrder(Base):
    __tablename__ = 'purchase_orders'
    
    id = Column(Integer, primary_key=True)
    order_number = Column(String, nullable=False, unique=True)  # Número de operación único
    cliente_nombre = Column(String, nullable=False)
    cliente_id = Column(Integer, nullable=True)
    amount = Column(Float, nullable=False)
    status = Column(String, nullable=False)

def setup_database():
    """Crea la conexión a la base de datos y retorna la sesión."""
    try:
        # Use the environment variable for database connection
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/knife_store")
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        logger.error(f"Error al conectar con la base de datos: {e}")
        raise

class PaymentAgent(KnifeStoreAgent):
    """
    Agente especializado en la validación de pagos a través de imágenes.
    Ahora con confirmación de número de orden de compra.
    """
    
    def __init__(self, db_session=None):
        super().__init__(
            name="Agente de Pagos",
            role="Validador de Pagos",
            goal="Verificar y validar comprobantes de pago enviados por los clientes",
            backstory="""
            Soy un especialista en verificación de pagos con amplia experiencia en procesamiento
            de comprobantes y validación de transacciones financieras. Mi trabajo es asegurar
            que los pagos realizados por los clientes sean procesados correctamente, verificando
            que coincidan con los montos esperados en las órdenes de compra.
            """,
            verbose=True
        )
        # If no session is provided, create one
        self.db_session = db_session if db_session is not None else setup_database()
        # Diccionario para almacenar el estado de confirmación de órdenes por usuario
        # Key: número de teléfono, Value: {'status': 'awaiting_confirmation', 'order_number': '123456'}
        self.confirmation_state = {}
    
    def extract_amount_from_image(self, image_path):
        """
        Extrae el monto de pago de una imagen de comprobante.
        
        Args:
            image_path (str): Ruta a la imagen del comprobante.
            
        Returns:
            float: El monto extraído como número flotante.
        """
        try:
            # Cargar la imagen
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen desde {image_path}")
                raise FileNotFoundError(f"No se pudo cargar la imagen desde {image_path}")
            
            # Convertir a escala de grises para mejor OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar umbral para obtener una imagen binaria
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Usar pytesseract para extraer texto de toda la imagen
            text = pytesseract.image_to_string(binary)
            
            # Buscar un patrón de monto ($ seguido de números, posiblemente con comas y punto decimal)
            amount_pattern = r'\$\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)'
            match = re.search(amount_pattern, text)
            
            if match:
                # Extraer el monto coincidente y normalizar el formato
                # Primero se eliminan comas y puntos
                amount_str = match.group(1).replace(',', '').replace('.', '')
                
                # Convertir a formato con 2 decimales:
                # Si la cadena tiene más de dos dígitos, se asume que los dos últimos son decimales.
                if len(amount_str) > 2:
                    amount_str = amount_str[:-2] + '.' + amount_str[-2:]
                
                amount = float(amount_str)
                logger.info(f"Monto extraído exitosamente: ${amount}")
                return amount
            else:
                logger.error("No se pudo encontrar un patrón de monto en la imagen")
                raise ValueError("Monto no encontrado en la imagen")
            
        except Exception as e:
            logger.error(f"Error al extraer monto de la imagen: {e}")
            raise
    
    def extract_client_name_from_image(self, image_path):
        """
        Extrae el nombre del cliente de la imagen de comprobante.
        Se buscan patrones tanto para el remitente como el destinatario.
        
        Args:
            image_path (str): Ruta a la imagen del comprobante.
            
        Returns:
            dict: Diccionario con la información extraída (nombres y CUIT/CUIL).
        """
        try:
            # Cargar la imagen
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen desde {image_path}")
                raise FileNotFoundError(f"No se pudo cargar la imagen desde {image_path}")
            
            # Convertir a escala de grises para mejor OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extraer texto de la imagen
            text = pytesseract.image_to_string(gray)
            
            result = {
                'sender_name': None,
                'sender_cuit': None,
                'receiver_name': None,
                'receiver_cuit': None
            }
            
            # Buscar el nombre del remitente (después de "De")
            sender_pattern = r'De\s*\n(.*?)\n'
            sender_match = re.search(sender_pattern, text, re.DOTALL)
            if sender_match:
                result['sender_name'] = sender_match.group(1).strip()
            
            # Buscar el CUIT/CUIL del remitente
            sender_cuit_pattern = r'CUIT/CUIL:\s*(\d{2}-\d{8}-\d{1})'
            sender_cuit_match = re.search(sender_cuit_pattern, text)
            if sender_cuit_match:
                result['sender_cuit'] = sender_cuit_match.group(1)
            
            # Buscar el nombre del destinatario (después de "Para")
            receiver_pattern = r'Para\s*\n(.*?)\n'
            receiver_match = re.search(receiver_pattern, text, re.DOTALL)
            if receiver_match:
                result['receiver_name'] = receiver_match.group(1).strip()
            
            # Buscar el CUIT/CUIL del destinatario (normalmente después del CUIT del remitente)
            # Esto es más complicado, así que intentamos buscar otro patrón similar de CUIT/CUIL
            receiver_cuit_pattern = r'CUIT/CUIL:\s*(\d{2}-\d{8}-\d{1})'
            all_cuits = re.findall(receiver_cuit_pattern, text)
            if len(all_cuits) > 1:  # Si hay al menos dos CUITs
                result['receiver_cuit'] = all_cuits[1]  # El segundo CUIT encontrado
            
            logger.info(f"Información de cliente extraída: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error al extraer información del cliente de la imagen: {e}")
            return {}
    
    def extract_operation_number_from_image(self, image_path):
        """
        Extrae el número de operación de Mercado Pago de la imagen.
        
        Args:
            image_path (str): Ruta a la imagen del comprobante.
            
        Returns:
            str: El número de operación extraído.
        """
        try:
            # Cargar la imagen
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen desde {image_path}")
                raise FileNotFoundError(f"No se pudo cargar la imagen desde {image_path}")
            
            # Convertir a escala de grises para mejor OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extraer texto de la imagen
            text = pytesseract.image_to_string(gray)
            
            # Buscar el número de operación
            operation_pattern = r'Número de operación .*?\n(\d+)'
            match = re.search(operation_pattern, text)
            
            if match:
                operation_number = match.group(1).strip()
                logger.info(f"Número de operación extraído: {operation_number}")
                return operation_number
            else:
                logger.error("No se pudo encontrar el número de operación en la imagen")
                raise ValueError("Número de operación no encontrado en la imagen")
            
        except Exception as e:
            logger.error(f"Error al extraer número de operación: {e}")
            raise
    
    async def validate_payment(self, image_path, purchase_order_number=None):
        """
        Valida que el monto de pago en la imagen coincida con el monto esperado en la base de datos.
        Ahora con validación adicional por número de operación.
        
        Args:
            image_path (str): Ruta a la imagen del comprobante de pago.
            purchase_order_number (str, opcional): Número de orden proporcionado por el cliente.
            
        Returns:
            dict: Resultado de la validación con estado y mensaje.
        """
        try:
            # Extraer monto de la imagen
            extracted_amount = self.extract_amount_from_image(image_path)
            
            # Extraer información del cliente (para referencia y posible almacenamiento)
            client_info = self.extract_client_name_from_image(image_path)
            
            # Extraer número de operación de la imagen
            operation_number = None
            try:
                operation_number = self.extract_operation_number_from_image(image_path)
            except:
                logger.warning("No se pudo extraer el número de operación de la imagen")
            
            # Verificar coincidencia del número de operación si se proporcionó
            if purchase_order_number and operation_number and purchase_order_number != operation_number:
                return {
                    "status": "error",
                    "message": f"El número de operación proporcionado ({purchase_order_number}) no coincide con el de la imagen ({operation_number})"
                }
            
            # Usar el número de operación extraído si no se proporcionó uno
            if not purchase_order_number and operation_number:
                purchase_order_number = operation_number
            
            # Si hay una sesión de base de datos, buscar la orden de compra
            if self.db_session is not None:
                # Primero intentamos buscar por número de orden
                if purchase_order_number:
                    purchase_order = self.db_session.query(PurchaseOrder).filter_by(order_number=purchase_order_number).first()
                    
                    if not purchase_order:
                        return {
                            "status": "error",
                            "message": f"No se encontró una orden de compra con el número: {purchase_order_number}"
                        }
                    
                    # Almacenar o actualizar información del cliente si tenemos datos suficientes
                    if client_info.get('sender_name') and client_info.get('sender_cuit'):
                        cliente = self.db_session.query(Cliente).filter_by(cuit_cuil=client_info['sender_cuit']).first()
                        if not cliente:
                            cliente = Cliente(
                                nombre=client_info['sender_name'],
                                cuit_cuil=client_info['sender_cuit']
                            )
                            self.db_session.add(cliente)
                            self.db_session.commit()
                            
                            # Actualizar la orden de compra con el ID del cliente
                            purchase_order.cliente_id = cliente.id
                            self.db_session.commit()
                    
                    expected_amount = purchase_order.amount
                    
                    tolerance = 0.01
                    if abs(extracted_amount - expected_amount) <= tolerance:
                        # Marcar la orden como pagada si la validación es exitosa
                        purchase_order.status = 'PAGADO'
                        self.db_session.commit()
                        
                        return {
                            "status": "success",
                            "message": f"Validación de pago exitosa. El monto ${extracted_amount} coincide con el monto esperado para la orden {purchase_order_number}."
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"Validación de pago fallida. El monto extraído ${extracted_amount} no coincide con el monto esperado ${expected_amount} para la orden {purchase_order_number}."
                        }
                else:
                    # Si no tenemos número de orden, intentamos buscar por monto y quizás por cliente
                    return {
                        "status": "warning",
                        "message": f"Se extrajo el monto ${extracted_amount} pero no se pudo identificar el número de orden. Por favor, proporciona el número de operación para completar la validación."
                    }
            else:
                # Si no hay sesión de base de datos, solo informamos el monto extraído
                return {
                    "status": "info",
                    "message": f"Se extrajo el monto ${extracted_amount} de la imagen, pero no se pudo validar con la base de datos."
                }
                
        except Exception as e:
            logger.error(f"Error durante la validación de pago: {e}")
            return {
                "status": "error",
                "message": f"Ocurrió un error durante la validación: {str(e)}"
            }
    
    def check_confirmation_state(self, phone_number):
        """
        Verifica si un usuario está esperando confirmación de número de orden.
        
        Args:
            phone_number (str): Número de teléfono del usuario.
            
        Returns:
            dict: Estado actual de confirmación o None si no hay estado activo.
        """
        return self.confirmation_state.get(phone_number)
    
    def set_confirmation_state(self, phone_number, state):
        """
        Establece el estado de confirmación para un usuario.
        
        Args:
            phone_number (str): Número de teléfono del usuario.
            state (dict): Estado a establecer (o None para eliminar).
        """
        if state is None and phone_number in self.confirmation_state:
            del self.confirmation_state[phone_number]
        else:
            self.confirmation_state[phone_number] = state
    
    async def handle_payment_image(self, message, image_path, phone_number=None):
        """
        Maneja el procesamiento de una imagen de comprobante de pago.
        Ahora con soporte para confirmación de número de orden.
        
        Args:
            message (str): Mensaje del cliente.
            image_path (str): Ruta a la imagen del comprobante.
            phone_number (str, opcional): Número de teléfono del cliente para seguimiento de estado.
            
        Returns:
            str: Respuesta para el cliente.
        """
        # Verificar si estamos esperando confirmación de este usuario
        confirmation_state = None
        if phone_number:
            confirmation_state = self.check_confirmation_state(phone_number)
        
        # Si hay un estado de confirmación pendiente y el usuario está respondiendo
        if confirmation_state and confirmation_state['status'] == 'awaiting_confirmation':
            # Intenta extraer un número del mensaje del usuario
            number_match = re.search(r'\b\d{7,12}\b', message)
            purchase_order_number = None
            
            if number_match:
                purchase_order_number = number_match.group(0)
                logger.info(f"Número de orden proporcionado por el usuario: {purchase_order_number}")
            else:
                # Si no se encontró un número, podemos intentar varias alternativas
                # 1. Ver si el usuario dice algo como "sí, es correcto" y usar el número que ya teníamos
                affirmative_patterns = [r'\bs[iíl]\b', r'\bcorrecto\b', r'\bese( es)?\b', r'\bexacto\b']
                if any(re.search(pattern, message.lower()) for pattern in affirmative_patterns):
                    purchase_order_number = confirmation_state.get('suggested_order_number')
                    logger.info(f"Usuario confirmó el número sugerido: {purchase_order_number}")
                else:
                    # Si no confirmó y no proporcionó un número, pedir clarificación
                    return "Por favor, necesito que me confirmes el número de operación exacto para validar tu pago."
            
            # Limpiar el estado de confirmación
            self.set_confirmation_state(phone_number, None)
            
            # Proceder con la validación usando el número proporcionado
            validation_result = await self.validate_payment(image_path, purchase_order_number)
        else:
            # Procesamiento normal sin número de orden específico
            # Intentamos extraer el número de operación de la imagen
            try:
                operation_number = self.extract_operation_number_from_image(image_path)
                
                if phone_number:
                    # Establecer un estado de confirmación para este usuario
                    self.set_confirmation_state(phone_number, {
                        'status': 'awaiting_confirmation',
                        'suggested_order_number': operation_number,
                        'image_path': image_path
                    })
                    
                    # Pedir confirmación al usuario
                    return f"He detectado que el número de operación en tu comprobante es {operation_number}. ¿Es correcto? Por favor, confirma o proporciónalo manualmente."
                
                # Si no podemos mantener estado (no hay número de teléfono), procedemos sin confirmación
                validation_result = await self.validate_payment(image_path, operation_number)
            except:
                # Si no pudimos extraer el número de operación, pedimos al usuario
                if phone_number:
                    # Establecer un estado de confirmación para este usuario
                    self.set_confirmation_state(phone_number, {
                        'status': 'awaiting_confirmation',
                        'image_path': image_path
                    })
                    
                    return "Por favor, indícame el número de operación o número de orden que aparece en tu comprobante para validar el pago."
                
                # Si no podemos mantener estado, procedemos sin número de orden
                validation_result = await self.validate_payment(image_path)
        
        # Generar respuesta basada en el resultado de la validación
        prompt = f"""
        Como especialista en verificación de pagos, responde a este mensaje sobre un comprobante de pago:
        
        Mensaje del cliente: {message}
        
        Resultado de la validación: {validation_result}
        
        Instrucciones importantes:
        - Sé breve y directo, máximo 2-3 oraciones
        - NO repitas la información técnica sobre la validación
        - Si la validación fue exitosa, confirma el pago de forma amigable y menciona que la orden ha sido marcada como pagada
        - Si hubo un error, explica el problema de manera clara y ofrece ayuda para resolverlo
        - Usa "vos" en lugar de "usted"
        - NO utilices emojis en tus respuestas
        - Mantén un tono profesional pero amigable
        """
        
        # Analizar y generar la respuesta
        return await self.analyze_message(prompt)
