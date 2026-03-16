"""Configuración de logging con salida coloreada para mejor legibilidad.

Este módulo proporciona un formateador con colores para logging en consola que hace más fácil escanear logs durante desarrollo y debugging.
Los colores se aplican según el nivel de log:
DEBUG=cyan, INFO=verde, WARNING=amarillo, ERROR=rojo.
"""

from __future__ import annotations

import logging

RESET = '\033[0m'
LEVEL_COLORS = {
    logging.DEBUG: '\033[36m',  # Cyan
    logging.INFO: '\033[32m',  # Verde
    logging.WARNING: '\033[33m',  # Amarillo
    logging.ERROR: '\033[31m',  # Rojo
}


class ColorFormatter(logging.Formatter):
    """Formateador personalizado que agrega códigos de color ANSI según el nivel de log.

    Los colores están definidos en el diccionario LEVEL_COLORS.
    Extiende el logging.Formatter estándar para aplicar colores antes de retornar la salida formateada.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Formatea el registro de log con el color apropiado.

        Args:
            record: LogRecord a formatear.

        Returns:
            String formateado con códigos de color ANSI.
        """
        color = LEVEL_COLORS.get(record.levelno, '')
        plain = super().format(record)
        return f'{color}{plain}{RESET}'


def get_logger(name: str = 'orquestador') -> logging.Logger:
    """Crea o recupera un logger con salida coloreada en consola.

    Configura un logger con:
    - Nivel INFO por defecto
    - ColorFormatter para salida legible en consola
    - StreamHandler escribiendo a stderr
    - Formato: timestamp | level | name | message

    El logger se configura una vez y se reutiliza en llamadas subsecuentes.

    Args:
        name: Nombre del logger (default: "asistente_soporte").

    Returns:
        Instancia de Logger configurada.

    Examples:
        >>> logger.error(f"Error de entrada: {e}")
        >>> 2026-02-16 15:11:54 | ERROR    | __main__ | Error de entrada: No se agregó la consulta. Usa --query 'tu consulta aquí'
        # Muestra log coloreado en consola
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColorFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
