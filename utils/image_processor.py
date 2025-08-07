"""
Image Processor Utilities
=========================

Utilitários básicos para processamento de imagens.
"""

import os
from pathlib import Path
from typing import Dict, Any


class ImageProcessor:
    """Classe para processamento básico de imagens."""
    
    @staticmethod
    def get_image_info(image_path: str) -> Dict[str, Any]:
        """
        Obtém informações básicas da imagem.
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Dicionário com informações da imagem
        """
        if not image_path or not os.path.exists(image_path):
            return {}
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        path_obj = Path(image_path)
        
        if path_obj.suffix.lower() not in valid_extensions:
            return {}
        
        return {
            'name': path_obj.name,
            'size_bytes': path_obj.stat().st_size,
            'extension': path_obj.suffix,
            'directory': str(path_obj.parent)
        }
