"""
Image Processor Utilities
=========================

Utilitários para processamento avançado de imagens.
"""

import os
from pathlib import Path
from typing import List, Dict, Any


class ImageProcessor:
    """Classe para processamento avançado de imagens."""
    
    @staticmethod
    def validate_image_path(image_path: str) -> bool:
        """
        Valida se o caminho da imagem existe e é válido.
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            True se válido, False caso contrário
        """
        if not image_path or not os.path.exists(image_path):
            return False
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return Path(image_path).suffix.lower() in valid_extensions
    
    @staticmethod
    def get_image_info(image_path: str) -> Dict[str, Any]:
        """
        Obtém informações básicas da imagem.
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Dicionário com informações da imagem
        """
        if not ImageProcessor.validate_image_path(image_path):
            return {}
        
        path_obj = Path(image_path)
        return {
            'name': path_obj.name,
            'size_bytes': path_obj.stat().st_size,
            'extension': path_obj.suffix,
            'directory': str(path_obj.parent)
        }
    
    @staticmethod
    def filter_images_by_size(image_paths: List[str], min_size_kb: int = 10) -> List[str]:
        """
        Filtra imagens por tamanho mínimo.
        
        Args:
            image_paths: Lista de caminhos de imagens
            min_size_kb: Tamanho mínimo em KB
            
        Returns:
            Lista filtrada de imagens
        """
        filtered_images = []
        min_size_bytes = min_size_kb * 1024
        
        for image_path in image_paths:
            if ImageProcessor.validate_image_path(image_path):
                if Path(image_path).stat().st_size >= min_size_bytes:
                    filtered_images.append(image_path)
        
        return filtered_images
    
    @staticmethod
    def group_images_by_extension(image_paths: List[str]) -> Dict[str, List[str]]:
        """
        Agrupa imagens por extensão.
        
        Args:
            image_paths: Lista de caminhos de imagens
            
        Returns:
            Dicionário agrupado por extensão
        """
        grouped = {}
        
        for image_path in image_paths:
            if ImageProcessor.validate_image_path(image_path):
                ext = Path(image_path).suffix.lower()
                if ext not in grouped:
                    grouped[ext] = []
                grouped[ext].append(image_path)
        
        return grouped
