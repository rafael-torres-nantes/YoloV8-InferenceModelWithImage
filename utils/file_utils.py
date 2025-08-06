"""
File Utilities
==============

Utilitários para manipulação de arquivos e diretórios.
"""

import os
import glob
from pathlib import Path
from typing import List
from config.settings import config


class FileUtils:
    """Utilitários para arquivos."""
    
    @staticmethod
    def find_images_in_folder(folder_path: str) -> List[str]:
        """
        Encontra todas as imagens em uma pasta.
        
        Args:
            folder_path: Caminho da pasta
            
        Returns:
            Lista de caminhos das imagens (sem duplicatas)
        """
        image_extensions = config.SUPPORTED_IMAGE_EXTENSIONS
        
        image_files = set()  # Usar set para evitar duplicatas
        folder = Path(folder_path)
        
        if not folder.exists():
            return []
        
        for ext in image_extensions:
            # Procurar por extensões em minúsculo e maiúsculo
            pattern_lower = f"*{ext.lower()}"
            pattern_upper = f"*{ext.upper()}"
            
            # Adicionar ao set (automaticamente evita duplicatas)
            image_files.update(folder.glob(pattern_lower))
            image_files.update(folder.glob(pattern_upper))
        
        # Converter caminhos Path para strings e ordenar
        return sorted([str(img_path) for img_path in image_files])
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> Path:
        """
        Garante que um diretório existe.
        
        Args:
            directory_path: Caminho do diretório
            
        Returns:
            Path object do diretório
        """
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """
        Retorna o tamanho do arquivo em MB.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Tamanho em MB
        """
        if os.path.exists(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)
        return 0.0
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Retorna lista de modelos disponíveis localmente.
        
        Returns:
            Lista de caminhos dos modelos
        """
        return config.get_available_local_models()
