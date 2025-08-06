"""
YoloV8 Inference Service
========================

Serviço principal para inferência com modelos YoloV8.
"""

import os
import cv2
import glob
import shutil
from pathlib import Path
from typing import List, Dict, Any
from ultralytics import YOLO
import numpy as np
from config.settings import config


class YoloV8InferenceService:
    """Serviço de inferência YoloV8."""
    
    def __init__(self, model_path: str = None):
        """
        Inicializa o serviço de inferência.
        
        Args:
            model_path (str): Caminho para o modelo (.pt). Se None, usa modelo padrão
        """
        self.model_path = config.get_model_path(model_path or config.DEFAULT_MODEL)
        self.model = None
        self.output_dir = config.OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        
        # Carrega o modelo
        self.load_model()
    
    def load_model(self):
        """
        Carrega o modelo YoloV8 local ou baixa automaticamente.
        
        Returns:
            None
        """
        try:
            print(f"🔄 Carregando modelo: {self.model_path}")
            
            # Caso 1: Modelo existe localmente
            if os.path.exists(self.model_path):
                self._load_local_model()
                return
            
            # Caso 2: Precisa baixar modelo pré-treinado
            self._download_and_organize_model()
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def _load_local_model(self):
        """
        Carrega modelo que já existe localmente.
        
        Returns:
            None
        """
        self.model = YOLO(self.model_path)
        print(f"✅ Modelo carregado de: {self.model_path}")
    
    def _download_and_organize_model(self):
        """
        Baixa modelo pré-treinado e organiza no diretório correto.
        
        Returns:
            None
        """
        print(f"⬇️  Baixando modelo pré-treinado: {self.model_path}")
        
        # Carrega o modelo (que fará o download automático na raiz)
        self.model = YOLO(self.model_path)
        
        # Verifica se precisa mover o modelo baixado
        downloaded_model = Path(self.model_path)
        if downloaded_model.exists() and not downloaded_model.is_absolute():
            self._move_downloaded_model(downloaded_model)
        
        print(f"✅ Modelo baixado e carregado: {self.model_path}")
    
    def _move_downloaded_model(self, downloaded_model: Path):
        """
        Move modelo baixado para diretório organizado.
        
        Args:
            downloaded_model (Path): Caminho do modelo baixado
            
        Returns:
            None
        """
        target_path = config.PRETRAINED_MODELS_DIR / downloaded_model.name
        
        try:
            shutil.move(str(downloaded_model), str(target_path))
            print(f"📁 Modelo movido para: {target_path}")
            
            # Atualiza o caminho do modelo e recarrega
            self.model_path = str(target_path)
            self.model = YOLO(str(target_path))
        except Exception as move_error:
            print(f"⚠️  Aviso: Não foi possível mover o modelo: {move_error}")
    
    def run_inference_on_image(self, image_path: str, conf: float = 0.5) -> Dict[str, Any]:
        """
        Executa inferência em uma única imagem.
        
        Args:
            image_path (str): Caminho para a imagem
            conf (float): Threshold de confiança
            
        Returns:
            result (Dict): Dicionário com resultados da inferência
        """
        try:
            # Executa inferência
            results = self.model(image_path, conf=conf)
            result = results[0]  # Primeira (única) imagem
            
            # Extrai informações das detecções
            detections = self._extract_detections(result)
            
            # Salva imagem com detecções
            output_path = self.save_annotated_image(result, image_path)
            
            return self._build_success_response(image_path, output_path, detections, conf)
            
        except Exception as e:
            print(f"❌ Erro na inferência de {image_path}: {e}")
            return self._build_error_response(image_path, str(e))
    
    def _extract_detections(self, result) -> List[Dict[str, Any]]:
        """
        Extrai informações das detecções do resultado YoloV8.
        
        Args:
            result: Resultado da inferência YoloV8
            
        Returns:
            detections (List): Lista de detecções extraídas
        """
        detections = []
        
        if result.boxes is None:
            return detections
        
        for i, box in enumerate(result.boxes):
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': result.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                'bbox_normalized': box.xywhn[0].tolist()  # [x_center, y_center, width, height] normalized
            }
            detections.append(detection)
        
        return detections
    
    def _build_success_response(self, image_path: str, output_path: str, 
                               detections: List[Dict], conf: float) -> Dict[str, Any]:
        """
        Constrói resposta de sucesso da inferência.
        
        Args:
            image_path (str): Caminho da imagem original
            output_path (str): Caminho da imagem anotada
            detections (List): Lista de detecções
            conf (float): Threshold de confiança usado
            
        Returns:
            response (Dict): Dicionário com resposta de sucesso
        """
        return {
            'image_path': image_path,
            'output_path': output_path,
            'detections_count': len(detections),
            'detections': detections,
            'model_used': self.model_path,
            'confidence_threshold': conf
        }
    
    def _build_error_response(self, image_path: str, error_message: str) -> Dict[str, Any]:
        """
        Constrói resposta de erro da inferência.
        
        Args:
            image_path (str): Caminho da imagem que causou erro
            error_message (str): Mensagem de erro
            
        Returns:
            response (Dict): Dicionário com resposta de erro
        """
        return {
            'image_path': image_path,
            'error': error_message,
            'detections_count': 0,
            'detections': []
        }
    
    def save_annotated_image(self, result, original_path: str) -> str:
        """
        Salva imagem com anotações das detecções.
        
        Args:
            result: Resultado da inferência YoloV8
            original_path (str): Caminho da imagem original
            
        Returns:
            output_path (str): Caminho da imagem anotada salva
        """
        try:
            # Cria imagem anotada
            annotated_img = result.plot()
            
            # Define caminho de saída
            original_name = Path(original_path).stem
            output_path = self.output_dir / f"{original_name}_detected.jpg"
            
            # Salva imagem
            cv2.imwrite(str(output_path), annotated_img)
            
            return str(output_path)
            
        except Exception as e:
            print(f"⚠️  Erro ao salvar imagem anotada: {e}")
            return ""
    
    def run_inference_on_folder(self, folder_path: str, conf: float = 0.5) -> List[Dict[str, Any]]:
        """
        Executa inferência em todas as imagens de uma pasta.
        
        Args:
            folder_path (str): Caminho para a pasta com imagens
            conf (float): Threshold de confiança
            
        Returns:
            results (List): Lista com resultados de todas as imagens
        """
        # Busca todas as imagens na pasta
        image_files = self._find_image_files(folder_path)
        
        if not image_files:
            print(f"⚠️  Nenhuma imagem encontrada em: {folder_path}")
            return []
        
        print(f"🖼️  Encontradas {len(image_files)} imagens para processar")
        
        # Processa cada imagem
        return self._process_all_images(image_files, conf)
    
    def _find_image_files(self, folder_path: str) -> List[str]:
        """
        Encontra todos os arquivos de imagem na pasta.
        
        Args:
            folder_path (str): Caminho da pasta
            
        Returns:
            image_files (List): Lista de caminhos de imagens encontradas
        """
        # Usa extensões da configuração
        image_extensions = [f"*{ext}" for ext in config.SUPPORTED_IMAGE_EXTENSIONS]
        
        # Encontra todas as imagens (usando set para evitar duplicatas)
        image_files = set()
        for ext in image_extensions:
            image_files.update(glob.glob(os.path.join(folder_path, ext)))
            image_files.update(glob.glob(os.path.join(folder_path, ext.upper())))
        
        # Converte para lista e ordena
        return sorted(list(image_files))
    
    def _process_all_images(self, image_files: List[str], conf: float) -> List[Dict[str, Any]]:
        """
        Processa lista de imagens com inferência.
        
        Args:
            image_files (List): Lista de caminhos de imagens
            conf (float): Threshold de confiança
            
        Returns:
            results (List): Lista com resultados de todas as imagens
        """
        all_results = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"📷 Processando ({i}/{len(image_files)}): {os.path.basename(image_path)}")
            result = self.run_inference_on_image(image_path, conf)
            all_results.append(result)
        
        return all_results
