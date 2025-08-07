"""
Image Controller
================

Controlador especializado para processamento e otimização de imagens para modelos YOLO.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageOps
import time


class ImageController:
    """Controlador para processamento e otimização de imagens para YOLO."""
    
    def __init__(self, target_size: int = 640, quality: int = 95):
        """
        Inicializa o controlador de imagens.
        
        Args:
            target_size: Tamanho alvo para redimensionamento (YOLO usa 640x640)
            quality: Qualidade de compressão para salvamento (1-100)
        """
        self.target_size = target_size
        self.quality = quality
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def validate_and_filter_images(self, image_paths: List[str]) -> List[str]:
        """
        Valida e filtra imagens válidas para processamento YOLO.
        
        Args:
            image_paths: Lista de caminhos de imagens
            
        Returns:
            Lista de imagens válidas filtradas
        """
        valid_images = []
        
        print("🔍 Validando imagens para YOLO...")
        
        for img_path in image_paths:
            if self._is_valid_for_yolo(img_path):
                valid_images.append(img_path)
            else:
                print(f"⚠️  Ignorando: {Path(img_path).name}")
        
        print(f"✅ {len(valid_images)} imagens válidas de {len(image_paths)} total")
        return valid_images
    
    def optimize_images_for_yolo(self, image_paths: List[str], 
                                output_dir: str = "img/optimized") -> List[str]:
        """
        Otimiza imagens para melhor performance com YOLO.
        
        Args:
            image_paths: Lista de caminhos das imagens originais
            output_dir: Diretório para salvar imagens otimizadas
            
        Returns:
            Lista de caminhos das imagens otimizadas
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        optimized_paths = []
        
        print(f"🖼️  Otimizando {len(image_paths)} imagens para YOLO...")
        
        for i, img_path in enumerate(image_paths, 1):
            try:
                print(f"   📷 ({i}/{len(image_paths)}) {Path(img_path).name}")
                
                optimized_path = self._optimize_single_image(img_path, output_path)
                if optimized_path:
                    optimized_paths.append(optimized_path)
                    
            except Exception as e:
                print(f"   ❌ Erro: {e}")
        
        print(f"✅ {len(optimized_paths)} imagens otimizadas salvas em: {output_dir}")
        return optimized_paths
    
    def analyze_image_properties(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Analisa propriedades das imagens para otimização.
        
        Args:
            image_paths: Lista de caminhos das imagens
            
        Returns:
            Análise das propriedades das imagens
        """
        analysis = {
            'total_images': len(image_paths),
            'sizes': [],
            'formats': {},
            'resolutions': [],
            'file_sizes_mb': [],
            'needs_optimization': []
        }
        
        for img_path in image_paths:
            try:
                # Análise básica do arquivo
                path_obj = Path(img_path)
                file_size_mb = path_obj.stat().st_size / (1024 * 1024)
                analysis['file_sizes_mb'].append(file_size_mb)
                
                # Formato
                ext = path_obj.suffix.lower()
                analysis['formats'][ext] = analysis['formats'].get(ext, 0) + 1
                
                # Dimensões da imagem
                with Image.open(img_path) as img:
                    width, height = img.size
                    analysis['resolutions'].append((width, height))
                    analysis['sizes'].append(width * height)
                    
                    # Verifica se precisa otimização
                    needs_opt = self._needs_optimization(img_path, width, height, file_size_mb)
                    if needs_opt:
                        analysis['needs_optimization'].append({
                            'path': img_path,
                            'reason': needs_opt,
                            'size': f"{width}x{height}",
                            'file_size_mb': round(file_size_mb, 2)
                        })
                        
            except Exception as e:
                print(f"⚠️  Erro ao analisar {img_path}: {e}")
        
        # Estatísticas
        if analysis['file_sizes_mb']:
            analysis['avg_file_size_mb'] = sum(analysis['file_sizes_mb']) / len(analysis['file_sizes_mb'])
            analysis['max_file_size_mb'] = max(analysis['file_sizes_mb'])
            
        if analysis['sizes']:
            analysis['avg_resolution'] = int(np.sqrt(sum(analysis['sizes']) / len(analysis['sizes'])))
            
        return analysis
    
    def batch_resize_for_yolo(self, image_paths: List[str], 
                             output_dir: str = "img/resized") -> List[str]:
        """
        Redimensiona imagens em lote mantendo aspect ratio.
        
        Args:
            image_paths: Lista de caminhos das imagens
            output_dir: Diretório de saída
            
        Returns:
            Lista de caminhos das imagens redimensionadas
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        resized_paths = []
        
        print(f"📐 Redimensionando {len(image_paths)} imagens...")
        
        for i, img_path in enumerate(image_paths, 1):
            try:
                print(f"   📷 ({i}/{len(image_paths)}) {Path(img_path).name}")
                
                resized_path = self._resize_single_image(img_path, output_path)
                if resized_path:
                    resized_paths.append(resized_path)
                    
            except Exception as e:
                print(f"   ❌ Erro: {e}")
        
        return resized_paths
    
    def create_optimization_report(self, original_paths: List[str], 
                                 optimized_paths: List[str]) -> Dict[str, Any]:
        """
        Cria relatório de otimização das imagens.
        
        Args:
            original_paths: Caminhos das imagens originais
            optimized_paths: Caminhos das imagens otimizadas
            
        Returns:
            Relatório de otimização
        """
        report = {
            'original_count': len(original_paths),
            'optimized_count': len(optimized_paths),
            'space_saved_mb': 0,
            'average_compression_ratio': 0,
            'processing_time': 0,
            'recommendations': []
        }
        
        # Calcula economia de espaço
        original_size = sum(Path(p).stat().st_size for p in original_paths if Path(p).exists())
        optimized_size = sum(Path(p).stat().st_size for p in optimized_paths if Path(p).exists())
        
        report['space_saved_mb'] = (original_size - optimized_size) / (1024 * 1024)
        report['average_compression_ratio'] = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0
        
        # Recomendações
        if report['space_saved_mb'] > 10:
            report['recommendations'].append("✅ Ótima economia de espaço alcançada")
        
        if report['average_compression_ratio'] > 30:
            report['recommendations'].append("⚡ Redução significativa no tamanho dos arquivos")
        
        if len(optimized_paths) == len(original_paths):
            report['recommendations'].append("✅ Todas as imagens foram processadas com sucesso")
        
        return report
    
    def _is_valid_for_yolo(self, image_path: str) -> bool:
        """Verifica se uma imagem é válida para YOLO."""
        try:
            if not os.path.exists(image_path):
                return False
            
            # Verifica extensão
            if Path(image_path).suffix.lower() not in self.valid_extensions:
                return False
            
            # Verifica se pode abrir a imagem
            with Image.open(image_path) as img:
                width, height = img.size
                
                # YOLO requer imagens com pelo menos 32x32 pixels
                if width < 32 or height < 32:
                    return False
                
                # Verifica se não é muito grande (>32MP pode causar problemas)
                if width * height > 32_000_000:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _optimize_single_image(self, image_path: str, output_dir: Path) -> Optional[str]:
        """Otimiza uma única imagem."""
        try:
            with Image.open(image_path) as img:
                # Converte para RGB se necessário
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Corrige orientação baseada em EXIF
                img = ImageOps.exif_transpose(img)
                
                original_size = img.size
                
                # Redimensiona se muito grande
                if max(original_size) > 2000:  # Redimensiona imagens > 2000px
                    img = self._smart_resize(img, max_size=1920)
                
                # Nome do arquivo otimizado
                original_name = Path(image_path).stem
                output_path = output_dir / f"{original_name}_optimized.jpg"
                
                # Salva com otimização
                img.save(
                    output_path,
                    'JPEG',
                    quality=self.quality,
                    optimize=True,
                    progressive=True
                )
                
                return str(output_path)
                
        except Exception as e:
            print(f"      ❌ Erro ao otimizar: {e}")
            return None
    
    def _resize_single_image(self, image_path: str, output_dir: Path) -> Optional[str]:
        """Redimensiona uma única imagem mantendo aspect ratio."""
        try:
            with Image.open(image_path) as img:
                # Converte para RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensiona mantendo proporção
                img = self._smart_resize(img, max_size=self.target_size)
                
                # Nome do arquivo redimensionado
                original_name = Path(image_path).stem
                output_path = output_dir / f"{original_name}_resized.jpg"
                
                # Salva
                img.save(output_path, 'JPEG', quality=90)
                
                return str(output_path)
                
        except Exception as e:
            print(f"      ❌ Erro ao redimensionar: {e}")
            return None
    
    def _smart_resize(self, img: Image.Image, max_size: int) -> Image.Image:
        """Redimensiona imagem mantendo aspect ratio."""
        width, height = img.size
        
        # Calcula o fator de escala
        scale = min(max_size / width, max_size / height)
        
        if scale < 1:  # Só redimensiona se for reduzir
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
    
    def _needs_optimization(self, image_path: str, width: int, height: int, 
                           file_size_mb: float) -> Optional[str]:
        """Verifica se uma imagem precisa de otimização."""
        reasons = []
        
        # Tamanho de arquivo muito grande
        if file_size_mb > 5:
            reasons.append(f"arquivo grande ({file_size_mb:.1f}MB)")
        
        # Resolução muito alta
        if width > 2000 or height > 2000:
            reasons.append(f"alta resolução ({width}x{height})")
        
        # Formato não otimizado
        ext = Path(image_path).suffix.lower()
        if ext in ['.bmp', '.tiff']:
            reasons.append(f"formato não otimizado ({ext})")
        
        return ", ".join(reasons) if reasons else None
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Imprime relatório de análise das imagens."""
        print("\n📊 ANÁLISE DAS IMAGENS")
        print("=" * 40)
        
        print(f"📷 Total de imagens: {analysis['total_images']}")
        
        if analysis.get('avg_file_size_mb'):
            print(f"💾 Tamanho médio: {analysis['avg_file_size_mb']:.1f}MB")
            print(f"💾 Maior arquivo: {analysis['max_file_size_mb']:.1f}MB")
        
        if analysis.get('avg_resolution'):
            print(f"📐 Resolução média: ~{analysis['avg_resolution']}px")
        
        if analysis['formats']:
            print(f"\n📁 Formatos encontrados:")
            for ext, count in analysis['formats'].items():
                print(f"   {ext}: {count} arquivo(s)")
        
        if analysis['needs_optimization']:
            print(f"\n⚠️  Imagens que precisam otimização ({len(analysis['needs_optimization'])}):")
            for img_info in analysis['needs_optimization'][:5]:  # Mostra apenas as 5 primeiras
                name = Path(img_info['path']).name
                print(f"   📷 {name}: {img_info['reason']}")
            
            if len(analysis['needs_optimization']) > 5:
                remaining = len(analysis['needs_optimization']) - 5
                print(f"   ... e mais {remaining} imagem(s)")
        else:
            print("\n✅ Todas as imagens já estão otimizadas!")
    
    def print_optimization_report(self, report: Dict[str, Any]):
        """Imprime relatório de otimização."""
        print("\n📊 RELATÓRIO DE OTIMIZAÇÃO")
        print("=" * 40)
        
        print(f"📷 Imagens processadas: {report['optimized_count']}/{report['original_count']}")
        
        if report['space_saved_mb'] > 0:
            print(f"💾 Espaço economizado: {report['space_saved_mb']:.1f}MB")
            print(f"📊 Taxa de compressão: {report['average_compression_ratio']:.1f}%")
        
        if report['recommendations']:
            print(f"\n💡 Recomendações:")
            for rec in report['recommendations']:
                print(f"   {rec}")
