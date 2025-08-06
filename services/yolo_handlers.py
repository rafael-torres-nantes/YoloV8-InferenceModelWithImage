"""
YoloV8 Handlers
===============

Handlers principais para as funcionalidades do sistema YoloV8.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from controller.inference_controller import InferenceController
from controller.benchmark_controller import BenchmarkController
from utils.file_utils import FileUtils
from utils.image_processor import ImageProcessor


class YoloV8InferenceHandler:
    """Handler para inferência YoloV8"""
    
    def __init__(self):
        self.response_header = {
            "status": "success",
            "timestamp": None,
            "model_info": None
        }
    
    def process_images(self, event: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Processa imagens com inferência YoloV8
        
        Args:
            event: Parâmetros de configuração
            context: Contexto de execução (opcional)
            
        Returns:
            Resultado do processamento
        """
        print('🚀 YoloV8 Inference Started')
        
        # Extrair parâmetros do evento
        model_path = event.get("model_name", "models/pretrained/yolov8n.pt")
        confidence = event.get("confidence", 0.5)
        analysis_type = event.get("analysis_type", "all_images")
        input_dir = event.get("input_dir", "img/inference_data")
        output_dir = event.get("output_dir", "output")
        
        print(f'📋 Model: {Path(model_path).name} | Confidence: {confidence} | Type: {analysis_type}')
        
        try:
            # Inicializar serviço de inferência
            inference_service = InferenceController(model_path)
            
            # Encontrar imagens
            available_images = FileUtils.find_images_in_folder(input_dir)
            print(f'📁 Found {len(available_images)} images in directory')
            
            if not available_images:
                return self._create_error_response(404, 'Nenhuma imagem encontrada no diretório.')
            
            # Processar imagens
            results = self._process_image_batch(
                inference_service, available_images, confidence, analysis_type
            )
            
            # Salvar resultados
            output_file = self._save_results(results, model_path, confidence, analysis_type, output_dir)
            
            return self._create_success_response(results, output_file)
            
        except Exception as e:
            print(f'❌ ERROR: {str(e)}')
            return self._create_error_response(500, f'Erro durante a inferência: {str(e)}')
    
    def _process_image_batch(self, inference_service, available_images, confidence, analysis_type):
        """Processa um lote de imagens"""
        all_results = []
        total_detections = 0
        
        for i, image_path in enumerate(available_images, 1):
            image_name = Path(image_path).name
            print(f'🖼️  Processing ({i}/{len(available_images)}): {image_name}')
            
            # Executar inferência baseada no tipo
            if analysis_type == "threshold_analysis":
                inference_results = inference_service.analyze_image_with_different_thresholds(
                    image_path=image_path,
                    thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]
                )
            elif analysis_type == "benchmark":
                inference_results = inference_service.run_benchmark(
                    test_image_path=image_path,
                    runs=3
                )
            else:  # all_images ou single_image
                inference_results = inference_service.run_single_image_inference(
                    image_path=image_path,
                    conf=confidence
                )
            
            # Adicionar informações da imagem
            image_info = ImageProcessor.get_image_info(image_path)
            result_data = {
                'image_name': image_name,
                'image_path': image_path,
                'inference_results': inference_results,
                'image_info': image_info
            }
            
            all_results.append(result_data)
            detections_count = inference_results.get('detections_count', 0)
            total_detections += detections_count
            print(f'   ✅ {detections_count} detections found')
        
        print(f'🎯 TOTAL: {total_detections} detections across {len(available_images)} images')
        
        return {
            'processed_images': len(available_images),
            'total_detections': total_detections,
            'results': all_results
        }
    
    def _save_results(self, results, model_path, confidence, analysis_type, output_dir):
        """Salva os resultados em arquivo JSON"""
        output_file = Path(output_dir) / "auto_inference_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'processed_images': results['processed_images'],
                    'total_detections': results['total_detections'],
                    'model_used': model_path,
                    'confidence_threshold': confidence,
                    'analysis_type': analysis_type,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'detailed_results': results['results']
            }, f, indent=2, ensure_ascii=False)
        
        print(f'💾 Results saved to: {output_file}')
        return output_file
    
    def _create_success_response(self, results, output_file):
        """Cria resposta de sucesso"""
        return {
            'statusCode': 200,
            'headers': self.response_header,
            'body': {
                'processed_images': results['processed_images'],
                'total_detections': results['total_detections'],
                'results': results['results'],
                'saved_to': str(output_file),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def _create_error_response(self, status_code, message):
        """Cria resposta de erro"""
        return {
            'statusCode': status_code,
            'headers': self.response_header,
            'body': json.dumps(message)
        }


class YoloV8BenchmarkHandler:
    """Handler para benchmark YoloV8"""
    
    def __init__(self):
        self.response_header = {
            "status": "success",
            "timestamp": None,
            "model_info": None
        }
    
    def run_benchmark(self, event: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Executa benchmark de múltiplos modelos
        
        Args:
            event: Parâmetros de configuração
            context: Contexto de execução (opcional)
            
        Returns:
            Resultado do benchmark
        """
        print('⚡ YoloV8 Benchmark Started')
        
        # Extrair parâmetros
        test_image = event.get("test_image", None)
        input_dir = event.get("input_dir", "img/inference_data")
        output_dir = event.get("output_dir", "output")
        
        try:
            # Inicializar serviço de benchmark
            benchmark_service = BenchmarkController()
            
            # Selecionar imagem de teste
            if test_image is None:
                available_images = FileUtils.find_images_in_folder(input_dir)
                if not available_images:
                    return self._create_error_response(404, 'Nenhuma imagem de teste encontrada.')
                test_image = available_images[0]
                print(f'📷 Using default test image: {Path(test_image).name}')
            
            # Executar benchmark
            benchmark_results = benchmark_service.run_multi_model_benchmark(test_image)
            
            # Salvar resultados
            output_file = self._save_benchmark_results(benchmark_results, output_dir)
            
            # Mostrar resumo
            models_tested = len(benchmark_results.get('model_results', {}))
            print(f'✅ SUCCESS: {models_tested} models benchmarked')
            
            return {
                'statusCode': 200,
                'headers': self.response_header,
                'body': {
                    'benchmark_results': benchmark_results,
                    'saved_to': str(output_file),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
        except Exception as e:
            print(f'❌ ERROR: {str(e)}')
            return self._create_error_response(500, f'Erro durante benchmark: {str(e)}')
    
    def _save_benchmark_results(self, benchmark_results, output_dir):
        """Salva resultados do benchmark"""
        output_file = Path(output_dir) / "benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        return output_file
    
    def _create_error_response(self, status_code, message):
        """Cria resposta de erro"""
        return {
            'statusCode': status_code,
            'headers': self.response_header,
            'body': json.dumps(message)
        }


class YoloV8BatchHandler:
    """Handler para processamento em lote"""
    
    def __init__(self):
        self.response_header = {
            "status": "success",
            "timestamp": None,
            "model_info": None
        }
    
    def process_batch(self, event: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Processa múltiplas imagens em lote
        
        Args:
            event: Parâmetros de configuração
            context: Contexto de execução (opcional)
            
        Returns:
            Resultado do processamento em lote
        """
        print('📁 YoloV8 Batch Processing Started')
        
        # Extrair parâmetros
        folder_path = event.get("folder_path", "img/inference_data")
        model_name = event.get("model_name", "models/pretrained/yolov8n.pt")
        confidence = event.get("confidence", 0.5)
        save_report = event.get("save_report", True)
        
        print(f'📋 Folder: {folder_path} | Model: {Path(model_name).name} | Confidence: {confidence}')
        
        try:
            # Inicializar serviço de inferência
            inference_service = InferenceController(model_name)
            
            # Executar inferência em lote
            batch_results = inference_service.run_folder_inference(
                folder_path=folder_path,
                conf=confidence,
                save_report=save_report
            )
            
            # Mostrar resumo
            summary = batch_results.get('summary', {})
            total_images = summary.get('total_images_processed', 0)
            total_detections = summary.get('total_detections', 0)
            
            print(f'✅ SUCCESS: {total_images} images processed, {total_detections} detections found')
            
            return {
                'statusCode': 200,
                'headers': self.response_header,
                'body': {
                    'batch_results': batch_results,
                    'processing_details': {
                        'folder_processed': folder_path,
                        'model_used': model_name,
                        'confidence_threshold': confidence,
                        'report_saved': save_report,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
            }
            
        except Exception as e:
            print(f'❌ ERROR: {str(e)}')
            return {
                'statusCode': 500,
                'headers': self.response_header,
                'body': json.dumps(f'Erro durante processamento em lote: {str(e)}')
            }
