"""
Exemplo Avançado - YoloV8 Inference
===================================

Demonstra funcionalidades avançadas do sistema de inferência YoloV8.
"""

import os
import json
import time
import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Importar as classes de serviços necessárias para o YoloV8 Inference
from controller.inference_controller import InferenceController
from controller.benchmark_controller import BenchmarkController

# Importar as classes de utilitários para processamento de imagens
from utils.file_utils import FileUtils
from utils.performance_utils import PerformanceUtils
from utils.image_processor import ImageProcessor
from utils.model_selector import ModelSelector

# Importar serviços de relatórios e configurações
from services.report_service import ReportService
from config.settings import config

# Obtém configurações do arquivo .env ou usa valores padrão
MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'models/pretrained/yolov8n.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
INPUT_DIR = os.getenv('INPUT_IMAGE_DIR', 'img/inference_data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')

# Criar um cabeçalho de resposta padrão
response_header = {
    "status": "success",
    "timestamp": None,
    "model_info": None
}

# ----------------------------------------------------------------------------
# Função principal para executar inferência YoloV8
# ----------------------------------------------------------------------------
def yolo_inference_handler(event, context):
    
    # 1 - Imprime o evento recebido
    print('🚀 YoloV8 Inference Started')
    
    # 2 - Instancia a classe InferenceController
    inference_service = InferenceController()

    # 3 - Obtém os valores do evento
    model_name = event.get("model_name", MODEL_PATH)
    confidence = event.get("confidence", CONFIDENCE_THRESHOLD)
    analysis_type = event.get("analysis_type", "all_images")
    
    print(f'📋 Model: {Path(model_name).name} | Confidence: {confidence} | Type: {analysis_type}')

    # 4 - Lista todas as imagens disponíveis no diretório
    available_images = FileUtils.find_images_in_folder(INPUT_DIR)
    print(f'📁 Found {len(available_images)} images in directory')
    
    if not available_images:
        print('❌ ERROR: No images found in directory')
        return {
            'statusCode': 404,
            'headers': response_header,
            'body': json.dumps('Nenhuma imagem encontrada no diretório.')
        }

    # 5 - Atualiza o modelo se especificado
    if model_name != MODEL_PATH:
        inference_service = InferenceController(model_name)

    # 6 - Executa a inferência em todas as imagens
    try:
        all_results = []
        total_detections = 0
        
        for i, image_path in enumerate(available_images, 1):
            image_name = Path(image_path).name
            print(f'🖼️  Processing ({i}/{len(available_images)}): {image_name}')
            
            # Executa a inferência baseada no tipo de análise
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
            
            # Adiciona informações da imagem
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

        # 7 - Salva os resultados automaticamente
        output_file = Path(OUTPUT_DIR) / "auto_inference_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'processed_images': len(available_images),
                    'total_detections': total_detections,
                    'model_used': model_name,
                    'confidence_threshold': confidence,
                    'analysis_type': analysis_type,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'detailed_results': all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f'💾 Results saved to: {output_file}')

        # 8 - Prepara a resposta final
        response_data = {
            'processed_images': len(available_images),
            'total_detections': total_detections,
            'results': all_results,
            'saved_to': str(output_file),
            'processing_details': {
                'model_used': model_name,
                'confidence_threshold': confidence,
                'analysis_type': analysis_type,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
            
        # 9 - Retorna os resultados da inferência
        return {
            'statusCode': 200,
            'headers': response_header,
            'body': response_data
        }

    # 10 - Caso ocorra algum erro durante a inferência
    except Exception as e:
        print(f'❌ ERROR: {str(e)}')
        return {
            'statusCode': 500,
            'headers': response_header,
            'body': json.dumps(f'Erro durante a inferência: {str(e)}')
        }


# ----------------------------------------------------------------------------
# Função para executar benchmark de múltiplos modelos
# ----------------------------------------------------------------------------
def yolo_benchmark_handler(event, context):
    
    # 1 - Imprime o evento recebido
    print('⚡ YoloV8 Benchmark Started')
    
    # 2 - Instancia a classe BenchmarkController
    benchmark_service = BenchmarkController()

    # 3 - Obtém os valores do evento
    test_image = event.get("test_image", None)
    
    # 4 - Se não especificada, usa a primeira imagem disponível
    if test_image is None:
        available_images = FileUtils.find_images_in_folder(INPUT_DIR)
        if not available_images:
            return {
                'statusCode': 404,
                'headers': response_header,
                'body': json.dumps('Nenhuma imagem de teste encontrada.')
            }
        test_image = available_images[0]
        print(f'📷 Using default test image: {Path(test_image).name}')
    
    # 5 - Executa o benchmark
    try:
        benchmark_results = benchmark_service.run_multi_model_benchmark(test_image)
        
        # Resumo dos resultados
        models_tested = len(benchmark_results.get('model_results', {}))
        print(f'✅ SUCCESS: {models_tested} models benchmarked')
        
        # 6 - Salva os resultados
        output_file = Path(OUTPUT_DIR) / "benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        
        # 7 - Retorna os resultados do benchmark
        return {
            'statusCode': 200,
            'headers': response_header,
            'body': {
                'benchmark_results': benchmark_results,
                'saved_to': str(output_file),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    except Exception as e:
        print(f'❌ ERROR: {str(e)}')
        return {
            'statusCode': 500,
            'headers': response_header,
            'body': json.dumps(f'Erro durante benchmark: {str(e)}')
        }


# ----------------------------------------------------------------------------
# Função para processar múltiplas imagens em lote
# ----------------------------------------------------------------------------
def yolo_batch_handler(event, context):
    
    # 1 - Imprime o evento recebido
    print('📁 YoloV8 Batch Processing Started')
    
    # 2 - Instancia a classe InferenceController
    inference_service = InferenceController()

    # 3 - Obtém os valores do evento
    folder_path = event.get("folder_path", INPUT_DIR)
    model_name = event.get("model_name", MODEL_PATH)
    confidence = event.get("confidence", CONFIDENCE_THRESHOLD)
    save_report = event.get("save_report", True)
    
    print(f'📋 Folder: {folder_path} | Model: {Path(model_name).name} | Confidence: {confidence}')
    
    # 4 - Atualiza o modelo se especificado
    if model_name != MODEL_PATH:
        inference_service = InferenceController(model_name)
    
    # 5 - Executa inferência em lote
    try:
        batch_results = inference_service.run_folder_inference(
            folder_path=folder_path,
            conf=confidence,
            save_report=save_report
        )
        
        # Resumo dos resultados
        results = batch_results.get('results', [])
        summary = batch_results.get('summary', {})
        total_images = summary.get('total_images_processed', 0)
        total_detections = summary.get('total_detections', 0)
        
        print(f'✅ SUCCESS: {total_images} images processed, {total_detections} detections found')
        
        # 6 - Retorna os resultados do processamento em lote
        return {
            'statusCode': 200,
            'headers': response_header,
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
            'headers': response_header,
            'body': json.dumps(f'Erro durante processamento em lote: {str(e)}')
        }


# ============================================================================
# EXEMPLOS DE USO COM SELEÇÃO INTERATIVA DE MODELO
# ============================================================================

def main():
    """Função principal com seleção interativa de modelo"""
    
    print("\n" + "="*60)
    print("🚀 YOLOV8 INFERENCE SYSTEM - SELEÇÃO INTERATIVA")
    print("="*60)
    
    # Seleção interativa do modelo usando o utilitário
    model_selector = ModelSelector()
    selected_model = model_selector.select_model_interactive()
    
    if not selected_model:
        print("❌ Operação cancelada - nenhum modelo selecionado")
        return
    
    print(f"\n🎯 Usando modelo: {selected_model}")
    print("="*60)
    
    # Exemplo 1: Inferência automática em todas as imagens
    print("\n1️⃣ AUTO IMAGE INFERENCE:")
    response_single = yolo_inference_handler(event={
        "model_name": selected_model,
        "confidence": 0.5,
        "analysis_type": "all_images"
    }, context=None)
    
    print(f"   Status: {response_single['statusCode']}")
    if response_single['statusCode'] == 200:
        images_processed = response_single['body'].get('processed_images', 0)
        total_detections = response_single['body'].get('total_detections', 0)
        print(f"   Result: {images_processed} images processed, {total_detections} total detections")
    
    # Exemplo 2: Benchmark com modelo selecionado
    print("\n2️⃣ MODEL BENCHMARK:")
    response_benchmark = yolo_benchmark_handler(event={
        "models_to_test": [selected_model]  # Usar apenas o modelo selecionado
    }, context=None)
    
    print(f"   Status: {response_benchmark['statusCode']}")
    if response_benchmark['statusCode'] == 200:
        models_count = len(response_benchmark['body']['benchmark_results'].get('model_results', {}))
        print(f"   Result: {models_count} models benchmarked")
    
    # Exemplo 3: Processamento em lote
    print("\n3️⃣ BATCH PROCESSING:")
    response_batch = yolo_batch_handler(event={
        "folder_path": "img/inference_data",
        "model_name": selected_model,
        "confidence": 0.4,
        "save_report": True
    }, context=None)
    
    print(f"   Status: {response_batch['statusCode']}")
    if response_batch['statusCode'] == 200:
        summary = response_batch['body']['batch_results'].get('summary', {})
        images = summary.get('total_images_processed', 0)
        detections = summary.get('total_detections', 0)
        print(f"   Result: {images} images, {detections} total detections")
    
    print("\n" + "="*60)
    print("✅ TODOS OS TESTES CONCLUÍDOS!")
    print(f"🎯 Modelo usado: {Path(selected_model).name}")
    print("📁 Todas as imagens em img/inference_data foram processadas!")
    print("="*60)


if __name__ == "__main__":
    main()