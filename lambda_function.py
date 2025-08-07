#!/usr/bin/env python3
"""
Detecção de Objetos - Otimizado
===============================

Sistema otimizado para detectar objetos com YoloV8 incluindo pré-processamento de imagens.
"""

from utils.model_selector import ModelSelector
from controller.inference_controller import InferenceController
from controller.image_controller import ImageController
from utils.file_utils import FileUtils
from pathlib import Path


def lambda_handler(event, context):
    """Detecção de objetos com otimização de imagens"""
    
    print("🔍 DETECÇÃO DE OBJETOS OTIMIZADA")
    print("="*40)
    
    # Passo 1: Escolher modelo
    print("\n📦 Escolha um modelo:")
    model_selector = ModelSelector()
    selected_model = model_selector.select_model_interactive()
    
    if not selected_model:
        print("❌ Nenhum modelo selecionado. Cancelando processo.")
        return
    
    print(f"\n🎯 Usando modelo: {selected_model}")
    
    # Passo 2: Analisar e otimizar imagens
    print("\n🖼️  Analisando imagens...")
    image_controller = ImageController()
    
    # Encontrar todas as imagens
    image_paths = FileUtils.find_images_in_folder("img/inference_data")
    
    if not image_paths:
        print("❌ Nenhuma imagem encontrada em img/inference_data")
        return
    
    print(f"� Encontradas {len(image_paths)} imagens")
    
    # Validar imagens para YOLO
    valid_images = image_controller.validate_and_filter_images(image_paths)
    
    if not valid_images:
        print("❌ Nenhuma imagem válida para processamento YOLO")
        return
    
    # Analisar propriedades das imagens
    analysis = image_controller.analyze_image_properties(valid_images)
    image_controller.print_analysis_report(analysis)
    
    # Otimizar imagens se necessário
    optimized_images = valid_images
    if analysis['needs_optimization']:
        print(f"\n🔧 Otimizando {len(analysis['needs_optimization'])} imagens...")
        optimized_images = image_controller.optimize_images_for_yolo(valid_images)
        
        # Relatório de otimização
        optimization_report = image_controller.create_optimization_report(valid_images, optimized_images)
        image_controller.print_optimization_report(optimization_report)
    else:
        print("\n✅ Imagens já estão otimizadas!")
    
    # Passo 3: Executar inferência YOLO
    print(f"\n🚀 Executando inferência YOLO...")
    print("="*50)
    
    try:
        controller = InferenceController(selected_model)
        
        # Usar imagens otimizadas se existirem, senão usar originais
        images_to_process = optimized_images if len(optimized_images) == len(valid_images) else valid_images
        
        # Processar cada imagem individualmente para melhor controle
        total_detections = 0
        successful_images = 0
        
        for i, img_path in enumerate(images_to_process, 1):
            print(f"📷 ({i}/{len(images_to_process)}) Processando: {Path(img_path).name}")
            
            result = controller.run_single_image_inference(img_path, conf=0.5)
            
            if 'error' not in result:
                detections = result.get('detections_count', 0)
                total_detections += detections
                successful_images += 1
                print(f"   ✅ {detections} objetos detectados")
            else:
                print(f"   ❌ Erro: {result.get('error', 'Desconhecido')}")
        
        # Passo 4: Mostrar resultados finais
        print(f"\n✅ RESULTADO FINAL:")
        print(f"   📷 Imagens processadas: {successful_images}/{len(images_to_process)}")  
        print(f"   🎯 Total de objetos detectados: {total_detections}")
        print(f"   📁 Resultados salvos em: output/")
        
        if analysis['needs_optimization'] and optimized_images:
            print(f"   � Imagens otimizadas salvas em: img/optimized/")
        
    except Exception as e:
        print(f"❌ Erro durante inferência: {e}")

    print(f"\n✅ Processo concluído!")

if __name__ == "__main__":
    lambda_handler(event={}, context={})
