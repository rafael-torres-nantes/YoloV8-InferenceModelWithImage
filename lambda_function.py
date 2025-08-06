#!/usr/bin/env python3
"""
Detecção de Objetos - Ultra Simples
==================================

O script mais simples possível para detectar objetos com YoloV8.
Apenas 3 passos: Selecionar modelo → Processar imagens → Ver resultados
"""

from utils.model_selector import ModelSelector
from controller.inference_controller import InferenceController


def lambda_handler(event, context):
    """Detecção de objetos ultra simples"""
    
    print("🔍 DETECÇÃO DE OBJETOS")
    print("="*30)
    
    # Passo 1: Escolher modelo
    print("\n📦 Escolha um modelo:")
    model_selector = ModelSelector()
    selected_model = model_selector.select_model_interactive()
    
    if not selected_model:
        print("❌ Nenhum modelo selecionado. Cancelando testes.")
        return
    
    print(f"\n🎯 Usando modelo: {selected_model}")
    print("="*50)
    
    # Passo 2: Detectar objetos
    print("\n🚀 Detectando objetos...")
    try:
        controller = InferenceController(selected_model)
        results = controller.run_folder_inference("img/inference_data", conf=0.5)
        
        # Passo 3: Mostrar resultados
        summary = results.get('summary', {})
        total_images = summary.get('total_images_processed', 0)
        total_detections = summary.get('total_detections', 0)
        classes = summary.get('classes_detected', {})
        
        print(f"\n✅ RESULTADO:")
        print(f"   📷 {total_images} imagens")  
        print(f"   🎯 {total_detections} objetos")
        
        if classes:
            print(f"   🏷️  Encontrados:")
            for obj_type, count in classes.items():
                print(f"      • {count}x {obj_type}")
        
        print(f"   📁 Salvos em: output/")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

    print("\n✅ Todos os testes individuais concluídos!")

if __name__ == "__main__":
    lambda_handler(event={}, context={})
