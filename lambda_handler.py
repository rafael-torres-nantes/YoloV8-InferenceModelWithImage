"""
Exemplo de Uso das Funções Individuais
======================================

Demonstra como usar as funções individuais do sistema refatorado.
"""

from services.examples_orchestrator import YoloV8ExamplesOrchestrator
from utils.model_selector import ModelSelector


def test_individual_functions():
    """Testa as funções individuais com seleção de modelo"""
    
    print("🧪 TESTE DAS FUNÇÕES INDIVIDUAIS")
    print("="*50)
    
    # Seleção interativa de modelo
    print("\n🤖 Selecionando modelo para os testes:")
    model_selector = ModelSelector()
    selected_model = model_selector.select_model_interactive()
    
    if not selected_model:
        print("❌ Nenhum modelo selecionado. Cancelando testes.")
        return
    
    print(f"\n🎯 Usando modelo: {selected_model}")
    print("="*50)
    
    # Teste 1: Apenas inferência
    print("\n1️⃣ Testando inferência individual:")
    try:
        orchestrator = YoloV8ExamplesOrchestrator()
        result = orchestrator.run_single_inference(
            model_path=selected_model,
            confidence=0.5
        )
        print(f"   Status: {result.get('statusCode', 'N/A')}")
        if result.get('statusCode') == 200:
            body = result.get('body', {})
            print(f"   Processadas: {body.get('processed_images', 0)} imagens")
            print(f"   Detecções: {body.get('total_detections', 0)}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    # Teste 2: Apenas benchmark
    print("\n2️⃣ Testando benchmark individual:")
    try:
        result = orchestrator.run_single_benchmark()
        print(f"   Status: {result.get('statusCode', 'N/A')}")
        if result.get('statusCode') == 200:
            body = result.get('body', {})
            benchmark_results = body.get('benchmark_results', {})
            models_count = len(benchmark_results.get('model_results', {}))
            print(f"   Modelos testados: {models_count}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    # Teste 3: Apenas processamento em lote
    print("\n3️⃣ Testando processamento em lote:")
    try:
        result = orchestrator.run_single_batch(
            model_path=selected_model,
            confidence=0.4
        )
        print(f"   Status: {result.get('statusCode', 'N/A')}")
        if result.get('statusCode') == 200:
            body = result.get('body', {})
            batch_results = body.get('batch_results', {})
            summary = batch_results.get('summary', {})
            images = summary.get('total_images_processed', 0)
            detections = summary.get('total_detections', 0)
            print(f"   Processadas: {images} imagens")
            print(f"   Detecções: {detections}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    print("\n✅ Todos os testes individuais concluídos!")


def show_modular_structure():
    """Mostra a nova estrutura modular"""
    
    print("\n📁 NOVA ESTRUTURA MODULAR")
    print("="*50)
    print("advanced_example.py          ← Ponto de entrada limpo")
    print("├─ services/")
    print("│  ├─ yolo_handlers.py       ← Handlers principais")
    print("│  └─ examples_orchestrator.py ← Orquestração")
    print("├─ config/")
    print("│  └─ app_config.py          ← Configuração centralizada")
    print("└─ utils/")
    print("   └─ model_selector.py      ← Seleção de modelos")
    
    print("\n🎯 VANTAGENS:")
    print("✅ Código limpo e organizados")
    print("✅ Responsabilidades separadas")
    print("✅ Fácil manutenção")
    print("✅ Funções reutilizáveis")
    print("✅ Configuração centralizada")


if __name__ == "__main__":
    show_modular_structure()
    test_individual_functions()
