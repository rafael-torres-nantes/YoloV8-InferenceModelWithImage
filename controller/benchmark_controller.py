"""
YoloV8 Benchmark Controller
===========================

Controller para benchmark de múltiplos modelos YoloV8.
"""

from typing import Dict, Any

from utils.file_utils import FileUtils
from utils.benchmark_utils import BenchmarkUtils


class BenchmarkController:
    """Controller para benchmark de múltiplos modelos."""
    
    def __init__(self):
        """Inicializa o controller de benchmark."""
        self.available_models = FileUtils.get_available_models()
    
    def run_multi_model_benchmark(self, test_image_path: str) -> Dict[str, Any]:
        """
        Executa benchmark de múltiplos modelos.
        
        Args:
            test_image_path: Caminho da imagem de teste
            
        Returns:
            Resultados comparativos
        """
        # Importação local para evitar importação circular
        from .inference_controller import InferenceController
        
        if not self.available_models:
            print("⚠️  Nenhum modelo local encontrado.")
            return {}
        
        print("⚡ Benchmark de Múltiplos Modelos")
        print("=" * 50)
        
        model_results = {}
        
        for model_path in self.available_models:
            try:
                controller = InferenceController(model_path)
                benchmark_result = controller.run_benchmark(test_image_path)
                model_results[benchmark_result['model_name']] = benchmark_result
                
            except Exception as e:
                print(f"   ❌ Erro com modelo {model_path}: {e}")
        
        # Compara resultados
        if model_results:
            comparisons = BenchmarkUtils.compare_models(model_results)
            BenchmarkUtils.print_benchmark_comparison(comparisons, model_results)
        
        return {
            'model_results': model_results,
            'comparisons': comparisons if model_results else {}
        }
