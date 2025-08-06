"""
Performance Utilities
=====================

Utilitários para medição de performance e benchmark.
"""

import time
from typing import Dict, Any, List
from pathlib import Path


class PerformanceUtils:
    """Utilitários de performance."""
    
    @staticmethod
    def time_function(func, *args, **kwargs) -> tuple:
        """
        Mede o tempo de execução de uma função.
        
        Args:
            func: Função a ser executada
            *args: Argumentos posicionais
            **kwargs: Argumentos nomeados
            
        Returns:
            Tupla (resultado, tempo_execucao)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def calculate_performance_metrics(results: List[Dict[str, Any]], total_time: float) -> Dict[str, float]:
        """
        Calcula métricas de performance.
        
        Args:
            results: Lista de resultados de inferência
            total_time: Tempo total de processamento
            
        Returns:
            Dicionário com métricas de performance
        """
        total_images = len(results)
        total_detections = sum(r.get('detections_count', 0) for r in results)
        
        metrics = {
            'total_time': total_time,
            'images_per_second': total_images / total_time if total_time > 0 else 0,
            'detections_per_second': total_detections / total_time if total_time > 0 else 0,
            'average_time_per_image': total_time / total_images if total_images > 0 else 0,
            'total_images': total_images,
            'total_detections': total_detections
        }
        
        return metrics
    
    @staticmethod
    def print_performance_metrics(metrics: Dict[str, float]):
        """Imprime métricas de performance formatadas."""
        print(f"\n⚡ MÉTRICAS DE PERFORMANCE:")
        print(f"   ⏱️  Tempo total: {metrics['total_time']:.1f}s")
        print(f"   📷 Imagens/segundo: {metrics['images_per_second']:.1f}")
        print(f"   🎯 Detecções/segundo: {metrics['detections_per_second']:.1f}")
        print(f"   📈 Tempo médio/imagem: {metrics['average_time_per_image']:.2f}s")


class BenchmarkUtils:
    """Utilitários para benchmark de modelos."""
    
    @staticmethod
    def compare_models(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Compara resultados entre diferentes modelos.
        
        Args:
            model_results: Dicionário {modelo: resultados}
            
        Returns:
            Dicionário com comparações
        """
        if not model_results:
            return {}
        
        # Encontra o melhor em cada categoria
        fastest_model = min(model_results.keys(), 
                          key=lambda x: model_results[x].get('avg_inference_time', float('inf')))
        
        most_accurate = max(model_results.keys(), 
                          key=lambda x: model_results[x].get('avg_detections', 0))
        
        smallest_model = min(model_results.keys(), 
                           key=lambda x: model_results[x].get('model_size', float('inf')))
        
        return {
            'fastest': fastest_model,
            'most_accurate': most_accurate,
            'smallest': smallest_model
        }
    
    @staticmethod
    def print_benchmark_comparison(comparisons: Dict[str, str], model_results: Dict[str, Dict[str, Any]]):
        """Imprime comparação de benchmark."""
        print(f"\n📊 COMPARATIVO DE MODELOS:")
        print("=" * 50)
        
        if 'fastest' in comparisons:
            fastest = comparisons['fastest']
            time_val = model_results[fastest]['avg_inference_time']
            print(f"🚀 Mais rápido: {fastest} ({time_val:.2f}s)")
        
        if 'most_accurate' in comparisons:
            accurate = comparisons['most_accurate']
            detections = model_results[accurate]['avg_detections']
            print(f"🎯 Mais detecções: {accurate} ({detections:.1f})")
        
        if 'smallest' in comparisons:
            smallest = comparisons['smallest']
            size = model_results[smallest]['model_size']
            print(f"💾 Menor tamanho: {smallest} ({size:.1f}MB)")
