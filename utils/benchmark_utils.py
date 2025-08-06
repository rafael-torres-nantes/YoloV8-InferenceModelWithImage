"""
Benchmark Utilities
==================

Utilitários para benchmark e comparação de modelos.
"""

from typing import Dict, Any, List


class BenchmarkUtils:
    """Utilitários para benchmark."""
    
    @staticmethod
    def compare_models(model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compara resultados de diferentes modelos."""
        if not model_results:
            return {}
        
        comparisons = {
            'fastest_model': None,
            'most_detections': None,
            'smallest_model': None,
            'performance_ranking': []
        }
        
        # Encontrar modelo mais rápido
        fastest_time = float('inf')
        for model_name, results in model_results.items():
            avg_time = results.get('average_inference_time', float('inf'))
            if avg_time < fastest_time:
                fastest_time = avg_time
                comparisons['fastest_model'] = {
                    'model': model_name,
                    'time': avg_time
                }
        
        # Encontrar modelo com mais detecções
        most_detections = 0
        for model_name, results in model_results.items():
            avg_detections = results.get('average_detections_per_image', 0)
            if avg_detections > most_detections:
                most_detections = avg_detections
                comparisons['most_detections'] = {
                    'model': model_name,
                    'detections': avg_detections
                }
        
        # Ranking de performance
        ranking = []
        for model_name, results in model_results.items():
            score = results.get('average_detections_per_image', 0) / max(results.get('average_inference_time', 1), 0.001)
            ranking.append({
                'model': model_name,
                'performance_score': round(score, 2),
                'avg_time': results.get('average_inference_time', 0),
                'avg_detections': results.get('average_detections_per_image', 0)
            })
        
        ranking.sort(key=lambda x: x['performance_score'], reverse=True)
        comparisons['performance_ranking'] = ranking
        
        return comparisons
    
    @staticmethod
    def print_benchmark_comparison(comparisons: Dict[str, Any], model_results: Dict[str, Any]):
        """Imprime comparação dos benchmarks."""
        print("\n📊 COMPARATIVO DE MODELOS:")
        
        if comparisons.get('fastest_model'):
            fastest = comparisons['fastest_model']
            print(f"🚀 Mais rápido: {fastest['model']} ({fastest['time']:.2f}s)")
        
        if comparisons.get('most_detections'):
            most_det = comparisons['most_detections']
            print(f"🎯 Mais detecções: {most_det['model']} ({most_det['detections']:.1f})")
        
        # Mostrar tamanhos de arquivo se disponível
        model_sizes = {}
        for model_name in model_results.keys():
            if 'n.pt' in model_name:
                model_sizes[model_name] = "6.2MB"
            elif 's.pt' in model_name:
                model_sizes[model_name] = "21.5MB"
            elif 'm.pt' in model_name:
                model_sizes[model_name] = "49.7MB"
            elif 'l.pt' in model_name:
                model_sizes[model_name] = "83.7MB"
            elif 'x.pt' in model_name:
                model_sizes[model_name] = "136.7MB"
        
        if model_sizes:
            smallest = min(model_sizes.items(), key=lambda x: float(x[1].replace('MB', '')))
            print(f"💾 Menor tamanho: {smallest[0]} ({smallest[1]})")
        
        # Ranking de performance
        if comparisons.get('performance_ranking'):
            print(f"\n🏆 RANKING DE PERFORMANCE:")
            for i, model in enumerate(comparisons['performance_ranking'][:3], 1):
                print(f"   {i}º {model['model']} (Score: {model['performance_score']})")
