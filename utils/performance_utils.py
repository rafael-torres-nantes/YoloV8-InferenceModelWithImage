"""
Performance Utilities
=====================

Utilitários para medição de performance.
"""

from typing import Dict, Any, List


class PerformanceUtils:
    """Utilitários de performance."""
    
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
