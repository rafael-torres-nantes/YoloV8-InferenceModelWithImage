"""
Report Service
==============

Serviço para geração de relatórios de inferência.
"""

from typing import List, Dict, Any


class ReportService:
    """Serviço de geração de relatórios."""
    
    @staticmethod
    def generate_summary_report(results: List[Dict[str, Any]], model_path: str = None) -> Dict[str, Any]:
        """Gera relatório resumido dos resultados."""
        total_images = len(results)
        total_detections = sum(r.get('detections_count', 0) for r in results)
        successful_inferences = len([r for r in results if 'error' not in r])
        
        # Conta classes detectadas
        class_counts = {}
        for result in results:
            for detection in result.get('detections', []):
                class_name = detection.get('class_name', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary = {
            'total_images_processed': total_images,
            'successful_inferences': successful_inferences,
            'failed_inferences': total_images - successful_inferences,
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / max(total_images, 1),
            'classes_detected': class_counts,
            'model_used': model_path or 'unknown'
        }
        
        return summary
