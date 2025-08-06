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
    
    @staticmethod
    def print_summary(summary: Dict[str, Any]):
        """Imprime resumo formatado no console."""
        print(f"\n📊 RESUMO DOS RESULTADOS:")
        print(f"   📷 Imagens processadas: {summary['total_images_processed']}")
        print(f"   ✅ Sucessos: {summary['successful_inferences']}")
        print(f"   ❌ Falhas: {summary['failed_inferences']}")
        print(f"   🎯 Total de detecções: {summary['total_detections']}")
        print(f"   📈 Média por imagem: {summary['average_detections_per_image']:.2f}")
        
        if summary['classes_detected']:
            print(f"\n🏷️  Classes detectadas:")
            for class_name, count in summary['classes_detected'].items():
                print(f"      - {class_name}: {count}")
    
    @staticmethod
    def print_detailed_results(results: List[Dict[str, Any]], max_examples: int = 3):
        """Imprime exemplos detalhados de detecções."""
        successful_results = [r for r in results if 'error' not in r and r['detections_count'] > 0]
        
        if successful_results:
            print(f"\n🔍 Exemplos de detecções:")
            for result in successful_results[:max_examples]:
                image_name = result['image_path'].split('/')[-1].split('\\')[-1]
                print(f"   📷 {image_name}: {result['detections_count']} objetos")
                for detection in result['detections'][:3]:  # Mostra até 3 detecções por imagem
                    conf = detection['confidence']
                    class_name = detection['class_name']
                    print(f"      🎯 {class_name} ({conf:.2f})")
