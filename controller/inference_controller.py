"""
YoloV8 Inference Controller
===========================

Controller principal para orquestrar inferência YoloV8.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

from services.inference_service import YoloV8InferenceService
from services.report_service import ReportService
from utils.file_utils import FileUtils
from utils.performance_utils import PerformanceUtils
from config.settings import config


class InferenceController:
    """Controller principal para inferência YoloV8."""
    
    def __init__(self, model_path: str = None):
        """
        Inicializa o controller.
        
        Args:
            model_path: Caminho do modelo a usar
        """
        self.inference_service = YoloV8InferenceService(model_path)
        self.report_service = ReportService()
        
    def run_single_image_inference(self, image_path: str, conf: float = 0.5) -> Dict[str, Any]:
        """
        Executa inferência em uma única imagem.
        
        Args:
            image_path: Caminho da imagem
            conf: Threshold de confiança
            
        Returns:
            Resultado da inferência
        """
        return self.inference_service.run_inference_on_image(image_path, conf)
    
    def run_folder_inference(self, folder_path: str, conf: float = 0.5, 
                           save_report: bool = True) -> Dict[str, Any]:
        """
        Executa inferência em uma pasta de imagens.
        
        Args:
            folder_path: Caminho da pasta
            conf: Threshold de confiança
            save_report: Se deve salvar relatório
            
        Returns:
            Dicionário com resultados e métricas
        """
        # Executa inferência com medição de tempo
        start_time = time.time()
        results = self.inference_service.run_inference_on_folder(folder_path, conf)
        total_time = time.time() - start_time
        
        # Gera relatório resumido
        summary = self.report_service.generate_summary_report(results, self.inference_service.model_path)
        
        # Calcula métricas de performance
        performance_metrics = PerformanceUtils.calculate_performance_metrics(results, total_time)
        
        # Salva relatório se solicitado
        if save_report and results:
            self._save_detailed_report(results, summary, performance_metrics)
        
        return {
            'results': results,
            'summary': summary,
            'performance': performance_metrics,
            'model_used': self.inference_service.model_path
        }
    
    def run_benchmark(self, test_image_path: str, runs: int = 3) -> Dict[str, Any]:
        """
        Executa benchmark do modelo atual.
        
        Args:
            test_image_path: Caminho da imagem de teste
            runs: Número de execuções para média
            
        Returns:
            Resultados do benchmark
        """
        print(f"🤖 Benchmark do modelo: {Path(self.inference_service.model_path).name}")
        
        # Mede tempo de carregamento (já foi carregado, mas simula)
        load_time = 0.0  # Modelo já carregado
        
        # Executa múltiplas inferências
        inference_times = []
        total_detections = 0
        
        for i in range(runs):
            start_time = time.time()
            result = self.inference_service.run_inference_on_image(test_image_path, conf=0.5)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            total_detections += result.get('detections_count', 0)
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        avg_detections = total_detections / runs
        model_size = FileUtils.get_file_size_mb(self.inference_service.model_path)
        
        benchmark_results = {
            'model_name': Path(self.inference_service.model_path).name,
            'load_time': load_time,
            'avg_inference_time': avg_inference_time,
            'avg_detections': avg_detections,
            'model_size': model_size,
            'runs': runs
        }
        
        # Imprime resultados
        print(f"   ⏱️  Carregamento: {load_time:.2f}s")
        print(f"   🚀 Inferência média: {avg_inference_time:.2f}s")
        print(f"   🎯 Detecções médias: {avg_detections:.1f}")
        print(f"   💾 Tamanho: {model_size:.1f}MB")
        
        return benchmark_results
    
    def _save_detailed_report(self, results: List[Dict[str, Any]], 
                            summary: Dict[str, Any], 
                            performance: Dict[str, Any]):
        """Salva relatório detalhado em arquivo JSON."""
        report_data = {
            'summary': summary,
            'performance': performance,
            'detailed_results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_file = config.OUTPUT_DIR / "inference_report.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Relatório salvo em: {output_file}")
