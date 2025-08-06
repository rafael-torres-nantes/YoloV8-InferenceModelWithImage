"""
YoloV8 Examples Orchestrator
=============================

Orquestra a execução dos exemplos do sistema YoloV8.
"""

from pathlib import Path
from typing import Optional

from utils.model_selector import ModelSelector
from services.yolo_handlers import YoloV8InferenceHandler, YoloV8BenchmarkHandler, YoloV8BatchHandler


class YoloV8ExamplesOrchestrator:
    """Orquestrador dos exemplos YoloV8"""
    
    def __init__(self):
        self.inference_handler = YoloV8InferenceHandler()
        self.benchmark_handler = YoloV8BenchmarkHandler()
        self.batch_handler = YoloV8BatchHandler()
        self.model_selector = ModelSelector()
    
    def run_interactive_examples(self) -> None:
        """
        Executa todos os exemplos com seleção interativa de modelo
        """
        print("\n" + "="*60)
        print("🚀 YOLOV8 INFERENCE SYSTEM - SELEÇÃO INTERATIVA")
        print("="*60)
        
        # Seleção interativa do modelo
        selected_model = self.model_selector.select_model_interactive()
        
        if not selected_model:
            print("❌ Operação cancelada - nenhum modelo selecionado")
            return
        
        print(f"\n🎯 Usando modelo: {selected_model}")
        print("="*60)
        
        # Executar todos os exemplos
        self._run_inference_example(selected_model)
        self._run_benchmark_example(selected_model)
        self._run_batch_example(selected_model)
        
        # Mostrar conclusão
        self._show_completion_summary(selected_model)
    
    def _run_inference_example(self, selected_model: str) -> None:
        """Executa exemplo de inferência automática"""
        print("\n1️⃣ AUTO IMAGE INFERENCE:")
        
        response = self.inference_handler.process_images(event={
            "model_name": selected_model,
            "confidence": 0.5,
            "analysis_type": "all_images"
        })
        
        self._show_response_summary(response, "inference")
    
    def _run_benchmark_example(self, selected_model: str) -> None:
        """Executa exemplo de benchmark"""
        print("\n2️⃣ MODEL BENCHMARK:")
        
        response = self.benchmark_handler.run_benchmark(event={
            "models_to_test": [selected_model]
        })
        
        self._show_response_summary(response, "benchmark")
    
    def _run_batch_example(self, selected_model: str) -> None:
        """Executa exemplo de processamento em lote"""
        print("\n3️⃣ BATCH PROCESSING:")
        
        response = self.batch_handler.process_batch(event={
            "folder_path": "img/inference_data",
            "model_name": selected_model,
            "confidence": 0.4,
            "save_report": True
        })
        
        self._show_response_summary(response, "batch")
    
    def _show_response_summary(self, response: dict, operation_type: str) -> None:
        """Mostra resumo da resposta"""
        print(f"   Status: {response['statusCode']}")
        
        if response['statusCode'] == 200:
            body = response.get('body', {})
            
            if operation_type == "inference":
                images_processed = body.get('processed_images', 0)
                total_detections = body.get('total_detections', 0)
                print(f"   Result: {images_processed} images processed, {total_detections} total detections")
            
            elif operation_type == "benchmark":
                benchmark_results = body.get('benchmark_results', {})
                models_count = len(benchmark_results.get('model_results', {}))
                print(f"   Result: {models_count} models benchmarked")
            
            elif operation_type == "batch":
                batch_results = body.get('batch_results', {})
                summary = batch_results.get('summary', {})
                images = summary.get('total_images_processed', 0)
                detections = summary.get('total_detections', 0)
                print(f"   Result: {images} images, {detections} total detections")
        else:
            print(f"   Error: {response.get('body', 'Unknown error')}")
    
    def _show_completion_summary(self, selected_model: str) -> None:
        """Mostra resumo final"""
        print("\n" + "="*60)
        print("✅ TODOS OS TESTES CONCLUÍDOS!")
        print(f"🎯 Modelo usado: {Path(selected_model).name}")
        print("📁 Todas as imagens em img/inference_data foram processadas!")
        print("="*60)
    
    def run_single_inference(self, model_path: Optional[str] = None, **kwargs) -> dict:
        """
        Executa apenas inferência única
        
        Args:
            model_path: Caminho do modelo (opcional, usa seleção interativa se None)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Resultado da inferência
        """
        if model_path is None:
            model_path = self.model_selector.select_model_interactive()
            if not model_path:
                return {"error": "Nenhum modelo selecionado"}
        
        event = {
            "model_name": model_path,
            "confidence": kwargs.get("confidence", 0.5),
            "analysis_type": kwargs.get("analysis_type", "all_images"),
            **kwargs
        }
        
        return self.inference_handler.process_images(event)
    
    def run_single_benchmark(self, test_image: Optional[str] = None, **kwargs) -> dict:
        """
        Executa apenas benchmark
        
        Args:
            test_image: Imagem de teste (opcional)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Resultado do benchmark
        """
        event = {
            "test_image": test_image,
            **kwargs
        }
        
        return self.benchmark_handler.run_benchmark(event)
    
    def run_single_batch(self, model_path: Optional[str] = None, **kwargs) -> dict:
        """
        Executa apenas processamento em lote
        
        Args:
            model_path: Caminho do modelo (opcional, usa seleção interativa se None)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Resultado do processamento
        """
        if model_path is None:
            model_path = self.model_selector.select_model_interactive()
            if not model_path:
                return {"error": "Nenhum modelo selecionado"}
        
        event = {
            "model_name": model_path,
            "folder_path": kwargs.get("folder_path", "img/inference_data"),
            "confidence": kwargs.get("confidence", 0.5),
            "save_report": kwargs.get("save_report", True),
            **kwargs
        }
        
        return self.batch_handler.process_batch(event)
