"""
YoloV8 Application Configuration
================================

Configuração centralizada da aplicação YoloV8.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()


class AppConfig:
    """Configuração da aplicação"""
    
    # Caminhos padrão
    MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'models/pretrained/yolov8n.pt')
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    INPUT_DIR = os.getenv('INPUT_IMAGE_DIR', 'img/inference_data')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
    
    # Diretórios do projeto
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    PRETRAINED_DIR = MODELS_DIR / "pretrained"
    TRAINED_DIR = MODELS_DIR / "trained"
    IMG_DIR = PROJECT_ROOT / "img"
    INFERENCE_DATA_DIR = IMG_DIR / "inference_data"
    OUTPUT_DATA_DIR = PROJECT_ROOT / "output"
    
    # Configurações de inferência
    DEFAULT_ANALYSIS_TYPES = ["all_images", "threshold_analysis", "benchmark"]
    DEFAULT_CONFIDENCE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
    DEFAULT_BENCHMARK_RUNS = 3
    
    # Configurações de saída
    AUTO_INFERENCE_FILE = "auto_inference_results.json"
    BENCHMARK_FILE = "benchmark_results.json"
    BATCH_FILE = "inference_report.json"
    
    @classmethod
    def ensure_directories_exist(cls):
        """Garante que todos os diretórios necessários existam"""
        directories = [
            cls.MODELS_DIR,
            cls.PRETRAINED_DIR,
            cls.TRAINED_DIR,
            cls.IMG_DIR,
            cls.INFERENCE_DATA_DIR,
            cls.OUTPUT_DATA_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_default_event_config(cls) -> dict:
        """Retorna configuração padrão para eventos"""
        return {
            "model_name": cls.MODEL_PATH,
            "confidence": cls.CONFIDENCE_THRESHOLD,
            "input_dir": cls.INPUT_DIR,
            "output_dir": cls.OUTPUT_DIR,
            "analysis_type": "all_images",
            "save_report": True
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Valida se a configuração está correta"""
        try:
            # Verificar se diretórios essenciais existem
            if not cls.INFERENCE_DATA_DIR.exists():
                print(f"⚠️  Aviso: Diretório de imagens não encontrado: {cls.INFERENCE_DATA_DIR}")
                return False
            
            # Verificar se pelo menos um modelo existe
            pretrained_models = list(cls.PRETRAINED_DIR.glob("*.pt"))
            trained_models = list(cls.TRAINED_DIR.glob("*.pt"))
            
            if not pretrained_models and not trained_models:
                print("⚠️  Aviso: Nenhum modelo encontrado nas pastas models/")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na validação da configuração: {e}")
            return False


# Instância global da configuração
app_config = AppConfig()
