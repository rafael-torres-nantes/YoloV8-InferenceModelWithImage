# config/settings.py

"""
Configurações Essenciais - YoloV8 Inference
==========================================
Configure aqui os diretórios e parâmetros principais do sistema
"""
from pathlib import Path

# =============================================================================
# 📁 DIRETÓRIOS DO PROGRAMA (ALTERE AQUI CONFORME NECESSÁRIO)
# =============================================================================
# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent

# 📷 Diretórios de imagens
IMG_DIR = PROJECT_ROOT / "img"
INFERENCE_DATA_DIR = IMG_DIR / "inference_data"    # ← Suas imagens para inferência

# 🤖 Diretórios de modelos
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"  # ← Modelos baixados automaticamente
TRAINED_MODELS_DIR = MODELS_DIR / "trained"        # ← Seus modelos customizados

# 📊 Diretório de resultados
OUTPUT_DIR = PROJECT_ROOT / "output"                # ← Resultados JSON e imagens anotadas

# =============================================================================
# ⚙️ CONFIGURAÇÕES DE INFERÊNCIA (ALTERE SE NECESSÁRIO)
# =============================================================================
# Modelo padrão (será baixado automaticamente se não existir)
DEFAULT_MODEL = "yolov8n.pt"

# Confiança mínima para detecções (0.0 a 1.0)
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

# IoU threshold para Non-Maximum Suppression
DEFAULT_IOU_THRESHOLD = 0.45

# =============================================================================
# 🤖 MODELOS YOLOV8 DISPONÍVEIS
# =============================================================================
AVAILABLE_PRETRAINED_MODELS = [
    "yolov8n.pt",  # Nano - Mais rápido, menor precisão
    "yolov8s.pt",  # Small
    "yolov8m.pt",  # Medium  
    "yolov8l.pt",  # Large
    "yolov8x.pt"   # Extra Large - Mais lento, maior precisão
]

# =============================================================================
# 📄 EXTENSÕES DE IMAGEM SUPORTADAS
# =============================================================================
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# =============================================================================
# 🏷️ CLASSES COCO (80 classes do dataset COCO)
# =============================================================================
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# =============================================================================
# 🔧 CLASSE DE CONFIGURAÇÃO (NÃO ALTERE ESTA PARTE)
# =============================================================================
class Config:
    """Configurações centralizadas do sistema."""
    
    # Diretórios principais  
    PROJECT_ROOT = PROJECT_ROOT
    IMG_DIR = IMG_DIR
    INFERENCE_DATA_DIR = INFERENCE_DATA_DIR
    MODELS_DIR = MODELS_DIR
    PRETRAINED_MODELS_DIR = PRETRAINED_MODELS_DIR
    TRAINED_MODELS_DIR = TRAINED_MODELS_DIR
    OUTPUT_DIR = OUTPUT_DIR
    
    # Configurações de inferência
    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_CONFIDENCE_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD
    DEFAULT_IOU_THRESHOLD = DEFAULT_IOU_THRESHOLD
    
    # Modelos e extensões
    AVAILABLE_PRETRAINED_MODELS = AVAILABLE_PRETRAINED_MODELS
    SUPPORTED_IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS
    
    # Configurações de saída
    SAVE_ANNOTATED_IMAGES = True
    SAVE_JSON_RESULTS = True
    
    def __init__(self):
        """Inicializa e cria diretórios necessários."""
        self.ensure_directories()
    
    def ensure_directories(self):
        """Garante que todos os diretórios necessários existam."""
        directories = [
            self.INFERENCE_DATA_DIR,
            self.PRETRAINED_MODELS_DIR,
            self.TRAINED_MODELS_DIR,
            self.OUTPUT_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_available_local_models(self):
        """Retorna lista de modelos disponíveis localmente."""
        models = []
        
        # Verifica modelos pré-treinados
        if self.PRETRAINED_MODELS_DIR.exists():
            models.extend(list(self.PRETRAINED_MODELS_DIR.glob("*.pt")))
        
        # Verifica modelos treinados
        if self.TRAINED_MODELS_DIR.exists():
            models.extend(list(self.TRAINED_MODELS_DIR.glob("*.pt")))
        
        return [str(model) for model in models]
    
    def get_model_path(self, model_name: str):
        """
        Retorna o caminho completo para um modelo.
        
        Args:
            model_name: Nome do modelo (ex: "yolov8n.pt")
            
        Returns:
            Caminho para o modelo local ou nome para download
        """
        import os
        
        # Verifica se é um caminho absoluto
        if os.path.isabs(model_name):
            return model_name
        
        # Verifica modelos treinados primeiro
        trained_path = self.TRAINED_MODELS_DIR / model_name
        if trained_path.exists():
            return str(trained_path)
        
        # Verifica modelos pré-treinados locais  
        pretrained_path = self.PRETRAINED_MODELS_DIR / model_name
        if pretrained_path.exists():
            return str(pretrained_path)
        
        # Se não encontrou local, retorna apenas o nome (para download automático)
        return model_name

# Instância global da configuração
config = Config()
