"""
Model Selector Utility
======================

Utilitário para listagem e seleção interativa de modelos YoloV8.
Permite ao usuário escolher entre modelos disponíveis nas pastas models/pretrained e models/trained.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union


class ModelSelector:
    """
    Classe responsável por listar e permitir seleção interativa de modelos YoloV8
    """
    
    def __init__(self):
        self.pretrained_dir = Path('models/pretrained')
        self.trained_dir = Path('models/trained')
    
    def list_available_models(self) -> Dict[str, List[Dict]]:
        """
        Lista todos os modelos disponíveis nas pastas models/pretrained e models/trained
        
        Returns:
            Dict contendo listas de modelos pré-treinados e customizados
        """
        models = {
            'pretrained': [],
            'trained': []
        }
        
        # Verificar modelos pré-treinados
        if self.pretrained_dir.exists():
            for model_file in self.pretrained_dir.glob('*.pt'):
                try:
                    models['pretrained'].append({
                        'name': model_file.name,
                        'path': str(model_file),
                        'size': model_file.stat().st_size / (1024*1024),  # MB
                        'type': 'pretrained'
                    })
                except (OSError, IOError) as e:
                    print(f"⚠️  Aviso: Não foi possível acessar {model_file.name}: {e}")
        
        # Verificar modelos treinados/customizados
        if self.trained_dir.exists():
            for model_file in self.trained_dir.glob('*.pt'):
                try:
                    models['trained'].append({
                        'name': model_file.name,
                        'path': str(model_file),
                        'size': model_file.stat().st_size / (1024*1024),  # MB
                        'type': 'trained'
                    })
                except (OSError, IOError) as e:
                    print(f"⚠️  Aviso: Não foi possível acessar {model_file.name}: {e}")
        
        # Ordenar modelos por nome
        models['pretrained'].sort(key=lambda x: x['name'])
        models['trained'].sort(key=lambda x: x['name'])
        
        return models
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict:
        """
        Obtém informações detalhadas sobre um modelo específico
        
        Args:
            model_path: Caminho para o modelo
            
        Returns:
            Dict com informações do modelo
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            return {
                'name': model_path.name,
                'path': str(model_path),
                'exists': False,
                'error': 'Arquivo não encontrado'
            }
        
        try:
            size_mb = model_path.stat().st_size / (1024*1024)
            model_type = 'pretrained' if 'pretrained' in str(model_path) else 'trained'
            
            return {
                'name': model_path.name,
                'path': str(model_path),
                'size': size_mb,
                'type': model_type,
                'exists': True
            }
        except (OSError, IOError) as e:
            return {
                'name': model_path.name,
                'path': str(model_path),
                'exists': False,
                'error': str(e)
            }
    
    def display_models_menu(self, models: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Exibe o menu de modelos disponíveis
        
        Args:
            models: Dict contendo listas de modelos
            
        Returns:
            Lista unificada de todos os modelos
        """
        all_models = []
        
        print("\n📦 MODELOS DISPONÍVEIS:")
        
        # Listar modelos pré-treinados
        if models['pretrained']:
            print("\n   🚀 Pré-treinados (models/pretrained/):")
            for model in models['pretrained']:
                idx = len(all_models) + 1
                all_models.append(model)
                print(f"      {idx}. {model['name']} ({model['size']:.1f}MB)")
        
        # Listar modelos treinados/customizados
        if models['trained']:
            print("\n   🎯 Customizados (models/trained/):")
            for model in models['trained']:
                idx = len(all_models) + 1
                all_models.append(model)
                print(f"      {idx}. {model['name']} ({model['size']:.1f}MB)")
        
        return all_models
    
    def select_model_interactive(self) -> Optional[str]:
        """
        Interface interativa para seleção de modelo
        
        Returns:
            Caminho do modelo selecionado ou None se cancelado
        """
        print("\n" + "="*60)
        print("🤖 SELEÇÃO DE MODELO YOLOV8")
        print("="*60)
        
        models = self.list_available_models()
        all_models = self.display_models_menu(models)
        
        if not all_models:
            print("\n❌ Nenhum modelo encontrado nas pastas models/!")
            print("   Certifique-se de ter modelos .pt em:")
            print("   - models/pretrained/")
            print("   - models/trained/")
            print("\n💡 Dica: Você pode adicionar modelos YoloV8 (.pt) nessas pastas")
            return None
        
        # Seleção do usuário
        while True:
            try:
                print(f"\n🔧 Digite o número do modelo (1-{len(all_models)}) ou Enter para usar o primeiro:")
                choice = input(">>> ").strip()
                
                if choice == "":
                    selected_model = all_models[0]
                    print(f"✅ Usando modelo padrão: {selected_model['name']}")
                    break
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(all_models):
                    selected_model = all_models[choice_num - 1]
                    print(f"✅ Modelo selecionado: {selected_model['name']}")
                    break
                else:
                    print(f"❌ Número inválido! Digite um valor entre 1 e {len(all_models)}")
            
            except ValueError:
                print("❌ Por favor, digite apenas números!")
            except KeyboardInterrupt:
                print("\n\n❌ Cancelado pelo usuário")
                return None
        
        return selected_model['path']
    
    def get_default_model(self) -> Optional[str]:
        """
        Obtém o primeiro modelo disponível como padrão
        
        Returns:
            Caminho do primeiro modelo encontrado ou None
        """
        models = self.list_available_models()
        
        # Priorizar modelos pré-treinados
        if models['pretrained']:
            return models['pretrained'][0]['path']
        elif models['trained']:
            return models['trained'][0]['path']
        
        return None
    
    def validate_model_path(self, model_path: Union[str, Path]) -> bool:
        """
        Valida se um caminho de modelo é válido
        
        Args:
            model_path: Caminho para validar
            
        Returns:
            True se o modelo existe e é válido
        """
        model_path = Path(model_path)
        
        # Verificar se existe
        if not model_path.exists():
            return False
        
        # Verificar extensão
        if model_path.suffix.lower() != '.pt':
            return False
        
        # Verificar se não está vazio
        try:
            if model_path.stat().st_size == 0:
                return False
        except (OSError, IOError):
            return False
        
        return True
    
    def find_models_by_pattern(self, pattern: str) -> List[str]:
        """
        Encontra modelos que correspondem a um padrão
        
        Args:
            pattern: Padrão para buscar (ex: "yolov8n", "custom_*")
            
        Returns:
            Lista de caminhos de modelos que correspondem ao padrão
        """
        matches = []
        
        # Buscar em ambas as pastas
        for directory in [self.pretrained_dir, self.trained_dir]:
            if directory.exists():
                for model_file in directory.glob(f"{pattern}*.pt"):
                    matches.append(str(model_file))
        
        return sorted(matches)


# Função de conveniência para uso direto
def select_model() -> Optional[str]:
    """
    Função de conveniência para seleção interativa de modelo
    
    Returns:
        Caminho do modelo selecionado ou None se cancelado
    """
    selector = ModelSelector()
    return selector.select_model_interactive()


def get_available_models() -> Dict[str, List[Dict]]:
    """
    Função de conveniência para obter lista de modelos disponíveis
    
    Returns:
        Dict contendo listas de modelos pré-treinados e customizados
    """
    selector = ModelSelector()
    return selector.list_available_models()
