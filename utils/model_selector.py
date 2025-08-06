"""
Model Selector Utility
======================

Utilitário para listagem e seleção interativa de modelos YoloV8.
Permite ao usuário escolher entre modelos disponíveis nas pastas models/pretrained e models/trained.
Inclui funcionalidade de download automático de modelos quando necessário.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union


class ModelSelector:
    """
    Classe responsável por listar e permitir seleção interativa de modelos YoloV8
    Inclui funcionalidade de download automático quando não há modelos disponíveis
    """
    
    # Modelos YoloV8 oficiais disponíveis para download
    OFFICIAL_MODELS = {
        'yolov8n.pt': {'size': '6.2MB', 'speed': '🚀🚀🚀🚀🚀', 'accuracy': '⭐⭐⭐', 'desc': 'Nano - Ultra rápido'},
        'yolov8s.pt': {'size': '21.5MB', 'speed': '🚀🚀🚀🚀', 'accuracy': '⭐⭐⭐⭐', 'desc': 'Small - Balanceado'},
        'yolov8m.pt': {'size': '49.7MB', 'speed': '🚀🚀🚀', 'accuracy': '⭐⭐⭐⭐⭐', 'desc': 'Medium - Boa precisão'},
        'yolov8l.pt': {'size': '83.7MB', 'speed': '🚀🚀', 'accuracy': '⭐⭐⭐⭐⭐⭐', 'desc': 'Large - Alta precisão'},
        'yolov8x.pt': {'size': '136.7MB', 'speed': '🚀', 'accuracy': '⭐⭐⭐⭐⭐⭐⭐', 'desc': 'Extra Large - Máxima precisão'}
    }
    
    def __init__(self):
        self.pretrained_dir = Path('models/pretrained')
        self.trained_dir = Path('models/trained')
        
        # Garantir que os diretórios existem
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        self.trained_dir.mkdir(parents=True, exist_ok=True)
    
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
        Inclui opção de download automático quando não há modelos disponíveis
        
        Returns:
            Caminho do modelo selecionado ou None se cancelado
        """
        print("\n" + "="*60)
        print("🤖 SELEÇÃO DE MODELO YOLOV8")
        print("="*60)
        
        models = self.list_available_models()
        all_models = self.display_models_menu(models)
        
        # Se não há modelos, oferece download
        if not all_models:
            return self.handle_no_models_scenario()
        
        # Adicionar opção de download mesmo quando há modelos
        print(f"\n   � {len(all_models) + 1}. Baixar novo modelo oficial")
        
        # Seleção do usuário
        while True:
            try:
                print(f"\n🔧 Digite o número do modelo (1-{len(all_models) + 1}) ou Enter para usar o primeiro:")
                choice = input(">>> ").strip()
                
                if choice == "":
                    selected_model = all_models[0]
                    print(f"✅ Usando modelo padrão: {selected_model['name']}")
                    return selected_model['path']
                
                choice_num = int(choice)
                
                # Opção de download
                if choice_num == len(all_models) + 1:
                    selected_model_name = self.display_download_menu()
                    if selected_model_name:
                        if self.download_model(selected_model_name):
                            model_path = self.pretrained_dir / selected_model_name
                            return str(model_path)
                        else:
                            print("❌ Falha no download. Tente novamente.")
                            continue
                    else:
                        continue
                
                # Seleção de modelo existente
                elif 1 <= choice_num <= len(all_models):
                    selected_model = all_models[choice_num - 1]
                    print(f"✅ Modelo selecionado: {selected_model['name']}")
                    return selected_model['path']
                else:
                    print(f"❌ Número inválido! Digite um valor entre 1 e {len(all_models) + 1}")
            
            except ValueError:
                print("❌ Por favor, digite apenas números!")
            except KeyboardInterrupt:
                print("\n\n❌ Cancelado pelo usuário")
                return None
    
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

    def download_model(self, model_name: str) -> bool:
        """
        Baixa um modelo YoloV8 oficial usando ultralytics
        
        Args:
            model_name: Nome do modelo (ex: 'yolov8n.pt')
            
        Returns:
            True se o download foi bem-sucedido
        """
        try:
            print(f"\n📥 Baixando modelo {model_name}...")
            print("   Isso pode levar alguns minutos dependendo da sua conexão...")
            
            # Importar YOLO aqui para evitar dependência circular
            try:
                from ultralytics import YOLO
            except ImportError:
                print("❌ Erro: ultralytics não instalado!")
                print("   Execute: pip install ultralytics")
                return False
            
            # Baixar o modelo (YOLO automaticamente baixa se não existir)
            model = YOLO(model_name)
            
            # Verificar se o modelo foi baixado no diretório correto
            downloaded_path = Path(model_name)
            target_path = self.pretrained_dir / model_name
            
            # Mover para o diretório correto se necessário
            if downloaded_path.exists() and downloaded_path != target_path:
                downloaded_path.rename(target_path)
                print(f"✅ Modelo movido para: {target_path}")
            elif target_path.exists():
                print(f"✅ Modelo baixado com sucesso: {target_path}")
            else:
                print("❌ Erro: Não foi possível localizar o modelo baixado")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Erro ao baixar modelo {model_name}: {e}")
            return False

    def display_download_menu(self) -> Optional[str]:
        """
        Exibe menu de download de modelos oficiais
        
        Returns:
            Nome do modelo escolhido para download ou None se cancelado
        """
        print("\n" + "="*70)
        print("📥 DOWNLOAD DE MODELOS YOLOV8 OFICIAIS")
        print("="*70)
        print("\n🤖 Modelos disponíveis para download:")
        
        models_list = list(self.OFFICIAL_MODELS.keys())
        
        for i, model_name in enumerate(models_list, 1):
            info = self.OFFICIAL_MODELS[model_name]
            print(f"   {i}. {model_name}")
            print(f"      📊 {info['desc']}")
            print(f"      💾 Tamanho: {info['size']}")
            print(f"      ⚡ Velocidade: {info['speed']}")
            print(f"      🎯 Precisão: {info['accuracy']}")
            print()
        
        print("   0. ❌ Cancelar")
        
        while True:
            try:
                choice = input(f"\n🔧 Digite o número do modelo (1-{len(models_list)}) ou 0 para cancelar: ").strip()
                
                if choice == "0":
                    print("❌ Download cancelado")
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(models_list):
                    selected_model = models_list[choice_num - 1]
                    print(f"✅ Selecionado para download: {selected_model}")
                    return selected_model
                else:
                    print(f"❌ Número inválido! Digite um valor entre 1 e {len(models_list)}")
            
            except ValueError:
                print("❌ Por favor, digite apenas números!")
            except KeyboardInterrupt:
                print("\n\n❌ Cancelado pelo usuário")
                return None

    def handle_no_models_scenario(self) -> Optional[str]:
        """
        Lida com o cenário onde não há modelos disponíveis
        Oferece opção de download automático
        
        Returns:
            Caminho do modelo baixado ou None se cancelado
        """
        print("\n❌ NENHUM MODELO ENCONTRADO!")
        print("="*50)
        print("   Não foram encontrados modelos YoloV8 nas pastas:")
        print("   📁 models/pretrained/")
        print("   📁 models/trained/")
        
        print("\n💡 OPÇÕES DISPONÍVEIS:")
        print("   1. 📥 Baixar modelo oficial YoloV8")
        print("   2. 📋 Ver instruções para adicionar modelos manualmente")
        print("   3. ❌ Cancelar")
        
        while True:
            try:
                choice = input("\n🔧 Digite sua escolha (1-3): ").strip()
                
                if choice == "1":
                    # Opção de download
                    selected_model = self.display_download_menu()
                    if selected_model:
                        if self.download_model(selected_model):
                            model_path = self.pretrained_dir / selected_model
                            return str(model_path)
                        else:
                            print("❌ Falha no download. Tente novamente.")
                            continue
                    else:
                        continue
                
                elif choice == "2":
                    # Instruções manuais
                    self.show_manual_instructions()
                    return None
                
                elif choice == "3":
                    print("❌ Operação cancelada")
                    return None
                
                else:
                    print("❌ Opção inválida! Digite 1, 2 ou 3")
            
            except ValueError:
                print("❌ Por favor, digite apenas números!")
            except KeyboardInterrupt:
                print("\n\n❌ Cancelado pelo usuário")
                return None

    def show_manual_instructions(self):
        """
        Mostra instruções para adicionar modelos manualmente
        """
        print("\n" + "="*60)
        print("📋 COMO ADICIONAR MODELOS MANUALMENTE")
        print("="*60)
        
        print("\n🤖 MODELOS PRÉ-TREINADOS:")
        print("   📁 Pasta: models/pretrained/")
        print("   🌐 Download manual: https://github.com/ultralytics/ultralytics")
        print("   📝 Formatos aceitos: .pt (PyTorch)")
        
        print("\n🎯 MODELOS CUSTOMIZADOS:")
        print("   📁 Pasta: models/trained/")
        print("   📝 Coloque seus modelos treinados aqui")
        print("   📝 Formatos aceitos: .pt (PyTorch)")
        
        print("\n📥 DOWNLOAD DIRETO:")
        for model_name, info in self.OFFICIAL_MODELS.items():
            print(f"   • {model_name} - {info['desc']} ({info['size']})")
        
        print("\n💡 DICAS:")
        print("   1. Certifique-se de que os arquivos têm extensão .pt")
        print("   2. Execute este script novamente após adicionar modelos")
        print("   3. Os modelos são carregados automaticamente na próxima execução")


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
