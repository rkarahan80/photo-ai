from PIL import Image, ImageDraw
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import json

@dataclass
class Layer:
    image: Image.Image
    name: str
    opacity: float = 1.0
    visible: bool = True
    blend_mode: str = 'normal'
    mask: Optional[Image.Image] = None
    position: Tuple[int, int] = (0, 0)

class LayerManager:
    def __init__(self):
        self.layers: List[Layer] = []
        self.active_layer_index: int = -1
        self.canvas_size: Tuple[int, int] = (800, 600)
        
    def add_layer(self, image: Image.Image, name: str = None) -> int:
        """Yeni katman ekle"""
        if name is None:
            name = f"Katman {len(self.layers) + 1}"
            
        # Görüntüyü canvas boyutuna ayarla
        if image.size != self.canvas_size:
            image = image.resize(self.canvas_size)
            
        layer = Layer(image=image, name=name)
        self.layers.append(layer)
        self.active_layer_index = len(self.layers) - 1
        return self.active_layer_index
    
    def delete_layer(self, index: int) -> bool:
        """Katmanı sil"""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            if self.active_layer_index >= index:
                self.active_layer_index = max(0, self.active_layer_index - 1)
            return True
        return False
    
    def move_layer(self, from_index: int, to_index: int) -> bool:
        """Katmanı taşı"""
        if 0 <= from_index < len(self.layers) and 0 <= to_index < len(self.layers):
            layer = self.layers.pop(from_index)
            self.layers.insert(to_index, layer)
            if self.active_layer_index == from_index:
                self.active_layer_index = to_index
            return True
        return False
    
    def set_layer_opacity(self, index: int, opacity: float) -> bool:
        """Katman opaklığını ayarla"""
        if 0 <= index < len(self.layers):
            self.layers[index].opacity = max(0.0, min(1.0, opacity))
            return True
        return False
    
    def set_layer_visibility(self, index: int, visible: bool) -> bool:
        """Katman görünürlüğünü ayarla"""
        if 0 <= index < len(self.layers):
            self.layers[index].visible = visible
            return True
        return False
    
    def set_layer_blend_mode(self, index: int, mode: str) -> bool:
        """Katman karışım modunu ayarla"""
        valid_modes = ['normal', 'multiply', 'screen', 'overlay', 'darken', 'lighten']
        if 0 <= index < len(self.layers) and mode in valid_modes:
            self.layers[index].blend_mode = mode
            return True
        return False
    
    def add_layer_mask(self, index: int, mask: Optional[Image.Image] = None) -> bool:
        """Katmana maske ekle"""
        if 0 <= index < len(self.layers):
            if mask is None:
                # Beyaz maske oluştur (tamamen görünür)
                mask = Image.new('L', self.canvas_size, 255)
            elif mask.size != self.canvas_size:
                mask = mask.resize(self.canvas_size)
            
            self.layers[index].mask = mask
            return True
        return False
    
    def apply_mask(self, layer: Layer) -> Image.Image:
        """Maskeyi katmana uygula"""
        if layer.mask:
            mask_array = np.array(layer.mask.convert('L')) / 255.0
            image_array = np.array(layer.image)
            
            # Alpha kanalı varsa
            if image_array.shape[-1] == 4:
                image_array[:, :, 3] *= mask_array
            else:
                # RGB görüntü için alpha kanalı ekle
                alpha = np.ones(mask_array.shape) * mask_array
                image_array = np.dstack((image_array, alpha * 255))
                
            return Image.fromarray(image_array.astype(np.uint8))
        return layer.image
    
    def blend_layers(self) -> Image.Image:
        """Tüm katmanları birleştir"""
        if not self.layers:
            return Image.new('RGBA', self.canvas_size, (0, 0, 0, 0))
            
        result = Image.new('RGBA', self.canvas_size, (0, 0, 0, 0))
        
        for layer in self.layers:
            if not layer.visible:
                continue
                
            current = self.apply_mask(layer)
            
            # Opaklık uygula
            if layer.opacity < 1.0:
                current.putalpha(int(255 * layer.opacity))
            
            # Karışım modunu uygula
            if layer.blend_mode == 'normal':
                result = Image.alpha_composite(result, current)
            elif layer.blend_mode == 'multiply':
                result = Image.blend(result, current, 0.5)
            # Diğer karışım modları buraya eklenebilir
            
        return result
    
    def save_project(self, filename: str):
        """Proje dosyasını kaydet"""
        project_data = {
            'canvas_size': self.canvas_size,
            'active_layer': self.active_layer_index,
            'layers': []
        }
        
        for layer in self.layers:
            layer_data = {
                'name': layer.name,
                'opacity': layer.opacity,
                'visible': layer.visible,
                'blend_mode': layer.blend_mode,
                'position': layer.position
            }
            # Görüntü ve maskeyi ayrı dosyalar olarak kaydet
            project_data['layers'].append(layer_data)
            
        with open(filename, 'w') as f:
            json.dump(project_data, f)
    
    def load_project(self, filename: str):
        """Proje dosyasını yükle"""
        with open(filename, 'r') as f:
            project_data = json.load(f)
            
        self.canvas_size = tuple(project_data['canvas_size'])
        self.active_layer_index = project_data['active_layer']
        # Katmanları yükle
        
    def get_active_layer(self) -> Optional[Layer]:
        """Aktif katmanı getir"""
        if 0 <= self.active_layer_index < len(self.layers):
            return self.layers[self.active_layer_index]
        return None
    
    def duplicate_layer(self, index: int) -> int:
        """Katmanı çoğalt"""
        if 0 <= index < len(self.layers):
            layer = self.layers[index]
            new_layer = Layer(
                image=layer.image.copy(),
                name=f"{layer.name} kopya",
                opacity=layer.opacity,
                blend_mode=layer.blend_mode,
                mask=layer.mask.copy() if layer.mask else None,
                position=layer.position
            )
            self.layers.insert(index + 1, new_layer)
            return index + 1
        return -1 