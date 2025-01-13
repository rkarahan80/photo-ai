from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple, List, Optional
import math

class Brush:
    def __init__(self, size: int = 10, hardness: float = 0.5, opacity: float = 1.0,
                 color: Tuple[int, int, int] = (0, 0, 0)):
        self.size = size
        self.hardness = hardness
        self.opacity = opacity
        self.color = color
        self.pressure_sensitive = True
        self._create_brush_tip()
        
    def _create_brush_tip(self):
        """Fırça ucunu oluştur"""
        size = self.size * 2  # Kenar yumuşatma için iki kat boyut
        center = size // 2
        self.tip = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(self.tip)
        
        # Yumuşak kenar için gradyan oluştur
        for x in range(size):
            for y in range(size):
                distance = math.sqrt((x - center) ** 2 + (y - center) ** 2)
                # Mesafeye göre opaklık hesapla
                if distance < self.size:
                    opacity = int(255 * (1 - (distance / self.size) ** (1/self.hardness)))
                    draw.point((x, y), opacity)
        
        self.tip = self.tip.resize((self.size, self.size), Image.LANCZOS)

class BrushManager:
    def __init__(self):
        self.brushes: List[Brush] = []
        self.active_brush: Optional[Brush] = None
        self._create_default_brushes()
        
    def _create_default_brushes(self):
        """Varsayılan fırçaları oluştur"""
        # Normal fırça
        self.add_brush(Brush(size=10, hardness=0.8, opacity=1.0))
        # Yumuşak fırça
        self.add_brush(Brush(size=20, hardness=0.3, opacity=0.8))
        # Sert fırça
        self.add_brush(Brush(size=5, hardness=1.0, opacity=1.0))
        
    def add_brush(self, brush: Brush):
        """Yeni fırça ekle"""
        self.brushes.append(brush)
        if self.active_brush is None:
            self.active_brush = brush
            
    def set_active_brush(self, index: int):
        """Aktif fırçayı ayarla"""
        if 0 <= index < len(self.brushes):
            self.active_brush = self.brushes[index]
            
    def create_stroke(self, points: List[Tuple[int, int]], pressure: List[float] = None) -> Image.Image:
        """Fırça darbesi oluştur"""
        if not self.active_brush or not points:
            return None
            
        # Darbe için boş görüntü oluştur
        stroke_image = Image.new('RGBA', (800, 600), (0, 0, 0, 0))
        draw = ImageDraw.Draw(stroke_image)
        
        # Basınç değerleri yoksa varsayılan olarak 1.0 kullan
        if pressure is None:
            pressure = [1.0] * len(points)
            
        # Noktalar arasına ara noktalar ekle
        interpolated_points = []
        interpolated_pressure = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            p1, p2 = pressure[i], pressure[i + 1]
            
            # İki nokta arasındaki mesafe
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            steps = max(1, int(distance))
            
            # Ara noktaları hesapla
            for step in range(steps):
                t = step / steps
                x = x1 + (x2 - x1) * t
                y = y1 + (y2 - y1) * t
                p = p1 + (p2 - p1) * t
                interpolated_points.append((x, y))
                interpolated_pressure.append(p)
        
        # Her nokta için fırça ucunu çiz
        for point, p in zip(interpolated_points, interpolated_pressure):
            if self.active_brush.pressure_sensitive:
                size = int(self.active_brush.size * p)
                opacity = int(self.active_brush.opacity * 255 * p)
            else:
                size = self.active_brush.size
                opacity = int(self.active_brush.opacity * 255)
            
            # Fırça ucunu yeniden oluştur
            brush_tip = self.active_brush.tip.resize((size, size), Image.LANCZOS)
            # Opaklık uygula
            brush_tip = Image.eval(brush_tip, lambda x: int(x * opacity / 255))
            
            # Fırça ucunu konuma yapıştır
            x, y = point
            pos = (int(x - size/2), int(y - size/2))
            stroke_image.paste(brush_tip, pos, brush_tip)
            
        return stroke_image

class EraserTool(Brush):
    def __init__(self, size: int = 10, hardness: float = 0.5):
        super().__init__(size, hardness, 1.0, (0, 0, 0))
        self.mode = 'erase'

class SmudgeTool(Brush):
    def __init__(self, size: int = 10, strength: float = 0.5):
        super().__init__(size, 0.5, 1.0, (0, 0, 0))
        self.strength = strength
        self.last_position = None
        self.buffer = None
        
    def apply(self, image: Image.Image, x: int, y: int) -> Image.Image:
        """Smudge efektini uygula"""
        if self.last_position is None:
            self.last_position = (x, y)
            # Fırça alanındaki görüntüyü tampona al
            left = x - self.size//2
            top = y - self.size//2
            self.buffer = image.crop((left, top, left + self.size, top + self.size))
            return image
            
        # Yeni pozisyon ile son pozisyon arasında interpolasyon yap
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        # Tampondaki görüntüyü yeni konuma karıştır
        if self.buffer:
            x1, y1 = self.last_position
            dx = x - x1
            dy = y - y1
            
            left = x - self.size//2
            top = y - self.size//2
            
            # Tamponu yeni konuma yapıştır
            result.paste(self.buffer, (left, top), self.tip)
            
            # Yeni tamponu al
            self.buffer = image.crop((left, top, left + self.size, top + self.size))
            
        self.last_position = (x, y)
        return result

class BrushPreset:
    def __init__(self, name: str, brush: Brush):
        self.name = name
        self.brush = brush
        self.settings = {
            'size': brush.size,
            'hardness': brush.hardness,
            'opacity': brush.opacity,
            'color': brush.color
        }
    
    def apply(self) -> Brush:
        """Ön ayarı uygula"""
        return Brush(
            size=self.settings['size'],
            hardness=self.settings['hardness'],
            opacity=self.settings['opacity'],
            color=self.settings['color']
        )

class BrushLibrary:
    def __init__(self):
        self.presets: List[BrushPreset] = []
        self._create_default_presets()
        
    def _create_default_presets(self):
        """Varsayılan fırça ön ayarlarını oluştur"""
        # Yumuşak fırça
        self.add_preset("Yumuşak Fırça", Brush(size=20, hardness=0.3, opacity=0.8))
        # Sert fırça
        self.add_preset("Sert Fırça", Brush(size=5, hardness=1.0, opacity=1.0))
        # Airbrush
        self.add_preset("Airbrush", Brush(size=30, hardness=0.1, opacity=0.3))
        
    def add_preset(self, name: str, brush: Brush):
        """Yeni ön ayar ekle"""
        preset = BrushPreset(name, brush)
        self.presets.append(preset)
        
    def get_preset(self, name: str) -> Optional[BrushPreset]:
        """İsme göre ön ayar getir"""
        for preset in self.presets:
            if preset.name == name:
                return preset
        return None
        
    def save_preset(self, name: str, brush: Brush):
        """Mevcut fırçayı ön ayar olarak kaydet"""
        preset = BrushPreset(name, brush)
        self.presets.append(preset)
        
    def delete_preset(self, name: str) -> bool:
        """Ön ayarı sil"""
        for i, preset in enumerate(self.presets):
            if preset.name == name:
                self.presets.pop(i)
                return True
        return False 