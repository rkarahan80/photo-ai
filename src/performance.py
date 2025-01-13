import multiprocessing
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
from PIL import Image
import importlib.util

class PerformanceOptimizer:
    def __init__(self):
        self.num_cpus = multiprocessing.cpu_count()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_threshold = 0.8  # %80 bellek kullanımı eşiği
        
    def optimize_gpu_memory(self):
        """GPU belleğini optimize et"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def check_memory_usage(self):
        """Sistem bellek kullanımını kontrol et"""
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_threshold * 100:
            gc.collect()
            return False
        return True
    
    def parallel_process_image(self, image, operation, chunks=4):
        """Görüntüyü paralel işle"""
        height, width = image.size[1], image.size[0]
        chunk_height = height // chunks
        
        # Görüntüyü parçalara böl
        image_chunks = []
        for i in range(chunks):
            start = i * chunk_height
            end = start + chunk_height if i < chunks-1 else height
            chunk = image.crop((0, start, width, end))
            image_chunks.append(chunk)
        
        # Parçaları paralel işle
        with ProcessPoolExecutor(max_workers=self.num_cpus) as executor:
            processed_chunks = list(executor.map(operation, image_chunks))
        
        # Parçaları birleştir
        final_image = Image.new('RGB', (width, height))
        for i, chunk in enumerate(processed_chunks):
            final_image.paste(chunk, (0, i * chunk_height))
        
        return final_image
    
    def batch_process_threaded(self, images, operation, max_workers=None):
        """Çoklu görüntüyü thread'lerle işle"""
        if max_workers is None:
            max_workers = self.num_cpus * 2
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(operation, images))
        return results
    
    def optimize_image_size(self, image, max_size=4096):
        """Görüntü boyutunu optimize et"""
        width, height = image.size
        if width > max_size or height > max_size:
            ratio = min(max_size/width, max_size/height)
            new_size = (int(width*ratio), int(height*ratio))
            return image.resize(new_size)
        return image
    
    @staticmethod
    def get_optimal_chunk_size(image_size, available_memory):
        """Optimal parça boyutunu hesapla"""
        # Görüntü başına yaklaşık bellek kullanımı (RGB = 3 kanal)
        memory_per_pixel = 3 * 8  # bytes
        total_pixels = image_size[0] * image_size[1]
        total_memory_needed = total_pixels * memory_per_pixel
        
        # Kullanılabilir belleğe göre parça sayısını hesapla
        chunks = max(1, int(np.ceil(total_memory_needed / (available_memory * 0.8))))
        return chunks
    
    def process_with_gpu(self, image, operation):
        """GPU ile işleme"""
        if torch.cuda.is_available():
            # Görüntüyü GPU'ya taşı
            img_tensor = torch.from_numpy(np.array(image)).to(self.device)
            
            # İşlemi GPU'da gerçekleştir
            with torch.cuda.amp.autocast():  # Otomatik karışık hassasiyet
                result = operation(img_tensor)
            
            # Sonucu CPU'ya geri taşı
            result = result.cpu().numpy()
            return Image.fromarray(np.uint8(result))
        else:
            return operation(image)
    
    def clear_memory(self):
        """Belleği temizle"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_system_info(self):
        """Sistem bilgilerini getir"""
        return {
            'cpu_count': self.num_cpus,
            'gpu_available': torch.cuda.is_available(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        } 