import os
import json
import requests
from typing import Optional, Dict, List, Any
from PIL import Image
import io
import base64
import importlib.util
from flask import Flask, request, jsonify, send_file

class CloudStorage:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('CLOUD_API_KEY')
        self.endpoints = {
            'google_drive': 'https://www.googleapis.com/drive/v3',
            'dropbox': 'https://api.dropboxapi.com/2',
            'onedrive': 'https://graph.microsoft.com/v1.0'
        }
        
    def upload_to_drive(self, image: Image.Image, filename: str) -> Dict[str, Any]:
        """Google Drive'a yükle"""
        if not self.api_key:
            raise ValueError("Google Drive API anahtarı gerekli")
            
        # Görüntüyü byte dizisine dönüştür
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'image/png'
        }
        
        # Google Drive API'sine yükle
        response = requests.post(
            f"{self.endpoints['google_drive']}/files",
            headers=headers,
            data=img_byte_arr
        )
        
        return response.json()
    
    def upload_to_dropbox(self, image: Image.Image, filename: str) -> Dict[str, Any]:
        """Dropbox'a yükle"""
        if not self.api_key:
            raise ValueError("Dropbox API anahtarı gerekli")
            
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/octet-stream',
            'Dropbox-API-Arg': json.dumps({
                'path': f'/{filename}',
                'mode': 'add'
            })
        }
        
        response = requests.post(
            f"{self.endpoints['dropbox']}/files/upload",
            headers=headers,
            data=img_byte_arr
        )
        
        return response.json()

class SocialMediaShare:
    def __init__(self, credentials: Dict[str, str] = None):
        self.credentials = credentials or {}
        
    def share_to_instagram(self, image: Image.Image, caption: str = "") -> Dict[str, Any]:
        """Instagram'da paylaş"""
        if 'instagram_token' not in self.credentials:
            raise ValueError("Instagram API token'ı gerekli")
            
        # Görüntüyü Instagram formatına uygun hale getir
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_str = base64.b64encode(img_byte_arr.getvalue()).decode()
        
        payload = {
            'image': img_str,
            'caption': caption
        }
        
        headers = {
            'Authorization': f"Bearer {self.credentials['instagram_token']}"
        }
        
        response = requests.post(
            'https://graph.instagram.com/me/media',
            json=payload,
            headers=headers
        )
        
        return response.json()
    
    def share_to_twitter(self, image: Image.Image, tweet: str = "") -> Dict[str, Any]:
        """Twitter'da paylaş"""
        if 'twitter_token' not in self.credentials:
            raise ValueError("Twitter API token'ı gerekli")
            
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_data = img_byte_arr.getvalue()
        
        # Önce medyayı yükle
        headers = {
            'Authorization': f"Bearer {self.credentials['twitter_token']}"
        }
        
        files = {
            'media': img_data
        }
        
        media_response = requests.post(
            'https://upload.twitter.com/1.1/media/upload.json',
            headers=headers,
            files=files
        )
        
        media_id = media_response.json()['media_id']
        
        # Tweet'i oluştur
        payload = {
            'status': tweet,
            'media_ids': [media_id]
        }
        
        response = requests.post(
            'https://api.twitter.com/1.1/statuses/update.json',
            headers=headers,
            json=payload
        )
        
        return response.json()

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, Any] = {}
        self.plugin_dir = 'plugins'
        
    def load_plugin(self, plugin_name: str) -> bool:
        """Plugin yükle"""
        try:
            plugin_path = os.path.join(self.plugin_dir, f"{plugin_name}.py")
            if not os.path.exists(plugin_path):
                return False
                
            # Plugin'i dinamik olarak içe aktar
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Plugin'i kaydet
            self.plugins[plugin_name] = module
            return True
        except Exception as e:
            print(f"Plugin yükleme hatası: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """Plugin'i getir"""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """Yüklü plugin'leri listele"""
        return list(self.plugins.keys())
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Plugin'i kaldır"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            return True
        return False

class APIEndpoint:
    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        """API rotalarını ayarla"""
        @self.app.route('/process', methods=['POST'])
        def process_image():
            if 'image' not in request.files:
                return jsonify({'error': 'Görüntü bulunamadı'}), 400
                
            file = request.files['image']
            image = Image.open(file.stream)
            
            # İşlem parametrelerini al
            params = request.form.to_dict()
            
            # İşlemi uygula
            # ... işlem kodları ...
            
            # Sonucu döndür
            output = io.BytesIO()
            image.save(output, format='PNG')
            
            return send_file(
                output,
                mimetype='image/png',
                as_attachment=True,
                download_name='processed.png'
            )
    
    def start(self):
        """API sunucusunu başlat"""
        self.app.run(host=self.host, port=self.port) 