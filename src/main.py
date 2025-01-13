import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps, ImageChops
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from rembg import remove
import face_recognition
from scipy.ndimage import gaussian_filter, rotate
import io
import colorsys
from skimage import exposure
import random
from datetime import datetime

class AIPhotoshop:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.style_model = VGG16(weights='imagenet', include_top=False)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Geçmiş yönetimi için
        self.history = []
        self.current_step = -1
        self.max_history = 20
        # Özel filtreler için
        self.custom_filters = {}
        
    def save_state(self, image, operation_name):
        """Geçmişe durum kaydet"""
        # Geçmişin sonundan itibaren silme
        if self.current_step < len(self.history) - 1:
            self.history = self.history[:self.current_step + 1]
        
        # Geçmişe ekle
        self.history.append({
            'image': image.copy(),
            'operation': operation_name,
            'timestamp': datetime.now()
        })
        self.current_step += 1
        
        # Maksimum geçmiş limitini kontrol et
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_step -= 1
    
    def undo(self):
        """Son işlemi geri al"""
        if self.current_step > 0:
            self.current_step -= 1
            return self.history[self.current_step]['image']
        return None
    
    def redo(self):
        """Son geri alınan işlemi yinele"""
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            return self.history[self.current_step]['image']
        return None
    
    def get_history(self):
        """İşlem geçmişini getir"""
        return [(item['operation'], item['timestamp']) for item in self.history]
    
    def save_image(self, image, format='PNG', quality=95):
        """Görüntüyü kaydet"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format, quality=quality)
        return img_byte_arr.getvalue()
    
    def batch_process(self, images, operations):
        """Toplu işlem uygula"""
        results = []
        for img in images:
            current_img = img
            for op_name, op_params in operations:
                if hasattr(self, op_name):
                    operation = getattr(self, op_name)
                    current_img = operation(current_img, **op_params)
            results.append(current_img)
        return results
    
    def create_custom_filter(self, name, operations):
        """Özel filtre oluştur"""
        self.custom_filters[name] = operations
    
    def apply_custom_filter(self, image, filter_name):
        """Özel filtreyi uygula"""
        if filter_name not in self.custom_filters:
            return image
            
        current_img = image
        for op_name, op_params in self.custom_filters[filter_name]:
            if hasattr(self, op_name):
                operation = getattr(self, op_name)
                current_img = operation(current_img, **op_params)
        return current_img
    
    def combine_effects(self, image, effects, weights=None):
        """Efekt kombinasyonu uygula"""
        if weights is None:
            weights = [1.0/len(effects)] * len(effects)
            
        results = []
        for effect, params in effects:
            if hasattr(self, effect):
                operation = getattr(self, effect)
                results.append(np.array(operation(image, **params)))
        
        # Ağırlıklı ortalama al
        final_result = np.zeros_like(results[0])
        for res, weight in zip(results, weights):
            final_result += res * weight
            
        return Image.fromarray(np.uint8(np.clip(final_result, 0, 255)))
    
    def load_image(self, image):
        if isinstance(image, (str, bytes, np.ndarray)):
            return Image.open(image)
        return image
    
    def enhance_image(self, image, brightness=1.0, contrast=1.0, sharpness=1.0):
        """Gelişmiş görüntü iyileştirme"""
        img = Image.fromarray(np.array(image))
        
        # Parlaklık ayarı
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
        
        # Kontrast ayarı
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        
        # Keskinlik ayarı
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)
        
        # Gürültü azaltma
        img_array = np.array(img)
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        
        return Image.fromarray(denoised)
    
    def apply_style_transfer(self, content_image, style_image, alpha=0.5):
        """Gelişmiş stil transferi"""
        content = np.array(content_image)
        style = np.array(style_image)
        
        content = tf.image.resize(content, (224, 224))
        style = tf.image.resize(style, (224, 224))
        
        content = tf.keras.applications.vgg16.preprocess_input(content)
        style = tf.keras.applications.vgg16.preprocess_input(style)
        
        content_features = self.style_model(tf.expand_dims(content, axis=0))
        style_features = self.style_model(tf.expand_dims(style, axis=0))
        
        result = content * (1-alpha) + style * alpha
        result = tf.clip_by_value(result, 0, 255)
        
        return Image.fromarray(np.uint8(result))
    
    def remove_background(self, image, alpha_matting=False):
        """Gelişmiş arka plan kaldırma"""
        img_array = np.array(image)
        output = remove(img_array, alpha_matting=alpha_matting)
        return Image.fromarray(output)
    
    def edit_face(self, image, smoothing=1.0, brightness=1.1, contrast=1.1):
        """Gelişmiş yüz düzenleme"""
        img_array = np.array(image)
        face_locations = face_recognition.face_locations(img_array)
        
        for (top, right, bottom, left) in face_locations:
            face_region = img_array[top:bottom, left:right]
            
            # Cildi yumuşat
            face_region = gaussian_filter(face_region, sigma=smoothing)
            
            # Parlaklık ve kontrast
            face_region = cv2.convertScaleAbs(face_region, alpha=contrast, beta=brightness*10)
            
            img_array[top:bottom, left:right] = face_region
        
        return Image.fromarray(img_array)
    
    def apply_artistic_filter(self, image, filter_type):
        """Sanatsal filtreler"""
        img = Image.fromarray(np.array(image))
        
        if filter_type == "Siyah Beyaz":
            return img.convert('L')
        elif filter_type == "Sepya":
            img_array = np.array(img.convert('RGB'))
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            sepia_img = cv2.transform(img_array, sepia_matrix)
            sepia_img = np.clip(sepia_img, 0, 255)
            return Image.fromarray(np.uint8(sepia_img))
        elif filter_type == "Bulanık":
            return img.filter(ImageFilter.BLUR)
        elif filter_type == "Kabartma":
            return img.filter(ImageFilter.EMBOSS)
        return img
    
    def add_watermark(self, image, text):
        """Filigran ekleme"""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Yarı saydam gri renk
        fillcolor = (128,128,128,128)
        
        # Metin boyutunu resim boyutuna göre ayarla
        fontsize = int(min(width, height) / 20)
        
        try:
            font = ImageFont.truetype("arial.ttf", fontsize)
        except:
            font = ImageFont.load_default()
        
        # Metni çapraz yerleştir
        textwidth, textheight = draw.textsize(text, font)
        x = width/2 - textwidth/2
        y = height/2 - textheight/2
        draw.text((x, y), text, font=font, fill=fillcolor)
        
        return img
    
    def add_frame(self, image, frame_size=10, frame_color=(255,255,255)):
        """Çerçeve ekleme"""
        img = Image.fromarray(np.array(image))
        width, height = img.size
        new_img = Image.new('RGB', (width + 2*frame_size, height + 2*frame_size), frame_color)
        new_img.paste(img, (frame_size, frame_size))
        return new_img
    
    def adjust_color_temperature(self, image, temperature):
        """Renk sıcaklığını ayarla"""
        img_array = np.array(image)
        
        # Sıcaklık değerini -100 (soğuk) ile 100 (sıcak) arasında al
        temperature = max(-100, min(100, temperature))
        
        # Renk kanallarını ayarla
        if temperature > 0:
            # Sıcak tonlar
            blue_factor = 1 - (temperature / 100) * 0.4
            red_factor = 1 + (temperature / 100) * 0.4
        else:
            # Soğuk tonlar
            blue_factor = 1 + (abs(temperature) / 100) * 0.4
            red_factor = 1 - (abs(temperature) / 100) * 0.4
            
        img_array[:,:,0] = np.clip(img_array[:,:,0] * red_factor, 0, 255)  # Kırmızı
        img_array[:,:,2] = np.clip(img_array[:,:,2] * blue_factor, 0, 255) # Mavi
        
        return Image.fromarray(np.uint8(img_array))
    
    def apply_vintage_effect(self, image):
        """Vintage efekti uygula"""
        img = np.array(image)
        
        # Renk doygunluğunu azalt
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 0.7  # Doygunluğu azalt
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Hafif sarımsı ton ekle
        img = img * [1.1, 1.0, 0.9]
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Vinyetleme efekti
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/4)
        kernel_y = cv2.getGaussianKernel(rows, rows/4)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        
        for i in range(3):
            img[:,:,i] = img[:,:,i] * mask
            
        return Image.fromarray(np.uint8(img))
    
    def create_double_exposure(self, image1, image2):
        """Çift pozlama efekti"""
        # İki görüntüyü aynı boyuta getir
        img1 = np.array(image1)
        img2 = np.array(image2.resize(image1.size))
        
        # İkinci görüntüyü siyah-beyaz yap ve kontrastını artır
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        img2_gray = exposure.adjust_gamma(img2_gray, 2.0)
        
        # Görüntüleri birleştir
        result = np.zeros_like(img1)
        for c in range(3):
            result[:,:,c] = cv2.addWeighted(img1[:,:,c], 0.7, img2_gray, 0.3, 0)
            
        return Image.fromarray(np.uint8(result))
    
    def apply_glitch_effect(self, image):
        """Glitch efekti uygula"""
        img = np.array(image)
        channels = []
        
        # Her renk kanalını ayrı ayrı işle
        for i in range(3):
            channel = img[:,:,i].copy()
            # Rastgele kaymalar ekle
            shift = random.randint(-20, 20)
            if shift > 0:
                channel[:,shift:] = channel[:,:-shift]
            else:
                channel[:,:shift] = channel[:,-shift:]
            channels.append(channel)
        
        # Kanalları birleştir
        glitched = np.stack(channels, axis=2)
        
        # Rastgele bloklar ekle
        for _ in range(10):
            x = random.randint(0, img.shape[1]-50)
            y = random.randint(0, img.shape[0]-50)
            w = random.randint(10, 50)
            h = random.randint(10, 50)
            glitched[y:y+h, x:x+w] = random.randint(0, 255)
            
        return Image.fromarray(np.uint8(glitched))
    
    def create_tilt_shift(self, image, blur_amount=3, focus_position=0.5, focus_width=0.2):
        """Tilt-shift efekti"""
        img = np.array(image)
        height = img.shape[0]
        
        # Bulanıklık maskesi oluştur
        mask = np.zeros((height, img.shape[1]))
        focus_center = int(height * focus_position)
        focus_range = int(height * focus_width)
        
        for i in range(height):
            if abs(i - focus_center) < focus_range:
                mask[i,:] = 1
            else:
                distance = abs(i - focus_center) - focus_range
                mask[i,:] = max(0, 1 - distance / (height * 0.1))
                
        # Orijinal ve bulanık görüntüyü birleştir
        blurred = cv2.GaussianBlur(img, (0,0), blur_amount)
        mask = np.stack([mask]*3, axis=2)
        result = img * mask + blurred * (1 - mask)
        
        return Image.fromarray(np.uint8(result))
    
    def create_kaleidoscope(self, image, segments=8):
        """Kaleydoskop efekti"""
        img = np.array(image)
        height, width = img.shape[:2]
        
        # Görüntüyü merkeze göre döndür
        center = (width//2, height//2)
        segment_angle = 360 / segments
        
        # İlk segmenti oluştur
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, center, (width//2, height//2), 0, 0, segment_angle, 255, -1)
        segment = cv2.bitwise_and(img, img, mask=mask)
        
        # Segmenti döndürerek kaleydoskop oluştur
        result = np.zeros_like(img)
        for i in range(segments):
            angle = i * segment_angle
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(segment, rotation_matrix, (width, height))
            result = cv2.add(result, rotated)
            
        return Image.fromarray(np.uint8(result))
    
    def create_cartoon(self, image):
        """Karikatür efekti"""
        img = np.array(image)
        
        # Kenarları belirginleştir
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        
        # Renkleri basitleştir
        color = cv2.bilateralFilter(img, 9, 300, 300)
        
        # Kenarları ve renkleri birleştir
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        return Image.fromarray(cartoon)
    
    def create_oil_painting(self, image, radius=4, levels=8):
        """Yağlı boya efekti"""
        img = np.array(image)
        
        # Görüntüyü HSV'ye dönüştür
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Yoğunluk histogramı hesapla
        shape = v.shape
        v_flat = v.flatten()
        intensity_level = np.linspace(0, 255, levels+1)
        
        # Her piksel için yoğunluk seviyesini hesapla
        oil_paint = np.zeros_like(img)
        for y in range(radius, shape[0]-radius):
            for x in range(radius, shape[1]-radius):
                kernel = v[y-radius:y+radius+1, x-radius:x+radius+1]
                kernel_h = h[y-radius:y+radius+1, x-radius:x+radius+1]
                kernel_s = s[y-radius:y+radius+1, x-radius:x+radius+1]
                
                # En yaygın yoğunluk seviyesini bul
                hist = np.histogram(kernel, bins=intensity_level)[0]
                max_level = np.argmax(hist)
                
                # Ortalama renk değerlerini hesapla
                mask = (kernel >= intensity_level[max_level]) & (kernel < intensity_level[max_level+1])
                if np.any(mask):
                    oil_paint[y,x,0] = np.mean(kernel_h[mask])
                    oil_paint[y,x,1] = np.mean(kernel_s[mask])
                    oil_paint[y,x,2] = np.mean(kernel[mask])
        
        return Image.fromarray(np.uint8(oil_paint))
    
    def create_mosaic(self, image, tile_size=20):
        """Mozaik efekti"""
        img = np.array(image)
        height, width = img.shape[:2]
        
        # Görüntüyü karelere böl
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                h = min(tile_size, height - y)
                w = min(tile_size, width - x)
                if h > 0 and w > 0:
                    tile = img[y:y+h, x:x+w]
                    # Karenin ortalama rengini hesapla
                    color = np.mean(tile, axis=(0,1))
                    # Kareyi tek renkle doldur
                    img[y:y+h, x:x+w] = color
                    
        return Image.fromarray(np.uint8(img))
    
    def create_neon_effect(self, image):
        """Neon efekti"""
        img = np.array(image)
        
        # Kenarları tespit et
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Kenarları kalınlaştır ve yumuşat
        edges = cv2.dilate(edges, None)
        edges = cv2.GaussianBlur(edges, (5,5), 0)
        
        # Renkli neon efekti
        result = np.zeros_like(img, dtype=np.float32)
        
        # Her kanal için farklı renk tonu
        result[:,:,0] = edges * 0.5  # Kırmızı
        result[:,:,1] = edges * 0.8  # Yeşil
        result[:,:,2] = edges        # Mavi
        
        # Parlaklık efekti
        result = cv2.GaussianBlur(result, (5,5), 0)
        result = np.clip(result * 2, 0, 255)
        
        return Image.fromarray(np.uint8(result))
    
    def create_pixel_sorting(self, image, threshold=128, direction='vertical'):
        """Piksel sıralama efekti"""
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if direction == 'vertical':
            for x in range(img.shape[1]):
                column = img[:,x]
                gray_column = gray[:,x]
                # Eşik değerinin üstündeki pikselleri sırala
                mask = gray_column > threshold
                if np.any(mask):
                    sorted_column = np.sort(column[mask], axis=0)
                    column[mask] = sorted_column
                    img[:,x] = column
        else:  # horizontal
            for y in range(img.shape[0]):
                row = img[y,:]
                gray_row = gray[y,:]
                mask = gray_row > threshold
                if np.any(mask):
                    sorted_row = np.sort(row[mask], axis=0)
                    row[mask] = sorted_row
                    img[y,:] = row
                    
        return Image.fromarray(np.uint8(img))
    
    def create_wave_distortion(self, image, amplitude=20, frequency=0.05):
        """Dalga distorsiyon efekti"""
        img = np.array(image)
        height, width = img.shape[:2]
        
        # Yeni görüntü matrisi
        result = np.zeros_like(img)
        
        # Her piksel için yeni pozisyon hesapla
        for y in range(height):
            for x in range(width):
                # Sinüs dalgası ile offset hesapla
                offset_x = int(amplitude * np.sin(2 * np.pi * frequency * y))
                # Yeni x pozisyonunu hesapla
                new_x = (x + offset_x) % width
                result[y, new_x] = img[y, x]
        
        return Image.fromarray(np.uint8(result))
    
    def create_rgb_shift(self, image, shift_amount=10):
        """RGB Kanal Kayması efekti"""
        img = np.array(image)
        height, width = img.shape[:2]
        
        # Her kanal için ayrı görüntü oluştur
        r_channel = np.zeros_like(img)
        g_channel = np.zeros_like(img)
        b_channel = np.zeros_like(img)
        
        # Kanalları kaydır
        r_channel[:-shift_amount, shift_amount:, 0] = img[shift_amount:, :-shift_amount, 0]  # Kırmızı
        g_channel[:, :, 1] = img[:, :, 1]  # Yeşil
        b_channel[shift_amount:, :-shift_amount, 2] = img[:-shift_amount, shift_amount:, 2]  # Mavi
        
        # Kanalları birleştir
        result = r_channel + g_channel + b_channel
        
        return Image.fromarray(np.uint8(result))
    
    def create_duotone(self, image, color1=(255,0,0), color2=(0,0,255)):
        """Duotone efekti"""
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        normalized = gray.astype(float) / 255
        
        # İki renk arasında interpolasyon
        result = np.zeros_like(img, dtype=float)
        for i in range(3):
            result[:,:,i] = normalized * color1[i] + (1-normalized) * color2[i]
            
        return Image.fromarray(np.uint8(np.clip(result, 0, 255)))

def main():
    st.set_page_config(layout="wide")
    st.title("AI Photoshop - Açık Kaynak Görüntü Düzenleme")
    
    # Sol sidebar için ayarlar
    st.sidebar.title("Ayarlar")
    
    # Çoklu görüntü yükleme
    uploaded_files = st.sidebar.file_uploader("Görüntü(ler) Yükleyin", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    editor = AIPhotoshop()
    
    if uploaded_files:
        # Aktif görüntüyü seç
        if len(uploaded_files) > 1:
            selected_image_index = st.sidebar.selectbox("Düzenlenecek Görüntüyü Seçin", range(len(uploaded_files)), format_func=lambda x: f"Görüntü {x+1}")
            image = Image.open(uploaded_files[selected_image_index])
        else:
            image = Image.open(uploaded_files[0])
        
        # Ana görüntüyü göster
        st.image(image, caption='Orijinal Görüntü', use_column_width=True)
        
        # Sekmeler oluştur
        tabs = st.tabs(["Temel Düzenleme", "Yüz Düzenleme", "Stil Transferi", 
                       "Sanatsal Efektler", "Özel Efektler", "Profesyonel Efektler",
                       "Deneysel Efektler", "Özel Filtreler", "Toplu İşlem"])
        
        # Temel Düzenleme Sekmesi
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                brightness = st.slider("Parlaklık", 0.0, 2.0, 1.0)
                contrast = st.slider("Kontrast", 0.0, 2.0, 1.0)
                sharpness = st.slider("Keskinlik", 0.0, 2.0, 1.0)
                
                if st.button('Görüntüyü İyileştir'):
                    enhanced_image = editor.enhance_image(image, brightness, contrast, sharpness)
                    st.image(enhanced_image, caption='İyileştirilmiş Görüntü')
            
            with col2:
                alpha_matting = st.checkbox("Gelişmiş Arka Plan Kaldırma")
                if st.button('Arka Planı Kaldır'):
                    no_bg_image = editor.remove_background(image, alpha_matting)
                    st.image(no_bg_image, caption='Arka Planı Kaldırılmış Görüntü')
        
        # Yüz Düzenleme Sekmesi
        with tabs[1]:
            smoothing = st.slider("Yumuşatma", 0.0, 2.0, 1.0)
            face_brightness = st.slider("Yüz Parlaklığı", 0.5, 1.5, 1.1)
            face_contrast = st.slider("Yüz Kontrastı", 0.5, 1.5, 1.1)
            
            if st.button('Yüz Düzenleme Uygula'):
                face_edited = editor.edit_face(image, smoothing, face_brightness, face_contrast)
                st.image(face_edited, caption='Yüz Düzenlenmiş Görüntü')
        
        # Stil Transferi Sekmesi
        with tabs[2]:
            style_file = st.file_uploader("Stil transferi için bir görüntü yükleyin", type=['png', 'jpg', 'jpeg'])
            if style_file is not None:
                style_image = Image.open(style_file)
                st.image(style_image, caption='Stil Görüntüsü', width=200)
                
                style_strength = st.slider("Stil Gücü", 0.0, 1.0, 0.5)
                if st.button('Stil Transferi Uygula'):
                    styled_image = editor.apply_style_transfer(image, style_image, style_strength)
                    st.image(styled_image, caption='Stil Transferi Uygulanmış Görüntü')
        
        # Sanatsal Efektler Sekmesi
        with tabs[3]:
            filter_type = st.selectbox(
                "Filtre Seç",
                ["Siyah Beyaz", "Sepya", "Bulanık", "Kabartma"]
            )
            
            if st.button('Filtreyi Uygula'):
                filtered_image = editor.apply_artistic_filter(image, filter_type)
                st.image(filtered_image, caption=f'{filter_type} Filtresi Uygulanmış Görüntü')
        
        # Özel Efektler Sekmesi
        with tabs[4]:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button('Vintage Efekti Uygula'):
                    vintage_image = editor.apply_vintage_effect(image)
                    st.image(vintage_image, caption='Vintage Efekti')
                
                if st.button('Glitch Efekti Uygula'):
                    glitch_image = editor.apply_glitch_effect(image)
                    st.image(glitch_image, caption='Glitch Efekti')
            
            with col2:
                second_image = st.file_uploader("Çift Pozlama için İkinci Görüntü", type=['png', 'jpg', 'jpeg'])
                if second_image is not None:
                    img2 = Image.open(second_image)
                    if st.button('Çift Pozlama Uygula'):
                        double_exp = editor.create_double_exposure(image, img2)
                        st.image(double_exp, caption='Çift Pozlama')
        
        # Profesyonel Efektler Sekmesi
        with tabs[5]:
            col1, col2 = st.columns(2)
            
            with col1:
                blur_amount = st.slider("Bulanıklık Miktarı", 1, 10, 3)
                focus_pos = st.slider("Odak Pozisyonu", 0.0, 1.0, 0.5)
                focus_width = st.slider("Odak Genişliği", 0.1, 0.5, 0.2)
                
                if st.button('Tilt-Shift Efekti Uygula'):
                    tilt_shift = editor.create_tilt_shift(image, blur_amount, focus_pos, focus_width)
                    st.image(tilt_shift, caption='Tilt-Shift Efekti')
                
                segments = st.slider("Segment Sayısı", 4, 16, 8)
                if st.button('Kaleydoskop Efekti Uygula'):
                    kaleidoscope = editor.create_kaleidoscope(image, segments)
                    st.image(kaleidoscope, caption='Kaleydoskop Efekti')
            
            with col2:
                if st.button('Karikatür Efekti Uygula'):
                    cartoon = editor.create_cartoon(image)
                    st.image(cartoon, caption='Karikatür Efekti')
                
                radius = st.slider("Fırça Yarıçapı", 1, 8, 4)
                levels = st.slider("Renk Seviyeleri", 4, 16, 8)
                if st.button('Yağlı Boya Efekti Uygula'):
                    oil_painting = editor.create_oil_painting(image, radius, levels)
                    st.image(oil_painting, caption='Yağlı Boya Efekti')
        
        # Deneysel Efektler Sekmesi
        with tabs[6]:
            col1, col2 = st.columns(2)
            
            with col1:
                tile_size = st.slider("Mozaik Boyutu", 5, 50, 20)
                if st.button('Mozaik Efekti Uygula'):
                    mosaic = editor.create_mosaic(image, tile_size)
                    st.image(mosaic, caption='Mozaik Efekti')
                
                if st.button('Neon Efekti Uygula'):
                    neon = editor.create_neon_effect(image)
                    st.image(neon, caption='Neon Efekti')
                
                wave_amp = st.slider("Dalga Genliği", 5, 50, 20)
                wave_freq = st.slider("Dalga Frekansı", 0.01, 0.1, 0.05)
                if st.button('Dalga Distorsiyonu Uygula'):
                    wave = editor.create_wave_distortion(image, wave_amp, wave_freq)
                    st.image(wave, caption='Dalga Distorsiyonu')
            
            with col2:
                sort_threshold = st.slider("Piksel Sıralama Eşiği", 0, 255, 128)
                sort_direction = st.selectbox("Sıralama Yönü", ['vertical', 'horizontal'])
                if st.button('Piksel Sıralama Uygula'):
                    sorted_pixels = editor.create_pixel_sorting(image, sort_threshold, sort_direction)
                    st.image(sorted_pixels, caption='Piksel Sıralama')
                
                rgb_shift = st.slider("RGB Kayma Miktarı", 1, 30, 10)
                if st.button('RGB Kayması Uygula'):
                    shifted = editor.create_rgb_shift(image, rgb_shift)
                    st.image(shifted, caption='RGB Kayması')
                
                color1 = st.color_picker("Birinci Renk", "#ff0000")
                color2 = st.color_picker("İkinci Renk", "#0000ff")
                if st.button('Duotone Efekti Uygula'):
                    # Renk kodlarını RGB'ye çevir
                    c1 = tuple(int(color1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    c2 = tuple(int(color2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    duotone = editor.create_duotone(image, c1, c2)
                    st.image(duotone, caption='Duotone Efekti')
        
        # Özel Filtreler Sekmesi
        with tabs[7]:
            st.subheader("Özel Filtre Oluştur")
            
            col1, col2 = st.columns(2)
            with col1:
                filter_name = st.text_input("Filtre Adı")
                selected_effects = st.multiselect("Efektler", 
                    ["enhance_image", "apply_vintage_effect", "create_neon_effect", 
                     "create_cartoon", "create_oil_painting", "create_mosaic"])
                
                if st.button("Filtre Oluştur"):
                    operations = []
                    for effect in selected_effects:
                        if effect == "enhance_image":
                            operations.append((effect, {"brightness": 1.2, "contrast": 1.1, "sharpness": 1.1}))
                        else:
                            operations.append((effect, {}))
                    editor.create_custom_filter(filter_name, operations)
                    st.success(f"{filter_name} filtresi oluşturuldu!")
            
            with col2:
                if editor.custom_filters:
                    selected_filter = st.selectbox("Kayıtlı Filtreleri Uygula", list(editor.custom_filters.keys()))
                    if st.button("Filtreyi Uygula"):
                        filtered = editor.apply_custom_filter(image, selected_filter)
                        st.image(filtered, caption=f'{selected_filter} Uygulanmış Görüntü')
        
        # Toplu İşlem Sekmesi
        with tabs[8]:
            if len(uploaded_files) > 1:
                st.subheader("Toplu İşlem")
                
                selected_ops = st.multiselect("Uygulanacak İşlemler",
                    ["Görüntü İyileştirme", "Vintage Efekti", "Karikatür", "Yağlı Boya"])
                
                if st.button("Toplu İşlem Uygula"):
                    operations = []
                    for op in selected_ops:
                        if op == "Görüntü İyileştirme":
                            operations.append(("enhance_image", {"brightness": 1.2, "contrast": 1.1, "sharpness": 1.1}))
                        elif op == "Vintage Efekti":
                            operations.append(("apply_vintage_effect", {}))
                        elif op == "Karikatür":
                            operations.append(("create_cartoon", {}))
                        elif op == "Yağlı Boya":
                            operations.append(("create_oil_painting", {}))
                    
                    images = [Image.open(f) for f in uploaded_files]
                    results = editor.batch_process(images, operations)
                    
                    cols = st.columns(3)
                    for idx, result in enumerate(results):
                        with cols[idx % 3]:
                            st.image(result, caption=f'Görüntü {idx+1}')
            else:
                st.info("Toplu işlem için birden fazla görüntü yükleyin.")
        
        # İşlem Geçmişi ve Geri Alma/Yineleme
        st.sidebar.subheader("İşlem Geçmişi")
        history = editor.get_history()
        if history:
            for op, timestamp in history:
                st.sidebar.text(f"{op} - {timestamp.strftime('%H:%M:%S')}")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("Geri Al"):
                    prev_image = editor.undo()
                    if prev_image:
                        image = prev_image
                        st.experimental_rerun()
            
            with col2:
                if st.button("Yinele"):
                    next_image = editor.redo()
                    if next_image:
                        image = next_image
                        st.experimental_rerun()
        
        # Görüntü Kaydetme Seçenekleri
        st.sidebar.subheader("Görüntüyü Kaydet")
        save_format = st.sidebar.selectbox("Format", ["PNG", "JPEG", "BMP"])
        quality = st.sidebar.slider("Kalite", 1, 100, 95)
        
        if st.sidebar.button("Kaydet"):
            img_bytes = editor.save_image(image, format=save_format, quality=quality)
            st.sidebar.download_button(
                label="İndir",
                data=img_bytes,
                file_name=f"edited_image.{save_format.lower()}",
                mime=f"image/{save_format.lower()}"
            )
        
        # Efekt Kombinasyonu
        st.sidebar.subheader("Efekt Kombinasyonu")
        combined_effects = st.sidebar.multiselect("Efektleri Seç",
            ["Vintage", "Neon", "Karikatür", "Yağlı Boya"])
        
        if st.sidebar.button("Efektleri Birleştir"):
            effects = []
            for effect in combined_effects:
                if effect == "Vintage":
                    effects.append(("apply_vintage_effect", {}))
                elif effect == "Neon":
                    effects.append(("create_neon_effect", {}))
                elif effect == "Karikatür":
                    effects.append(("create_cartoon", {}))
                elif effect == "Yağlı Boya":
                    effects.append(("create_oil_painting", {}))
            
            weights = [1.0/len(effects)] * len(effects)
            combined = editor.combine_effects(image, effects, weights)
            st.sidebar.image(combined, caption="Birleştirilmiş Efektler", use_column_width=True)

if __name__ == '__main__':
    main() 