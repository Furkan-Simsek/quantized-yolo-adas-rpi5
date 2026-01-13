import torch
from ultralytics import YOLO
import multiprocessing
import os

def main():
    # --- 1. Donanım ve Hızlandırma Ayarları ---
    if torch.cuda.is_available():
        print(f"🚀 Eğitim NVIDIA GPU üzerinde başlayacak: {torch.cuda.get_device_name(0)}")
        device = '0'
        workers = 8   # BDD100K büyük olduğu için veri yükleme hızı önemli (Linux'ta artırılabilir)
        batch_size = 16 # VRAM'ine göre burayı artır/azalt.
    elif torch.backends.mps.is_available():
        print("🍎 Eğitim Apple Silicon (Metal) üzerinde başlayacak.")
        device = 'mps'
        workers = 4
        batch_size = 16 
    else:
        print("⚠️ GPU bulunamadı! CPU kullanılıyor (Çok yavaş olacak).")
        device = 'cpu'
        workers = 0
        batch_size = 4

    # Windows için Safe Mode Kontrolü
    if os.name == 'nt': 
        print("🔧 Windows algılandı. Workers sayısı güvenli moda (4) çekiliyor.")
        # Workers=0 çok yavaştır. Windows'ta genelde 4 çalışır, hata verirse 0 yaparsın.
        workers = 4 

    # --- 2. Model Yükleme ---
    # RPi 5 için 'nano' (n) idealdir. Biraz daha başarım istersen 'small' (s) deneyebilirsin ama FPS düşer.
    model = YOLO('yolo11n.pt') 

    # --- 3. Eğitim Başlatma ---
    print(f"🎯 Eğitim Başlıyor... Batch Size: {batch_size}, Workers: {workers}")
    
    model.train(
        data='bdd100k/data.yaml',    # Aşağıda vereceğim YAML dosyasının adı
        epochs=100,             # BDD100K için 100 iyidir, veri çoksa 50 bile yetebilir.
        imgsz=640,
        batch=batch_size,       # Senin hesapladığın dinamik batch size'ı buraya bağladım!
        device=device,
        workers=workers,
        project='adas_training_nano',
        name='bdd100k_v11_nano', # İsimlendirmeyi veri setine uygun yaptım
        cache=True,            # DİKKAT: BDD100K 100 bin fotoğraftır. RAM'in 64GB+ değilse bunu False yap, yoksa RAM taşar!
        amp=True,               
        exist_ok=True,
        patience=20,            # Veri seti zorlu olduğu için sabrı biraz artırdım (10 -> 20)
        optimizer="AdamW",
        plots=True,
    )

    # --- 4. RPi 5 için Export ---
    print("📦 Model NCNN formatına dönüştürülüyor (RPi için)...")
    # NCNN, Raspberry Pi üzerindeki Vulkan GPU hızlandırması için en iyisidir.
    model.export(format='ncnn', imgsz=640)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
