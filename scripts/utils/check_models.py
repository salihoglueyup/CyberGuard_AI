# check_models.py

import google.generativeai as genai
import os

# .env'den API key'i oku
def load_env():
    env_vars = {}
    if os.path.exists('.env'):
        # encoding='utf-8' ekle ve errors='ignore'
        with open('.env', 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars

env = load_env()
api_key = env.get('GOOGLE_API_KEY')

if not api_key:
    print("❌ GOOGLE_API_KEY bulunamadı!")
    print("💡 .env dosyasını kontrol edin")
    exit(1)

print(f"🔑 API Key: {api_key[:10]}...{api_key[-5:]}")
print()

# Configure
genai.configure(api_key=api_key)

print("🤖 Mevcut Gemini Modelleri:\n")
print("="*70)

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ {model.name}")
            print(f"   Display: {model.display_name}")
            print()
except Exception as e:
    print(f"❌ Hata: {e}")
    print("\n💡 API Key geçersiz olabilir!")
    print("   Yeni bir key oluşturun: https://aistudio.google.com/app/apikey")

print("="*70)