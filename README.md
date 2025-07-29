<div align="center">

🤖 Yapay Zeka Destekli Diyalogsal Ürün Öneri Sistemi
Kullanıcıların niyetlerini, duygularını ve problemlerini anlayarak onlara anlamlı ve kişiselleştirilmiş ürün önerileri sunan akıllı bir alışveriş asistanı.

</div>

<div align="center">

</div>

Bu proje, standart anahtar kelime tabanlı aramaların ötesine geçerek, bir kullanıcının "huzurlu bir ortam istiyorum" gibi soyut veya "uykusuzluk çekiyorum" gibi problem odaklı ifadelerinden yola çıkarak asıl ihtiyacını anlayan ve buna yönelik öneriler sunan bir sistemdir.

🌟 Temel Yetenekler
🧠 Derin Anlam Çıkarma: Kullanıcı girdilerindeki soyut kavramları ve problemleri anlar.

🕸️ Bilgi Grafiği (Knowledge Graph) Mimarisi: Ürünler, kavramlar ve aralarındaki karmaşık ilişkiler Neo4j graf veritabanı üzerinde modellenmiştir.

🔗 Akıllı İlişki Analizi:

İYİ_GİDER: Birbiriyle uyumlu ürünleri önererek çapraz satış fırsatları yaratır.

ÇÖZÜM_OLABİLİR: Belirli bir probleme çözüm sunan ürünleri doğrudan hedefler.

💬 Diyalogsal Etkileşim: Google Gemini API kullanarak, önerileri daraltmak ve kullanıcıyı daha iyi anlamak için bağlama uygun, akıllı takip soruları üretir.

🛠️ Teknoloji Yığını
Bileşen

Teknoloji

Ana Dil



Veritabanı



Yapay Zeka



Kütüphaneler

py2neo, pandas, google-generativeai

⚙️ Sistem Mimarisi
Sistem, kullanıcının girdisinden nihai öneriye giden yolda aşağıdaki adımları izler:

Girdi: Kullanıcı, doğal dilde bir istek girer (örn: "uyumama yardımcı olacak bir şeyler arıyorum").

Anlama (NLU): Kullanıcının girdisi, kavramları (Uykusuzluk), niyeti (problem_çözme) ve duyguyu çıkarmak üzere Google Gemini API'ye gönderilir.

Sorgu Oluşturma: Python betiği, NLU çıktısını kullanarak Neo4j graf veritabanını sorgulamak için dinamik bir Cypher sorgusu oluşturur.

Graf Analizi: Oluşturulan sorgu, Neo4j üzerinde ÇÖZÜM_OLABİLİR gibi anlamsal ilişkileri takip ederek en alakalı ürünleri ve alaka skorlarını bulur.

Öneri Sunma: Bulunan ürünler kullanıcıya sunulur (örn: Akşam Çayı Seti, Lavanta Kokulu Mum).

Diyalog Yönetimi: Sistem, konuşmanın bağlamına göre kullanıcıya bir sonraki adımı sormak için "Bu ürünlerin yanında, odanızda yumuşak bir ışık ister misiniz?" gibi akıllı bir takip sorusu üretir.

🚀 Projeyi Başlatma
Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1. Ön Gereksinimler
Python 3.8 veya üstü

Docker (Neo4j'i kolayca kurmak için önerilir)

Git

2. Kurulum
Repoyu Klonlayın:

git clone https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
cd REPO_ADINIZ

Python Sanal Ortamı Oluşturun ve Aktive Edin:

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

Gerekli Kütüphaneleri Yükleyin:

pip install -r requirements.txt

Not: Henüz bir requirements.txt dosyanız yoksa, sanal ortam aktifken pip freeze > requirements.txt komutuyla oluşturabilirsiniz.

Neo4j Veritabanını Başlatın (Docker ile):

docker run \
    --name neo4j-ai-project \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/test_password \
    neo4j:latest

Bu komut, test_password şifresiyle bir Neo4j veritabanı başlatır. Veritabanı arayüzüne http://localhost:7474 adresinden erişebilirsiniz.

API Anahtarını Ayarlayın:
Proje ana dizininde .env adında bir dosya oluşturun ve içine Google Gemini API anahtarınızı ekleyin:

GOOGLE_API_KEY="BURAYA_API_ANAHTARINIZI_YAPISTIRIN"

Kodumuz bu anahtarı otomatik olarak ortam değişkeni olarak yükleyecektir. .gitignore dosyamız bu dosyanın GitHub'a gönderilmesini engelleyecektir.

3. Kullanım
Tüm kurulum adımları tamamlandıktan sonra, ana Python betiğini çalıştırarak interaktif konsolu başlatabilirsiniz:

python app.py

Sistem sizi bir "Merhaba!" mesajıyla karşılayacak ve sizden ilk isteğinizi girmenizi bekleyecektir.

📈 Gelecek Geliştirmeler
[ ] Kullanıcı Profilleri: Her kullanıcı için bir profil oluşturarak geçmiş tercihlerine ve davranışlarına göre daha derin kişiselleştirme sağlamak.

[ ] Esnek Diyalog Yönetimi: Kullanıcının mevcut önerileri daraltmasına ("daha ucuz olanları göster") veya karşılaştırma yapmasına olanak tanımak.

[ ] Görsel Entegrasyon: Ürün önerileriyle birlikte image_url kullanarak görsellerini de göstermek.

[ ] Web Arayüzü: Projeyi bir web arayüzü (Flask veya FastAPI ile) üzerinden sunmak.

🤝 Katkıda Bulunma
Katkılarınız projenin gelişimi için çok değerlidir! Lütfen bir "issue" açarak veya "pull request" göndererek katkıda bulunun.

Projeyi Fork'layın.

Yeni bir özellik dalı oluşturun (git checkout -b ozellik/HarikaBirOzellik).

Değişikliklerinizi Commit'leyin (git commit -m 'Harika bir özellik eklendi').

Dalınızı Push'layın (git push origin ozellik/HarikaBirOzellik).

Bir Pull Request açın.

📄 Lisans
Bu proje MIT Lisansı altında lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasına bakınız.
