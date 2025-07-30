import pandas as pd
from py2neo import Graph, Node, Relationship
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os



products_data = {
    'product_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010', 'P011', 'P012'],
    'product_name': [
        'Lavanta Kokulu Mum', 'Yumuşak Işık Lambader', 'Saksı Salon Bitkisi',
        'Pamuklu Battaniye', 'Ahşap Tütsülük Seti', 'Minimalist Seramik Vazo',
        'Doğal Taş Detaylı Yastık', 'Akşam Çayı Seti', 'Zen Bahçesi Kiti',
        'Aromaterapi Difüzörü', 'Portakal Çiçeği Oda Spreyi', 'Sedir Ağacı Tütsü Koku Kartı'
    ],
    'description': [
        'Evde huzurlu bir atmosfer yaratmak için ideal lavanta kokulu mum. Uzun süreli yanma sağlar.',
        'Okuma köşeniz veya yatak odanız için sıcak ve yumuşak ışık sağlayan modern tasarımlı lambader.',
        'Oksijen üreten ve havayı temizleyen, bakımı kolay salon bitkisi. İç mekanlara doğal bir dokunuş katar.',
        'Sıcak ve konforlu bir his veren %100 pamuklu, yumuşacık battaniye. Soğuk kış akşamları için mükemmel.',
        'Meditasyon ve rahatlama için tasarlanmış ahşap tütsülük ve çeşitli koku çubukları.',
        'Modern ve sade bir estetiğe sahip, el yapımı seramik vazo. Tek bir çiçekle bile şık durur.',
        'Doğal taş işlemeleriyle dikkat çeken, yumuşak dokulu dekoratif yastık. Otantik ve sakin bir hava katar.',
        'Uyku öncesi rahatlamak için ideal, sakinleştirici bitki çayları ve zarif porselen fincanlardan oluşan set.',
        'Masanızda veya çalışma alanınızda sakinlik ve odaklanma sağlamak için minyatür zen bahçesi.',
        'Esansiyel yağlarla birlikte kullanarak ortamdaki havayı güzelleştiren ve ruh halini dengeleyen difüzör.',
        'Portakal çiçeği kokulu, anında ferahlık veren oda spreyi. Çiçeksi ve narenciye notaları içerir.',
        'Sedir ağacı kokulu, odunsu ve sakinleştirici notalara sahip tütsü koku kartı. Evde huzurlu bir ortam için.'
    ],
    'category': [
        'Ev Dekoru', 'Aydınlatma', 'Bitki', 'Tekstil', 'Ev Dekoru',
        'Ev Dekoru', 'Tekstil', 'Mutfak & Sofra', 'Ev Dekoru', 'Ev Gereçleri',
        'Ev Gereçleri', 'Ev Dekoru'
    ],
    'sub_category': [
        'Mumlar', 'Lambaderler', 'Salon Bitkileri', 'Battaniyeler', 'Tütsüler',
        'Vazolar', 'Yastıklar', 'Çay Setleri', 'Bahçe Kitleri', 'Difüzörler',
        'Oda Spreyi', 'Tütsüler'
    ],
    'price': [45.00, 320.00, 80.00, 150.00, 75.00, 60.00, 95.00, 120.00, 110.00, 180.00, 65.00, 50.00],
    'image_url': [
        'url_mum.jpg', 'url_lambader.jpg', 'url_bitki.jpg', 'url_battaniye.jpg',
        'url_tutsu.jpg', 'url_vazo.jpg', 'url_yastik.jpg', 'url_cay.jpg',
        'url_zen.jpg', 'url_difuzor.jpg', 'url_sprey.jpg', 'url_koku_karti.jpg'
    ]
}
products_df = pd.DataFrame(products_data)

concepts = [
    {"name": "Huzur", "type": "Duygu/Atmosfer"}, {"name": "Sakinlik", "type": "Duygu/Atmosfer"},
    {"name": "Odaklanma", "type": "Duygu/Atmosfer"}, {"name": "Konfor", "type": "Duygu/Atmosfer"},
    {"name": "Doğallık", "type": "Özellik/Estetik"}, {"name": "Modern", "type": "Özellik/Estetik"},
    {"name": "Meditasyon", "type": "Aktivite"}, {"name": "Okuma", "type": "Aktivite"},
    {"name": "Uykusuzluk", "type": "Sorun"}, {"name": "Pozitif Enerji", "type": "Duygu/Atmosfer"},
    {"name": "Minimalist", "type": "Özellik/Estetik"}, {"name": "Otantik", "type": "Özellik/Estetik"},
    {"name": "Lavanta Kokusu", "type": "Koku"}, {"name": "Yumuşak Işık", "type": "Işık"},
    {"name": "Pamuklu", "type": "Malzeme"}, {"name": "Yumuşak Doku", "type": "Dokunsal"},
    {"name": "Ahşap", "type": "Malzeme"}, {"name": "Seramik", "type": "Malzeme"},
    {"name": "Doğal Taş", "type": "Malzeme"}, {"name": "Bitki Çayı", "type": "İçerik"},
    {"name": "Minyatür Bahçe", "type": "Form"}, {"name": "Esansiyel Yağ", "type": "Kullanım"},
    {"name": "Huzurlu", "type": "Duygu/Atmosfer"}, {"name": "Sakinleştirici", "type": "Duygu/Atmosfer"},
    {"name": "Odaklan", "type": "Duygu/Atmosfer"}, {"name": "Rahat", "type": "Duygu/Atmosfer"},
    {"name": "Doğal", "type": "Özellik/Estetik"}, {"name": "Minimal", "type": "Özellik/Estetik"},
    {"name": "Ortam", "type": "Genel Kavram"}, {"name": "Ev", "type": "Mekan"},
    {"name": "Dekorasyon", "type": "Aktivite"}, {"name": "Renk", "type": "Özellik"},
    {"name": "Kırmızı", "type": "Renk"}, {"name": "Mavi", "type": "Renk"}, {"name": "Yeşil", "type": "Renk"},
    {"name": "Metal", "type": "Malzeme"}, {"name": "Çiçeksi Koku", "type": "Koku Profili"},
    {"name": "Odunsu Koku", "type": "Koku Profili"}, {"name": "Narenciye Kokusu", "type": "Koku Profili"},
    {"name": "Sprey", "type": "Form"}, {"name": "Koku Kartı", "type": "Form"},
    {"name": "Ferahlık", "type": "Duygu/Atmosfer"}, {"name": "Enerji", "type": "Duygu/Atmosfer"},
    {"name": "Fonksiyonel", "type": "Özellik"}, {"name": "Dekoratif", "type": "Özellik"},
    {"name": "Yatak Odası", "type": "Mekan"}, {"name": "Salon", "type": "Mekan"},
    {"name": "Okuma Köşesi", "type": "Mekan"}
]

relationships_data = [
    ("Huzur", "İLİŞKİLİDİR", "Sakinlik", {"weight": 0.9}), ("Sakinlik", "İLİŞKİLİDİR", "Meditasyon", {"weight": 0.8}),
    ("Huzur", "SAĞLAR", "Pozitif Enerji", {"weight": 0.7}), ("Odaklanma", "İLİŞKİLİDİR", "Minimalist", {"weight": 0.7}),
    ("Huzurlu", "İLİŞKİLİDİR", "Huzur", {"weight": 0.95}), ("Sakinleştirici", "İLİŞKİLİDİR", "Sakinlik", {"weight": 0.95}),
    ("Rahat", "İLİŞKİLİDİR", "Konfor", {"weight": 0.9}), ("Doğal", "İLİŞKİLİDİR", "Doğallık", {"weight": 0.95}),
    ("Minimal", "İLİŞKİLİDİR", "Minimalist", {"weight": 0.95}), ("Ortam", "SAĞLAR", "Huzur", {"strength": 0.3}),
    ("Ev", "SAĞLAR", "Konfor", {"strength": 0.4}), ("Ev", "İLİŞKİLİDİR", "Dekorasyon", {"weight": 0.6}),
    ("Kırmızı", "İLİŞKİLİDİR", "Renk", {"weight": 0.9}), ("Çiçeksi Koku", "SAĞLAR", "Ferahlık", {"strength": 0.7}),
    ("Odunsu Koku", "SAĞLAR", "Sakinlik", {"strength": 0.8}), ("Narenciye Kokusu", "SAĞLAR", "Enerji", {"strength": 0.7}),
    ("Sprey", "İLİŞKİLİDİR", "Ev Gereçleri", {"weight": 0.6}), ("Koku Kartı", "İLİŞKİLİDİR", "Ev Dekoru", {"weight": 0.6}),
    ("Yumuşak Işık Lambader", "SAĞLAR", "Okuma Köşesi", {"strength": 0.8}), ("Yumuşak Işık Lambader", "SAĞLAR", "Yatak Odası", {"strength": 0.7}),
    ("Yumuşak Işık Lambader", "SAHİPTİR", "Fonksiyonel", {"strength": 0.9}), ("Yumuşak Işık Lambader", "SAĞLAR", "Sakinlik", {"strength": 0.7}),
    ("Fonksiyonel", "İLİŞKİLİDİR", "Okuma", {"weight": 0.7}), ("Dekoratif", "İLİŞKİLİDİR", "Estetik", {"weight": 0.7}),
    ("Aydınlatma", "SAĞLAR", "Fonksiyonel", {"strength": 0.6}), ("Aydınlatma", "SAĞLAR", "Dekoratif", {"strength": 0.6}),
    ("Lavanta Kokulu Mum", "SAĞLAR", "Huzur", {"strength": 0.9, "reason": "koku"}), ("Yumuşak Işık Lambader", "SAĞLAR", "Sakinlik", {"strength": 0.7, "reason": "aydınlatma"}),
    ("Saksı Salon Bitkisi", "SAĞLAR", "Doğallık", {"strength": 0.9, "reason": "bitki"}), ("Saksı Salon Bitkisi", "SAĞLAR", "Huzur", {"strength": 0.6, "reason": "doğal varlık"}),
    ("Pamuklu Battaniye", "SAĞLAR", "Konfor", {"strength": 0.9, "reason": "dokunsal"}), ("Ahşap Tütsülük Seti", "SAĞLAR", "Meditasyon", {"strength": 0.9, "reason": "ritual"}),
    ("Minimalist Seramik Vazo", "SAHİPTİR", "Minimalist", {"strength": 0.9}), ("Minimalist Seramik Vazo", "SAĞLAR", "Estetik", {"strength": 0.7}),
    ("Doğal Taş Detaylı Yastık", "SAĞLAR", "Otantik", {"strength": 0.8}), ("Akşam Çayı Seti", "SAĞLAR", "Sakinlik", {"strength": 0.7, "reason": "içecek"}),
    ("Zen Bahçesi Kiti", "SAĞLAR", "Odaklanma", {"strength": 0.9, "reason": "aktivite"}), ("Aromaterapi Difüzörü", "SAĞLAR", "Huzur", {"strength": 0.9, "reason": "koku"}),
    ("Aromaterapi Difüzörü", "SAĞLAR", "Sakinlik", {"strength": 0.7, "reason": "koku"}), ("Lavanta Kokulu Mum", "SAHİPTİR", "Lavanta Kokusu", {"type": "koku"}),
    ("Yumuşak Işık Lambader", "SAHİPTİR", "Yumuşak Işık", {"type": "ışık"}), ("Saksı Salon Bitkisi", "SAHİPTİR", "Doğal", {"type": "malzeme"}),
    ("Pamuklu Battaniye", "SAHİPTİR", "Pamuklu", {"type": "malzeme"}), ("Pamuklu Battaniye", "SAHİPTİR", "Yumuşak Doku", {"type": "dokunsal"}),
    ("Ahşap Tütsülük Seti", "SAHİPTİR", "Ahşap", {"type": "malzeme"}), ("Minimalist Seramik Vazo", "SAHİPTİR", "Seramik", {"type": "malzeme"}),
    ("Minimalist Seramik Vazo", "SAHİPTİR", "Minimalist", {"type": "stil"}), ("Doğal Taş Detaylı Yastık", "SAHİPTİR", "Doğal Taş", {"type": "malzeme"}),
    ("Akşam Çayı Seti", "SAHİPTİR", "Bitki Çayı", {"type": "içerik"}), ("Zen Bahçesi Kiti", "SAHİPTİR", "Minyatür Bahçe", {"type": "form"}),
    ("Aromaterapi Difüzörü", "SAHİPTİR", "Esansiyel Yağ", {"type": "kullanım"}), ("Portakal Çiçeği Oda Spreyi", "SAĞLAR", "Ferahlık", {"strength": 0.8, "reason": "koku"}),
    ("Portakal Çiçeği Oda Spreyi", "SAHİPTİR", "Çiçeksi Koku", {"type": "koku_profili"}), ("Portakal Çiçeği Oda Spreyi", "SAHİPTİR", "Narenciye Kokusu", {"type": "koku_profili"}),
    ("Portakal Çiçeği Oda Spreyi", "SAHİPTİR", "Sprey", {"type": "form"}), ("Sedir Ağacı Tütsü Koku Kartı", "SAĞLAR", "Sakinlik", {"strength": 0.8, "reason": "koku"}),
    ("Sedir Ağacı Tütsü Koku Kartı", "SAHİPTİR", "Odunsu Koku", {"type": "koku_profili"}), ("Sedir Ağacı Tütsü Koku Kartı", "SAHİPTİR", "Koku Kartı", {"type": "form"}),
]

# --- 2. Neo4j Veri Aktarımı
def import_data_to_neo4j_for_testing_updated(products_df, concepts, relationships_data, graph_conn):
    try:
        graph_conn.run("MATCH (n) DETACH DELETE n")
        print("Neo4j bağlantısı kuruldu ve mevcut veriler temizlendi (test amaçlı).")
    except Exception as e:
        print(f"Neo4j bağlantı hatası: {e}")
        print("Lütfen Neo4j Docker konteynerinizin çalıştığından ve kimlik bilgilerinizin doğru olduğundan emin olun.")
        return False

    product_nodes = {}
    concept_nodes = {}

    for index, row in products_df.iterrows():
        node = Node("Product", product_id=row['product_id'], name=row['product_name'], description=row['description'], category=row['category'], sub_category=row['sub_category'], price=row['price'], image_url=row['image_url'])
        graph_conn.create(node)
        product_nodes[row['product_name']] = node

    for concept in concepts:
        node = Node("Concept", name=concept["name"], type=concept["type"])
        graph_conn.create(node)
        concept_nodes[concept["name"]] = node

    relationships_created_count = 0
    for rel_data in relationships_data:
        source_name, rel_type, target_name, properties = rel_data
        source_node = product_nodes.get(source_name) or concept_nodes.get(source_name)
        target_node = concept_nodes.get(target_name)

        if not target_node and target_name in product_nodes:
            target_node = product_nodes.get(target_name)

        if source_node and target_node:
            relationship = Relationship(source_node, rel_type, target_node, **properties)
            graph_conn.create(relationship)
            relationships_created_count += 1
        else:
            pass
    print(f"Neo4j'ye {len(product_nodes)} ürün düğümü, {len(concept_nodes)} kavram düğümü ve {relationships_created_count} ilişki aktarıldı.")
    return True


# --- 3. Doğal Dil Anlama (NLU) Modülü  ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('models/gemini-2.5-pro')
    print("Gemini API Yapılandırıldı ve Model Yüklendi.")
except KeyError:
    print("HATA: GOOGLE_API_KEY ortam değişkeni ayarlanmamış. Lütfen API anahtarınızı ayarlayın.")
    exit()
except Exception as e:
    print(f"Gemini API yüklenirken bir hata oluştu: {e}")
    exit()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_nlu_with_gemini(user_input, defined_concepts_list, possible_intents):
    all_known_terms = defined_concepts_list + products_df['product_name'].tolist()
    prompt_text = f"""
    Sen bir alışveriş asistanısın. Kullanıcının "{user_input}" isteğini analiz et ve aşağıdaki bilgileri bir JSON formatında döndür:
    - user_intent (kullanıcının amacı): {', '.join(possible_intents)} listesinden en uygun olanı seç.
    - sentiment (duygu): 'positive', 'negative', 'neutral'
    - extracted_concepts (çıkarılan anahtar kavramlar): Kullanıcının ifadesindeki temel kavramlar, duygular, atmosferler, stiller, materyaller, koku profilleri, ürün formları veya spesifik ürün adları. Lütfen Türkçe kelimeler kullan. İşte bilinen bazı terimler: {', '.join(all_known_terms)}.
    - product_attributes (çıkarılan ürün özellikleri): Kullanıcının bahsettiği spesifik özellikler ve değerleri. (örn. "malzeme: ahşap", "renk: kırmızı", "oda: yatak odası").
    
    Örnek JSON Çıktısı:
    {{
      "user_intent": "atmosfer_yaratma",
      "sentiment": "positive",
      "extracted_concepts": ["huzur", "sakinlik", "doğal"],
      "product_attributes": []
    }}
    Lütfen yalnızca JSON çıktısını sağla ve başka hiçbir metin ekleme.
    """
    try:
        response = gemini_model.generate_content(prompt_text, request_options={"timeout": 120})
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        nlu_result = json.loads(json_str)
        
        all_defined_keywords = set(defined_concepts_list + products_df['product_name'].tolist())
        processed_known_terms = {preprocess_text(k) for k in all_defined_keywords}
        
        filtered_concepts = []
        if "extracted_concepts" in nlu_result and nlu_result["extracted_concepts"] is not None:
             for c in nlu_result.get("extracted_concepts", []):
                if preprocess_text(c) in processed_known_terms:
                    original_concept_name = next((k for k in all_defined_keywords if preprocess_text(k) == preprocess_text(c)), c)
                    filtered_concepts.append(original_concept_name)
        
        nlu_result["extracted_concepts"] = list(set(filtered_concepts))
        return nlu_result
    except Exception as e:
        print(f"Gemini NLU analizi hatası: {e}")
        return {"user_intent": "bilgi_sorgulama", "sentiment": "neutral", "extracted_concepts": [], "product_attributes": []}

# --- 4. Öneri Motoru Fonksiyonu (CYPHER SORGUSU DÜZELTİLDİ) ---
def get_recommendations(nlu_result, num_recommendations=5):
    try:
        graph_recommendation = Graph("bolt://localhost:7687", auth=("neo4j", "test_password"))
    except Exception as e:
        print(f"Hata: Neo4j öneri için bağlanamadı: {e}")
        return []

    extracted_concepts = nlu_result.get("extracted_concepts", [])
    product_attributes = nlu_result.get("product_attributes", [])
    
    print(f"NLU Analizi - Çıkarılan Kavramlar: {extracted_concepts}")
    print(f"NLU Analizi - Çıkarılan Özellikler: {product_attributes}")

    all_query_union_parts = [] 

    # 1. Kavramlara göre ürünleri bulma sorgusu
    if extracted_concepts:
        escaped_concepts = [c.replace("'", "\\'") for c in extracted_concepts]
        concept_list_str = "[" + ", ".join([f"'{c_escaped}'" for c_escaped in escaped_concepts]) + "]"

        concept_based_query_part = f"""
        UNWIND {concept_list_str} AS concept_name
        MATCH (c:Concept {{name: concept_name}})
        CALL {{
            WITH c
            OPTIONAL MATCH (c)<-[r1:SAĞLAR|SAHİPTİR]-(p1:Product)
            WHERE p1 IS NOT NULL
            RETURN p1 AS product, (coalesce(r1.strength, 0.0) + coalesce(r1.weight, 0.0)) AS score
            UNION
            WITH c
            OPTIONAL MATCH (c)-[:İLİŞKİLİDİR]->(:Concept)<-[r2:SAĞLAR|SAHİPTİR]-(p2:Product)
            WHERE p2 IS NOT NULL
            RETURN p2 AS product, ((coalesce(r2.strength, 0.0) + coalesce(r2.weight, 0.0)) * 0.5) AS score
        }}
        RETURN product AS p, score AS total_score
        """
        all_query_union_parts.append(concept_based_query_part)

    # 2. Ürün özelliklerine göre filtreleme/skorlama sorguları
    if product_attributes:
        for attr_str in product_attributes:
            if ":" in attr_str:
                key, value = attr_str.split(":", 1)
                key = key.strip().lower()
                value = value.strip().replace("'", "\\'")
                if key == "renk":
                    all_query_union_parts.append(f"MATCH (p:Product)-[:SAHİPTİR]->(:Concept {{name: '{value}', type: 'Renk'}}) RETURN p, 0.8 AS total_score")
                elif key == "malzeme":
                    all_query_union_parts.append(f"MATCH (p:Product)-[:SAHİPTİR]->(:Concept {{name: '{value}', type: 'Malzeme'}}) RETURN p, 0.7 AS total_score")
                elif key == "koku":
                    all_query_union_parts.append(f"MATCH (p:Product)-[:SAHİPTİR]->(:Concept {{name: '{value}', type: 'Koku Profili'}}) RETURN p, 0.9 AS total_score")
                    all_query_union_parts.append(f"MATCH (p:Product)-[:SAHİPTİR]->(:Concept {{name: '{value}', type: 'Koku'}}) RETURN p, 0.9 AS total_score")
                elif key == "form":
                    all_query_union_parts.append(f"MATCH (p:Product)-[:SAHİPTİR]->(:Concept {{name: '{value}', type: 'Form'}}) RETURN p, 0.7 AS total_score")
                elif key == "kategori":
                    all_query_union_parts.append(f"MATCH (p:Product) WHERE p.category = '{value}' RETURN p, 0.6 AS total_score")
                elif key == "özellik":
                    all_query_union_parts.append(f"MATCH (p:Product)-[:SAHİPTİR]->(:Concept {{name: '{value}', type: 'Özellik'}}) RETURN p, 0.6 AS total_score")
                elif key == "oda":
                    all_query_union_parts.append(f"MATCH (p:Product)-[:SAĞLAR]->(:Concept {{name: '{value}', type: 'Mekan'}}) RETURN p, 0.7 AS total_score")

    # 3. Nihai Cypher Sorgusu Oluşturma
    if not all_query_union_parts: 
        print("Uyarı: Anlamlı kavram veya özellik çıkarılamadı. Varsayılan öneriler sunuluyor.")
        final_cypher_query = f"MATCH (p:Product) RETURN p, 0.0 AS total_score ORDER BY p.price DESC LIMIT {num_recommendations}"
    else:
        # HATA DÜZELTİLDİ: Her bir sorgu parçasını çevreleyen gereksiz parantezler kaldırıldı.
        combined_queries = " UNION ALL ".join([part.strip() for part in all_query_union_parts])
        
        final_cypher_query = f"""
        {combined_queries}
        WITH p, SUM(total_score) AS aggregated_total_score
        RETURN p, aggregated_total_score AS total_score
        ORDER BY total_score DESC, p.price DESC
        LIMIT {num_recommendations}
        """

    print(f"\nOluşturulan Cypher Sorgusu:\n{final_cypher_query}\n")

    try:
        results = graph_recommendation.run(final_cypher_query)
        recommendations = []
        for record in results:
            product = record["p"]
            recommendations.append({
                "id": product["product_id"], "name": product["name"],
                "description": product["description"], "category": product["category"],
                "price": product["price"], "image_url": product["image_url"],
                "relevance_score": record["total_score"]
            })
        return recommendations
    except Exception as e:
        print(f"Neo4j sorgu hatası: {e}")
        print("Sorgu hatası oluştu, genel öneriler sunuluyor.")
        fallback_query = f"MATCH (p:Product) RETURN p, 0.0 AS total_score ORDER BY p.price DESC LIMIT {num_recommendations}"
        try:
            results = graph_recommendation.run(fallback_query)
            recommendations = []
            for record in results:
                product = record["p"]
                recommendations.append({
                    "id": product["product_id"], "name": product["name"],
                    "description": product["description"], "category": product["category"],
                    "price": product["price"], "image_url": product["image_url"],
                    "relevance_score": record["total_score"]
                })
            return recommendations
        except Exception as fallback_e:
            print(f"Fallback sorgusu da başarısız oldu: {fallback_e}")
            return []

# --- 5. Prompt Chaining ve Ana Çalışma Bloğu (Değişiklik yok) ---
# Bu kısımlar olduğu gibi kalabilir.

def get_follow_up_question_gemini(nlu_result, recommended_products_count, turn_history, previous_query_analysis=None):
    user_intent = nlu_result.get("user_intent", "bilgi_sorgulama")
    extracted_concepts = nlu_result.get("extracted_concepts", [])
    sentiment = nlu_result.get("sentiment", "neutral")
    prompt_messages = []
    for entry in turn_history:
        prompt_messages.append({"role": entry["role"], "parts": [entry["parts"]]})
    system_prompt = f"""
    Sen bir alışveriş asistanısın. Kullanıcının durumu: Niyet: {user_intent}, Kavramlar: {', '.join(extracted_concepts) if extracted_concepts else 'Yok'}, Öneri Sayısı: {recommended_products_count}.
    Bu duruma ve diyalog geçmişine göre, kullanıcıya en uygun, açık ve yönlendirici bir takip sorusu sor.
    Eğer yeterli ürün önerilmişse memnuniyetini sorgula. Az öneri varsa daha detaylı sorular sor (örn. koku profilleri, kullanım amacı, oda tipi).
    Cevabın sadece soru olmalı.
    """
    prompt_messages.append({"role": "user", "parts": [system_prompt]})
    try:
        response = gemini_model.generate_content(prompt_messages, request_options={"timeout": 120})
        return response.text.strip(), "dynamic"
    except Exception as e:
        print(f"Gemini takip sorusu üretme hatası: {e}")
        return "Size nasıl daha iyi yardımcı olabilirim?", "genel_detaylandırma"

def initiate_prompt_chaining_interactive(initial_query, max_turns=5):
    current_query = initial_query
    all_recommendations = []
    turn_history = []
    print(f"\nBaşlangıç Sorgusu: '{initial_query}'")
    for turn in range(max_turns):
        print(f"\n--- Diyalog Turu {turn + 1} ---")
        defined_concepts_names = [c["name"] for c in concepts]
        possible_intents = ["alışveriş", "dekorasyon", "bilgi_sorgulama", "ürün_bulma", "atmosfer_yaratma"]
        nlu_result = extract_nlu_with_gemini(current_query, defined_concepts_names, possible_intents)
        recommendations = get_recommendations(nlu_result, num_recommendations=3)
        if recommendations:
            print("\nÖnerilen Ürünler:")
            for rec in recommendations:
                print(f"  - {rec['name']} (Alaka: {rec['relevance_score']:.2f})")
                if rec['id'] not in [item['id'] for item in all_recommendations]:
                    all_recommendations.append(rec)
        else:
            print("\nBu aşamada bir öneri bulunamadı.")
        turn_history.append({"role": "user", "parts": current_query})
        question, _ = get_follow_up_question_gemini(nlu_result, len(recommendations), turn_history)
        print(f"\nSistem: {question}")
        turn_history.append({"role": "model", "parts": question})
        user_response = input("Cevabınız: ").lower().strip()
        if user_response in ["çıkış", "teşekkürler", "yeterli", "hayır"]:
            print("\nMemnuniyetiniz için teşekkür ederiz.")
            break
        if not user_response:
            continue
        current_query += " " + user_response
    return all_recommendations

if __name__ == "__main__":
    print("--- Neo4j Veri Aktarımı Başlıyor ---")
    try:
        graph_global_for_import = Graph("bolt://localhost:7687", auth=("neo4j", "test_password"))
        if import_data_to_neo4j_for_testing_updated(products_df, concepts, relationships_data, graph_global_for_import):
            print("\n--- Neo4j Veri Aktarımı Tamamlandı ---\n")
        else:
            print("\n--- Neo4j Veri Aktarımı Başarısız. Uygulama Durduruluyor. ---\n")
            exit()
    except Exception as e:
        print(f"Ana blokta Neo4j bağlantı hatası: {e}")
        exit()

    print("\nMerhaba! Size nasıl yardımcı olabilirim? (Çıkmak için 'çıkış' yazın)")
    while True:
        user_initial_input = input("\nSiz: ").strip()
        if user_initial_input.lower() == 'çıkış':
            print("Güle güle!")
            break
        final_recommendations_from_dialogue = initiate_prompt_chaining_interactive(user_initial_input)
        if final_recommendations_from_dialogue:
            unique_final_recommendations = list({rec['id']: rec for rec in final_recommendations_from_dialogue}.values())
            print("\nFinal Önerilen Ürünler (Diyalog Sonucu):")
            for rec in unique_final_recommendations:
                print(f"  - {rec['name']} (Fiyat: {rec['price']:.2f} TL, Alaka: {rec['relevance_score']:.2f})")
        else:
            print("\nFinal Öneri: Bu sorgu için nihai bir öneri bulunamadı.")
    print("\n--- Uygulama Tamamlandı ---")
