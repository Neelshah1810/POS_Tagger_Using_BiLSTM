"""
Streamlit App for POS Tagging
Interactive interface for Multi-lingual POS tagging using trained BiLSTM models
Supports 12 morphologically rich languages
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import os
import random
import urllib.parse

# Language configuration with sample sentences in native scripts
LANGUAGE_CONFIG = {
    "Arabic": {
        "code": "arabic",
        "samples": [
            "الكتاب على الطاولة في الغرفة",
            "ذهب الولد إلى المدرسة صباحاً",
            "الشمس تشرق من الشرق",
            "القطة تنام على السرير",
            "أكل الطفل التفاحة الحمراء",
            "الطقس جميل اليوم",
            "يقرأ الطالب الكتاب بجد",
            "السماء زرقاء والغيوم بيضاء",
            "يلعب الأطفال في الحديقة",
            "أحب القراءة كثيراً",
            "الماء ضروري للحياة",
            "المعلم يشرح الدرس للتلاميذ"
        ],
        "direction": "rtl",
        "description": "Arabic - العربية"
    },
    "Basque": {
        "code": "basque",
        "samples": [
            "Gizona etxera doa arratsaldean",
            "Umeak parkean jolasten ari dira",
            "Liburua mahai gainean dago",
            "Eguzkia goizean ateratzen da",
            "Txakurra kalean korrika dabil",
            "Emakumeak janaria prestatzen du",
            "Mendiak ederrak dira udaberrian",
            "Ikaslea ikastolan ikasten ari da",
            "Ilargia gauean agertzen da",
            "Katua etxean lo egiten du",
            "Sagarra gorria eta gozoa da",
            "Haizea indartsua da gaur"
        ],
        "direction": "ltr",
        "description": "Basque - Euskara"
    },
    "Czech": {
        "code": "czech",
        "samples": [
            "Rychlá hnědá liška skáče přes líného psa",
            "Děti si hrají v parku",
            "Slunce svítí na modré obloze",
            "Kočka spí na pohovce",
            "Student čte zajímavou knihu",
            "Vlak přijíždí na nádraží",
            "Babička peče chutný koláč",
            "Ptáci zpívají na stromě",
            "Řeka teče přes město",
            "Sníh padá v zimě",
            "Učitel vysvětluje novou látku",
            "Květiny kvetou na jaře"
        ],
        "direction": "ltr",
        "description": "Czech - Čeština"
    },
    "English": {
        "code": "english",
        "samples": [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "The beautiful garden blooms in spring.",
            "Children are playing happily in the park.",
            "The scientist discovered a new species.",
            "Music brings joy to our lives.",
            "The old man walked slowly down the street.",
            "Books open doors to new worlds.",
            "The sun sets behind the mountains.",
            "Technology changes rapidly every year.",
            "Fresh bread smells absolutely wonderful.",
            "The river flows peacefully through the valley.",
            "Students study hard for their exams.",
            "Birds fly south during winter.",
            "The chef prepared a delicious meal."
        ],
        "direction": "ltr",
        "description": "English"
    },
    "Estonian": {
        "code": "estonian",
        "samples": [
            "Kiire pruun rebane hüppab üle laisa koera",
            "Lapsed mängivad pargis",
            "Päike paistab eredalt taevas",
            "Kass magab diivanil",
            "Õpilane loeb huvitavat raamatut",
            "Lilled õitsevad aias",
            "Lumi sajab talvel",
            "Linnud laulavad puudel",
            "Jõgi voolab läbi linna",
            "Vanaema küpsetab kooki",
            "Rong saabub jaama",
            "Tuul puhub tugevalt"
        ],
        "direction": "ltr",
        "description": "Estonian - Eesti"
    },
    "Finnish": {
        "code": "finnish",
        "samples": [
            "Nopea ruskea kettu hyppää laiskan koiran yli",
            "Lapset leikkivät puistossa",
            "Aurinko paistaa kirkkaasti",
            "Kissa nukkuu sohvalla",
            "Opiskelija lukee mielenkiintoista kirjaa",
            "Kukat kukkivat puutarhassa",
            "Lumi sataa talvella",
            "Linnut laulavat puissa",
            "Joki virtaa kaupungin läpi",
            "Isoäiti leipoo kakkua",
            "Juna saapuu asemalle",
            "Tuuli puhaltaa voimakkaasti"
        ],
        "direction": "ltr",
        "description": "Finnish - Suomi"
    },
    "Hindi": {
        "code": "hindi",
        "samples": [
            "राम बाजार में सेब खरीदता है।",
            "बच्चे पार्क में खेल रहे हैं।",
            "सूरज पूर्व से उगता है।",
            "बिल्ली सोफे पर सो रही है।",
            "छात्र किताब पढ़ रहा है।",
            "फूल बगीचे में खिल रहे हैं।",
            "बारिश हो रही है।",
            "पक्षी पेड़ पर गा रहे हैं।",
            "नदी शहर से होकर बहती है।",
            "माँ खाना बना रही है।",
            "ट्रेन स्टेशन पर आ रही है।",
            "हवा तेज चल रही है।",
            "दादी कहानी सुना रही है।",
            "लड़का साइकिल चला रहा है।"
        ],
        "direction": "ltr",
        "description": "Hindi - हिन्दी"
    },
    "Korean": {
        "code": "korean",
        "samples": [
            "빠른 갈색 여우가 게으른 개를 뛰어넘습니다",
            "아이들이 공원에서 놀고 있습니다",
            "해가 밝게 빛납니다",
            "고양이가 소파에서 자고 있습니다",
            "학생이 책을 읽고 있습니다",
            "꽃들이 정원에 피어 있습니다",
            "겨울에 눈이 내립니다",
            "새들이 나무에서 노래합니다",
            "강이 도시를 통해 흐릅니다",
            "할머니가 케이크를 굽습니다",
            "기차가 역에 도착합니다",
            "바람이 세게 붑니다"
        ],
        "direction": "ltr",
        "description": "Korean - 한국어"
    },
    "Latvian": {
        "code": "latvian",
        "samples": [
            "Ātrā brūnā lapsa lec pāri slinkajam sunim",
            "Bērni spēlējas parkā",
            "Saule spīd spilgti debesīs",
            "Kaķis guļ uz dīvāna",
            "Students lasa interesantu grāmatu",
            "Ziedi zied dārzā",
            "Ziemā snieg",
            "Putni dzied kokos",
            "Upe tek caur pilsētu",
            "Vecmāmiņa cep kūku",
            "Vilciens ierodas stacijā",
            "Vējš pūš stipri"
        ],
        "direction": "ltr",
        "description": "Latvian - Latviešu"
    },
    "Polish": {
        "code": "polish",
        "samples": [
            "Szybki brązowy lis przeskakuje nad leniwym psem",
            "Dzieci bawią się w parku",
            "Słońce świeci jasno na niebie",
            "Kot śpi na kanapie",
            "Student czyta ciekawą książkę",
            "Kwiaty kwitną w ogrodzie",
            "Zimą pada śnieg",
            "Ptaki śpiewają na drzewach",
            "Rzeka płynie przez miasto",
            "Babcia piecze ciasto",
            "Pociąg przyjeżdża na stację",
            "Wiatr wieje mocno"
        ],
        "direction": "ltr",
        "description": "Polish - Polski"
    },
    "Russian": {
        "code": "russian",
        "samples": [
            "Быстрая коричневая лиса прыгает через ленивую собаку",
            "Дети играют в парке",
            "Солнце ярко светит на небе",
            "Кошка спит на диване",
            "Студент читает интересную книгу",
            "Цветы цветут в саду",
            "Зимой идёт снег",
            "Птицы поют на деревьях",
            "Река течёт через город",
            "Бабушка печёт пирог",
            "Поезд прибывает на станцию",
            "Ветер дует сильно",
            "Мама готовит ужин",
            "Дедушка рассказывает историю"
        ],
        "direction": "ltr",
        "description": "Russian - Русский"
    },
    "Turkish": {
        "code": "turkish",
        "samples": [
            "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar",
            "Çocuklar parkta oynuyor",
            "Güneş gökyüzünde parlıyor",
            "Kedi kanepede uyuyor",
            "Öğrenci ilginç bir kitap okuyor",
            "Çiçekler bahçede açıyor",
            "Kışın kar yağıyor",
            "Kuşlar ağaçlarda şarkı söylüyor",
            "Nehir şehrin içinden akıyor",
            "Büyükanne pasta yapıyor",
            "Tren istasyona geliyor",
            "Rüzgar kuvvetli esiyor"
        ],
        "direction": "ltr",
        "description": "Turkish - Türkçe"
    }
}

# Set page configuration
st.set_page_config(
    page_title="Multi-lingual POS Tagger",
    page_icon="🏷️",
    layout="wide"
)

# Model Architecture Classes (same as training)
class CharCNN(nn.Module):
    """Character-level CNN for morphological features"""
    
    def __init__(self, char_vocab_size, char_embed_dim, num_filters, kernel_sizes):
        super(CharCNN, self).__init__()
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, k) for k in kernel_sizes
        ])
        
        self.output_dim = num_filters * len(kernel_sizes)
    
    def forward(self, x):
        batch_size, seq_len, max_word_len = x.size()
        x = x.view(-1, max_word_len)
        
        char_embed = self.char_embedding(x)
        char_embed = char_embed.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(char_embed))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        output = torch.cat(conv_outputs, dim=1)
        output = output.view(batch_size, seq_len, -1)
        
        return output


class EnhancedBiLSTMPOSTagger(nn.Module):
    """Enhanced BiLSTM with Character CNN and Attention"""
    
    def __init__(self, vocab_size, embedding_dim, char_vocab_size, char_embed_dim,
                 char_num_filters, char_kernel_sizes, hidden_dim, num_layers,
                 tagset_size, dropout=0.5, use_attention=True):
        super(EnhancedBiLSTMPOSTagger, self).__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.char_cnn = CharCNN(char_vocab_size, char_embed_dim,
                               char_num_filters, char_kernel_sizes)
        
        input_dim = embedding_dim + self.char_cnn.output_dim
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
    
    def forward(self, words, chars, lengths):
        word_embed = self.word_embedding(words)
        word_embed = self.dropout(word_embed)
        
        char_embed = self.char_cnn(chars)
        combined_embed = torch.cat([word_embed, char_embed], dim=2)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            combined_embed, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        lstm_out = self.layer_norm(lstm_out)
        
        if self.use_attention:
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            lstm_out = lstm_out * attention_weights
        
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        
        return output


class Vocabulary:
    """Enhanced vocabulary with character-level support"""
    
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.tag2idx = {'<PAD>': 0}
        self.idx2tag = {0: '<PAD>'}
        
        # Character vocabulary
        self.char2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2char = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}


@st.cache_resource
def load_model(model_path, device):
    """Load trained model and vocabulary"""
    try:
        # Add Vocabulary to safe globals for PyTorch 2.6+
        torch.serialization.add_safe_globals([Vocabulary])
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        vocab = checkpoint['vocab']
        config = checkpoint['config']
        
        model = EnhancedBiLSTMPOSTagger(
            vocab_size=len(vocab.word2idx),
            embedding_dim=config['embedding_dim'],
            char_vocab_size=len(vocab.char2idx),
            char_embed_dim=config['char_embed_dim'],
            char_num_filters=config['char_num_filters'],
            char_kernel_sizes=config['char_kernel_sizes'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            tagset_size=len(vocab.tag2idx),
            dropout=config['dropout'],
            use_attention=config['use_attention']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, vocab
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def encode_sentence(sentence, vocab):
    """Encode a sentence for model input"""
    words = sentence.strip().split()
    
    # Encode words
    word_ids = [vocab.word2idx.get(word.lower(), vocab.word2idx['<UNK>']) 
                for word in words]
    
    # Encode characters
    chars = []
    max_word_len = 0
    for word in words:
        char_ids = [vocab.char2idx['<START>']]
        for char in word:
            char_ids.append(vocab.char2idx.get(char, vocab.char2idx['<UNK>']))
        char_ids.append(vocab.char2idx['<END>'])
        chars.append(char_ids)
        max_word_len = max(max_word_len, len(char_ids))
    
    return word_ids, chars, words, max_word_len


def predict_pos_tags(sentence, model, vocab, device):
    """Predict POS tags for a sentence"""
    word_ids, chars, words, max_word_len = encode_sentence(sentence, vocab)
    
    # Create tensors
    words_tensor = torch.tensor([word_ids], dtype=torch.long).to(device)
    
    # Pad characters
    chars_tensor = torch.zeros(1, len(chars), max_word_len, dtype=torch.long).to(device)
    for j, char_ids in enumerate(chars):
        chars_tensor[0, j, :len(char_ids)] = torch.tensor(char_ids, dtype=torch.long)
    
    lengths = torch.tensor([len(word_ids)]).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(words_tensor, chars_tensor, lengths)
        pred_tags = torch.argmax(output, dim=-1).squeeze().cpu().numpy()
    
    # Convert indices to tags
    if len(words) == 1:
        pred_tags = [pred_tags.item()]
    else:
        pred_tags = pred_tags.tolist()
    
    pos_tags = [vocab.idx2tag[idx] for idx in pred_tags]
    
    return list(zip(words, pos_tags))


def get_tag_color(tag):
    """Get background and text color for POS tag visualization with readable contrast"""
    # Format: (background_color, text_color)
    color_map = {
        'NOUN': ('#FF6B6B', '#FFFFFF'),
        'VERB': ('#4ECDC4', '#1A1A1A'),
        'ADJ': ('#45B7A0', '#FFFFFF'),
        'ADV': ('#F4A460', '#1A1A1A'),
        'PRON': ('#CD853F', '#FFFFFF'),
        'DET': ('#8B4513', '#FFFFFF'),
        'ADP': ('#D4C4A8', '#1A1A1A'),
        'PROPN': ('#E76F51', '#FFFFFF'),
        'NUM': ('#2A9D8F', '#FFFFFF'),
        'CONJ': ('#264653', '#FFFFFF'),
        'CCONJ': ('#3A5A6B', '#FFFFFF'),
        'SCONJ': ('#457B7D', '#FFFFFF'),
        'PART': ('#4A90D9', '#FFFFFF'),
        'AUX': ('#DAA520', '#1A1A1A'),
        'INTJ': ('#F72585', '#FFFFFF'),
        'PUNCT': ('#808080', '#FFFFFF'),
        'SYM': ('#7209B7', '#FFFFFF'),
        'X': ('#6B6B6B', '#FFFFFF')
    }
    return color_map.get(tag, ('#E0E0E0', '#1A1A1A'))


def generate_chatgpt_prompt(sentence, results, language, available_tags):
    """Generate a structured prompt for ChatGPT to verify POS tags"""
    # Format available tags as a list
    tags_list = ", ".join(sorted(available_tags))
    
    prompt = f"""I need you to verify the POS (Part-of-Speech) tags for this {language} sentence.

The tags follow the Universal POS tagset (UPOS). Please check if each tag is correct.

**Sentence:** "{sentence}"

**Available POS Tags for this {language} model:** {tags_list}

**My BiLSTM Model's Predictions:**
"""
    
    for word, tag in results:
        prompt += f"- {word} → {tag}\n"
    
    prompt += f"""
**Please verify each tag and respond in this exact format:**

| Word | Model Tag | Correct? | Right Tag (if wrong) | Explanation |
|------|-----------|----------|----------------------|-------------|

**At the end, provide:**
1. Overall accuracy score (X/Y correct)
2. If there are any errors, explain WHY the BiLSTM model might have made these mistakes. Consider:
   - Ambiguous words that can be multiple parts of speech
   - Context-dependent tagging challenges
   - Morphologically complex words in {language}
   - Words that might be out-of-vocabulary for the model
   - Similar word patterns that the model might confuse

**POS Tags Reference:**
- NOUN (noun), VERB (verb), ADJ (adjective), ADV (adverb)
- PRON (pronoun), DET (determiner), ADP (adposition/preposition)
- PROPN (proper noun), NUM (numeral), AUX (auxiliary verb)
- CCONJ (coordinating conjunction), SCONJ (subordinating conjunction)
- PART (particle), INTJ (interjection), PUNCT (punctuation), SYM (symbol), X (other)
"""
    return prompt


def main():
    # Title and description
    st.title("🏷️ Multi-lingual POS Tagger")
    st.markdown("""
    This application performs **Part-of-Speech (POS) tagging** using trained BiLSTM models.
    
    **Features:**
    - Support for **12 morphologically rich languages**
    - Character-level embeddings for better handling of unknown words
    - Deep BiLSTM architecture with attention mechanism
    - Interactive visualization of tagged sentences
    """)
    
    # Custom CSS to increase button height
    st.markdown("""
    <style>
    .stButton > button {
        height: 3.4rem;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.header("⚙️ Settings")
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"**Device:** {device}")
    
    # Language selection with all 12 languages
    languages = list(LANGUAGE_CONFIG.keys())
    language = st.sidebar.selectbox(
        "Select Language",
        languages,
        format_func=lambda x: LANGUAGE_CONFIG[x]["description"],
        help="Choose the language for POS tagging"
    )
    
    # Get language config
    lang_config = LANGUAGE_CONFIG[language]
    
    # Detect language change and automatically set a random sample
    if 'previous_language' not in st.session_state:
        st.session_state['previous_language'] = language
        st.session_state['sample_text'] = random.choice(lang_config['samples'])
    elif st.session_state['previous_language'] != language:
        # Language changed, select a new random sample for the new language
        st.session_state['previous_language'] = language
        st.session_state['sample_text'] = random.choice(lang_config['samples'])
    
    # Model path based on language (from models folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", f"{lang_config['code']}_enhanced_best_model.pt")
    
    # Load model
    with st.spinner(f"Loading {language} model..."):
        model, vocab = load_model(model_path, device)
    
    if model is None or vocab is None:
        st.error(f"❌ Failed to load {language} model.")
        st.info(f"💡 **Tip:** Make sure '{lang_config['code']}_enhanced_best_model.pt' exists in the 'models' folder.")
        return
    
    st.sidebar.success(f"✅ {language} model loaded!")
    
    # Test accuracy for each language model
    MODEL_ACCURACY = {
        "English": 0.9670,
        "Hindi": 0.9680,
        "Turkish": 0.9119,
        "Finnish": 0.9545,
        "Arabic": 0.9594,
        "Korean": 0.9376,
        "Estonian": 0.9649,
        "Latvian": 0.9671,
        "Basque": 0.9405,
        "Russian": 0.9774,
        "Polish": 0.9765,
        "Czech": 0.9622
    }
    
    # Display model statistics
    with st.sidebar.expander("📊 Model Statistics", expanded=False):
        accuracy = MODEL_ACCURACY.get(language, 0.95)
        st.metric("Test Accuracy", f"{accuracy:.2%}")
        st.write(f"**Vocabulary Size:** {len(vocab.word2idx):,}")
        st.write(f"**Character Vocabulary:** {len(vocab.char2idx):,}")
        st.write(f"**Number of POS Tags:** {len(vocab.tag2idx) - 1}")  # Exclude <PAD>
        st.write(f"**Model Parameters:** {sum(p.numel() for p in model.parameters()):,}")
    
    # Display POS tags legend for the selected language
    with st.sidebar.expander("🎨 POS Tags for This Language", expanded=True):
        unique_tags = sorted([tag for tag in vocab.tag2idx.keys() if tag != '<PAD>'])
        cols = st.columns(3)
        for i, tag in enumerate(unique_tags):
            with cols[i % 3]:
                bg_color, text_color = get_tag_color(tag)
                st.markdown(
                    f"<span style='background-color: {bg_color}; color: {text_color}; "
                    f"padding: 4px 10px; border-radius: 4px; font-size: 11px; "
                    f"display: inline-block; margin: 3px 0; white-space: nowrap; "
                    f"font-weight: 500;'>{tag}</span>",
                    unsafe_allow_html=True
                )
    
    # POS tag meanings dictionary - includes all possible Universal POS tags
    POS_TAG_MEANINGS = {
        '<UNK>': ('Unknown', 'Unknown or unrecognized token that was not in training vocabulary'),
        'ADJ': ('Adjective', 'Words that describe or modify nouns (e.g., big, beautiful, quick)'),
        'ADP': ('Adposition', 'Prepositions and postpositions (e.g., in, on, at, to, from)'),
        'ADV': ('Adverb', 'Words that modify verbs, adjectives, or other adverbs (e.g., quickly, very, well)'),
        'AUX': ('Auxiliary Verb', 'Helping verbs used with main verbs (e.g., is, has, will, can, must)'),
        'CCONJ': ('Coordinating Conjunction', 'Words that connect equal elements (e.g., and, or, but)'),
        'CONJ': ('Conjunction', 'Words that connect clauses or sentences (e.g., and, but, or, because)'),
        'DET': ('Determiner', 'Words that introduce nouns (e.g., the, a, this, my, some)'),
        'INTJ': ('Interjection', 'Exclamations or expressions of emotion (e.g., oh, wow, hello, ouch)'),
        'NOUN': ('Noun', 'Words for people, places, things, or ideas (e.g., dog, city, happiness)'),
        'NUM': ('Numeral', 'Numbers and numeric expressions (e.g., one, 2, third, 100)'),
        'PART': ('Particle', 'Function words that don\'t fit other categories (e.g., not, \'s, to-infinitive)'),
        'PRON': ('Pronoun', 'Words that replace nouns (e.g., I, you, he, she, it, they, this)'),
        'PROPN': ('Proper Noun', 'Names of specific entities (e.g., John, London, Google, Monday)'),
        'PUNCT': ('Punctuation', 'Punctuation marks (e.g., . , ! ? : ; - \' \")'),
        'SCONJ': ('Subordinating Conjunction', 'Words that introduce dependent clauses (e.g., if, because, when, although)'),
        'SYM': ('Symbol', 'Symbols and special characters (e.g., $, %, @, +, =)'),
        'VERB': ('Verb', 'Action or state words (e.g., run, eat, think, is, become)'),
        'X': ('Other', 'Foreign words, typos, or unclassifiable tokens'),
        '_': ('Underscore', 'Placeholder or unspecified tag')
    }
    
    # Display POS tag meanings for this language
    with st.sidebar.expander(f"📖 Tag Meanings ({len(unique_tags)} tags)", expanded=False):
        for tag in unique_tags:
            if tag in POS_TAG_MEANINGS:
                name, description = POS_TAG_MEANINGS[tag]
                bg_color, text_color = get_tag_color(tag)
                st.markdown(
                    f"<div style='margin-bottom: 10px;'>"
                    f"<span style='background-color: {bg_color}; color: {text_color}; "
                    f"padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;'>{tag}</span> "
                    f"<strong>{name}</strong><br>"
                    f"<span style='font-size: 12px; color: #666;'>{description}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
    # Main input area
    st.header("📝 Input Sentence")
    
    # Show language info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"**Selected Language:** {lang_config['description']}")
    with col2:
        if st.button("📋 Use Sample", use_container_width=True):
            # Randomly select a sample from the list of samples for this language
            st.session_state['sample_text'] = random.choice(lang_config['samples'])
            st.rerun()
    
    # Get sample text from session state
    default_text = st.session_state.get('sample_text', lang_config['samples'][0])
    
    # Text input with RTL support for Arabic
    text_direction = lang_config['direction']
    sentence = st.text_area(
        "Enter a sentence:",
        value=default_text,
        height=100,
        help=f"Enter a sentence in {language} for POS tagging"
    )
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        predict_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Perform prediction
    if predict_button and sentence.strip():
        with st.spinner("Analyzing..."):
            try:
                results = predict_pos_tags(sentence, model, vocab, device)
                
                # Display results
                st.header("📊 Results")
                
                # Visualization with colored tags
                st.subheader("Tagged Sentence")
                html_output = "<div style='display: flex; flex-wrap: wrap; gap: 10px; align-items: center;'>"
                for word, tag in results:
                    bg_color, text_color = get_tag_color(tag)
                    html_output += f"<span style='display: inline-flex; flex-direction: column; align-items: center; padding: 8px 12px; background-color: {bg_color}; color: {text_color}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'><strong style='font-size: 16px;'>{word}</strong><small style='font-size: 12px; opacity: 0.9;'>{tag}</small></span>"
                html_output += "</div>"
                
                st.markdown(html_output, unsafe_allow_html=True)
                
                # Table view
                st.subheader("Detailed View")
                
                import pandas as pd
                df = pd.DataFrame(results, columns=['Word', 'POS Tag'])
                df.index = df.index + 1  # Start index from 1
                df.index.name = '#'
                st.dataframe(df, use_container_width=True)
                
                # Download results
                st.subheader("💾 Export & Verify")
                
                # Create downloadable content
                output_text = "Word\tPOS Tag\n" + "-"*30 + "\n"
                output_text += "\n".join([f"{word}\t{tag}" for word, tag in results])
                
                col_download, col_verify = st.columns(2)
                
                with col_download:
                    st.download_button(
                        label="📥 Download as TXT",
                        data=output_text,
                        file_name=f"pos_tags_{language.lower()}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col_verify:
                    # Get available POS tags for this language (excluding PAD)
                    available_tags = [tag for tag in vocab.tag2idx.keys() if tag != '<PAD>']
                    
                    # Generate ChatGPT verification prompt with language-specific tags
                    chatgpt_prompt = generate_chatgpt_prompt(sentence, results, language, available_tags)
                    # Use quote_plus for URL encoding (handles spaces as + and special chars)
                    encoded_prompt = urllib.parse.quote_plus(chatgpt_prompt, safe='')
                    chatgpt_url = f"https://chat.openai.com/?q={encoded_prompt}"
                    
                    # Custom green button using markdown
                    st.markdown(
                        f'''<a href="{chatgpt_url}" target="_blank" style="
                            display: block;
                            width: 100%;
                            padding: 0.75rem 1rem;
                            background-color: #10a37f;
                            color: #ffffff !important;
                            text-align: center;
                            text-decoration: none !important;
                            border-radius: 0.5rem;
                            font-weight: 600;
                            font-size: 14px;
                            box-sizing: border-box;
                        ">🔍 Verify with ChatGPT</a>
                        <style>
                            a[href*="chat.openai.com"] {{
                                color: #ffffff !important;
                            }}
                            a[href*="chat.openai.com"]:hover {{
                                color: #ffffff !important;
                                background-color: #0d8a6f !important;
                            }}
                            a[href*="chat.openai.com"]:visited {{
                                color: #ffffff !important;
                            }}
                        </style>''',
                        unsafe_allow_html=True
                    )
                
                # Statistics
                st.subheader("📈 Statistics")
                tag_counts = defaultdict(int)
                for _, tag in results:
                    tag_counts[tag] += 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Words", len(results))
                with col2:
                    st.metric("Unique POS Tags", len(tag_counts))
                with col3:
                    most_common = max(tag_counts.items(), key=lambda x: x[1])
                    st.metric("Most Common Tag", f"{most_common[0]} ({most_common[1]})")
                
                # Tag distribution
                with st.expander("📊 Tag Distribution"):
                    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(results)) * 100
                        st.write(f"**{tag}:** {count} ({percentage:.1f}%)")
                        st.progress(percentage / 100)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please check your input and try again.")
    
    elif predict_button and not sentence.strip():
        st.warning("⚠️ Please enter a sentence to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Powered by Enhanced BiLSTM with Character-level CNN and Self Attention Mechanism</p>
        <p>Supporting 12 Morphologically Rich Languages</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()