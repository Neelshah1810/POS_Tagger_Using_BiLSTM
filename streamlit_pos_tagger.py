"""
Streamlit App for POS Tagging
Interactive interface for Hindi and English POS tagging using trained models
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# Set page configuration
st.set_page_config(
    page_title="POS Tagger",
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
    """Get color for POS tag visualization"""
    color_map = {
        'NOUN': '#FF6B6B',
        'VERB': '#4ECDC4',
        'ADJ': '#95E1D3',
        'ADV': '#FFA07A',
        'PRON': '#DDA15E',
        'DET': '#BC6C25',
        'ADP': '#FEFAE0',
        'PROPN': '#E76F51',
        'NUM': '#2A9D8F',
        'CONJ': '#264653',
        'CCONJ': '#264653',
        'SCONJ': '#287271',
        'PART': '#F4A261',
        'AUX': '#E9C46A',
        'INTJ': '#F72585',
        'PUNCT': '#B5B5B5',
        'SYM': '#7209B7',
        'X': '#D3D3D3'
    }
    return color_map.get(tag, '#E0E0E0')


def main():
    # Title and description
    st.title("🏷️ Multi-lingual POS Tagger")
    st.markdown("""
    This application performs **Part-of-Speech (POS) tagging** using trained BiLSTM models.
    
    **Features:**
    - Support for Hindi and English languages
    - Character-level embeddings for better handling of unknown words
    - Deep BiLSTM architecture with attention mechanism
    - Interactive visualization of tagged sentences
    """)
    
    # Sidebar for model selection
    st.sidebar.header("⚙️ Settings")
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"**Device:** {device}")
    
    # Language selection
    language = st.sidebar.selectbox(
        "Select Language",
        ["Hindi", "English"],
        help="Choose the language for POS tagging"
    )
    
    # Model path based on language
    model_path = f"{language.lower()}_enhanced_best_model.pt"
    
    # Load model
    with st.spinner(f"Loading {language} model..."):
        model, vocab = load_model(model_path, device)
    
    if model is None or vocab is None:
        st.error(f"❌ Failed to load {language} model. Please ensure '{model_path}' exists in the current directory.")
        st.info("💡 **Tip:** Make sure you have the trained model file in the same directory as this script.")
        return
    
    st.sidebar.success(f"✅ {language} model loaded successfully!")
    
    # Display model statistics
    with st.sidebar.expander("📊 Model Statistics"):
        st.write(f"**Vocabulary Size:** {len(vocab.word2idx):,}")
        st.write(f"**Character Vocabulary:** {len(vocab.char2idx):,}")
        st.write(f"**Number of POS Tags:** {len(vocab.tag2idx)}")
        st.write(f"**Model Parameters:** {sum(p.numel() for p in model.parameters()):,}")
    
    # Display POS tags legend
    with st.sidebar.expander("🎨 POS Tags Legend"):
        unique_tags = sorted([tag for tag in vocab.tag2idx.keys() if tag != '<PAD>'])
        for tag in unique_tags:
            color = get_tag_color(tag)
            st.markdown(
                f"<span style='background-color: {color}; padding: 2px 8px; "
                f"border-radius: 3px; margin: 2px;'>{tag}</span>",
                unsafe_allow_html=True
            )
    
    # Main input area
    st.header("📝 Input Sentence")
    
    # Example sentences
    if language == "Hindi":
        example = "राम बाजार में सेब खरीदता है।"
    else:
        example = "The quick brown fox jumps over the lazy dog."
    
    # Text input
    sentence = st.text_area(
        "Enter a sentence:",
        value=example,
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
                    color = get_tag_color(tag)
                    html_output += f"<span style='display: inline-flex; flex-direction: column; align-items: center; padding: 8px 12px; background-color: {color}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'><strong style='font-size: 16px;'>{word}</strong><small style='font-size: 12px; opacity: 0.8;'>{tag}</small></span>"
                html_output += "</div>"
                
                st.markdown(html_output, unsafe_allow_html=True)
                
                # Table view
                st.subheader("Detailed View")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Word**")
                    for word, _ in results:
                        st.write(word)
                
                with col2:
                    st.markdown("**POS Tag**")
                    for _, tag in results:
                        st.write(tag)
                
                # Download results
                st.subheader("💾 Export Results")
                
                # Create downloadable content
                output_text = "Word\tPOS Tag\n" + "-"*30 + "\n"
                output_text += "\n".join([f"{word}\t{tag}" for word, tag in results])
                
                st.download_button(
                    label="📥 Download as TXT",
                    data=output_text,
                    file_name=f"pos_tags_{language.lower()}.txt",
                    mime="text/plain"
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
        <p>Powered by Enhanced BiLSTM with Character-level CNN</p>
        <p>Accuracy: ~96% on test datasets</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()