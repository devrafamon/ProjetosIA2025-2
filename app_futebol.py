import streamlit as st
import math
from collections import Counter

# 0. CONFIGURAÇÃO (Streamlit)
# =============================================================================
st.set_page_config(page_title="Tática de Escanteios v2.1", page_icon="⚽")

# =============================================================================
# 1. MÓDULO DO ALGORITMO ID3 (Matemática Pura)
# =============================================================================

def entropy(examples, target_attr):
    if not examples: return 0
    targets = [e[target_attr] for e in examples]
    counts = Counter(targets)
    total = len(examples)
    ent = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def remainder(attribute, examples, target_attr):
    values = set(e[attribute] for e in examples)
    total_examples = len(examples)
    rem = 0
    for v in values:
        subset = [e for e in examples if e[attribute] == v]
        weight = len(subset) / total_examples
        rem += weight * entropy(subset, target_attr)
    return rem

def importance(attribute, examples, target_attr):
    return entropy(examples, target_attr) - remainder(attribute, examples, target_attr)

def plurality_value(examples, target_attr):
    if not examples: return None
    targets = [e[target_attr] for e in examples]
    return Counter(targets).most_common(1)[0][0]

def learn_decision_tree(examples, attributes, parent_examples, target_attr='Decisao'):
    if not examples:
        return plurality_value(parent_examples, target_attr)
    first_class = examples[0][target_attr]
    if all(e[target_attr] == first_class for e in examples):
        return first_class
    if not attributes:
        return plurality_value(examples, target_attr)

    best_attr = max(attributes, key=lambda a: importance(a, examples, target_attr))
    tree = {best_attr: {}}
    possible_values = set(e[best_attr] for e in examples)
    
    for v in possible_values:
        exs = [e for e in examples if e[best_attr] == v]
        new_attributes = [attr for attr in attributes if attr != best_attr]
        subtree = learn_decision_tree(exs, new_attributes, examples, target_attr)
        tree[best_attr][v] = subtree
            
    return tree

def predict_with_explanation(node, observation, path=[]):
    if not isinstance(node, dict):
        return node, path
    
    attribute = next(iter(node))
    value = observation.get(attribute)
    new_path = path + [f"Como '{attribute}' é '{value}'"]
    
    if value not in node[attribute]:
        return "Situação Desconhecida (Não treinado para isso)", new_path
    
    return predict_with_explanation(node[attribute][value], observation, new_path)

# =============================================================================
# 2. DATASET ATUALIZADO (Incluindo CRUZAMENTO_SEGUNDO_PAU)
# 

dataset_futebol = [
    # --- NOVIDADE: SEGUNDO PAU (Fugir da zona congestionada) ---
    # Lógica: Se é Zona/Mista e o Goleiro não sai, a bola longa nas costas da defesa é mortal.
    {'Marcacao': 'Zona',       'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Altos',  'Pressao': 'Nao', 'Decisao': 'Cruzamento_Segundo_Pau'},
    {'Marcacao': 'Mista',      'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Altos',  'Pressao': 'Nao', 'Decisao': 'Cruzamento_Segundo_Pau'},
    {'Marcacao': 'Zona',       'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Baixos', 'Estatura_Adversario': 'Altos',  'Pressao': 'Nao', 'Decisao': 'Cruzamento_Segundo_Pau'}, 

    # --- CENÁRIOS DE VANTAGEM AÉREA (Nós Altos vs Eles Baixos) ---
    {'Marcacao': 'Individual', 'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Baixos', 'Pressao': 'Nao', 'Decisao': 'Cruzamento_Ponto_Penalti'},
    {'Marcacao': 'Zona',       'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Baixos', 'Pressao': 'Nao', 'Decisao': 'Cruzamento_Ponto_Penalti'}, 
    {'Marcacao': 'Mista',      'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Baixos', 'Pressao': 'Nao', 'Decisao': 'Cruzamento_Ponto_Penalti'},
    
    # --- CENÁRIOS DE DESVANTAGEM AÉREA (Nós Baixos vs Eles Altos) ---
    {'Marcacao': 'Zona',       'Goleiro': 'Sai_Bem',  'Estatura_Nosso_Time': 'Baixos', 'Estatura_Adversario': 'Altos',  'Pressao': 'Nao', 'Decisao': 'Jogada_Curta'},
    {'Marcacao': 'Individual', 'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Baixos', 'Estatura_Adversario': 'Altos',  'Pressao': 'Nao', 'Decisao': 'Jogada_Curta'},
    {'Marcacao': 'Mista',      'Goleiro': 'Sai_Bem',  'Estatura_Nosso_Time': 'Baixos', 'Estatura_Adversario': 'Altos',  'Pressao': 'Nao', 'Decisao': 'Cruzamento_Primeiro_Pau'},

    # --- JOGO PARELHO (Altos vs Altos) - Ajustado para evitar conflito com Segundo Pau ---
    
    {'Marcacao': 'Individual', 'Goleiro': 'Sai_Bem',  'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Altos',  'Pressao': 'Nao', 'Decisao': 'Cruzamento_Primeiro_Pau'},
    {'Marcacao': 'Mista',      'Goleiro': 'Sai_Bem',  'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Altos',  'Pressao': 'Nao', 'Decisao': 'Cruzamento_Primeiro_Pau'},

    # --- JOGO PARELHO (Baixos vs Baixos) ---
    {'Marcacao': 'Zona',       'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Baixos', 'Estatura_Adversario': 'Baixos', 'Pressao': 'Nao', 'Decisao': 'Cruzamento_Primeiro_Pau'},
    {'Marcacao': 'Individual', 'Goleiro': 'Sai_Bem',  'Estatura_Nosso_Time': 'Baixos', 'Estatura_Adversario': 'Baixos', 'Pressao': 'Nao', 'Decisao': 'Jogada_Curta'},

    # --- PRESSÃO NA BANDEIRA (Prioridade Máxima de Segurança) ---
    {'Marcacao': 'Zona',       'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Baixos', 'Pressao': 'Sim', 'Decisao': 'Cruzamento_Primeiro_Pau'},
    {'Marcacao': 'Individual', 'Goleiro': 'Sai_Bem',  'Estatura_Nosso_Time': 'Baixos', 'Estatura_Adversario': 'Altos',  'Pressao': 'Sim', 'Decisao': 'Cruzamento_Primeiro_Pau'},
    {'Marcacao': 'Mista',      'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Altos',  'Pressao': 'Sim', 'Decisao': 'Cruzamento_Primeiro_Pau'},
    # Adicionando um caso onde, mesmo com pressão, se o goleiro fica, os altos podem tentar o segundo pau (arriscado, mas válido)
    {'Marcacao': 'Zona',       'Goleiro': 'Fica_Gol', 'Estatura_Nosso_Time': 'Altos',  'Estatura_Adversario': 'Altos',  'Pressao': 'Sim', 'Decisao': 'Cruzamento_Segundo_Pau'}
]

atributos_disponiveis = ['Marcacao', 'Goleiro', 'Estatura_Nosso_Time', 'Estatura_Adversario', 'Pressao']

# =============================================================================
# 3. INTERFACE VISUAL (STREAMLIT)
# =============================================================================

st.title("⚽ IA Tática de Escanteios v2.1")
st.markdown("Consultor inteligente considerando a estatura de ambos os times.")

# Treinando a IA
arvore_tatica = learn_decision_tree(dataset_futebol, atributos_disponiveis, [], target_attr='Decisao')

col1, col2 = st.columns(2)

with col1:
    marcacao = st.selectbox("Marcação Adversária", ["Individual", "Zona", "Mista"])
    goleiro = st.selectbox("Estilo do Goleiro", ["Sai_Bem", "Fica_Gol"])
    pressao = st.selectbox("Pressão na Bandeira?", ["Sim", "Nao"])

with col2:
    estatura_nosso = st.selectbox("Nosso Time é:", ["Altos", "Baixos"])
    estatura_adv = st.selectbox("Adversário é:", ["Altos", "Baixos"])

if st.button("Definir Jogada", type="primary"):
    
    situacao_usuario = {
        'Marcacao': marcacao,
        'Goleiro': goleiro,
        'Estatura_Nosso_Time': estatura_nosso,
        'Estatura_Adversario': estatura_adv,
        'Pressao': pressao
    }
    
    try:
        decisao, motivo = predict_with_explanation(arvore_tatica, situacao_usuario)
        
        st.divider()
        st.subheader(f"🎯 {decisao.replace('_', ' ').upper()}")
        
        # Colorindo a resposta para ficar mais fácil de ler
        if "Curta" in decisao:
            st.info("💡 Jogada de Inteligência (Bola no chão)")
        elif "Segundo" in decisao:
            st.warning("🚀 Bola Longa (Nas costas da defesa)")
        elif "Penalti" in decisao:
            st.success("⚔️ Bola de Força (Disputa no alto)")
        else:
            st.error("⚡ Bola de Velocidade (Antecipação)")

        st.write("📝 **Lógica utilizada:**")
        for i, passo in enumerate(motivo):
            st.caption(f"{i+1}. {passo}")
            
    except Exception as e:
        st.error(f"Erro: {e}")
