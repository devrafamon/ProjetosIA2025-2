import streamlit as st
import math
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle



# 0 Campo de Futebol (Matplotlib)
# ----------------------------------------------------------------------------
def desenhar_campo(cor_campo, cor_linha):
    """
    Desenha um campo de futebol com legendas para os times.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Cores
    fig.patch.set_facecolor(cor_campo)
    ax.set_facecolor(cor_campo)

    # Dimensões (escala 120x80)
    xmax = 120
    ymax = 80
    
    # --- Desenho das Linhas (Igual ao anterior) ---
    plt.plot([0, 0], [0, ymax], color=cor_linha, linewidth=2)
    plt.plot([0, xmax], [ymax, ymax], color=cor_linha, linewidth=2)
    plt.plot([xmax, xmax], [ymax, 0], color=cor_linha, linewidth=2)
    plt.plot([xmax, 0], [0, 0], color=cor_linha, linewidth=2)
    plt.plot([xmax/2, xmax/2], [0, ymax], color=cor_linha, linewidth=2)

    # Círculos
    centro = Circle((xmax/2, ymax/2), 9.15, color=cor_linha, fill=False, linewidth=2)
    ponto_central = Circle((xmax/2, ymax/2), 0.6, color=cor_linha)
    ax.add_patch(centro)
    ax.add_patch(ponto_central)

    # Áreas Esquerda
    ax.add_patch(Rectangle((0, ymax/2 - 20.16), 16.5, 40.32, edgecolor=cor_linha, facecolor='none', linewidth=2))
    ax.add_patch(Rectangle((0, ymax/2 - 9.16), 5.5, 18.32, edgecolor=cor_linha, facecolor='none', linewidth=2))
    ax.add_patch(Arc((11, ymax/2), height=18.3, width=18.3, angle=0, theta1=308, theta2=52, color=cor_linha, linewidth=2))
    ax.add_patch(Circle((11, ymax/2), 0.6, color=cor_linha))

    # Áreas Direita
    ax.add_patch(Rectangle((xmax-16.5, ymax/2 - 20.16), 16.5, 40.32, edgecolor=cor_linha, facecolor='none', linewidth=2))
    ax.add_patch(Rectangle((xmax-5.5, ymax/2 - 9.16), 5.5, 18.32, edgecolor=cor_linha, facecolor='none', linewidth=2))
    ax.add_patch(Arc((xmax-11, ymax/2), height=18.3, width=18.3, angle=0, theta1=128, theta2=232, color=cor_linha, linewidth=2))
    ax.add_patch(Circle((xmax-11, ymax/2), 0.6, color=cor_linha))

    # --- NOVA PARTE: LEGENDAS DOS TIMES ---
    
    # Legenda: Nosso Time (Lado Esquerdo - x=30 é 1/4 do campo)
    ax.text(30, 40, "NOSSO TIME\n(Defesa)", 
            color=cor_linha, 
            fontsize=25, 
            ha='center', 
            va='center', 
            alpha=0.15,  # Transparência para não atrapalhar
            fontweight='bold')

    # Legenda: Time Adversário (Lado Direito - x=90 é 3/4 do campo)
    ax.text(90, 40, "TIME\nADVERSÁRIO", 
            color=cor_linha, 
            fontsize=25, 
            ha='center', 
            va='center', 
            alpha=0.15, 
            fontweight='bold')

    # Flecha indicando ataque (Opcional, mas ajuda visualmente)
    ax.arrow(45, 75, 30, 0, head_width=2, head_length=3, fc=cor_linha, ec=cor_linha, alpha=0.5)
    ax.text(60, 78, "Sentido do Ataque", color=cor_linha, ha='center', fontsize=10, alpha=0.8)

    # Configurações finais
    plt.xlim(-5, xmax+5)
    plt.ylim(-5, ymax+5)
    plt.axis('off')
    ax.set_aspect('equal')
    
    return fig


# 0 CONFIGURAÇÃO (Streamlit)
# =============================================================================
st.set_page_config(page_title="Tática de Escanteios v2.1", page_icon="⚽")

# ===========================================================================
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

def classificar_densidade(nos, eles):
    diferenca = nos - eles
    if diferenca >= 2:
        return "Superioridade"
    elif diferenca <= -2:
        return "Inferioridade"
    else:
        return "Equilibrado"
    
def classificar_cobrador(nota):
    if nota >= 8:
        return "Elite"
    elif nota >= 5:
        return "Medio"
    else:
        return "Fraco"

# =============================================================================
# 2. DATASET ATUALIZADO (Incluindo CRUZAMENTO_SEGUNDO_PAU)

dataset_futebol = [

# ==========================================================
# 1️⃣ PRESSÃO = SIM  → PRIORIDADE: EXECUÇÃO RÁPIDA
# ==========================================================

# Inferioridade sob pressão → evitar duelo
{'Marcacao':'Zona','Goleiro':'Sai_Bem','Estatura_Nosso_Time':'Baixos','Estatura_Adversario':'Altos','Pressao':'Sim','Densidade_Area':'Inferioridade','Qualidade_Cobrador':'Fraco','Decisao':'Jogada_Curta'},

# Equilibrado sob pressão → antecipação
{'Marcacao':'Individual','Goleiro':'Sai_Bem','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Altos','Pressao':'Sim','Densidade_Area':'Equilibrado','Qualidade_Cobrador':'Medio','Decisao':'Cruzamento_Primeiro_Pau'},

# Superioridade sob pressão → atacar rápido
{'Marcacao':'Mista','Goleiro':'Fica_Gol','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Baixos','Pressao':'Sim','Densidade_Area':'Superioridade','Qualidade_Cobrador':'Elite','Decisao':'Cruzamento_Primeiro_Pau'},

# Goleiro fica e cobrador elite → segundo pau mesmo sob pressão
{'Marcacao':'Zona','Goleiro':'Fica_Gol','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Altos','Pressao':'Sim','Densidade_Area':'Superioridade','Qualidade_Cobrador':'Elite','Decisao':'Cruzamento_Segundo_Pau'},

# ==========================================================
# 2️⃣ PRESSÃO = NÃO  → ANÁLISE ESTRUTURAL COMPLETA
# ==========================================================

# --------------------------
# A) SUPERIORIDADE NUMÉRICA
# --------------------------

# Vantagem aérea clara
{'Marcacao':'Individual','Goleiro':'Fica_Gol','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Baixos','Pressao':'Nao','Densidade_Area':'Superioridade','Qualidade_Cobrador':'Medio','Decisao':'Cruzamento_Ponto_Penalti'},

# Zona + goleiro não sai → explorar segundo pau
{'Marcacao':'Zona','Goleiro':'Fica_Gol','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Altos','Pressao':'Nao','Densidade_Area':'Superioridade','Qualidade_Cobrador':'Elite','Decisao':'Cruzamento_Segundo_Pau'},

# Goleiro sai bem → bola rápida primeiro pau
{'Marcacao':'Mista','Goleiro':'Sai_Bem','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Altos','Pressao':'Nao','Densidade_Area':'Superioridade','Qualidade_Cobrador':'Medio','Decisao':'Cruzamento_Primeiro_Pau'},

# Cobrador fraco → reduzir risco
{'Marcacao':'Zona','Goleiro':'Sai_Bem','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Baixos','Pressao':'Nao','Densidade_Area':'Superioridade','Qualidade_Cobrador':'Fraco','Decisao':'Cruzamento_Primeiro_Pau'},

# --------------------------
# B) EQUILIBRADO
# --------------------------

# Jogo parelho + vantagem aérea
{'Marcacao':'Individual','Goleiro':'Fica_Gol','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Altos','Pressao':'Nao','Densidade_Area':'Equilibrado','Qualidade_Cobrador':'Medio','Decisao':'Cruzamento_Ponto_Penalti'},

# Zona congestionada → segundo pau
{'Marcacao':'Zona','Goleiro':'Fica_Gol','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Altos','Pressao':'Nao','Densidade_Area':'Equilibrado','Qualidade_Cobrador':'Elite','Decisao':'Cruzamento_Segundo_Pau'},

# Desvantagem aérea leve → antecipação
{'Marcacao':'Mista','Goleiro':'Sai_Bem','Estatura_Nosso_Time':'Baixos','Estatura_Adversario':'Altos','Pressao':'Nao','Densidade_Area':'Equilibrado','Qualidade_Cobrador':'Medio','Decisao':'Cruzamento_Primeiro_Pau'},

# Cobrador fraco → jogada construída
{'Marcacao':'Individual','Goleiro':'Sai_Bem','Estatura_Nosso_Time':'Baixos','Estatura_Adversario':'Baixos','Pressao':'Nao','Densidade_Area':'Equilibrado','Qualidade_Cobrador':'Fraco','Decisao':'Jogada_Curta'},

# --------------------------
# C) INFERIORIDADE NUMÉRICA
# --------------------------

# Desvantagem total → evitar duelo
{'Marcacao':'Zona','Goleiro':'Sai_Bem','Estatura_Nosso_Time':'Baixos','Estatura_Adversario':'Altos','Pressao':'Nao','Densidade_Area':'Inferioridade','Qualidade_Cobrador':'Fraco','Decisao':'Jogada_Curta'},

# Inferioridade mas cobrador elite → tentar segundo pau
{'Marcacao':'Zona','Goleiro':'Fica_Gol','Estatura_Nosso_Time':'Baixos','Estatura_Adversario':'Altos','Pressao':'Nao','Densidade_Area':'Inferioridade','Qualidade_Cobrador':'Elite','Decisao':'Cruzamento_Segundo_Pau'},

# Inferioridade leve + marcação individual → primeiro pau
{'Marcacao':'Individual','Goleiro':'Sai_Bem','Estatura_Nosso_Time':'Baixos','Estatura_Adversario':'Altos','Pressao':'Nao','Densidade_Area':'Inferioridade','Qualidade_Cobrador':'Medio','Decisao':'Cruzamento_Primeiro_Pau'},

# Inferioridade mas vantagem aérea isolada
{'Marcacao':'Mista','Goleiro':'Fica_Gol','Estatura_Nosso_Time':'Altos','Estatura_Adversario':'Baixos','Pressao':'Nao','Densidade_Area':'Inferioridade','Qualidade_Cobrador':'Medio','Decisao':'Cruzamento_Ponto_Penalti'}

]

atributos_disponiveis = [
    'Marcacao',
    'Goleiro',
    'Estatura_Nosso_Time',
    'Estatura_Adversario',
    'Pressao',
    'Densidade_Area',
    'Qualidade_Cobrador'
]

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

st.divider()
st.subheader("📊 Variáveis Avançadas")

col3, col4 = st.columns(2)

with col3:
    nossos_jogadores = st.number_input(
        "Jogadores nossos na área",
        min_value=0,
        max_value=9,
        value=5
    )
    
    jogadores_adv = st.number_input(
        "Jogadores adversários na área",
        min_value=0,
        max_value=10,
        value=5
    )

with col4:
    nota_cobrador = st.slider(
        "Qualidade do Cobrador (0-10)",
        min_value=0,
        max_value=10,
        value=7
    )
    
if st.button("Definir Jogada", type="primary"):
    
    situacao_usuario = {
        'Marcacao': marcacao,
        'Goleiro': goleiro,
        'Estatura_Nosso_Time': estatura_nosso,
        'Estatura_Adversario': estatura_adv,
        'Pressao': pressao,
        'Densidade_Area': classificar_densidade(nossos_jogadores, jogadores_adv),
        'Qualidade_Cobrador': classificar_cobrador(nota_cobrador)
    }
    
    try:
        # 1. Primeiro fazemos a predição para saber ONDE posicionar o jogador
        decisao, motivo = predict_with_explanation(arvore_tatica, situacao_usuario)
        
        # 2. Definimos as coordenadas baseadas na decisão da IA
        # Assumindo que o escanteio está sendo cobrado do canto superior direito (x=120, y=80)
        if decisao == 'Cruzamento_Primeiro_Pau':
            pos_x, pos_y = 114, 44  # Próximo à pequena área, lado de cima
            label_jogador = "1º Pau"
        elif decisao == 'Cruzamento_Segundo_Pau':
            pos_x, pos_y = 114, 36  # Próximo à pequena área, lado de baixo
            label_jogador = "2º Pau"
        elif decisao == 'Cruzamento_Ponto_Penalti':
            pos_x, pos_y = 109, 40  # Marca do pênalti exata (11m do gol)
            label_jogador = "Pênalti"
        elif decisao == 'Jogada_Curta':
            pos_x, pos_y = 112, 75  # Perto do escanteio para receber o passe
            label_jogador = "Apoio Curto"
        else:
            pos_x, pos_y = 109, 40  # Fallback padrão
            label_jogador = "Atacante"

        # 3. Desenhamos o campo
        figura_campo = desenhar_campo(cor_campo="#1B5E20", cor_linha="#FFFFFF")

        # 4. Desenhamos o jogador na posição calculada e removemos os sliders manuais
        plt.plot(pos_x, pos_y, 'o', color='red', markersize=12, markeredgecolor='white', markeredgewidth=2)
        plt.text(pos_x, pos_y + 3, label_jogador, color='white', ha='center', fontweight='bold')

        # Mostra o campo no Streamlit
        st.pyplot(figura_campo)
        
        # 5. Exibe os textos explicativos da decisão
        st.divider()
        st.subheader(f"🎯 {decisao.replace('_', ' ').upper()}")
        
        # Colorindo a resposta
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