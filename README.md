# NLPBasics

Implementações do zero e estudos guiados de tarefas clássicas em Processamento de Linguagem Natural (PLN).

O objetivo deste repositório é revisar, consolidar e operacionalizar conceitos fundamentais da área — partindo de técnicas estatísticas tradicionais (como Bag-of-Words e N-Grams) até arquiteturas neurais modernas como Transformers e Large Language Models (LLMs).

Mais do que utilizar bibliotecas prontas, a proposta aqui é:

- Entender os algoritmos em nível matemático;

- Implementar versões simplificadas from scratch;

> 📚 Este repositório segue de perto o conteúdo do livro *Speech and Language Processing* (Jurafsky & Martin, 3ª ed.), servindo como caderno de estudos prático.

---

## 🗂️ Estrutura de Diretórios

Devido à amplitude da área, o repositório está organizado em módulos temáticos. Cada diretório contém implementações, anotações e experimentos referentes ao seu respectivo tema.

| # | Tópico | Status |
|---|--------|--------|
| 01 | Introdução | 🔲 To Do |
| 02 | Palavras e Tokens | 🔲 To Do |
| 03 | Modelos de Linguagem com N-Grams | 🔲 To Do |
| 04 | Regressão Logística e Classificação de Texto | 🔲 To Do |
| 05 | Embeddings | 🔲 To Do |
| 06 | Redes Neurais em PLN | 🔲 To Do |
| 07 | Large Language Models | 🔲 To Do |
| 08 | Transformers | 🔲 To Do |
| 09 | Post-training: Instruction Tuning, Alignment e Test-Time Compute | 🔲 To Do |
| 10 | Masked Language Models | 🔲 To Do |
| 11 | Recuperação da Informação | 🟡 Em andamento |
| 12 | Machine Translation | 🔲 To Do |
| 13 | RNNs e LSTMs | 🔲 To Do |
| 14 | Fonética e Extração de Features de Áudio | 🔲 To Do |

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**
- **uv** — gerenciamento de dependências e ambientes virtuais
- **NLTK / spaCy** — pré-processamento e tarefas clássicas de PLN
- **scikit-learn** — modelos de machine learning
- **PyTorch / HuggingFace Transformers** — redes neurais e LLMs
- **Jupyter Notebooks** — experimentos e visualizações

---

## 🚀 Como Usar

### Pré-requisitos

- [Python 3.12](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Instalação
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/NLPBasics.git
cd NLPBasics

# Cria o ambiente virtual e instala as dependências declaradas no pyproject.toml
uv sync
```

### Executando um módulo
```bash
# Ativa o ambiente virtual gerenciado pelo uv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Ou execute diretamente sem ativar o ambiente
uv run python modulo/script.py
```

---

## 📖 Referências

- Jurafsky, D. & Martin, J. H. — *Speech and Language Processing* (3ª ed.)
- [HuggingFace Course](https://huggingface.co/learn/nlp-course)
- [fast.ai NLP](https://www.fast.ai/)
- [Documentação do uv](https://docs.astral.sh/uv/)