# 📊 Advanced Analytics & Growth Economics: Segmentação RFM-NPS e Otimização de LTV/Retention para Streaming (Claro tv+)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Business-Impact](https://img.shields.io/badge/Impact-R%24%204.0M%20Margem%2F90d-green?style=flat-square)

## 📌 Executive Summary

Este repositório contém um estudo fim a fim de **Advanced Analytics, Pricing & Retention Engineering** aplicado a uma base sintética de **1,92 milhão de assinaturas de streaming (24 meses de histórico)** e **734 mil assinantes ativos**[cite: 3].

O objetivo central foi diagnosticar e estancar a evasão de receita no horizonte de 90 dias — onde **22,4% da base ativa cancelava (164 mil clientes e R$ 12,6M/mês de receita em risco)**[cite: 3].

A solução integra:
1. **Modelagem de Segmentação Comportamental ($k$-Means RFM-NPS)** em 6 grupos acionáveis[cite: 2].
2. **Análise de Vazamentos de Canal & Unit Economics** (Fatura vs. Cartão de Crédito, Mídia Paga vs. Orgânico, Elasticidade de Cupons)[cite: 2, 3].
3. **Plano Operacional de CRM & Testes A/B (Pilotagem em 90 dias)** projetando um impacto líquido de **R$ 4,0M por ciclo** (-R$ 0,1M no cenário pessimista a R$ 5,4M no otimista) com um teto orçamentário de R$ 150k[cite: 3].

---

## 📐 Formulação Matemática & Metodologia

### 1. Construção das Variáveis de Entrada
A segmentação foi realizada na janela observada $T_0$ (26/07/2026), rastreando a taxa de churn nos 90 dias subsequentes[cite: 2]. As variáveis do vetor de características $\mathbf{X}$ foram definidas como:

* **Recência ($R$):** $R_i = \text{dias sem acesso (login/streaming)}$[cite: 2].
* **Frequência ($F$):** Ciclos faturados + consumo transactional ($F_i = \text{tenure}_i + (\text{aluguéis TVOD} + \text{compras TVOD}) \times \text{frac}_i$)[cite: 2].
* **Valor Monetário ($M$):** Receita acumulada paga corrigida por promoções[cite: 2].
* **Exposição a Promoção ($\text{cupom}$):** Total de cupons de entrada e TVOD utilizados[cite: 2].
* **Engajamento Especial ($\text{presente}$):** Indicador binário de adesão via degustação/presente[cite: 2].
* **Satisfação ($\text{NPS}$):** Nota atribuída no levantamento de satisfação do cliente[cite: 2].

### 2. Preprocessamento e Transformação Log-Standard
Para corrigir a assimetria à direita inerente a métricas de engajamento e receita, aplicou-se a transformação logarítmica $x' = \ln(1 + x)$ seguida de padronização por $Z$-score:

$$Z = \frac{\ln(1 + X) - \mu}{\sigma}$$

### 3. Validação do Número de Clusters ($k$) & Estabilidade
Foram avaliados $k \in [3, 7]$ utilizando **Silhouette Score**, **Inércia**, **Davies-Bouldin**, **Calinski-Harabasz** e o **Adjusted Rand Index (ARI)** entre 5 sementes/amostras distintas ($n = 150.000$)[cite: 2].

| $k$ | Inércia | Silhouette | Davies-Bouldin | ARI Médio (Estabilidade) | Leitura Técnica[cite: 2] |
|:---:|:-------:|:----------:|:--------------:|:------------------------:|:------------------|
| 3 | 537.352 | **0.279** | 1.251 | 0.766 | Estrutura instável (ARI min 0,43)[cite: 2]. |
| 4 | 450.122 | 0.254 | 1.304 | **0.995** | Aglutina públicos *Esfriando* e *Novatos*[cite: 2]. |
| 5 | 401.839 | 0.241 | 1.390 | 0.951 | Estabilidade equivalente a $k=6$[cite: 2]. |
| **6** | **367.848** | **0.222** | **1.357** | **0.947** | **Escolha Ótima: Separa *Esfriando (Alto Valor)* de *Quase Perdidos***[cite: 2]. |
| 7 | 341.657 | 0.216 | 1.301 | 0.867 | Perde estabilidade e dispersa a operação[cite: 2]. |

> **Nota de Advanced Analytics:** O Silhouette Score decresce continuamente ($0.28 \to 0.22$), indicando que a base possui uma estrutura comportamental contínua (sem "ilhas" isoladas)[cite: 2]. A escolha de $k=6$ foi orientada pela **estabilidade do ARI (0.947)** e pela **utilidade de decisão de negócios**, permitindo diferenciar estratégias para clientes com alto valor retido vs. clientes em esgotamento de ciclo[cite: 2].

---

## 🎯 Resultados da Segmentação Comportamental ($k=6$)

A base ativa de 670,6k assinantes analisada gerou os seguintes perfis comportamentais e métricas de risco[cite: 2]:


---

## 💰 Modelo de Simulação Econômica (Plano de Testes 90 Dias)

Para sanar os gargalos, foi formulado um plano piloto em **10% da base** com simulação Monte Carlo/Cenários para 90 dias, limitado a R$ 150k de custo de contato/desconto variável[cite: 3]:

### Estrutura de Retorno do Caso Base (R$ 4,0M Líquidos / Ciclo)[cite: 3]
* **Margem Incremental Bruta:** +R$ 5,47M (28 mil assinantes salvos: 15k via CRM/Ações de Retenção + 13k via Dunning/Recuperação de Pagamento)[cite: 3].
* **Custo Incremental de Desconto:** -R$ 1,30M[cite: 3].
* **Custo de Contato Operacional:** -R$ 0,17M[cite: 3].
* **Margem Líquida Projetada:** **+R$ 4,00M** por ciclo[cite: 3].

| Cenário Simulado | Premissa de Salvamento | Retenção Incremental | Margem Líquida Calculada[cite: 3] |
|:---|:---|:---:|:---:|
| **Pessimista** | Salvamento reduzido pela metade e permanência extra de apenas 3 meses | ~14.000 clientes | **-R$ 0,1 M**[cite: 3] |
| **Base** | Eficiência nominal de dunning e réguas segmentadas por $k$-means | ~28.000 clientes | **+R$ 4,0 M**[cite: 3] |
| **Otimista** | Salvamento 25% superior com conversão de Presenteados | ~35.000 clientes | **+R$ 5,4 M**[cite: 3] |

---

## 🛠️ Arquitetura do Repositório

```bash
.
├── data/
│   ├── clientes_ativos_segmentados.csv.gz  # Dataset final de saída com scores/grupos
│   └── Estudo_RFM_ClaroTV_v3.xlsx          # Planilha modelo com calculadoras financeiras
├── src/
│   └── segmentacao_crm.py                  # Pipeline completo de Engenharia de Dados & ML
├── docs/
│   ├── video+_resumo_executivo_cmo.pptx     # Apresentação executiva para C-Level
│   └── CRM_Segmentacao_Comportamental.xlsx # Matriz de validação dos clusters e ABRs
└── README.md                               # Documentação técnica do projeto
