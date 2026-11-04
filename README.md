# myproject
FINOR — Index Tracking
======================

Este repositório contém o notebook `finor_index_tracking.ipynb` e um dashboard
Streamlit para explorar o acompanhamento do S&P 100 e do Ibovespa.

## Executar o dashboard

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

O dashboard consulta o Yahoo Finance em tempo de execução, permite escolher o
índice, período e tamanho da carteira, e calcula uma carteira long-only por
mínimos quadrados não negativos. Os dados podem ser exportados pela aba
**Dados**.