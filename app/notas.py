import joblib
import pandas as pd
import streamlit as st


def cria_campos():
    df = pd.read_csv('../data/transformed/notas.csv')

    lista_campos = []

    colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
    colunas_numericas = [coluna for coluna in df.columns if coluna not in colunas_categoricas]

    print(colunas_numericas)

    col1, col2, col3 = st.columns(3)

    print(df.columns)

    for i, coluna in enumerate(df.columns):
        if coluna in colunas_numericas:
            minimo = min(df[coluna])
            maximo = max(df[coluna])

            if i % 3 == 0:
                campo = col1.slider(coluna, minimo, maximo)
            elif i % 3 == 1:
                campo = col2.slider(coluna, minimo, maximo)
            else:
                campo = col3.slider(coluna, minimo, maximo)
        else:
            niveis = df[coluna].unique()

            if i % 3 == 0:
                campo = col1.selectbox(coluna, niveis)
            elif i % 3 == 1:
                campo = col2.selectbox(coluna, niveis)
            else:
                campo = col3.selectbox(coluna, niveis)

        lista_campos.append(campo)

    df.loc[-1] = lista_campos  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index

    df = pd.get_dummies(df, drop_first=True)

    valores = df.iloc[0].values

    return valores


def predicao_notas(caracteristicas):
    '''
    Função para realizar a predição de notas
    '''

    modelo = joblib.load('./models/regressor.pkl')
    nota = modelo.predict(caracteristicas.reshape(1, -1))

    return nota


# Criando a interface

cabecalho = st.container()
features = st.container()
resultado = st.container()

with cabecalho:
    st.image('./img/header.png', width=900)
    st.write('\n')
    st.title("Previsão de Notas de Alunos")

with features:
    st.subheader('Informe os dados do aluno')

    campos = cria_campos()

    if st.button("Prever nota do aluno!"):
        caracteristicas = campos

        predicao = predicao_notas(caracteristicas)

        with resultado:
            st.text(f"SERÁ QUE PASSOU DE ANO???? NOTA {predicao}!")
