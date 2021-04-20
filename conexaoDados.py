import jaydebeapi
from tratamentos import pickles
import pandas as pd
from datetime import date, timedelta
## Variuaveis
pathDremioJDBC = 'dremio-jdbc-driver-14.0.0-202103011714040666-9a0c2e10.jar' ## Caminho para driver JDBC do Dremio.
usr_dremio = "" ## Usuario acesso Dremio
pswd_dremio = "" ## Senha acesso Dremio

## Consulta Base Dremio - Retorna dados do ultimo dia de recepção dos dados
sqlEOF =  'SELECT "Exercício do orçamento (Ano)" ' \
                    ' ,"Órgão (Código/Nome)"' \
                    ' ,"Órgão Sucessor Atual (Código/Nome)"' \
                    ' ,"Tipo Administração (Nome)"' \
                    ' ,"Tipo Poder (Nome)"' \
                    ' ,"Classificação orçamentária (Descrição)"' \
                    ' ,"Função (Cod/Nome)"' \
                    ' ,"Subfunção (Cod/Nome)"' \
                    ' ,"Programa (Cod/Nome)"' \
                    ' ,"Ação (Cod/Nome)"' \
                    ' ,"Natureza Despesa (Cod)"' \
                    ' ,"Natureza Despesa (Nome)"' \
                    ' ,"Grupo Despesa (Cod/Nome)"' \
                    ' ,"Elemento Despesa (Cod/Nome)"' \
                    ' ,"Formalidade (Nome)"' \
                    ' ,"Modalidade Licitação (Nome)"' \
                    ' ,"Fonte Recurso (Cod)"' \
                    ' ,"Fonte Recurso (Nome)"' \
                    ' ,"Beneficiário (Nome)"' \
                    ' ,"Beneficiário (CPF)"' \
                    ' ,"Beneficiário (CNPJ)"' \
                    ' ,"Beneficiário (CPF/CNPJ)"' \
                    ' ,"Período (Dia/Mes/Ano)"' \
                    ' ,"Empenho (Sequencial Empenho)"' \
                    ' ,"Empenho (Histórico)"' \
                    ' ,"Empenho (Número do Processo)"' \
                    ' ,"Valor Empenhado"' \
                    ' ,"Valor Anulação Empenho"' \
                    ' ,"Valor Estorno Anulação Empenho"' \
                    ' ,"Valor Cancelamento Empenho"' \
                    ' ,"Valor Anulação Cancelamento Empenho"' \
                    ' ,"Valor Liquidação Empenho"' \
                    ' ,"Valor Anulação Liquidacao Empenho"' \
                    ' ,"Valor Ordem de Pagamento"' \
                    ' ,"Valor Guia Recolhimento"' \
                    ' ,"Valor Anulação Ordem de Pagamento"' \
                    ' ,"Valor Estorno Anulação O. Pagamento"' \
                    ' ,"Valor Estorno Guia Recolhimento"' \
                    ' ,"Valor Saldo do Empenho"' \
                    ' ,"Valor Saldo Liquidado"' \
                    ' ,"Valor Saldo Pago"' \
                    ' ,"Valor Saldo a Pagar"' \
                    ' ,"Valor a Liquidar"' \
                    ' ,"Valor a Pagar Liquidado" ' \
                    ' FROM IE.EXTRACAO.EOFClassificacao c ' \

def todos_dados():
    conn = jaydebeapi.connect("com.dremio.jdbc.Driver", "jdbc:dremio:direct=bdata01.tce.go.gov.br:31010", [usr_dremio, pswd_dremio], pathDremioJDBC)
    curs = conn.cursor()
    curs.execute(sqlEOF+' WHERE c."Exercício do orçamento (Ano)" >= 2015')
    nomeCampos = [i[0] for i in curs.description]
    df = pd.DataFrame(curs.fetchall(),columns=nomeCampos)
    pickles.criaPickle(df,"df")
    return df

def range_dados(valor = 0):
    conn = jaydebeapi.connect("com.dremio.jdbc.Driver", "jdbc:dremio:direct=bdata01.tce.go.gov.br:31010", [usr_dremio, pswd_dremio], pathDremioJDBC)
    curs = conn.cursor()
     # Se nao receber parametros avalia a ultima data (ultima data e o dia anterior)
    if(valor == 0):
        valor = (date.today()-timedelta(days=1)).strftime('%Y/%m/%d')
        valor = "\'"+valor+"\'"
        curs.execute(sqlEOF+' WHERE c."Período (Dia/Mes/Ano)" = dia'.replace("dia",valor))
    elif("a" in valor):
        inicio = valor.split(" ")[0]
        fim =valor.split(" ")[2]
        curs.execute(sqlEOF+' WHERE c."Período (Dia/Mes/Ano)" >= inicio and "Período (Dia/Mes/Ano)" <= fim'.replace("inicio",inicio).replace("fim",fim))
    else:
        curs.execute(sqlEOF+' WHERE c."Período (Dia/Mes/Ano)" = dia'.replace("dia",valor))
    nomeCampos = [i[0] for i in curs.description]
    df = pd.DataFrame(curs.fetchall(),columns=nomeCampos)
    pickles.criaPickle(df,"dados_teste")
    return df
