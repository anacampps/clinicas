from shiny import App, ui, render
import os
import matplotlib.pyplot as plt
from dados_auxiliares import regioes


visualizacoes_dir = "output/visualizacoes"

def mostrar_imagem(nome_arquivo):
    caminho = os.path.join(visualizacoes_dir, nome_arquivo)
    if os.path.exists(caminho):
        return ui.img(src=f"/{caminho}", style="max-width: 100%;")
    else:
        return ui.p("Gráfico não encontrado.")
    

app_ui = ui.page_fluid(
    ui.div(
        ui.h1("Portal de Processos do CADE", style="text-align: center; font-size: 2.5rem; margin-bottom: 0.5rem;"),
        ui.h6("texto", style="text-align: center; font-size: 1.2rem; color: #555;"),
        class_="mb-4",
    ),
    ui.page_navbar(
        ui.nav_panel(
            "Início",
            ui.h2("Bem-vindo ao Portal de Processos do CADE"),
            ui.p("Este portal permite visualizar dados sobre os processos do CADE de forma interativa. Ele foi desenvolvido "
            "como projeto de Cliínicas II do curso de Direito do Insper.")
        ),
        ui.nav_panel(
            "Estatísticas",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "grafico",
                        "Escolha o gráfico:",
                        {
                            "boxplot": "Boxplot",
                            "dosimetria": "Dosimetria",
                            "evolucao": "Evolução",
                            "histograma": "Histograma",
                            "features": "Importância das Features",
                            "matriz": "Matriz de Confusão",
                        }
                    )
                ),
                [
                    ui.h2("Estatísticas"),
                    ui.output_ui("painel_grafico")
                ]
            )
        ),
        ui.nav_panel(
            "Dashboard",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_checkbox_group("periodo", "Período:", [
                        "2012", "2013", "2014", "2015", "2016", "2017", 
                        "2018", "2019", "2020", "2021", "2022"
                    ]),
                    ui.input_select("regiao", "Região:", {
                        "norte": "Norte",
                        "nordeste": "Nordeste",
                        "sudeste": "Sudeste",
                        "sul": "Sul",
                        "centro-oeste": "Centro-Oeste"
                    }),
                    ui.input_selectize(
                        "ramos",
                        "Ramos de Atividade:",
                        multiple=True,
                        choices=regioes,  # vindo do seu dados_auxiliares.py
                        selected=[],
                        width="100%",
                    ),
                    ui.input_checkbox_group("agravantes", "Agravantes:", [
                        "reincidência", "obstrução", "dolo"
                    ]),
                    ui.input_checkbox_group("atenuantes", "Atenuantes:", [
                        "colaboração", "baixa gravidade", "arrependimento"
                    ])
                ),
                ui.panel_main(
                    ui.h2("Dashboard Interativo"),
                    ui.output_plot("grafico_dashboard")
                )
            )
        )

    )
)

def server(input, output, session):
    @output
    @render.ui
    def painel_grafico():
        nome_arquivo = {
            "boxplot": "boxplot.png",
            "dosimetria": "dosimetria.png",
            "evolucao": "evolucao.png",
            "histograma": "histograma.png",
            "features": "importancia_features.png",
            "matriz": "matriz_confusao.png"
        }.get(input.grafico(), None)

        if nome_arquivo:
            return mostrar_imagem(nome_arquivo)
        else:
            return ui.p("Selecione um gráfico para visualizar.")
        
    @output
    @render.plot
    def grafico_dashboard():
        # Exemplo: muda cor com base no filtro de região
        cor = {
            "norte": "blue",
            "nordeste": "orange",
            "sudeste": "green",
            "sul": "red",
            "centro-oeste": "purple"
        }.get(input.regiao(), "gray")

        # Gráfico de exemplo
        fig, ax = plt.subplots()
        ax.bar(["A", "B", "C"], [1, 3, 2], color=cor)
        ax.set_title(f"Exemplo para {input.periodo()} - {input.regiao().capitalize()}")
        return fig
    
app = App(app_ui, server)


# Dicionário Ramos de Atividade



