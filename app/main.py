import os
import openai
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pypdf

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# FastAPI app setup
app = FastAPI()

# Mount static files (for public folder)
app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="app/templates")

# Configuração do Azure OpenAI
client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") # O nome do deployment

class AnalysisRequest(BaseModel):
    filename: str

class SaveRequest(BaseModel):
    filename: str
    content: str

# Helper function to read PDF content
def read_pdf(file_path: str):
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/files")
async def list_files():
    files = os.listdir("public/analise")
    return files

@app.post("/analyze")
async def analyze_request(req: AnalysisRequest):
    pdf_path = os.path.join("public/analise", req.filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found")

    pdf_text = read_pdf(pdf_path)

    # ====================================================================
    # ALTERAÇÕES AQUI: O prompt agora é formatado como uma lista de mensagens
    # ====================================================================
    messages = [
        {"role": "system", "content": """
    Baseado nos requisitos para treinamento da IA e nas portarias mencionadas, analise o seguinte pedido de diária em formato PDF.
    Verifique se o pedido está em conformidade com as normas, identificando pontos de aprovação e possíveis não conformidades.

    Requisitos e Portarias:
    1. Objetivo
    Fornecer instruções claras e juridicamente fundamentadas para programar a IA a analisar pedidos de diárias no âmbito do Ministério Público do Estado do Pará, garantindo aderência às normas vigentes.
    2. Base Normativa
    A IA deverá considerar, de forma integrada:
    1. Portaria nº 3196/2025-MP/PGJ – Estabelece normas gerais sobre concessão, pagamento e prestação de contas de diárias.
    2. Resolução nº 7/2011-CPJ – Concessão e pagamento de diárias a membros.
    3. Resolução nº 8/2011-CPJ – Concessão e pagamento de diárias a servidores.
    4. Portaria nº 5247/2022-MP/PGJ – Valores de diárias para membros.
    5. Portaria nº 5248/2022-MP/PGJ – Valores de diárias para servidores e militares.
    3. Liberação (Com base na Portaria nº 3196/2025-MP/PGJ):
    	Tipo de solicitante: membro, servidor, militar à disposição, colaborador eventual.
    	Verificar se o requerente possui pendências em outras solicitações de diária:
    	Ausência de prestação de contas no prazo de 15 dias úteis do término da missão (Portaria nº 3196/2025-MP/PGJ, art. 1º, § 5º);
    	Nesta etapa é importante verificar se o processo está com o Requerente no GEDOC (“Responsável Atual”), pois há situações em que já está fora do prazo, porém, o processo encontra-se em outro setor, o que irá descaracterizar a pendência.
    	Finalidade do deslocamento (curso, reunião, mutirão, fiscalização, etc.).
    	Se capacitação, seminários, cursos, etc., haverá limitações na concessão (Portaria nº 3196/2025-MP/PGJ, Art. 8º, § 6º).
    	Origem, destino e distância
    	Não caberá diária para deslocamentos dentro da RMB (LCE nº 027/1995), ou de até 80 km de distância, por trecho rodoviário (Portaria nº 3196/2025-MP/PGJ, Art. 2º, § 1º, I);
    	Há previsão para indeferimento no caso de o deslocamento abranger transporte fluvial, cujo tempo de ida e retorno seja inferior a 4 horas, porém, é de difícil análise e, portanto, sem uso na prática (Portaria nº 3196/2025-MP/PGJ, Art. 2º, § 1º, IV);
    	Ressalta-se que, caso justificado o pernoite, poderá haver o pagamento de diárias ainda que em situações de vedação (Portaria nº 3196/2025-MP/PGJ, Art. 2º, § 2º);
    	Período do deslocamento, data de ida e de volta.
    	O artigo 2º da Portaria nº 3196/2025-MP/PGJ informa como é feito o cálculo do quantitativo. Detalhei no item 4 do presente documento;
    	Verificar se há outros pedidos, do mesmo requerente, de diárias em sobreposição com o analisado, ou seja, para o mesmo período.
    	Documentos obrigatórios anexados.
    	Documentação que motivou a viagem (Portaria nº 3196/2025-MP/PGJ, Art. 1º, § 2º);
    	Se servidor ou militar, ciência da chefia imediata ou autoridade competente (Portaria nº 3196/2025-MP/PGJ, Art. 1º, § 2º);
    	Caso o requerimento tenha como assunto, no GEDOC, pedido “com autorização prévia”, deverá ser apresentada a autorização prévia ou ato designativo (Portaria nº 3196/2025-MP/PGJ, Art. 3º, § 3º);
    	Em pedidos de Membros para atender à designação para exercer atribuições diversas da titularidade, em caráter de acumulação ou designação de oficiar em sessão do Tribunal do Júri, deverão conter o respectivo ato designativo, o qual deverá ser compatível com o período de deslocamento. Ressalta-se que há limite de diárias para Acumulação, salvo se Tribunal do Júri, de 8 diárias por mês (Portaria nº 3196/2025-MP/PGJ, Art. 3º, § 3º c/c Art. 2º, VI)
    	Em pedidos de capacitação, o pedido deverá vir instruído com a manifestação da chefia imediata, caso servidor ou militar, quanto à relevância do curso para as atividades desenvolvidas no órgão. No caso de membro, a instrução será feita pelo próprio requerente quanto à relevância (Portaria nº 3196/2025-MP/PGJ, Art. 8º, § 1º). Além disso, o CEAF deverá se manifestar quanto à adequação do pedido (Portaria nº 3196/2025-MP/PGJ, Art. 8º, § 2º). A solicitação deverá conter a programação completa do evento (Portaria nº 3196/2025-MP/PGJ, Art. 8º, § 1º, IV);
    	Não há fundamento nas normativas internas, porém, as solicitações de diárias com a finalidade de “prestar apoio administrativo/substituição de servidor”, deverão ser instruídas com a autorização da Sub-TA para o deslocamento. Ressalta-se que essa regra se aplica exclusivamente às solicitações feitas por servidores.
    	Quando colaborador eventual e tratar-se de capacitação, o pedido deverá vir instruído com o currículo do colaborador.
    	Antecedência do pedido (prazo mínimo exigido).
    	Se missão, 7 dias úteis da data da atividade (Portaria nº 3196/2025-MP/PGJ, artigo 1º, § 2º);
    	Se a missão é sigilosa, o requerimento poderá ser feito em momento posterior ao deslocamento (Portaria nº 3196/2025-MP/PGJ, Art. 5º, § 2º);
    	Se colaborador eventual, 15 dias úteis da data do evento (Portaria nº 3196/2025-MP/PGJ, artigo 9º, § 1º);
    	Se capacitação, 30 dias corridos da data do evento (Portaria nº 3196/2025-MP/PGJ, artigo 8º, § 1º, I);

    4. Avaliar quantitativo (detalhamento)
    	Identificação do período de deslocamento
    	Considera-se do dia de início da viagem até o dia de chegada ao município de lotação (Art. 2º, I).
    	Inclui-se no cômputo dias de deslocamento e dias efetivos de missão oficial.
    	Percentual da diária a aplicar
    	100% do valor: quando houver pernoite fora da sede.
    	50% do valor: quando não houver pernoite/dia do retorno.
    	25% do valor: quando hospedagem e alimentação forem custeadas por órgão/entidade da Administração Pública.
    	Restrições que excluem o pagamento de diária
    	Deslocamento na Região Metropolitana de Belém ou até 80 km de distância, salvo se houver pernoite (Art. 2º, §1º, I).
    	Deslocamento que constitua exigência habitual do cargo, exceto nos casos específicos previstos (Art. 2º, §1º, II).
    	Deslocamento fluvial cujo tempo total de ida e volta seja inferior a 4 horas (Art. 2º, §1º, IV).
    	Situações especiais
    	Finais de semana e feriados: pagamento apenas se expressamente demonstrada a necessidade de continuidade da missão (Art. 2º, IV).
    	Os deslocamentos devem ocorrer, preferencialmente, no mesmo dia de início e término da atividade. Assim, do contrário, o requerente deverá demonstrar necessidade de deslocamento em outras datas (Art. 6º, § 3º).
    	Limite máximo de 10 diárias por ato autorizador, salvo exceções autorizadas pelo Procurador-Geral de Justiça (Art. 15).
    	Para qualificação, ressalvada conveniência da Administração Superior (Art. 8º, § 6º):
    	Até duas capacitações por exercício financeiro, com o pagamento máximo de três diárias por capacitação, quando realizadas em território nacional;
    	Alternativamente, uma capacitação internacional por exercício financeiro, com o pagamento de até três diárias.
    	Valores aplicáveis
    	Definidos conforme cargo/função e natureza do deslocamento:
    	Membros: Portaria nº 5247/2022-MP/PGJ.
    	Servidores e militares: Portaria nº 5248/2022-MP/PGJ.

    5. Prestação de Contas – Etapas e Requisitos (Portaria nº 3196/2025-MP/PGJ, Art. 10)
    	A prestação de contas é obrigatória para todos os beneficiários de diárias, devendo comprovar a realização do deslocamento e o cumprimento da missão institucional. O processo segue as etapas abaixo:
    	5.1. Prazo para Apresentação
    	Prazo geral: até 15 (quinze) dias úteis contados a partir do término da missão (Art. 1º, §5º).
    	Relatório de viagem: prazo máximo de 30 (trinta) dias úteis para saneamento de pendências; o descumprimento suspende o pagamento de futuras solicitações (Art. 10, §8º).
    	Nas Resoluções nº 7/2011 e nº 8/2011, o prazo para comprovação do deslocamento é de 15 dias corridos.

    	5.2. Documento Principal
    	Relatório de Viagem (Anexo II) contendo:
            1.	Identificação do beneficiário e dados da viagem.
            2.	Objetivo do deslocamento.
            3.	Atividades desenvolvidas.
            4.	Resultados alcançados.
            5.	Informações sobre o meio de transporte utilizado.
            6.	Documentação comprobatória do deslocamento.
            7.	Indenizações ou restituições, quando houver.
            8.	Assinatura do beneficiário e da chefia imediata.

    	5.3. Documentos Comprobatórios
    	São aceitos como prova do deslocamento (Art. 10, §1º):
            Cartões de embarque ou bilhetes de passagens rodoviárias, fluviais ou aéreas. (GEDOC nº 138671/2025).
            Ficha de circulação do veículo oficial (bastante comum).
            Exemplos de GEDOCs com uso de carro oficial e, consequentemente, relatório de circulação: 128175/2025; 128142/2025; 127302/2025; 130602/2025; 
            Certificados de participação em cursos, seminários ou eventos.
            Outros documentos válidos que comprovem a execução da missão institucional.
            Declaração de próprio punho, quando não houver documentação, devendo ser referendada pela chefia imediata (Art. 10, §2º).
            Importante: Documentos judiciais ou extrajudiciais eletrônicos não são aceitos como comprovação (Art. 10, §5º).
            A partir da análise da documentação é possível identificar se o requerente deverá devolver valores, decorrente de deslocamento efetuado em período com quantitativo inferior ao concedido.
            Ressalta-se que nem sempre os documentos comprobatórios do deslocamento vêm anexos ao Relatório de Viagem, ou seja, não estão presentes logo abaixo dessa etapa no protocolo. Tal fato ocorre principalmente em viagens pretéritas, em que o requerente anexa, já na etapa inicial de solicitação, os documentos que comprovam o deslocamento. Ex.: GEDOC nº 122841/2025.
            Necessário identificar, caso ocorra alteração no período de deslocamento, se o requerente não apresenta solicitação de diárias para o novo período.

    	5.4. Análise e Aprovação
    	O ordenador de despesas aprovará expressamente ou impugnará as contas apresentadas, com base na análise da documentação (Art. 10, §§ 3º e 9º).
    	A Administração pode exigir apresentação dos originais por até 5 anos a contar da data do final evento (Art. 10, §6º).
    	A ausência de comprovação dentro do prazo implica devolução dos valores recebidos, com atualização monetária, podendo ser descontado em folha (Art. 10, §7º).

    	5.5. Restituição e Complementação
    	Complementação de diárias ou ressarcimento de despesas com alimentação e transporte: solicitados no relatório de viagem (Art. 11).
    	Restituição de valores: depósito em conta do MP, protocolizado com o relatório (Art. 13). O não pagamento em até 15 dias úteis gera desconto em folha.

    - [Portaria n° 3196/2025-MP/PGJ]
    - [Resolução n° 7/2011-CPJ]
    - [Resolução n° 8/2011-CPJ]
    - [Portaria n° 5247/2022-MP/PGJ]
    - [Portaria n° 5248/2022-MP/PGJ]
    
    Conteúdo do Pedido de Diária:
    {pdf_text}

    Análise detalhada:
    """},
        {"role": "user", "content": f"""Analise o seguinte pedido de diária:

        --- Início do Pedido ---
        {pdf_text}
        --- Fim do Pedido ---

        Forneça uma análise detalhada, destacando claramente se o pedido está em conformidade ou não, e justifique cada ponto com base nas regras e portarias fornecidas."""}
    ]

    try:
        # ====================================================================
        # ALTERAÇÃO AQUI: Mudança da chamada de 'completions.create' para 'chat.completions.create'
        # e o parâmetro 'prompt' foi substituído por 'messages'
        # ====================================================================
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages, # <-- Agora passamos a lista de mensagens
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
        )
        # ====================================================================
        # ALTERAÇÃO AQUI: Acessando o conteúdo da resposta para chat completions
        # ====================================================================
        analysis_result = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erro na chamada da API da OpenAI (chat completions): {e}") # Para depuração
        raise HTTPException(status_code=500, detail=f"Erro ao analisar o pedido: {str(e)}")

    return {"analysis": analysis_result}

@app.post("/save")
async def save_analysis(req: SaveRequest):
    save_path = os.path.join("public/resultados", req.filename)
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(req.content)
    return {"message": "Analysis saved successfully"}

# Main entry point for uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)