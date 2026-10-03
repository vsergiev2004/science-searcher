'''
РАБОТА АГЕНТА БАЗИРУЕТСЯ НА ДАННЫХ(ЭМБЕДДИНГАХ И BM25-КОРПУСЕ) ЗАПИСАННЫХ В ПАПКЕ TECHNICAL_DATA
И НА ДАННЫХ(КОРОТКИЕ ОПИСОНИЯ СТАТЕЙ) ЗАПИСАННЫХ В ПАПКЕ SCIENCE_DATA
ЭТИ ДАННЫЕ МОЖНО МЕНЯТЬ ТОЛЬКО СИНХРОННО И СОГЛАСОВАННО, 
ИНАЧЕ ПОД ИНДЕКСОМ ВЫДАВАЕМЫМ ПОИСКОМ ДУМЕТ ХРАНИТЬСЯ НЕ ТА СТАТЬЯ, КОТОРА НУЖНО

ПОДУМАТЬ НАД РАЗВИТИЕМ: ПОНРАВИВШИЕСЯ СТАТЬИ КАЧАТЬ(А НЕ ТОЛЬКО ЗАПИСЫВАТЬ В ФАЙЛ), ДАЛЕЕ ПЕРЕВОДИТЬ И ТЕХАТЬ
НАУЧИТЬ НАХОДИТЬ САМИ СТАТЬИ, ЧИТАТЬ, ТЕХАТЬ.
'''


import os
from dotenv import load_dotenv
import json
import numpy as np
import torch
import pickle
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch.nn.functional as F
from openai import OpenAI
import latex

load_dotenv()

# ─── LLM Client ───────────────────────────────────────────
# Вариант 1: Groq (бесплатно) — ключ на console.groq.com
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-120b"


#Подготовка данных для поиска(делаем перед работой агента, иначе на каждом действии агента будем грузить одно и то же)
SEARCH_MODEL = "allenai/specter2_base"
model = AutoAdapterModel.from_pretrained(SEARCH_MODEL)
tokenizer = AutoTokenizer.from_pretrained(SEARCH_MODEL)

#Грузим доки
docs = []
with open("science_data/only_mathAG_and_mathAC.json") as f:
    for line in f:
        doc = json.loads(line)
        docs.append(doc)

#Грузим эмбеддинги для нейро поиска
embs = np.load("technical_data/doc_embs.npy")

#собираем модель, чтобы правильно токенизировать запрос
ADAPTER = "allenai/specter2_adhoc_query"
model.load_adapter(ADAPTER, sourse='hf', set_active=True)

#Грузим статистический корпус для bm25-поиска
with open("technical_data/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

#################################################################################################
#ПОИСКОВИКИ НЕЙРОНКОЙ И BM25
#Поиск через нейросеть(параметры скачаны, эмбеддинги документов сделаны, остается только найти эмбеддинг запроса)
@torch.no_grad()
def neyro_search(query:str, top_k:int) -> list:
    
    tokens = tokenizer(query, padding=True, truncation=True,
                return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = model(**tokens)
    q_emb = output.last_hidden_state[:,0,:]
    q_emb = F.normalize(q_emb, p=2, dim=1)

    scores = q_emb.numpy() @ embs.T
    top_idx = np.argsort(scores[0])[-top_k:][::-1]
    return [docs[i] for i in top_idx]

#Поиск через bm25
def bm25_search(query: str, top_k:int) -> list:
    bm_tokens = tokenizer.tokenize(query)
    scores = bm25.get_scores(bm_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [docs[i] for i in top_idx]

###################################################################################################
#Тулы нейросети


def search_papers(query: str, k: int = 5, variant: str = "reranked") -> list[dict]:
    """
    variant='simple':   k от BM25 + k от SPECTER2 (раздельно)
    variant='reranked':  расширяем оба набора, реранкер, топ-k из объединённого
    """
    if variant == "simple":
        bm25_results = bm25_search(query, k)          # → list[{id, title, abstract}]
        specter_results = neyro_search(query, k)
        return {
            "bm25": _truncate(bm25_results),
            "specter2": _truncate(specter_results),
        }
    else:
        # расширяем выборку для реранкера
        expand_n = k * 3
        bm25_results = bm25_search(query, expand_n)
        specter_results = neyro_search(query, expand_n)
        combined = _merge_dedupe(bm25_results, specter_results)
        return _truncate(combined[:k])


def find_doc_index(paper_id:str):
    return [i for i, d in enumerate(docs) if d["id"] == paper_id][0]

def find_similar(paper_id: str, k: int = 5) -> list[dict]:
    """Поиск похожих статей через SPECTER2-эмбеддинги (косинусное сходство)."""
    emb = embs[find_doc_index(paper_id)]# вектор выбранной статьи
    scores = embs @ emb.T
    top_indx = np.argsort(scores)[::-1]
    similar = [docs[i] for i in top_indx if str(docs[i]["id"]) != str(paper_id)][:k]#берем k ближайших по косинусу статей
    return _truncate(similar)

# ─── Вспомогательные ─────────────────────────────────────

def _truncate(results, max_abstract=300):
    """Обрезаем abstract, чтобы экономить токены LLM."""
    if isinstance(results, dict):
        return {k: _truncate(v, max_abstract) for k, v in results.items()}
    for p in results:
        if len(p.get("abstract", "")) > max_abstract:
            p["abstract"] = p["abstract"][:max_abstract] + "…"
    return results


def _merge_dedupe(*lists):
    seen, merged, scores = set(), [], {}
    n = len(lists)
    for j in range(n):
        k = len(lists[j])
        for i in range(k):
            if lists[j][i]["id"] not in seen:
                seen.add(lists[j][i]["id"])
                merged.append(lists[j][i])
                scores[lists[j][i]["id"]] = i + k#индекс вхождения в первый список и во сторой(во второй пока не входит)
            else:
                scores[lists[j][i]["id"]] += j - k#меняем ранее неизвестный индекс вхождения во второй массив на реальный индекс

        sorted(merged, key=lambda x: 1/scores[x["id"]], reverse=True)
    return merged


def save_doc(paper_id: str, filename: str) -> str:
    '''
    хочется добавить файл cool_statiy/ids.txt,
    где будут храниться все id-шки в формате set
    при добавлении статьи они будут считываться и проверять,
    что статья не была добавлена раньше
    '''
    doc = docs[find_doc_index(paper_id)]
    with open(filename, "a") as f:
        print(doc, file=f)



def try_tex(text:str, tex_name:str) -> str:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    tex_file = os.path.join(SCRIPT_DIR, "tex", tex_name)
        
    with open(tex_file, "w") as f:
        f.write(text)
    return latex.try_tex(tex_file)


# ─── Описание инструментов для function calling ─────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Поиск научных статей по текстовому запросу. "
                "variant='reranked' — объединяет BM25 и SPECTER2, прогоняет через "
                "реранкер, возвращает топ-k. Используй по умолчанию. "
                "variant='simple' — возвращает k от BM25 и k от SPECTER2 раздельно. "
                "Используй, если пользователь хочет сравнить методы или нужен быстрый ответ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на естественном языке",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Количество результатов передавай именно число, без кавычек",
                        "default": 5,
                    },
                    "variant": {
                        "type": "string",
                        "enum": ["simple", "reranked"],
                        "default": "reranked",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar",
            "description": (
                "Поиск статей, похожих на выбранную. "
                "Используй, когда пользователь просит найти похожие "
                "или хочет копнуть глубже по одной из статей из результатов. "
                "Нужен id статьи из предыдущих результатов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "ID статьи-образца из предыдущих результатов",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Количество похожих статей",
                        "default": 5,
                    },
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_doc",
            "description": (
                "Сохранение выбранной статьи."
                "Используй, когда пользователю понравилась статья"
                "Нужен id статьи из прошлых результатов"
                "-Используй cool_statiy/cool.txt, если пользователь явно просит сохранить статью"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "ID статьи которую нужно сохранить из прошлых результатов",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Название файла, в который нужно сохранить статью.",
                        "default": "cool_statiy/cool.txt",
                    },
                },
                "required": ["paper_id"],
            }
        }
    },
    {
            "type": "function",
            "function": {
                "name": "try_tex",
                "description": (
                    "Создание LaTeX документа"
                    "Используй, когда пользователь просит записать что-то в LaTeX"
                    "Старайся написать коротко, но не упуская ничего важного"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "LaTeX код, который будет запускаться",
                        },
                        "tex_name": {
                            "type": "string",
                            "description": "Название файла в который будет записываться LaTeX код",
                            "default": "default_name",
                        }
                    },
                    "required": ["text"],
                }
            }
        }
]

tool_map = {
    "search_papers": search_papers,
    "find_similar": find_similar,
    "save_doc": save_doc,
    "try_tex": try_tex,
}


# ─── Системный промпт ─────────────────────────────────────

SYSTEM_PROMPT = """\
Ты — научный поисковый агент. Помогаешь исследователям находить статьи \
в базе из 40 000 научных публикаций.

Твои инструменты:
1. search_papers(query, k, variant) — поиск по запросу.
   - 'reranked' (по умолчанию): лучшее качество, реранкер.
   - 'simple': BM25 и SPECTER2 раздельно, для сравнения.
2. find_similar(paper_id, k) — найти похожие на конкретную статью.
3. save_doc(paper_id, filename) — сохранить данные о документе в указанный файл.
4. try_tex(text) — создать LaTeX документ. 

Правила:
- По умолчанию используй variant='reranked'.
- Если пользователь явно просит «сравнить методы» или «оба варианта» — бери 'simple'.
- Формулируй поисковый запрос на английском (статьи, скорее всего, на английском).
- В ответе показывай: id, название и краткое резюме абстракта (2-3 предложения).
- После выдачи результатов предлагай: «Могу найти похожие статьи на любую из них — назови id».
- Если пользователь называет номер/id статьи — вызывай find_similar.
- Не выдумывай статьи. Всё, что отдаёшь, должно приходить из инструментов.
"""


# ─── Агентский цикл ───────────────────────────────────────

def run_agent():
    #load data


    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 60)
    print("  Scientific Paper Search Agent")
    print("  Введите 'quit' для выхода")
    print("=" * 60, "\n")

    while True:
        user_input = input("\nВы: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("До встречи!")
            break

        messages.append({"role": "user", "content": user_input})

        # Цикл вызова инструментов (может быть несколько итераций)
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                print(f"\nАгент: {msg.content}")
                break

            # Выполняем каждый вызов инструмента
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  ⚙ {name}({json.dumps(args, ensure_ascii=False)})")

                try:
                    result = tool_map[name](**args)
                except Exception as e:
                    result = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })


if __name__ == "__main__":\
    run_agent()
