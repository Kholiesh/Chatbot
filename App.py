import streamlit as st
import asyncio
from openai import OpenAI
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
import numpy as np

# --- PENGATURAN KONFIGURASI ---
# Ambil kunci API dari Streamlit Secrets
try:
    OPENROUTER_API_KEY = st.secrets["LLM_BINDING_API_KEY"]
except KeyError:
    st.error("Kunci API LLM tidak ditemukan. Harap atur 'LLM_BINDING_API_KEY' di Streamlit Secrets.")
    st.stop()
    
# Inisialisasi klien OpenAI dengan host OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
DEFAULT_MODEL = "google/gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-3-large"
LIGHTRAG_EMBEDDING_DIM = 3072

# --- FUNGSI ASYNC UNTUK INJEKSI LIGHTRAG ---
async def openai_compatible_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Filter out unknown arguments from LightRAG
    kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['hashing_kv']}

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            **kwargs_filtered
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memanggil API LLM: {e}")
        return f"Terjadi kesalahan: {e}"

async def openai_embed(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return np.array([data.embedding for data in response.data])

# --- INITIALISASI LIGHTRAG ---
# PENTING: Lakukan inisialisasi di luar kelas agar hanya dijalankan sekali
rag_model = LightRAG(
    llm_model_func=openai_compatible_complete,
    embedding_func=EmbeddingFunc(
        embedding_dim=LIGHTRAG_EMBEDDING_DIM,
        func=openai_embed
    )
)
try:
    asyncio.run(rag_model.initialize_storages())
    asyncio.run(initialize_pipeline_status())
except RuntimeError:
    pass

# --- Kumpulan Template Prompt Simulasi ---
BASE_PROMPT_TEXT = """Anda adalah Alma Learn, seorang pelatih pegawai yang supportif dan ramah. Anda sedang berbicara dengan '{user_name}' ({user_role}). Balas SELALU dalam Bahasa Indonesia. Gunakan informasi yang diberikan untuk menjawab.
ATURAN UTAMA: Balas HANYA dalam BAHASA INDONESIA. Seluruh teks dari awal hingga akhir harus dalam Bahasa Indonesia."""

QA_INSTRUCTION_TEMPLATE = f"""{BASE_PROMPT_TEXT}
RIWAYAT OBROLAN SEBELUMNYA:
---
{{history_text}}
---
TUGASMU ADALAH MENJAWAB PERTANYAN PENGGUNA BERDASARKAN KONTEKS YANG DIBERIKAN. CANTUMKAN SUMBER YANG DIGUNAKAN.
"""

QUIZ_GENERATION_TEMPLATE = f"""{BASE_PROMPT_TEXT}
PERAN ANDA SEKARANG ADALAH PEMBUAT KUIS. Berdasarkan topik dan konteks yang Anda terima, buatlah SATU pertanyaan kuis (esai singkat atau pilihan ganda, pilih salah satu yang paling sesuai) yang relevan untuk menguji pemahaman pengguna. Ajukan pertanyaan itu secara langsung dan jelas. Anda boleh memberikan penjelasan umum yang membantu pengguna memahami pertanyaan, tetapi jangan berikan jawabannya dalam informasi tersebut.
"""

QUIZ_EVALUATION_TEMPLATE = f"""{BASE_PROMPT_TEXT}
PERAN ANDA SEKARANG ADALAH EVALUATOR. Pertanyaan kuis sebelumnya adalah: '{{quiz_question}}'. Jawaban dari pengguna adalah: '{{user_answer}}'. Berdasarkan konteks RAG, evaluasi apakah jawaban tersebut benar. Berikan apresiasi jika benar, atau berikan koreksi yang ramah dan semangat jika salah.
"""

SIM_GENERATE_PROMPT = """
ATURAN UTAMA: GUNAKAN BAHASA INDONESIA. Seluruh teks dari awal hingga akhir harus dalam Bahasa Indonesia.

Anda adalah seorang perancang skenario simulasi pelatihan. Berdasarkan informasi dan konteks yang diberikan, buatlah sebuah skenario simulasi yang detail.
Topik yang dirancang sebagai berikut: {topic}
Skenario ini harus mencakup:
1.  Peran yang akan Anda mainkan sebagai seorang pelanggan.
2.  Situasi spesifik yang terjadi.
ATURAN: 
1. GUNAKAN KONTEKS YANG DIBERIKAN SEBAGAI SUMBER DALAM MERANCANG SKENARIO NYA.
2. HANYA TERDAPAT DUA PERAN: PELANGGAN DAN {user_role}.

Setelah membuat skenario, buatlah satu kalimat percakapan yang mengawali peranmu sebagai pelanggan pada skenario yang telah dibuat.

Format balasan Anda HARUS seperti ini:
SCENARIO_START
[Tulis skenario detail di sini]
SCENARIO_END
[Tulis kalimat pembuka di sini]
"""

SIM_INTERACTION_PROMPT = """
Anda sedang berada dalam mode simulasi. Lanjutkan percakapan simulasi ini. 
SKENARIO UTAMA:
---
{scenario}
---

RIWAYAT PERCAKAPAN SEBELUMNYA:
---
{history}
---
Peran anda adalah sebagai pelanggan. Anda sedang berinteraksi dengan pegawai yang berperan sebagai {user_role} Anda HARUS tetap dalam peran Anda. Jangan keluar dari peran Anda.

Balas pesan terakhir dari pengguna sesuai dengan peran Anda dalam skenario. GUNAKAN BAHASA INDONESIA YANG BAIK DAN BENAR. Tidak perlu menjelaskan kembali skenario yang diberikan! Cukup lanjutkan percakapan dengan menanggapi pesan terakhir dari pengguna.
"""

SIM_EVALUATION_PROMPT = """
ATURAN UTAMA: Balas HANYA dalam BAHASA INDONESIA. Seluruh teks dari awal hingga akhir harus dalam Bahasa Indonesia.
Anda adalah Alma Learn, seorang pelatih pegawai di Toko GoBIG. Simulasi baru saja berakhir.

SKENARIO YANG DIJALANKAN:
---
{scenario}
---

TRANSKRIP LENGKAP SIMULASI:
---
{history}
---
PENTING: Dalam transkrip di atas, pesan dengan label 'user' adalah dari '{user_name}' yang berperan sebagai '{user_role}'. Pesan dengan label 'assistant' adalah dari AI yang berperan sebagai pelanggan.

Tugas Anda: Berdasarkan transkrip, berikan evaluasi yang **SINGKAT DAN TO THE POINT** terhadap kinerja '{user_name}' dalam perannya sebagai '{user_role}'.

Gunakan format poin-poin berikut:
- **👍 Poin Positif:** (Sebutkan 1-2 hal yang sudah bagus dalam satu kalimat).
- **💡 Area Peningkatan:** (Sebutkan 1 saran perbaikan paling penting dalam satu kalimat).
- **⭐ Kesimpulan:** (Berikan satu kalimat penyemangat).
Gunakan konteks yang ada untuk memberikan umpan balik yang relevan.
DORONG PENGGUNA UNTUK TERUS BELAJAR DAN MEMPERBAIKI DIRI.
"""

class Chatbot:
    
    def __init__(self, model: str):
        self.model = model
        self._initialize_session_state()

    def _initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.quiz_state = None
            st.session_state.simulation_context = None
    
    def login(self):
        col1, col2, col3, col4, col5 = st.columns((1, 2, 1, 2, 1))
        with col3:
            st.image("https://raw.githubusercontent.com/alifrizqullah/Chatbot-AlmaLearn-GoBIG/main/container/wave.png", width=100)
        st.info("Mari berkenalan dengan Alma!")
        with st.form("login_form"):
            name = st.text_input("Nama Anda", placeholder="Masukkan nama Anda")
            role = st.selectbox("Pilih Peran", ("Pramuniaga", "Admin Sosial Media", "Kasir","Host Live","Pemasang Senar"))
            mode = st.selectbox("Mode Interaksi", ("Tanya-Jawab", "Kuis Interaktif", "Simulasi"))
            submit_button = st.form_submit_button("Mulai Chat")
            if submit_button and name and role:
                st.session_state.user_name = name
                st.session_state.user_role = role
                st.session_state.user_mode = mode
                st.session_state.messages.append({"role": "assistant", "content": f"Halo {name}! Saya Alma, siap membantu Anda mengembangkan peran sebagai {role} di Toko GoBIG."})
                st.rerun()

    def get_response(self, user_query: str, instruction: str):
        final_prompt_structure = f"/[{instruction}] {user_query}"
        
        try:
            result = asyncio.run(rag_model.aquery(final_prompt_structure))
            return result
        except Exception as e:
            return f"Terjadi kesalahan: {e}"

    def tampilan(self):
        user_name = st.session_state.user_name
        user_role = st.session_state.user_role
        mode = st.session_state.user_mode

        st.sidebar.header("Status Sesi:")
        st.sidebar.info(f"**Pengguna:** {user_name}")
        st.sidebar.warning(f"**Peran:** {user_role}")
        st.sidebar.success(f"**Mode:** {mode}")
        
        if st.sidebar.button("Mulai Sesi Baru"):
            for key in ["user_name","user_role","user_mode","messages","quiz_state","simulation_context"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        is_waiting_for_answer = (mode == "Kuis Interaktif" and st.session_state.quiz_state and st.session_state.quiz_state['status'] == 'menunggu_jawaban')

        if is_waiting_for_answer:
            if st.button("🔄 Ganti Topik Kuis"):
                st.session_state.messages.append({"role": "assistant", "content": "Baik, kuis sebelumnya dibatalkan. Silakan masukkan topik baru untuk kuis di bawah ini."})
                st.session_state.quiz_state = None
                st.rerun()

        if is_waiting_for_answer:
            placeholder_text = f"Ketik jawaban Anda untuk kuis topik: {st.session_state.quiz_state['topik']}"
        elif mode == "Kuis Interaktif":
            placeholder_text = "Masukkan topik kuis, misalnya (SOP Refund)"
        elif mode == "Simulasi":
            placeholder_text = "Masukkan topik simulasi yang Anda inginkan..."
        else:
            placeholder_text = "Mau belajar apa hari ini?"

        if st.session_state.user_mode and mode != "Simulasi":
            if prompt := st.chat_input(placeholder_text, disabled=False):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("Alma learn sedang berpikir..."):
                        history_parts = []
                        for msg in st.session_state.messages[:-1]: 
                            history_parts.append(f"{msg['role']}: {msg['content']}")
                        history_text = "\n".join(history_parts)
                        
                        final_instruction = ""
                        query_for_rag = ""
                        is_quiz_generation = False

                        if mode == "Kuis Interaktif":
                            if is_waiting_for_answer:
                                quiz_info = st.session_state.quiz_state
                                final_instruction = QUIZ_EVALUATION_TEMPLATE.format(
                                    user_name=user_name, user_role=user_role,
                                    history_text=history_text, 
                                    quiz_question=quiz_info['soal'], 
                                    user_answer=prompt,
                                )
                                query_for_rag = quiz_info['topik']
                                st.session_state.quiz_state = None
                            else:
                                final_instruction = QUIZ_GENERATION_TEMPLATE.format(
                                    user_name=user_name, user_role=user_role,
                                    history_text=history_text
                                )
                                topic = prompt.split(":", 1)[-1].strip() if ":" in prompt else prompt.split("tentang", 1)[-1].strip() if "tentang" in prompt else prompt
                                query_for_rag = topic
                                st.session_state.quiz_state = {'status': 'menunggu_jawaban', 'topik': topic, 'soal': ''}
                                is_quiz_generation = True
                        else:
                            final_instruction = QA_INSTRUCTION_TEMPLATE.format(
                                user_name=user_name, user_role=user_role,
                                history_text=history_text
                            )
                            query_for_rag = prompt
                            if st.session_state.quiz_state:
                                st.session_state.quiz_state = None

                        assistant_response = self.get_response(user_query=query_for_rag, instruction=final_instruction)
                        
                        if is_quiz_generation:
                            st.session_state.quiz_state['soal'] = assistant_response

                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    st.rerun()
        else:
            if st.session_state.simulation_context is None:
                if prompt := st.chat_input(placeholder_text, disabled=False):
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    with st.chat_message("assistant"):
                        with st.spinner("Alma sedang merancang skenario..."):
                            instruction = SIM_GENERATE_PROMPT.format(topic=prompt, user_role=user_role)
                            response = self.get_response(user_query=prompt, instruction=instruction)

                            if response:
                                try:
                                    parts = response.split("SCENARIO_END")
                                    scenario_text = parts[0].replace("SCENARIO_START", "").strip()
                                    opening_message = parts[1].strip()

                                    st.session_state.simulation_context = {
                                        'status': 'active',
                                        'scenario': scenario_text
                                    }
                                    st.session_state.messages.append({"role": "assistant", "content": opening_message})
                                    st.rerun()
                                except Exception:
                                    error_msg = "Maaf, saya gagal membuat skenario. Coba topik lain."
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                    st.rerun()
            
            elif st.session_state.simulation_context['status'] == 'active':
                with st.expander("Lihat Skenario 📝", expanded=False):
                    st.warning(st.session_state.simulation_context['scenario'])

                if st.button("Selesaikan & Evaluasi Simulasi"):
                    st.session_state.simulation_context['status'] = 'evaluating'
                    st.rerun()

                if prompt := st.chat_input("Ketik respons Anda dalam simulasi..."):
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    with st.chat_message("assistant"):
                        with st.spinner("Alma (sebagai aktor) sedang berpikir..."):
                            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                            instruction = SIM_INTERACTION_PROMPT.format(
                                scenario=st.session_state.simulation_context['scenario'],
                                history=history_str,
                                user_role=user_role
                            )
                            response = self.get_response(user_query=prompt, instruction=instruction)
                            if response:
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.rerun()
            
            elif st.session_state.simulation_context['status'] == 'evaluating':
                with st.spinner("Simulasi selesai. Alma sedang menganalisis percakapan Anda..."):
                    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    instruction = SIM_EVALUATION_PROMPT.format(
                        scenario=st.session_state.simulation_context['scenario'],
                        history=history_str,
                        user_name=user_name, user_role=user_role
                    )
                    summary_query = "Tolong berikan evaluasi berdasarkan transkrip."
                    evaluation = self.get_response(user_query=summary_query, instruction=instruction)
                    
                    if evaluation:
                        st.success("Berikut adalah hasil evaluasi dari simulasi:")
                        st.markdown(evaluation)
                        st.session_state.messages.append({"role": "assistant", "content": evaluation})

                    st.session_state.simulation_context['status'] = 'finished'
                    st.rerun()

            elif st.session_state.simulation_context['status'] == 'finished':
                st.info("Sesi simulasi ini telah berakhir.")
                if st.button("🔁 Mulai Sesi Simulasi Baru"):
                    if "simulation_context" in st.session_state:
                        del st.session_state.simulation_context
                    if "quiz_state" in st.session_state:
                        del st.session_state.quiz_state
                    user_name = st.session_state.user_name
                    user_role = st.session_state.user_role
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"Sesi sebelumnya telah selesai. Halo lagi, {user_name}! Siap untuk memulai topik simulasi baru sebagai {user_role}?"}
                    ]
                    st.session_state.simulation_context = None
                    st.rerun()

    def run_ui(self):
        st.set_page_config(page_title="Alma Learn GoBIG", page_icon="🏸")
        st.title("Alma Learn GoBIG")
        st.write("Selamat datang! Mari tingkatkan kompetensi bersama Alma!")

        if "user_name" not in st.session_state:
            self.login()
        else:
            self.tampilan()

if __name__ == "__main__":
    alma = Chatbot(
        model=DEFAULT_MODEL
    )
    alma.run_ui()
